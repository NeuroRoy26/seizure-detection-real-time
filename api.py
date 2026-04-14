from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import os
import time
from typing import Optional, Tuple
from collections import deque

from model_fetch import download_if_needed

app = FastAPI()

class EEGData(BaseModel):
    data: list[float]

latest_data = None


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    try:
        return float(v)
    except ValueError:
        return default


MODEL_URL = os.environ.get("MODEL_URL", "").strip()
MODEL_LOCAL_PATH = os.environ.get("MODEL_LOCAL_PATH", "artifacts/model.onnx").strip()
MODEL_REFRESH_SECONDS = _env_float("MODEL_REFRESH_SECONDS", 0.0)
MODEL_HTTP_TIMEOUT_S = _env_float("MODEL_HTTP_TIMEOUT_S", 30.0)

# Inference/windowing config (matches your notebook defaults)
SFREQ_HZ = int(os.environ.get("SFREQ_HZ", "128"))
WINDOW_SECONDS = float(os.environ.get("WINDOW_SECONDS", "2"))
WINDOW_SAMPLES = int(max(1, round(SFREQ_HZ * WINDOW_SECONDS)))
INCOMING_CHANNELS = int(os.environ.get("INCOMING_CHANNELS", "23"))

# Notebook logic: standardize to first 20 channels, then pick best 10 indices.
STANDARDIZE_CHANNELS = int(os.environ.get("STANDARDIZE_CHANNELS", "20"))
BEST_CHANNELS = [0, 1, 5, 2, 13, 18, 14, 8, 19, 16]

# Model state (kept in-memory)
_model_last_loaded_at: Optional[float] = None
_model_source_fingerprint: Optional[str] = None
_onnx_session = None

# Rolling buffer of incoming samples (each sample is shape: (INCOMING_CHANNELS,))
_sample_buffer: deque[np.ndarray] = deque(maxlen=WINDOW_SAMPLES)


def _load_onnx_session(model_path: str):
    import onnxruntime as ort

    # Use CPU provider for portability.
    return ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])


def _predict_seizure_probability_from_buffer() -> Optional[float]:
    """
    Converts the rolling buffer into (channels, time), slices channels per notebook logic,
    and returns P(seizure) from the PyTorch model.
    """
    global _onnx_session
    if _onnx_session is None:
        return None
    if len(_sample_buffer) < WINDOW_SAMPLES:
        return None

    window_time_major = np.stack(_sample_buffer, axis=0)  # (time, channels)
    window = window_time_major.T  # (channels, time)

    if window.shape[0] < STANDARDIZE_CHANNELS:
        return None

    window = window[:STANDARDIZE_CHANNELS, :]  # (20, time)
    window = window[BEST_CHANNELS, :]  # (10, time)

    x = window.astype(np.float32)[None, :, :]  # (1, 10, time)

    input_name = _onnx_session.get_inputs()[0].name
    outputs = _onnx_session.run(None, {input_name: x})
    logits = outputs[0]  # expected shape: (1, 2)
    logits = np.asarray(logits, dtype=np.float32)[0]
    # softmax
    exps = np.exp(logits - np.max(logits))
    probs = exps / np.sum(exps)
    return float(probs[1])


def _download_if_needed(
    *,
    url: str,
    local_path: str,
    timeout_s: float,
    previous_fingerprint: Optional[str],
) -> Tuple[bool, Optional[str]]:
    return download_if_needed(
        url=url,
        local_path=local_path,
        timeout_s=timeout_s,
        previous_fingerprint=previous_fingerprint,
    )


def _maybe_refresh_model() -> None:
    global _model_last_loaded_at, _model_source_fingerprint, _onnx_session

    did_download, fp = _download_if_needed(
        url=MODEL_URL,
        local_path=MODEL_LOCAL_PATH,
        timeout_s=MODEL_HTTP_TIMEOUT_S,
        previous_fingerprint=_model_source_fingerprint,
    )

    # Load (or reload) ONNX if we downloaded new weights, or if not loaded yet.
    if os.path.exists(MODEL_LOCAL_PATH) and (_onnx_session is None or did_download):
        try:
            _onnx_session = _load_onnx_session(MODEL_LOCAL_PATH)
        except Exception:
            # Keep serving the API even if model load fails; we'll report via /health later if needed.
            _onnx_session = None

    # Mark “loaded” time if we downloaded, loaded for first time, or if this is first ensure.
    if did_download or _model_last_loaded_at is None:
        _model_last_loaded_at = time.time()
        _model_source_fingerprint = fp or _model_source_fingerprint


@app.on_event("startup")
async def _startup() -> None:
    # Download model at startup (if configured). This keeps “it runs” behavior for demos.
    _maybe_refresh_model()

@app.post("/ingest")
async def ingest(data: EEGData):
    global latest_data
    sample = np.array(data.data, dtype=float)
    latest_data = sample
    if sample.shape[0] >= INCOMING_CHANNELS:
        sample = sample[:INCOMING_CHANNELS]
    _sample_buffer.append(sample)
    return {"message": "Data ingested successfully"}

@app.get("/latest")
async def get_latest():
    global latest_data
    if latest_data is None:
        return {"message": "No data available"}

    # Optional periodic refresh of the weights file.
    if MODEL_URL and MODEL_REFRESH_SECONDS and MODEL_REFRESH_SECONDS > 0:
        now = time.time()
        if _model_last_loaded_at is None or (now - _model_last_loaded_at) >= MODEL_REFRESH_SECONDS:
            _maybe_refresh_model()

    seizure_probability = _predict_seizure_probability_from_buffer()
    if seizure_probability is None:
        # Not enough samples yet (or model not configured); return a stable default.
        seizure_probability = 0.0
    return {"data": latest_data.tolist(), "seizure_probability": seizure_probability}
