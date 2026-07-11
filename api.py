from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import os
import time
import yaml
from typing import Optional, Tuple
from collections import deque

from model_fetch import download_if_needed
from src.llm_client import LLMClient

# Load Master Configuration if present
config = {}
if os.path.exists("config.yaml"):
    try:
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
    except Exception:
        pass


app = FastAPI()
llm_client = LLMClient()

class EEGData(BaseModel):
    data: list[float]

class LLMReportRequest(BaseModel):
    seizure_probability: float
    active_state: str
    features: Optional[list[dict]] = None

class LLMExplainRequest(BaseModel):
    channel_idx: int
    features: dict[str, float]

class SimulatorState(BaseModel):
    state: str

class EEGWindow(BaseModel):
    data: list[list[float]]

latest_data = None
_simulator_state = "patient_normal"


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    try:
        return float(v)
    except ValueError:
        return default


MODEL_URL = os.environ.get("MODEL_URL", "").strip()
config_onnx_path = config.get("paths", {}).get("target_onnx_path")
default_local_path = config_onnx_path if (config_onnx_path and os.path.exists(config_onnx_path)) else "seizure_detector_mobilenetv2.onnx"
MODEL_LOCAL_PATH = os.environ.get("MODEL_LOCAL_PATH", default_local_path).strip()
MODEL_REFRESH_SECONDS = _env_float("MODEL_REFRESH_SECONDS", 0.0)
MODEL_HTTP_TIMEOUT_S = _env_float("MODEL_HTTP_TIMEOUT_S", 30.0)

# Inference/windowing config (matches master config fallback)
SFREQ_HZ = int(os.environ.get("SFREQ_HZ", str(int(config.get("signal_processing", {}).get("target_hz", 128)))))
WINDOW_SECONDS = float(os.environ.get("WINDOW_SECONDS", str(config.get("signal_processing", {}).get("window_sec", 2.0))))
WINDOW_SAMPLES = int(max(1, round(SFREQ_HZ * WINDOW_SECONDS)))
INCOMING_CHANNELS = int(os.environ.get("INCOMING_CHANNELS", "23"))

# Notebook logic: standardize to first 20 channels, then pick best 10 indices.
STANDARDIZE_CHANNELS = int(os.environ.get("STANDARDIZE_CHANNELS", "20"))
BEST_CHANNELS = config.get("signal_processing", {}).get("best_indices", [0, 1, 5, 2, 13, 18, 14, 8, 19, 16])


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

    x = window.astype(np.float32)[None, :, :, None]  # (1, 10, time, 1)

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
    
    # Pre-populate rolling buffer with realistic baseline normal EEG signals
    # to avoid prediction delays at startup.
    phase_offsets = np.array([i * 0.23 for i in range(INCOMING_CHANNELS)], dtype=float)
    for i in range(WINDOW_SAMPLES):
        t = i / SFREQ_HZ
        delta = 25.0 * np.sin(2 * np.pi * 1.5 * t + phase_offsets)
        theta = 15.0 * np.sin(2 * np.pi * 5.0 * t + phase_offsets * 1.3)
        alpha = 18.0 * np.sin(2 * np.pi * 10.0 * t + phase_offsets * 0.7)
        beta = 8.0 * np.sin(2 * np.pi * 20.0 * t + phase_offsets * 2.1)
        noise = np.random.normal(0, 15.0, size=(INCOMING_CHANNELS,))
        sample = (delta + theta + alpha + beta + noise) * 0.8
        _sample_buffer.append(sample)
        
    global latest_data
    latest_data = _sample_buffer[-1]

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
        
    buffer_list = [sample.tolist() for sample in _sample_buffer]
    return {"data": buffer_list, "seizure_probability": seizure_probability}

@app.get("/simulator/state")
async def get_simulator_state():
    return {"state": _simulator_state}

@app.post("/simulator/state")
async def set_simulator_state(data: SimulatorState):
    global _simulator_state
    if data.state in ["normal", "seizure", "patient_normal", "patient_seizure"]:
        _simulator_state = data.state
        return {"message": "Simulator state updated successfully", "state": _simulator_state}
    return {"error": "Invalid state. Choose 'normal', 'seizure', 'patient_normal', or 'patient_seizure'."}

@app.post("/classify_window")
async def classify_window(window: EEGWindow):
    global _onnx_session
    if _onnx_session is None:
        return {"error": "Model not loaded"}
    
    x = np.array(window.data, dtype=np.float32)[None, :, :, None]
    
    input_name = _onnx_session.get_inputs()[0].name
    outputs = _onnx_session.run(None, {input_name: x})
    logits = outputs[0]
    logits = np.asarray(logits, dtype=np.float32)[0]
    
    # Softmax
    exps = np.exp(logits - np.max(logits))
    probs = exps / np.sum(exps)
    return {"seizure_probability": float(probs[1])}

@app.get("/llm/health")
async def get_llm_health():
    return llm_client.check_health()

@app.post("/llm/report")
async def generate_llm_report(req: LLMReportRequest):
    if not llm_client.enabled:
        return {"error": "LLM features are disabled in config.yaml."}
    report = llm_client.generate_report(
        seizure_probability=req.seizure_probability,
        active_state=req.active_state,
        features_list=req.features
    )
    return {"report": report}

@app.post("/llm/explain")
async def explain_llm_features(req: LLMExplainRequest):
    if not llm_client.enabled:
        return {"error": "LLM features are disabled in config.yaml."}
    explanation = llm_client.explain_features(
        channel_idx=req.channel_idx,
        features=req.features
    )
    return {"explanation": explanation}

class LLMChatRequest(BaseModel):
    prompt: str

@app.post("/llm/chat")
async def explain_llm_chat(req: LLMChatRequest):
    if not llm_client.enabled:
        return {"error": "LLM features are disabled in config.yaml."}
    prompt = f"""<system>
You are an expert clinical neurophysiologist and AI engineering assistant. 
You are answering questions from a doctor or researcher who is using the Real-Time EEG Seizure Detection dashboard.
Keep your answers professional, concise, and clinically relevant.
Do not include conversational preambles.
</system>

User: {req.prompt}

Assistant:"""
    response_text = llm_client._query_api(prompt, max_tokens=400, temperature=0.7)
    return {"response": response_text}
