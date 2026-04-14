from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import random
import os
import time
from typing import Optional, Tuple

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
MODEL_LOCAL_PATH = os.environ.get("MODEL_LOCAL_PATH", "artifacts/model.h5").strip()
MODEL_REFRESH_SECONDS = _env_float("MODEL_REFRESH_SECONDS", 0.0)
MODEL_HTTP_TIMEOUT_S = _env_float("MODEL_HTTP_TIMEOUT_S", 30.0)

# Model state (kept in-memory); we’ll wire your real model in next.
_model_last_loaded_at: Optional[float] = None
_model_source_fingerprint: Optional[str] = None


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
    """
    Ensures the model file is present locally (if MODEL_URL is set).
    For now we don’t load TF/Keras here; we only manage the weights file lifecycle.
    """
    global _model_last_loaded_at, _model_source_fingerprint

    did_download, fp = _download_if_needed(
        url=MODEL_URL,
        local_path=MODEL_LOCAL_PATH,
        timeout_s=MODEL_HTTP_TIMEOUT_S,
        previous_fingerprint=_model_source_fingerprint,
    )

    # Mark “loaded” time if we downloaded or if this is the first successful ensure.
    if did_download or (_model_last_loaded_at is None and (not MODEL_URL or os.path.exists(MODEL_LOCAL_PATH))):
        _model_last_loaded_at = time.time()
        _model_source_fingerprint = fp or _model_source_fingerprint


@app.on_event("startup")
async def _startup() -> None:
    # Download model at startup (if configured). This keeps “it runs” behavior for demos.
    _maybe_refresh_model()

@app.post("/ingest")
async def ingest(data: EEGData):
    global latest_data
    latest_data = np.array(data.data)
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

    # Placeholder until we integrate your real model inference from Colab.
    seizure_probability = random.uniform(0.0, 1.0)
    return {"data": latest_data.tolist(), "seizure_probability": seizure_probability}
