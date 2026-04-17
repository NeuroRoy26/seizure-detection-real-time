import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TEST_SFREQ_HZ = "128"
TEST_WINDOW_SECONDS = "2"
TEST_INCOMING_CHANNELS = "23"
TEST_STANDARDIZE_CHANNELS = "20"


@pytest.fixture
def eeg_sample_23():
    return np.linspace(-1.0, 1.0, 23, dtype=float).tolist()


@pytest.fixture
def eeg_payload(eeg_sample_23):
    return {"data": eeg_sample_23}


@pytest.fixture
def eeg_window_256x23():
    t = np.linspace(0, 2.0, 256, endpoint=False)
    base = np.sin(2 * np.pi * 5.0 * t)
    return np.stack([base + (i * 0.01) for i in range(23)], axis=1).astype(np.float32)


@pytest.fixture
def api_module(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_URL", "")
    monkeypatch.setenv("MODEL_LOCAL_PATH", str(tmp_path / "model.onnx"))
    monkeypatch.setenv("MODEL_REFRESH_SECONDS", "0")
    monkeypatch.setenv("MODEL_HTTP_TIMEOUT_S", "1")
    monkeypatch.setenv("SFREQ_HZ", TEST_SFREQ_HZ)
    monkeypatch.setenv("WINDOW_SECONDS", TEST_WINDOW_SECONDS)
    monkeypatch.setenv("INCOMING_CHANNELS", TEST_INCOMING_CHANNELS)
    monkeypatch.setenv("STANDARDIZE_CHANNELS", TEST_STANDARDIZE_CHANNELS)

    import api

    module = importlib.reload(api)
    module.latest_data = None
    module._sample_buffer.clear()
    module._onnx_session = None
    module._model_last_loaded_at = None
    module._model_source_fingerprint = None
    return module
