import asyncio

import numpy as np


def test_maybe_refresh_model_loads_session_when_downloaded(api_module, monkeypatch):
    sentinel_session = object()
    monkeypatch.setattr(api_module, "_download_if_needed", lambda **kwargs: (True, "fp-new"))
    monkeypatch.setattr(api_module.os.path, "exists", lambda _path: True)
    monkeypatch.setattr(api_module, "_load_onnx_session", lambda _path: sentinel_session)

    api_module._maybe_refresh_model()

    assert api_module._onnx_session is sentinel_session
    assert api_module._model_source_fingerprint == "fp-new"
    assert api_module._model_last_loaded_at is not None


def test_get_latest_returns_default_probability_when_model_not_ready(api_module, eeg_sample_23):
    api_module.latest_data = np.array(eeg_sample_23, dtype=float)
    response = asyncio.run(api_module.get_latest())
    assert response["data"] == eeg_sample_23
    assert response["seizure_probability"] == 0.0
