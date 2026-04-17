import numpy as np
import pytest
import requests

from mock_streamer import generate_sample, post_with_retry


def test_generate_sample_scales_during_seizure():
    phase_offsets = np.zeros(4, dtype=float)
    normal = generate_sample(
        t=0.01,
        num_channels=4,
        frequency_hz=5.0,
        amplitude=2.0,
        phase_offsets=phase_offsets,
        seizure_active=False,
        seizure_multiplier=10.0,
    )
    seizure = generate_sample(
        t=0.01,
        num_channels=4,
        frequency_hz=5.0,
        amplitude=2.0,
        phase_offsets=phase_offsets,
        seizure_active=True,
        seizure_multiplier=10.0,
    )
    assert np.max(np.abs(seizure)) == pytest.approx(np.max(np.abs(normal)) * 10.0)


def test_post_with_retry_success(monkeypatch):
    class FakeResponse:
        status_code = 200

    monkeypatch.setattr("mock_streamer.requests.post", lambda *args, **kwargs: FakeResponse())
    status = post_with_retry("http://example.test/ingest", [0.1, 0.2], timeout_s=1.0)
    assert status == 200


def test_post_with_retry_handles_request_exception(monkeypatch):
    def _raise(*args, **kwargs):
        raise requests.RequestException("network down")

    monkeypatch.setattr("mock_streamer.requests.post", _raise)
    status = post_with_retry("http://example.test/ingest", [0.1, 0.2], timeout_s=1.0)
    assert status is None
