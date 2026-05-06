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


def test_post_with_retry_non_200(monkeypatch):
    class FakeResponse:
        status_code = 503

    monkeypatch.setattr("mock_streamer.requests.post", lambda *a, **kw: FakeResponse())
    status = post_with_retry("http://example.test/ingest", [0.1], timeout_s=1.0)
    assert status == 503


def test_post_with_retry_handles_request_exception(monkeypatch):
    def _raise(*args, **kwargs):
        raise requests.RequestException("network down")

    monkeypatch.setattr("mock_streamer.requests.post", _raise)
    status = post_with_retry("http://example.test/ingest", [0.1, 0.2], timeout_s=1.0)
    assert status is None


def test_generate_sample_zero_amplitude():
    offsets = np.zeros(2, dtype=float)
    result = generate_sample(
        t=1.0,
        num_channels=2,
        frequency_hz=10.0,
        amplitude=0.0,
        phase_offsets=offsets,
        seizure_active=True,
        seizure_multiplier=100.0,
    )
    assert result == [0.0, 0.0]


def test_generate_sample_channel_count():
    offsets = np.zeros(7, dtype=float)
    result = generate_sample(
        t=0.0,
        num_channels=7,
        frequency_hz=1.0,
        amplitude=1.0,
        phase_offsets=offsets,
        seizure_active=False,
        seizure_multiplier=5.0,
    )
    assert len(result) == 7


def test_main_parses_custom_args():
    from unittest.mock import patch

    import mock_streamer

    with patch("mock_streamer.post_with_retry", return_value=200) as mock_post, patch(
        "mock_streamer.time.sleep", side_effect=StopIteration
    ):
        try:
            mock_streamer.main.__globals__["sys"] = __import__("sys")
            import sys

            sys.argv = ["mock_streamer.py", "--channels", "5", "--hz", "1.0"]
            mock_streamer.main()
        except StopIteration:
            pass

    args, kwargs = mock_post.call_args
    assert len(args[1]) == 5
