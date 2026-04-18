import numpy as np
import pytest


def test_predict_builds_expected_model_input(api_module, eeg_window_256x23):
    class FakeInput:
        name = "eeg"

    class FakeSession:
        def __init__(self):
            self.last_feed = None

        def get_inputs(self):
            return [FakeInput()]

        def run(self, _outputs, feed_dict):
            self.last_feed = feed_dict
            return [np.array([[1.0, 3.0]], dtype=np.float32)]

    fake_session = FakeSession()
    api_module._onnx_session = fake_session
    for row in eeg_window_256x23:
        api_module._sample_buffer.append(row)

    prob = api_module._predict_seizure_probability_from_buffer()
    expected = np.exp(3.0) / (np.exp(1.0) + np.exp(3.0))
    assert prob == pytest.approx(expected, rel=1e-5)

    model_input = fake_session.last_feed["eeg"]
    assert model_input.shape == (1, len(api_module.BEST_CHANNELS), api_module.WINDOW_SAMPLES)


def test_predict_returns_none_when_not_enough_channels(api_module):
    class FakeInput:
        name = "eeg"

    class FakeSession:
        def get_inputs(self):
            return [FakeInput()]

        def run(self, _outputs, _feed_dict):
            return [np.array([[0.0, 1.0]], dtype=np.float32)]

    api_module._onnx_session = FakeSession()
    for _ in range(api_module.WINDOW_SAMPLES):
        api_module._sample_buffer.append(np.zeros(10, dtype=float))

    assert api_module._predict_seizure_probability_from_buffer() is None
