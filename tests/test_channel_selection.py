import importlib
import sys
from types import ModuleType, SimpleNamespace

import numpy as np


def _fake_scipy(monkeypatch, *, capture=None):
    """
    Provide minimal scipy.signal / scipy.stats shims so channel_selection can be
    imported and executed in CI without the real SciPy dependency.
    """

    def butter(order, wn, btype=None):
        if capture is not None:
            capture["butter"] = {"order": order, "wn": wn, "btype": btype}
        # The real butter returns filter coefficients; our filtfilt stub ignores them.
        return np.array([1.0]), np.array([1.0])

    def filtfilt(b, a, x, axis=-1):
        if capture is not None:
            capture.setdefault("filtfilt_shapes", []).append(np.asarray(x).shape)
        # Identity "filter" (keeps tests deterministic and fast).
        return np.asarray(x)

    def medfilt(x, kernel_size=3):
        # Very small/fast median filter replacement (good enough for our unit tests).
        x = np.asarray(x)
        if x.size == 0:
            return x
        k = int(kernel_size)
        if k <= 1:
            return x
        if k % 2 == 0:
            k += 1
        pad = k // 2
        xp = np.pad(x, (pad, pad), mode="edge")
        out = np.empty_like(x, dtype=float)
        for i in range(x.size):
            out[i] = np.median(xp[i : i + k])
        return out

    def median_abs_deviation(x, scale="normal"):
        x = np.asarray(x, dtype=float)
        med = np.median(x)
        mad = np.median(np.abs(x - med))
        if scale == "normal":
            # Consistent with SciPy's scale="normal" (approx. std for normal dist).
            mad = mad * 1.4826
        return mad

    scipy = ModuleType("scipy")
    signal = ModuleType("scipy.signal")
    stats = ModuleType("scipy.stats")

    signal.butter = butter
    signal.filtfilt = filtfilt
    signal.medfilt = medfilt
    stats.median_abs_deviation = median_abs_deviation

    scipy.signal = signal
    scipy.stats = stats

    monkeypatch.setitem(sys.modules, "scipy", scipy)
    monkeypatch.setitem(sys.modules, "scipy.signal", signal)
    monkeypatch.setitem(sys.modules, "scipy.stats", stats)


def _import_channel_selection(monkeypatch, *, capture=None):
    _fake_scipy(monkeypatch, capture=capture)
    sys.modules.pop("channel_selection", None)
    return importlib.import_module("channel_selection")


def test_calculate_channel_stability_ranks_stable_high_amplitude(monkeypatch):
    """
    Score = mean_rms / (cv_rms + eps). Channels with higher mean RMS and low
    variability across sessions should rank highest.
    """
    ch = _import_channel_selection(monkeypatch)

    sfreq = 256.0
    n_channels = 4
    channel_names = [f"Ch{i}" for i in range(n_channels)]
    t = np.linspace(0.0, 2.0, int(2.0 * sfreq), endpoint=False)

    # Channel 0: stable amplitude 2.0 across sessions (best)
    # Channel 1: stable amplitude 1.0 across sessions (second best)
    # Channel 2: unstable amplitude varies a lot across sessions (penalized by CV)
    # Channel 3: near-zero
    amps_by_session = [
        (2.0, 1.0, 0.5, 0.01),
        (2.0, 1.0, 3.0, 0.01),
        (2.0, 1.0, 0.2, 0.01),
    ]
    sessions = []
    for a0, a1, a2, a3 in amps_by_session:
        s0 = a0 * np.sin(2 * np.pi * 5.0 * t)
        s1 = a1 * np.sin(2 * np.pi * 5.0 * t + 0.1)
        s2 = a2 * np.sin(2 * np.pi * 5.0 * t + 0.2)
        s3 = a3 * np.sin(2 * np.pi * 5.0 * t + 0.3)
        sessions.append(np.vstack([s0, s1, s2, s3]).astype(float))

    result = ch.calculate_channel_stability(
        sessions_data=sessions,
        sfreq=sfreq,
        channel_names=channel_names,
        top_n=2,
    )

    assert result["best_indices"][:2] == [0, 1]
    assert result["best_names"][:2] == ["Ch0", "Ch1"]
    assert len(result["scores"]) == n_channels


def test_calculate_channel_stability_trims_long_sessions(monkeypatch):
    capture = {}
    ch = _import_channel_selection(monkeypatch, capture=capture)

    sfreq = 256.0
    channel_names = ["A", "B"]
    trim_samples = int(round(1.5 * sfreq))
    # Long enough to trim.
    n_samples = (2 * trim_samples) + 100
    raw = np.random.randn(2, n_samples).astype(float)

    ch.calculate_channel_stability([raw], sfreq=sfreq, channel_names=channel_names, top_n=1)

    # filtfilt should see trimmed length: n_samples - 2*trim_samples
    assert capture["filtfilt_shapes"][0] == (2, n_samples - 2 * trim_samples)


def test_calculate_channel_stability_no_trim_for_short_sessions(monkeypatch):
    capture = {}
    ch = _import_channel_selection(monkeypatch, capture=capture)

    sfreq = 256.0
    channel_names = ["A", "B"]
    trim_samples = int(round(1.5 * sfreq))
    # Too short to trim.
    n_samples = (2 * trim_samples) - 1
    raw = np.random.randn(2, n_samples).astype(float)

    ch.calculate_channel_stability([raw], sfreq=sfreq, channel_names=channel_names, top_n=1)
    assert capture["filtfilt_shapes"][0] == (2, n_samples)


def test_calculate_channel_stability_respects_nyquist_high_cut(monkeypatch):
    """
    For low sfreq, high_cut = min(50, nyquist-1) should be used in butter().
    """
    capture = {}
    ch = _import_channel_selection(monkeypatch, capture=capture)

    sfreq = 60.0  # nyquist=30 -> high_cut should become 29.0
    channel_names = ["A", "B"]
    raw = np.random.randn(2, int(10 * sfreq)).astype(float)

    ch.calculate_channel_stability([raw], sfreq=sfreq, channel_names=channel_names, top_n=1)

    wn = capture["butter"]["wn"]
    nyquist = sfreq / 2.0
    expected_low = 1.0 / nyquist
    expected_high = (nyquist - 1.0) / nyquist
    assert wn[0] == expected_low
    assert wn[1] == expected_high


def test_calculate_channel_stability_top_n_larger_than_channels(monkeypatch):
    ch = _import_channel_selection(monkeypatch)
    sfreq = 256.0
    channel_names = ["A", "B", "C"]
    raw = np.random.randn(3, int(5 * sfreq)).astype(float)
    result = ch.calculate_channel_stability([raw], sfreq=sfreq, channel_names=channel_names, top_n=10)
    # Should just return as many as exist.
    assert len(result["best_indices"]) == 3
    assert len(result["best_names"]) == 3

