import pytest
import numpy as np
from unittest.mock import patch


# ── helpers ──────────────────────────────────────────────────────────────

def _make_raw(n_channels=3, n_seconds=10, sfreq=256.0):
    mne = pytest.importorskip("mne", reason="mne not installed")
    data = np.random.randn(n_channels, int(n_seconds * sfreq)) * 1e-5
    info = mne.create_info(
        ch_names=[f"EEG{i}" for i in range(n_channels)],
        sfreq=sfreq,
        ch_types="eeg",
    )
    return mne.io.RawArray(data, info, verbose=False)


# ── parse_seizure_summary tests (no mne needed, always run in CI) ─────────

def test_parse_seizure_summary_extracts_times(tmp_path):
    content = (
        "File Name: chb01_03.edf\n"
        "Number of Seizures in File: 1\n"
        "Seizure Start Time: 2996 seconds\n"
        "Seizure End Time: 3036 seconds\n"
    )
    summary = tmp_path / "chb01-summary.txt"
    summary.write_text(content)

    from build_local_database import parse_seizure_summary

    result = parse_seizure_summary(str(summary))
    assert result["chb01_03.edf"] == [(2996, 3036)]


def test_parse_seizure_summary_no_seizures(tmp_path):
    content = "File Name: chb01_01.edf\nNumber of Seizures in File: 0\n"
    summary = tmp_path / "chb01-summary.txt"
    summary.write_text(content)

    from build_local_database import parse_seizure_summary

    result = parse_seizure_summary(str(summary))
    assert result["chb01_01.edf"] == []


# ── preprocess_and_window tests (skipped in CI if mne absent) ─────────────

def test_preprocess_and_window_shapes():
    from build_local_database import preprocess_and_window

    raw = _make_raw(n_channels=3, n_seconds=6, sfreq=256.0)
    X, y = preprocess_and_window(
        raw,
        seizure_times=[],
        best_indices=[0, 1],
        target_hz=128,
        win_sec=2,
    )
    assert X.shape[1] == 2  # 2 channels selected
    assert X.shape[2] == 256  # 128 Hz × 2 s
    assert len(X) == len(y)


def test_preprocess_and_window_labels_seizure_window():
    from build_local_database import preprocess_and_window

    raw = _make_raw(n_channels=3, n_seconds=6, sfreq=256.0)
    # seizure spans [0, 3) — first window (0-2s) overlaps, second (2-4s) overlaps, third (4-6s) doesn't
    X, y = preprocess_and_window(
        raw,
        seizure_times=[(0, 3)],
        best_indices=[0, 1],
        target_hz=128,
        win_sec=2,
    )
    assert y[0] == 1  # overlaps seizure [0,3)
    assert y[2] == 0  # window [4,6) is outside


# ── main() test (no mne needed — empty patient dir has no .edf files) ─────

def test_main_skips_missing_summary(tmp_path):
    patient_dir = tmp_path / "chb01"
    patient_dir.mkdir()

    with patch("build_local_database.LOCAL_DATA_DIR", str(tmp_path)), patch(
        "build_local_database.HDF5_DATABASE_PATH", str(tmp_path / "db.h5")
    ), patch("build_local_database._load_config") as load_config:
        load_config.return_value = {
            "paths": {
                "local_data_dir": str(tmp_path),
                "hdf5_database_path": str(tmp_path / "db.h5"),
            },
            "signal_processing": {
                "window_sec": 2,
                "target_hz": 128,
                "top_n_channels": 2,
            },
        }

        from build_local_database import main

        main()  # should not raise


def test_normalize_channel_name():
    from build_local_database import normalize_channel_name
    assert normalize_channel_name("FP1-F7") == "FP1F7"
    assert normalize_channel_name("FP1-F7.") == "FP1F7"
    assert normalize_channel_name("FP1 F7") == "FP1F7"
    assert normalize_channel_name("FP1-F7-Ref") == "FP1F7"


def test_align_channels():
    from build_local_database import align_channels
    raw_names = ["FP1-F7", "F7-T7", "T7-P7", "FP1-F3-Ref"]
    assert align_channels(raw_names, [0, 2]) == [0, 2]
    assert align_channels(raw_names, ["FP1-F7", "FP1-F3"]) == [0, 3]

    with pytest.raises(ValueError):
        align_channels(raw_names, ["INVALID_CHANNEL"])


def test_preprocess_and_window_under_sampling():
    from build_local_database import preprocess_and_window
    raw = _make_raw(n_channels=3, n_seconds=3000, sfreq=256.0)

    # 1 seizure at 2000-2050s
    X, y, meta = preprocess_and_window(
        raw,
        seizure_times=[(2000, 2050)],
        best_indices=[0, 1],
        target_hz=128,
        win_sec=2,
        sampling_config={"enabled": True, "negative_ratio": 2.0, "random_seed": 42},
        return_meta=True,
    )
    # Seizure is 50s. At 2s window -> 25 positive windows.
    # negative ratio = 2.0 -> 50 negative windows.
    # Total windows = 75
    assert len(y) == 75
    assert np.sum(y == 1) == 25
    assert np.sum(y == 0) == 50
    assert len(meta["seizure_windows"]) == 25
    assert len(meta["pre_ictal_windows"]) == 25
    assert len(meta["baseline_windows"]) == 25

