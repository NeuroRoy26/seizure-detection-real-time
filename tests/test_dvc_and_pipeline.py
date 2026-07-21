import os
import configparser
import yaml
import pytest
import numpy as np
import tempfile
from unittest.mock import patch, MagicMock

# ── 1. The "DVC Configuration" Test ──────────────────────────────────────────

def test_dvc_configuration():
    """
    Verifies that DVC is correctly initialized, configuration file exists,
    and remotes (gdrive-remote and s3-remote) are configured with correct URLs and parameters.
    """
    dvc_config_path = os.path.join(".dvc", "config")
    assert os.path.isfile(dvc_config_path), "DVC configuration file '.dvc/config' does not exist."

    # Parse .dvc/config in INI format
    config = configparser.ConfigParser()
    config.read(dvc_config_path)

    # Find the sections matching remote names (ignoring formatting quotes/brackets in key names)
    gdrive_section = None
    s3_section = None
    for section in config.sections():
        if "gdrive-remote" in section:
            gdrive_section = section
        elif "s3-remote" in section:
            s3_section = section

    assert gdrive_section is not None, "gdrive-remote section not found in DVC config."
    assert s3_section is not None, "s3-remote section not found in DVC config."

    # Check configurations
    assert config[gdrive_section]['url'] == 'gdrive://1ISmGMyjB40hIU9SKNKIfw9FWvyyKK2dh', "gdrive-remote URL is incorrect."
    assert config[s3_section]['url'] == 's3://seizure-detection--s-roy', "s3-remote URL is incorrect."
    assert config[s3_section]['region'] == 'eu-north-1', "s3-remote region is incorrect."

    # Check datasets.dvc outputs map to gdrive-remote
    assert os.path.isfile("datasets.dvc"), "datasets.dvc file does not exist."
    with open("datasets.dvc", "r") as f:
        datasets_dvc = yaml.safe_load(f)
    assert "outs" in datasets_dvc, "No outs section in datasets.dvc"
    assert len(datasets_dvc["outs"]) == 1, "Expected exactly one entry in outs section of datasets.dvc"
    assert datasets_dvc["outs"][0]["path"] == "datasets", "datasets.dvc does not track 'datasets' path."
    assert datasets_dvc["outs"][0]["remote"] == "gdrive-remote", "datasets.dvc does not route to gdrive-remote."

    # Check seizure_detector_mobilenetv2.onnx.dvc outputs map to s3-remote
    assert os.path.isfile("seizure_detector_mobilenetv2.onnx.dvc"), "seizure_detector_mobilenetv2.onnx.dvc file does not exist."
    with open("seizure_detector_mobilenetv2.onnx.dvc", "r") as f:
        latest_onnx_dvc = yaml.safe_load(f)
    assert "outs" in latest_onnx_dvc, "No outs section in seizure_detector_mobilenetv2.onnx.dvc"
    assert len(latest_onnx_dvc["outs"]) == 1, "Expected exactly one entry in outs section of seizure_detector_mobilenetv2.onnx.dvc"
    assert latest_onnx_dvc["outs"][0]["path"] == "seizure_detector_mobilenetv2.onnx", "seizure_detector_mobilenetv2.onnx.dvc does not track 'seizure_detector_mobilenetv2.onnx' path."
    assert latest_onnx_dvc["outs"][0]["remote"] == "s3-remote", "seizure_detector_mobilenetv2.onnx.dvc does not route to s3-remote."


# ── 2. The "Data Presence" Test ───────────────────────────────────────────────

def test_data_presence():
    """
    Verifies that the dataset configuration is present and that the datasets folder
    is ignored by Git while tracked by DVC. Also verifies the structure of the
    datasets folder if it exists locally.
    """
    # 1. Verify datasets.dvc file exists
    assert os.path.isfile("datasets.dvc"), "datasets.dvc metadata file is missing."

    # 2. Verify gitignore ignores the datasets directory
    gitignore_path = ".gitignore"
    assert os.path.isfile(gitignore_path), ".gitignore is missing."
    with open(gitignore_path, "r") as f:
        gitignore_content = f.read()

    assert any(pattern in gitignore_content for pattern in ["datasets", "datasets/", "/datasets"]), (
        "datasets directory is not ignored in .gitignore."
    )

    # 3. If datasets/ directory exists locally, verify its basic structure is valid
    if os.path.isdir("datasets"):
        files = os.listdir("datasets")
        assert len(files) > 0, "datasets/ directory is empty."
        
        # Verify it contains typical files/dirs expected for CHB-MIT (e.g. SUBJECT-INFO or patient subdirs)
        has_subdirs = any(os.path.isdir(os.path.join("datasets", f)) for f in files)
        has_metadata = any(f in ["SUBJECT-INFO", "RECORDS", "RECORDS-WITH-SEIZURES", "SHA256SUMS.txt"] for f in files)
        assert has_subdirs or has_metadata, "datasets/ folder does not contain expected CHB-MIT data files or subfolders."


# ── 3. The "Mock Data Pipeline" Test ──────────────────────────────────────────

def test_mock_data_pipeline(tmp_path, monkeypatch):
    """
    Simulates the end-to-end training data ETL and ONNX model training pipeline using dummy files.
    """
    mne = pytest.importorskip("mne", reason="mne not installed")
    h5py = pytest.importorskip("h5py", reason="h5py not installed")
    tf = pytest.importorskip("tensorflow", reason="tensorflow not installed")

    # 1. Create a dummy patient directory structure and mock files
    local_data_dir = tmp_path / "datasets"
    patient_dir = local_data_dir / "chb01"
    patient_dir.mkdir(parents=True)

    # Empty dummy EDF file
    edf_file = patient_dir / "chb01_01.edf"
    edf_file.write_bytes(b"")

    # Dummy summary text detailing a seizure
    summary_content = (
        "File Name: chb01_01.edf\n"
        "Number of Seizures in File: 1\n"
        "Seizure Start Time: 1 seconds\n"
        "Seizure End Time: 3 seconds\n"
    )
    summary_file = patient_dir / "chb01-summary.txt"
    summary_file.write_text(summary_content)

    hdf5_db_path = tmp_path / "train_database.h5"
    target_onnx_path = tmp_path / "seizure_detector_mobilenetv2.onnx"

    # Config dict mock
    mock_config = {
        "paths": {
            "local_data_dir": str(local_data_dir),
            "hdf5_database_path": str(hdf5_db_path),
            "target_onnx_path": str(target_onnx_path),
        },
        "signal_processing": {
            "window_sec": 2.0,
            "target_hz": 128.0,
            "top_n_channels": 10,
        },
        "training": {
            "batch_size": 2,
            "epochs": 1,
            "learning_rate": 0.001,
            "test_split": 0.5,
        }
    }

    # 2. Patch build_local_database configs & functions
    import build_local_database
    import local_train_onnx

    monkeypatch.setattr(build_local_database, "_load_config", lambda *args, **kwargs: mock_config)
    monkeypatch.setattr(local_train_onnx, "_load_config", lambda *args, **kwargs: mock_config)

    # Mock the mne raw reader to return a raw-like object
    monkeypatch.setattr(mne.io, "read_raw_edf", lambda *args, **kwargs: MagicMock())

    # Mock preprocess_and_window to return deterministic, balanced datasets to satisfy stratification constraints
    # (We need at least 2 samples per class for stratify split with 50% split size -> e.g. 4 class 0, 4 class 1)
    mock_X = np.random.randn(8, 10, 256).astype(np.float32)
    mock_y = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int32)
    monkeypatch.setattr(build_local_database, "preprocess_and_window", lambda *args, **kwargs: (mock_X, mock_y))

    mock_stability = {
        "best_indices": list(range(10)),
        "best_names": [f"EEG{i}" for i in range(10)]
    }
    import src.data.channel_selection as channel_selection
    monkeypatch.setattr(channel_selection, "calculate_channel_stability", lambda *args, **kwargs: mock_stability)

    # 3. Run build_local_database main() to process ETL
    build_local_database.main()

    # Verify HDF5 database was created and has the correct shape
    assert os.path.isfile(hdf5_db_path), "HDF5 database file was not created by ETL."
    with h5py.File(hdf5_db_path, "r") as h5f:
        x_key = "raw_signals/X" if "raw_signals/X" in h5f else "X"
        y_key = "raw_signals/y" if "raw_signals/y" in h5f else "y"
        assert x_key in h5f, "Dataset X not found in HDF5 file."
        assert y_key in h5f, "Dataset y not found in HDF5 file."
        assert h5f[x_key].shape == (8, 10, 256)
        assert len(h5f[y_key]) == 8

    # Define a clean Keras-compliant layer subclass to act as our Mock MobileNetV2 base model.
    # This prevents the need to download large pre-trained weights in unit tests.
    class MockMobileNetV2(tf.keras.layers.Layer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.resizer = tf.keras.layers.Resizing(7, 7)
            self.conv = tf.keras.layers.Conv2D(1280, (1, 1))
            self.trainable = False

        def call(self, inputs, training=None):
            return self.conv(self.resizer(inputs))

    # 4. Run local_train_onnx main() to run model training & ONNX export
    with patch("tensorflow.keras.applications.MobileNetV2") as mock_mobilenet:
        mock_mobilenet.return_value = MockMobileNetV2()
        local_train_onnx.main()

    # Verify ONNX model was successfully exported
    assert os.path.isfile(target_onnx_path), "ONNX model was not created by training pipeline."
