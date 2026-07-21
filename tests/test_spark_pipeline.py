import os
import pytest
import numpy as np
import tempfile
import yaml
from unittest.mock import patch, MagicMock
from pathlib import Path

# Skip these tests if pyspark is not installed
pyspark = pytest.importorskip("pyspark", reason="pyspark is not installed")
from pyspark.sql import SparkSession

import src.data.preprocess_spark as spark_preprocessor

def test_parse_seizure_summary(tmp_path):
    summary_file = tmp_path / "chb01-summary.txt"
    summary_content = (
        "File Name: chb01_01.edf\n"
        "Number of Seizures in File: 1\n"
        "Seizure Start Time: 10 seconds\n"
        "Seizure End Time: 20 seconds\n"
    )
    summary_file.write_text(summary_content)
    
    seizure_map = spark_preprocessor.parse_seizure_summary(str(summary_file))
    assert "chb01_01.edf" in seizure_map
    assert seizure_map["chb01_01.edf"] == [(10, 20)]

@patch("src.data.preprocess_spark.validate_eeg_data")
def test_process_file_in_spark_worker(mock_validate, tmp_path, monkeypatch):
    mne = pytest.importorskip("mne", reason="mne is not installed")
    
    # Mock Great Expectations validation to pass
    mock_validate.return_value = True
    
    # Create empty dummy EDF file
    patient_dir = tmp_path / "chb01"
    patient_dir.mkdir()
    edf_path = patient_dir / "chb01_01.edf"
    edf_path.write_bytes(b"")
    
    # Mock mne read_raw_edf
    mock_raw = MagicMock()
    mock_raw.info = {"sfreq": 256.0}
    # Mock raw.get_data() returning 10 channels, 512 samples (2 seconds at 256Hz)
    mock_raw.get_data.return_value = np.random.randn(10, 512).astype(np.float32)
    
    monkeypatch.setattr(mne.io, "read_raw_edf", lambda *args, **kwargs: mock_raw)
    
    # Process file
    best_indices = list(range(10))
    records = spark_preprocessor.process_file_in_spark_worker(
        edf_path=str(edf_path),
        best_indices=best_indices,
        target_hz=128.0,
        window_sec=2.0,
        amp_min=-1000.0,
        amp_max=1000.0,
        seizure_times=[(1, 2)]
    )
    
    # Verify records
    assert len(records) > 0
    # Record structure: (patient_id, file_name, window_idx, raw_flat, features_flat, label)
    patient_id, file_name, window_idx, raw_flat, features_flat, label = records[0]
    assert patient_id == "chb01"
    assert file_name == "chb01_01.edf"
    assert window_idx == 0
    # 10 channels * (128Hz * 2.0s) = 2560 values
    assert len(raw_flat) == 2560
    # 10 channels * 9 features = 90 values
    assert len(features_flat) == 90
    assert label == 1

def test_spark_session_preprocesses_and_writes_parquet(tmp_path, monkeypatch):
    mne = pytest.importorskip("mne", reason="mne is not installed")
    
    # Create mock configurations
    config_file = tmp_path / "config.yaml"
    local_data_dir = tmp_path / "datasets"
    parquet_path = tmp_path / "features.parquet"
    
    mock_config = {
        "paths": {
            "local_data_dir": str(local_data_dir),
            "parquet_dataset_path": str(parquet_path)
        },
        "signal_processing": {
            "window_sec": 2.0,
            "target_hz": 128.0,
            "top_n_channels": 10,
            "best_indices": list(range(10))
        },
        "validation": {
            "amplitude_min_microvolts": -1000.0,
            "amplitude_max_microvolts": 1000.0
        }
    }
    config_file.write_text(yaml.safe_dump(mock_config))
    
    # Create dummy patients and summaries
    patient_dir = local_data_dir / "chb01"
    patient_dir.mkdir(parents=True)
    edf_path = patient_dir / "chb01_01.edf"
    edf_path.write_bytes(b"")
    
    summary_file = patient_dir / "chb01-summary.txt"
    summary_file.write_text("File Name: chb01_01.edf\nNumber of Seizures: 0\n")
    
    # Mock out config load
    monkeypatch.setattr(spark_preprocessor, "_load_config", lambda *args, **kwargs: mock_config)
    
    # Mock mne raw and channel topography calculation
    mock_raw = MagicMock()
    mock_raw.info = {"sfreq": 256.0}
    mock_raw.ch_names = [f"EEG {i}" for i in range(23)]
    mock_raw.get_data.return_value = np.random.randn(23, 512).astype(np.float32)
    monkeypatch.setattr(mne.io, "read_raw_edf", lambda *args, **kwargs: mock_raw)
    
    # Mock worker return values to bypass full signal filters/Welch complexity in unit test
    dummy_records = [("chb01", "chb01_01.edf", 0, [0.1]*2560, [0.5]*90, 0)]
    monkeypatch.setattr(spark_preprocessor, "process_file_in_spark_worker", lambda *args, **kwargs: dummy_records)
    
    # Execute Spark preprocessing main
    spark_preprocessor.main()
    
    # Verify Parquet store is created
    assert os.path.exists(parquet_path)
    assert os.path.isdir(parquet_path)
    
    # Read back parquet using a local Spark session to verify data schema
    spark = SparkSession.builder.master("local[1]").appName("VerifyTest").getOrCreate()
    df = spark.read.parquet(str(parquet_path))
    assert df.count() == 1
    assert "raw_signal" in df.columns
    assert "engineered_features" in df.columns
    assert df.schema["label"].dataType.simpleString() == "int"
    spark.stop()
