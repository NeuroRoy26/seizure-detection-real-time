import os
import h5py
import pytest
import numpy as np
import yaml
from unittest.mock import patch
from src.data.validate_database_quality import main, _load_config

def test_load_config_fallback(tmp_path, monkeypatch):
    # Test loading config when config files exist
    config_file = tmp_path / "config.yaml"
    state_file = tmp_path / ".run_state.json"
    
    mock_config = {"paths": {"hdf5_database_path_2d": "store.h5"}}
    mock_state = {"best_indices": [0, 1, 2]}
    
    config_file.write_text(yaml.safe_dump(mock_config))
    
    import json
    state_file.write_text(json.dumps(mock_state))
    
    monkeypatch.chdir(tmp_path)
    config = _load_config()
    assert config["paths"]["hdf5_database_path_2d"] == "store.h5"
    assert config["signal_processing"]["best_indices"] == [0, 1, 2]

def test_validate_database_quality_success(tmp_path, monkeypatch):
    # Create valid dummy database
    db_file = tmp_path / "test_db.h5"
    with h5py.File(db_file, "w") as h5f:
        # 5 samples, 10 channels, 256 timepoints
        h5f.create_dataset("raw_signals/X", data=np.random.normal(0, 100.0, size=(5, 10, 256)))
        h5f.create_dataset("raw_signals/y", data=np.array([0, 1, 0, 1, 0]))

    mock_config = {
        "paths": {"hdf5_database_path_2d": str(db_file)},
        "validation": {"amplitude_min_microvolts": -1000.0, "amplitude_max_microvolts": 1000.0}
    }
    
    with patch("src.data.validate_database_quality._load_config", return_value=mock_config):
        success = main()
        assert success is True

def test_validate_database_quality_missing_db(tmp_path):
    mock_config = {
        "paths": {"hdf5_database_path_2d": "non_existent_file_xyz.h5"}
    }
    with patch("src.data.validate_database_quality._load_config", return_value=mock_config):
        success = main()
        assert success is False

def test_validate_database_quality_invalid_shape(tmp_path):
    db_file = tmp_path / "invalid_db.h5"
    with h5py.File(db_file, "w") as h5f:
        # Invalid shape: 9 channels instead of 10
        h5f.create_dataset("raw_signals/X", data=np.random.normal(0, 10.0, size=(5, 9, 256)))
        h5f.create_dataset("raw_signals/y", data=np.array([0, 1, 0, 1, 0]))

    mock_config = {"paths": {"hdf5_database_path_2d": str(db_file)}}
    with patch("src.data.validate_database_quality._load_config", return_value=mock_config):
        success = main()
        assert success is False

def test_validate_database_quality_missing_keys(tmp_path):
    db_file = tmp_path / "missing_keys_db.h5"
    with h5py.File(db_file, "w") as h5f:
        # Missing raw_signals/y
        h5f.create_dataset("raw_signals/X", data=np.random.normal(0, 10.0, size=(5, 10, 256)))

    mock_config = {"paths": {"hdf5_database_path_2d": str(db_file)}}
    with patch("src.data.validate_database_quality._load_config", return_value=mock_config):
        success = main()
        assert success is False

def test_validate_database_quality_validation_failure(tmp_path):
    db_file = tmp_path / "fail_db.h5"
    with h5py.File(db_file, "w") as h5f:
        # Generate NaN values to trigger validation failure
        X = np.random.normal(0, 10.0, size=(5, 10, 256))
        X[0, 0, 0] = np.nan
        h5f.create_dataset("raw_signals/X", data=X)
        h5f.create_dataset("raw_signals/y", data=np.array([0, 1, 0, 1, 0]))

    mock_config = {
        "paths": {"hdf5_database_path_2d": str(db_file)},
        "validation": {"amplitude_min_microvolts": -1000.0, "amplitude_max_microvolts": 1000.0}
    }
    with patch("src.data.validate_database_quality._load_config", return_value=mock_config):
        success = main()
        assert success is False
