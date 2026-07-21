"""
=============================================================================
  UNIT TEST SUITE FOR THE 2D PIPELINE
=============================================================================
Verifies the correctness of Great Expectations validation rules, feature
engineering, HDF5 Feature Store operations, and Keras 2D-CNN shape compatibility.
=============================================================================
"""

import os
import tempfile
import numpy as np
import pytest
import h5py

from src.data.validation import validate_eeg_data
from src.data.features import extract_eeg_features
from src.data.feature_store import LocalFeatureStore
from src.models.model import build_adapted_2d_cnn

def test_great_expectations_validation():
    # 1. Valid microvolt window (channels, samples) = (10, 256)
    valid_window = np.random.uniform(-100, 100, (10, 256))
    assert validate_eeg_data(valid_window, -1000.0, 1000.0) is True

    # 2. Invalid channel count
    invalid_channels = np.random.uniform(-100, 100, (9, 256))
    assert validate_eeg_data(invalid_channels, -1000.0, 1000.0) is False

    # 3. Out-of-bounds voltage values (simulates unscaled raw Volts or huge artifacts)
    unscaled_window = np.random.uniform(-100, 100, (10, 256))
    unscaled_window[0, :] = 1200.0  # Set entire channel out of bounds (10% of window) to exceed 1% threshold
    assert validate_eeg_data(unscaled_window, -1000.0, 1000.0) is False


    # 4. Window containing NaN values
    nan_window = np.random.uniform(-100, 100, (10, 256))
    nan_window[2, 100] = np.nan
    assert validate_eeg_data(nan_window, -1000.0, 1000.0) is False

def test_feature_engineering_extraction():
    # Create mock signal segment (10 channels, 256 samples)
    mock_window = np.random.uniform(-100, 100, (10, 256))
    
    # Run feature extraction
    features = extract_eeg_features(mock_window, sfreq=128.0)
    
    # Assert correct dimensions (10 channels, 9 features each)
    assert features.shape == (10, 9)
    assert np.all(np.isfinite(features))  # Check no NaNs or infs

def test_local_feature_store():
    # Create a temporary file path
    fd, temp_h5_path = tempfile.mkstemp(suffix=".h5")
    os.close(fd)
    
    try:
        store = LocalFeatureStore(temp_h5_path)
        
        # Initialize store
        store.initialize_store(n_channels=10, n_samples=256, n_features=9)
        shapes = store.get_dataset_shapes()
        assert shapes["raw_signals_X"] == (0, 10, 256)
        assert shapes["engineered_features_X"] == (0, 10, 9)
        
        # Append dummy batch
        batch_size = 5
        dummy_raw = np.random.uniform(-100, 100, (batch_size, 10, 256)).astype(np.float32)
        dummy_eng = np.random.uniform(0, 1, (batch_size, 10, 9)).astype(np.float32)
        dummy_y = np.array([0, 1, 0, 0, 1], dtype=np.int32)
        
        store.append_batch(dummy_raw, dummy_eng, dummy_y)
        
        # Verify shapes after appending
        new_shapes = store.get_dataset_shapes()
        assert new_shapes["raw_signals_X"] == (batch_size, 10, 256)
        assert new_shapes["raw_signals_y"] == (batch_size,)
        assert new_shapes["engineered_features_X"] == (batch_size, 10, 9)
        assert new_shapes["engineered_features_y"] == (batch_size,)
        
        # Read and verify content
        with h5py.File(temp_h5_path, "r") as h5f:
            np.testing.assert_array_almost_equal(h5f["raw_signals/X"][:], dummy_raw)
            np.testing.assert_array_equal(h5f["raw_signals/y"][:], dummy_y)
            np.testing.assert_array_almost_equal(h5f["engineered_features/X"][:], dummy_eng)
            
    finally:
        # Clean up temporary file
        if os.path.exists(temp_h5_path):
            os.remove(temp_h5_path)

def test_model_compilation():
    # Compile model
    model = build_adapted_2d_cnn(input_shape=(10, 256, 1))
    
    # Assert input and output dimensions
    assert model.input_shape == (None, 10, 256, 1)
    assert model.output_shape == (None, 2)
