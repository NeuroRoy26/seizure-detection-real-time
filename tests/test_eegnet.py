import pytest
from unittest.mock import patch
from src.models.model_eegnet import build_eegnet

def test_build_eegnet_default():
    tf = pytest.importorskip("tensorflow")
    
    # 1. Compile model with default shape
    model = build_eegnet(input_shape=(10, 256))
    
    # 2. Check type and name
    assert model.name == "EEGNet_Seizure"
    
    # 3. Check input shape: (None, 10, 256)
    assert model.input_shape == (None, 10, 256)
    
    # 4. Check output shape: (None, 2)
    assert model.output_shape == (None, 2)

def test_build_eegnet_3d_shape():
    tf = pytest.importorskip("tensorflow")
    
    # Compile with explicit 3D shape
    model = build_eegnet(input_shape=(10, 256, 1))
    assert model.input_shape == (None, 10, 256, 1)
    assert model.output_shape == (None, 2)

def test_build_eegnet_import_error():
    # Mock tensorflow module level variable tf = None to trigger the error check
    with patch("src.models.model_eegnet.tf", None):
        with pytest.raises(ImportError) as excinfo:
            build_eegnet()
        assert "TensorFlow is required to compile this model" in str(excinfo.value)
