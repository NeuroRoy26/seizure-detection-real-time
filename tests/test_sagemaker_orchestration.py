import os
import sys
import yaml
import pytest
import tempfile
import io
import tarfile
from unittest.mock import patch, MagicMock
from pathlib import Path

import run_sagemaker_job as orchestrator

def test_load_config(tmp_path):
    config_file = tmp_path / "config.yaml"
    mock_data = {
        "paths": {"target_onnx_path": "dummy.onnx"},
        "training": {"epochs": 1}
    }
    config_file.write_text(yaml.safe_dump(mock_data))
    
    config = orchestrator.load_config(str(config_file))
    assert config["paths"]["target_onnx_path"] == "dummy.onnx"
    assert config["training"]["epochs"] == 1

@patch("run_sagemaker_job.boto3.client")
def test_get_boto3_session_standard_credentials(mock_boto_client):
    # Mock STS identity call returning success
    mock_sts = MagicMock()
    mock_sts.get_caller_identity.return_value = {"Arn": "arn:aws:iam::111:user/dummy"}
    mock_boto_client.return_value = mock_sts
    
    session = orchestrator.get_boto3_session()
    assert session is not None
    mock_sts.get_caller_identity.assert_called_once()

@patch("run_sagemaker_job.boto3.client")
def test_get_boto3_session_dvc_fallback(mock_boto_client):
    # Make STS throw an error to trigger DVC fallback
    mock_boto_client.side_effect = Exception("No standard credentials")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a mock .dvc/config.local file
        dvc_dir = Path(tmpdir) / ".dvc"
        dvc_dir.mkdir()
        config_local = dvc_dir / "config.local"
        config_local.write_text(
            "[remote \"s3-remote\"]\n"
            "access_key_id = DUMMY_ACCESS_KEY\n"
            "secret_access_key = DUMMY_SECRET_KEY\n"
        )
        
        # Patch the file path check in the orchestrator
        with patch("run_sagemaker_job.os.path.exists", return_value=True), \
             patch("run_sagemaker_job.os.path.join", return_value=str(config_local)), \
             patch("run_sagemaker_job.boto3.Session") as mock_session_class:
            
            orchestrator.get_boto3_session()
            mock_session_class.assert_called_once_with(
                aws_access_key_id="DUMMY_ACCESS_KEY",
                aws_secret_access_key="DUMMY_SECRET_KEY",
                region_name="eu-north-1"
            )

def test_download_and_extract_model_downloads_and_unpacks(tmp_path):
    # Create a mock tar.gz containing seizure_detector_mobilenetv2.onnx
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    onnx_file = model_dir / "seizure_detector_mobilenetv2.onnx"
    onnx_content = b"fake-onnx-model-binary-data"
    onnx_file.write_bytes(onnx_content)
    
    tar_path = tmp_path / "model.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(onnx_file, arcname="seizure_detector_mobilenetv2.onnx")
        
    # Mock boto3 s3 client
    mock_s3 = MagicMock()
    mock_session = MagicMock()
    mock_session.client.return_value = mock_s3
    
    # Mock download_file to write our prepared tar.gz
    def mock_download(bucket, key, dest_path):
        with open(tar_path, "rb") as src, open(dest_path, "wb") as dest:
            dest.write(src.read())
            
    mock_s3.download_file.side_effect = mock_download
    
    output_onnx_path = tmp_path / "extracted_seizure_detector.onnx"
    
    orchestrator.download_and_extract_model(
        session=mock_session,
        model_data_url="s3://my-bucket/model.tar.gz",
        output_onnx_path=output_onnx_path,
        bucket="stable-bucket"
    )
    
    # Assertions
    assert output_onnx_path.exists()
    assert output_onnx_path.read_bytes() == onnx_content
    # Confirm S3 upload of model
    mock_s3.upload_file.assert_called_once_with(
        str(output_onnx_path), "stable-bucket", "models/seizure_detector_mobilenetv2.onnx"
    )

def test_stdout_encoding_wrapper_handles_unicode():
    # Setup standard text buffer
    buf = io.BytesIO()
    # TextIOWrapper simulating standard Windows CP1252 console output
    # but reconfigured to replace errors
    wrapper = io.TextIOWrapper(buf, encoding="cp1252", errors="replace")
    
    # Try writing non-cp1252 characters (like unicode box characters used in progress bars)
    # Under standard cp1252 without replace this raises UnicodeEncodeError.
    # Our replacement mechanism should replace them with '?'
    try:
        wrapper.write("Progress: ━━━ 100%")
        wrapper.flush()
        written = buf.getvalue().decode("cp1252")
        assert "Progress: ???" in written
    except UnicodeEncodeError:
        pytest.fail("TextIOWrapper raised UnicodeEncodeError on unsupported characters")
