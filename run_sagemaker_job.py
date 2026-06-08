"""
=============================================================================
  SAGEMAKER ORCHESTRATION & LAUNCH SCRIPT
=============================================================================
This script configures and launches the SageMaker training job.
It uploads the local preprocessed database to S3, configures a TensorFlow 
estimator with hyperparameters from config.yaml, triggers the training job, 
and extracts the final latest.onnx model from the SageMaker model artifact.
=============================================================================
"""

import os
import sys
import yaml
import argparse
import boto3
import tarfile
import shutil
import tempfile
from pathlib import Path

# Try to import SageMaker
try:
    import sagemaker
    from sagemaker.tensorflow import TensorFlow
except ImportError:
    print("SageMaker SDK is not imported. We will attempt to run it once installed.")
    sagemaker = None
    TensorFlow = None

def get_boto3_session():
    """Returns a boto3 Session, attempting local credentials and fallback to DVC configs."""
    try:
        # Check standard AWS config/environment credentials
        sts = boto3.client('sts')
        sts.get_caller_identity()
        print("[*] AWS credentials found in environment/credentials file.")
        return boto3.Session()
    except Exception as e:
        print(f"[*] Standard AWS authentication failed: {e}")
        print("[*] Checking DVC local configuration for S3 credentials...")
        
        import configparser
        config_path = os.path.join(os.path.dirname(__file__), '.dvc', 'config.local')
        if os.path.exists(config_path):
            try:
                cfg = configparser.ConfigParser()
                cfg.read(config_path)
                for section in cfg.sections():
                    if 'remote' in section and 's3' in section:
                        access_key = cfg.get(section, 'access_key_id', fallback=None)
                        secret_key = cfg.get(section, 'secret_access_key', fallback=None)
                        if access_key and secret_key:
                            print("[SUCCESS] Loaded AWS credentials from .dvc/config.local")
                            return boto3.Session(
                                aws_access_key_id=access_key,
                                aws_secret_access_key=secret_key,
                                region_name='eu-north-1'
                            )
            except Exception as ex:
                print(f"Error parsing .dvc/config.local: {ex}")
        
        raise RuntimeError(
            "Could not authenticate with AWS. Please run 'aws configure' or set AWS environment variables, "
            "or ensure S3 credentials exist in .dvc/config.local"
        )

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def upload_dataset_if_missing(session, bucket, local_h5_path, s3_key):
    """Checks if the database is in S3. If not, uploads it."""
    s3 = session.client('s3')
    try:
        s3.head_object(Bucket=bucket, Key=s3_key)
        print(f"[SUCCESS] Dataset already exists on S3: s3://{bucket}/{s3_key}")
    except Exception:
        print(f"[*] Dataset not found on S3. Uploading local {local_h5_path} to s3://{bucket}/{s3_key}...")
        if not os.path.exists(local_h5_path):
            raise FileNotFoundError(f"Local training database not found at {local_h5_path}. Run build_local_database.py first!")
        
        # Determine file size to show progress or info
        size_mb = os.path.getsize(local_h5_path) / (1024 * 1024)
        print(f"  -> File size: {size_mb:.2f} MB")
        
        s3.upload_file(local_h5_path, bucket, s3_key)
        print(f"[SUCCESS] Upload complete!")

def download_and_extract_model(session, model_data_url, output_onnx_path, bucket):
    """Downloads model.tar.gz from S3, extracts latest.onnx, and uploads it to stable S3 path."""
    print("\n" + "=" * 60)
    print("RETRIEVING TRAINED ONNX MODEL FROM SAGEMAKER ARTIFACTS")
    print("=" * 60)
    print(f"Model data URL: {model_data_url}")
    
    # Parse bucket and key from s3://model_data_url
    if not model_data_url.startswith("s3://"):
        raise ValueError(f"Invalid model URL: {model_data_url}")
        
    parts = model_data_url[5:].split("/", 1)
    artifact_bucket = parts[0]
    artifact_key = parts[1]
    
    s3 = session.client('s3')
    
    # Download tarball to temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        tar_local_path = os.path.join(tmpdir, "model.tar.gz")
        print(f"[*] Downloading tarball from s3://{artifact_bucket}/{artifact_key}...")
        s3.download_file(artifact_bucket, artifact_key, tar_local_path)
        
        print("[*] Extracting model.tar.gz...")
        with tarfile.open(tar_local_path, "r:gz") as tar:
            tar.extractall(path=tmpdir)
            
        extracted_files = os.listdir(tmpdir)
        print(f"  -> Extracted files: {extracted_files}")
        
        onnx_filename = "latest.onnx"
        local_onnx = os.path.join(tmpdir, onnx_filename)
        
        if not os.path.exists(local_onnx):
            raise FileNotFoundError(f"ONNX model '{onnx_filename}' was not found inside the model.tar.gz archive.")
            
        # 1. Copy to local workspace root
        shutil.copy(local_onnx, output_onnx_path)
        print(f"[SUCCESS] Copied trained model to local workspace: {output_onnx_path}")
        
        # 2. Upload to stable S3 model path
        s3_stable_key = "models/latest.onnx"
        print(f"[*] Uploading model to stable S3 path: s3://{bucket}/{s3_stable_key}")
        s3.upload_file(str(output_onnx_path), bucket, s3_stable_key)
        print(f"[SUCCESS] Model uploaded to s3://{bucket}/{s3_stable_key}")

def main():
    parser = argparse.ArgumentParser(description="Orchestrates the SageMaker EEG training job.")
    parser.add_argument('--role-arn', type=str, default=None,
                        help="AWS IAM execution role ARN for SageMaker. (Required for cloud runs)")
    parser.add_argument('--instance-type', type=str, default=None,
                        help="SageMaker instance type (e.g. ml.m5.large, ml.g4dn.xlarge, local). Defaults to ml.m5.large (or local if --local is set)")
    parser.add_argument('--local', action='store_true',
                        help="Run the training job in SageMaker Local Mode (requires Docker running locally)")
    parser.add_argument('--bucket', type=str, default='seizure-detection--s-roy',
                        help="S3 bucket name for data and model storage.")
    
    args = parser.parse_known_args()[0]

    # Verify SageMaker is installed
    global sagemaker, TensorFlow
    if sagemaker is None:
        try:
            import sagemaker
            from sagemaker.tensorflow import TensorFlow
        except ImportError:
            print("[ERROR] SageMaker Python SDK is not installed in the active environment.")
            print("Please wait for the installation to finish or run: venv\\Scripts\\pip install sagemaker")
            sys.exit(1)

    class CustomSageMakerSession(sagemaker.Session):
        def __init__(self, custom_bucket, **kwargs):
            super().__init__(**kwargs)
            self.custom_bucket = custom_bucket
            
        def default_bucket(self):
            return self.custom_bucket

    print("=" * 60)
    print("SAGEMAKER TRAINING ORCHESTRATION")
    print("=" * 60)
    
    # 1. AWS Authentication
    session_boto = get_boto3_session()
    bucket = args.bucket
    
    # Create Custom SageMaker session to override default S3 bucket checks
    if args.local:
        from sagemaker.local import LocalSession
        class CustomLocalSession(LocalSession):
            def __init__(self, custom_bucket, **kwargs):
                super().__init__(**kwargs)
                self.custom_bucket = custom_bucket
                
            def default_bucket(self):
                return self.custom_bucket
        sagemaker_session = CustomLocalSession(custom_bucket=bucket, boto_session=session_boto)
    else:
        sagemaker_session = CustomSageMakerSession(custom_bucket=bucket, boto_session=session_boto)
    
    # Load parameters from config.yaml
    config = load_config()
    
    # Get hyperparameters
    epochs = config["training"]["epochs"]
    batch_size = config["training"]["batch_size"]
    learning_rate = config["training"]["learning_rate"]
    top_n_channels = config["signal_processing"]["top_n_channels"]
    target_hz = config["signal_processing"]["target_hz"]
    window_sec = config["signal_processing"]["window_sec"]
    test_split = config["training"]["test_split"]
    
    local_h5 = config["paths"]["hdf5_database_path"]
    local_onnx = config["paths"]["target_onnx_path"]
    
    bucket = args.bucket
    s3_data_key = "data/train_database.h5"
    s3_data_uri = f"s3://{bucket}/data/"

    # 2. Upload Dataset if missing
    upload_dataset_if_missing(session_boto, bucket, local_h5, s3_data_key)
    
    # 3. Determine Execution Role and Instance Type
    instance_type = args.instance_type
    if args.local:
        print("[*] Running in SageMaker Local Mode...")
        instance_type = instance_type or 'local'
        # Local mode doesn't strictly validate IAM role ARN
        role_arn = args.role_arn or "arn:aws:iam::116584140401:role/service-role/AmazonSageMaker-ExecutionRole-Dummy"
    else:
        print("[*] Running in SageMaker Cloud Mode...")
        instance_type = instance_type or 'ml.m5.large'
        if not args.role_arn:
            print("\n[ERROR] SageMaker Cloud Mode requires an execution role ARN.")
            print("Please look up your SageMaker role ARN in the AWS console and provide it like so:")
            print("python run_sagemaker_job.py --role-arn arn:aws:iam::116584140401:role/service-role/AmazonSageMaker-ExecutionRole-XXXXXXXX")
            sys.exit(1)
        role_arn = args.role_arn
        
    print(f"  -> IAM Role: {role_arn}")
    print(f"  -> Instance Type: {instance_type}")
    print(f"  -> S3 Input URI: {s3_data_uri}")
    
    # 4. Set up Estimator
    hyperparameters = {
        'epochs': epochs,
        'batch-size': batch_size,
        'learning-rate': learning_rate,
        'top-n-channels': top_n_channels,
        'target-hz': target_hz,
        'window-sec': window_sec,
        'test-split': test_split
    }
    
    print(f"\n[*] Configuring TensorFlow Estimator with hyperparameters:")
    for k, v in hyperparameters.items():
        print(f"  - {k}: {v}")
        
    # Standard TensorFlow DLC image configuration
    # Note: version 2.14.1 with py310 is highly compatible
    tf_estimator = TensorFlow(
        entry_point='sagemaker_train.py',
        role=role_arn,
        instance_count=1,
        instance_type=instance_type,
        framework_version='2.14.1',
        py_version='py310',
        hyperparameters=hyperparameters,
        sagemaker_session=sagemaker_session,
        output_path=f"s3://{bucket}/output",
        code_location=f"s3://{bucket}/code"
    )
    
    # 5. Start Training Job
    print("\n[*] Submitting SageMaker training job...")
    tf_estimator.fit({'training': s3_data_uri})
    
    # 6. Retrieve Model Artifact
    model_data_url = tf_estimator.model_data
    download_and_extract_model(session_boto, model_data_url, local_onnx, bucket)
    
    print("\n[SUCCESS] SageMaker run completed! Model is deployed at latest.onnx and stable S3 folder.")

if __name__ == '__main__':
    main()
