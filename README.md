# Production-Grade Real-Time Seizure Detection Pipeline

[![CI](https://github.com/NeuroRoy26/seizure-detection-real-time/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/NeuroRoy26/seizure-detection-real-time/actions/workflows/ci.yml)
[![codecov](https://codecov.io/github/NeuroRoy26/seizure-detection-real-time/graph/badge.svg?token=KIM3PCNSMP)](https://codecov.io/github/NeuroRoy26/seizure-detection-real-time)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-blueviolet)](https://mlflow.org/)
[![AWS SageMaker](https://img.shields.io/badge/AWS-SageMaker-orange)](https://aws.amazon.com/sagemaker/)
[![Great Expectations](https://img.shields.io/badge/Data_Quality-Great_Expectations-green)](https://greatexpectations.io/)

This repository implements a production-grade, end-to-end MLOps pipeline for real-time seizure detection from multi-channel EEG signals. The architecture scales from local data processing to cloud-based distributed training, featuring robust data validation, structured feature storage, experiment tracking, and real-time ONNX inference.

---

## 🏗️ Architecture & Data Flow

```mermaid
flowchart TD
    subgraph Data Pipeline (ETL & Validation)
        EDF["Raw EDF Files (CHB-MIT)"] -->|Parallel Worker Pools| ETL["Producer-Consumer ETL (src/preprocess.py)"]
        ETL -->|Data Quality Validation| GE["Great Expectations Gate (src/validation.py)"]
        GE -->|Passed Check| FS["HDF5 Feature Store (src/feature_store.py)"]
    end

    subgraph Training & MLOps
        FS -->|Single/Tune Runs| LT["Local Training (src/train.py / src/tune.py)"]
        LT -->|Log Params, Metrics & Models| MLF["MLflow Tracking (SQLite DB)"]
        
        FS -->|Upload to S3| S3["AWS S3 Data Bucket"]
        S3 -->|Orchestrated Job| SM["AWS SageMaker Training (ml.m5.large / CPU-GPU)"]
        SM -->|Train MobileNetV2 Transfer Learning| SMT["sagemaker_train.py"]
        SMT -->|Export ONNX| S3Out["S3 Model Artifacts (model.tar.gz)"]
    end

    subgraph Real-Time Inference
        S3Out -->|Retrieve & Extract| ONNX["latest.onnx"]
        LT -->|Direct Export| ONNX
        
        ONNX -->|ONNX Runtime Inference| API["FastAPI Backend (api.py)"]
        Streamer["Mock EEG Streamer (mock_streamer.py)"] -->|POST /ingest (10-Ch Signal)| API
        API -->|GET /latest (Probabilities)| Dash["Streamlit Dashboard (dashboard.py)"]
    end
```

---

## 🚀 Key Technical Highlights (MLOps Engineering)

* **Robust Data Validation Gate**: Integrates **Great Expectations v1.18.0** to enforce data-quality schemas (e.g. range bounds, standard deviations, scaling limits) on raw EEG microvolt values, preventing bad channel data or sensor noise from poisoning downstream models.
* **Parallelized Producer-Consumer ETL**: Processing large raw binary `.edf` files is heavily CPU-bound. To bypass the Global Interpreter Lock (GIL) and prevent database corruption (HDF5 does not support concurrent writes), the ETL pipeline uses a parallel `ProcessPoolExecutor` for signal filtering and validation (Producers) feeding a single-threaded writer queue (Consumer), achieving a **4x–8x processing speedup**.
* **Structured Local Feature Store**: A unified **HDF5 Feature Store** separates processed raw signals (for deep learning models) and pre-calculated time-frequency features (RMS, line length, spectral band powers) for traditional machine learning baselines.
* **Industrial Experiment Tracking (MLflow)**: Leverages an **MLflow SQLite backend** (`sqlite:///mlflow.db`) to record hyperparameters, evaluation metrics (Accuracy, Precision, Seizure Sensitivity/Recall, F1-Score, RMSE), and serializes the compiled ONNX model artifacts directly inside the active run.
* **AWS SageMaker Cloud & Local Mode**: Supports both standard cloud training and containerized Local Mode (using Docker Desktop) via **AWS SageMaker**. It uploads preprocessed, down-sampled balanced datasets to **AWS S3**, provisions ephemeral training instances, executes the training script, and automatically downloads and extracts the compiled `latest.onnx` model.
* **2D CNN Transfer Learning**: Maps 10-channel 1D EEG time-series windows to 2D representations using a `(1,1)` spatial channel expansion layer, resizes to `(224,224,3)` via bilinear interpolation, and utilizes a frozen **MobileNetV2** backbone pre-trained on ImageNet to perform transfer learning on neural signals.
* **ONNX Compilation & Deployment**: Replaces heavy framework dependencies (`TensorFlow`/`PyTorch`) with a lightweight `onnxruntime` engine on the FastAPI backend for fast, low-latency, real-time predictions.

---

## 🛠️ Codebase Structure

```text
├── config.yaml                     # Centralized project configurations (ETL, paths, tuning, models)
├── api.py                          # FastAPI server serving real-time model inference
├── dashboard.py                    # Streamlit real-time visualization dashboard
├── mock_streamer.py                # Simulated multi-channel EEG streamer
├── run_sagemaker_job.py            # SageMaker orchestrator (handles S3 upload, Docker launch, download)
├── sagemaker_train.py              # Cloud/Container training script (runs inside SageMaker TensorFlow container)
├── requirements.txt                # Production dependencies (FastAPI, Streamlit, OnnxRuntime, etc.)
├── requirements-dev.txt            # Development & pipeline dependencies (MNE, TensorFlow, Great Expectations, MLflow)
├── src/
│   ├── preprocess.py               # Multiprocess ETL orchestrator
│   ├── validation.py               # Great Expectations data validator
│   ├── features.py                 # Time-frequency domain feature extraction
│   ├── feature_store.py            # Local HDF5 Feature Store manager
│   ├── model.py                    # Adapted 2D-CNN Model architecture
│   ├── train.py                    # Local training loop & MLflow logger
│   └── tune.py                     # Local Grid Search hyperparameter tuner
└── tests/                          # 50+ unit and integration tests (Pytest)
```

---

## 💻 Getting Started (Installation & Run)

### 1. Environment Setup & Pulling Data (DVC)
Create your virtual environment, install dependencies, and pull the version-controlled hospital datasets:
```powershell
# Create & activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1   # Windows PowerShell

# Install dev/mlops dependencies
pip install -r requirements.txt -r requirements-dev.txt

# Pull hospital datasets from Google Drive remote
dvc pull datasets.dvc
```

### 2. Run the Data Preprocessing Pipeline (ETL)
Execute the ETL script to select the 10 most stable channels dynamically, apply bandpass/notch filters, run data validation checks, and save windows to the HDF5 Feature Store:
```powershell
python src/preprocess.py
```

### 3. Training & Orchestration

#### Local Training & Hyperparameter Tuning
Train a model locally and track it with MLflow (runs local SQLite server):
```powershell
# Run local training loop
python src/train.py

# Run grid search hyperparameter sweep
python src/tune.py

# View training metrics and ONNX artifacts locally
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

#### AWS SageMaker Training (Cloud / Local Container Mode)
Verify and test the training job locally inside a Docker container (requires Docker Desktop running), or deploy directly to AWS SageMaker instances (completely free under the AWS Free Tier allowance of 50 training hours/month):

```powershell
# Option A: Run in SageMaker Local Mode using your local Docker engine (for debugging)
python run_sagemaker_job.py --local

# Option B: Spin up a managed SageMaker cloud instance (requires AWS execution role)
python run_sagemaker_job.py --role-arn arn:aws:iam::116584140401:role/service-role/AmazonSageMaker-ExecutionRole-XXXXXXXX
```

---

## ⚡ Real-Time System Run (Local Demo)

To run the full end-to-end real-time dashboard, start the following processes in three separate terminals:

### 1) Start Uvicorn FastAPI Server (Terminal A)
```powershell
python -m uvicorn api:app --reload
```
FastAPI runs locally at `http://127.0.0.1:8000`. Access interactive API documentation at `/docs`.

### 2) Launch the Mock EEG Streamer (Terminal B)
```powershell
python mock_streamer.py --hz 128 --seizure-every 60 --seizure-duration 5
```
Simulates real-time 10-channel streaming data, injecting mock seizure anomalies every 60 seconds and posting samples directly to the FastAPI server.

### 3) Run the Streamlit Dashboard (Terminal C)
```powershell
streamlit run dashboard.py
```
Open `http://localhost:8501`, press **Start**, and watch the rolling raw signal waveforms update alongside the live seizure probability outputs.

---

## 🧪 Testing & Code Quality
The codebase is validated by a rigorous Pytest test suite containing **50 unit and integration tests** covering mock streaming, validation constraints, preprocessing transformations, API routes, and model serialization.

```powershell
# Run the test suite
pytest -v

# Run with coverage reports
pytest --cov=. --cov-report=term-missing
```
