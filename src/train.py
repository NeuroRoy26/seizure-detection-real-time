"""
=============================================================================
  LOCAL TRAINING & ONNX EXPORT ENGINE
=============================================================================
Streams EEG windows from the local Feature Store, trains the adapted 2D-CNN,
evaluates clinical metrics, and cross-compiles the Keras graph to a
production-ready ONNX model.
=============================================================================
"""

import os
import sys
import h5py
import numpy as np
import yaml
import tempfile
import subprocess
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error

# Ensure root directory is in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import MLflow for model tracking
try:
    import mlflow
    import mlflow.tensorflow
except ImportError:
    mlflow = None


# Optional heavy dependencies
try:
    import tensorflow as tf
    from tensorflow.keras.utils import Sequence
except Exception:
    tf = None
    class Sequence:
        pass

from src.model import build_adapted_2d_cnn

class HDF5SignalGenerator(Sequence):
    """
    Memory-safe batch generator that streams raw EEG signal tensors from the
    Feature Store, expanding dimensions to match Keras Conv2D input shapes.
    """
    def __init__(self, h5_path: str, indices: np.ndarray, batch_size: int = 64, **kwargs):
        super().__init__(**kwargs)
        self.h5_path = h5_path
        self.indices = indices
        self.batch_size = batch_size
        
        # Load expected signal length from config.yaml dynamically
        try:
            with open("config.yaml", "r") as f:
                cfg = yaml.safe_load(f)
            self.expected_samples = int(
                cfg["signal_processing"]["target_hz"] * cfg["signal_processing"]["window_sec"]
            )
        except Exception:
            self.expected_samples = 256  # Fallback to standard 2-sec @ 128Hz

    def __len__(self) -> int:
        return int(np.ceil(len(self.indices) / float(self.batch_size)))

    def __getitem__(self, idx: int) -> tuple:
        batch_indices = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        sorted_idx = np.sort(batch_indices)
        
        with h5py.File(self.h5_path, 'r') as h5f:
            x_key = 'raw_signals/X' if 'raw_signals/X' in h5f else 'X'
            y_key = 'raw_signals/y' if 'raw_signals/y' in h5f else 'y'
            X_batch = h5f[x_key][sorted_idx]
            y_batch = h5f[y_key][sorted_idx]
            
        # Dynamically slice or pad if the HDF5 has a different window size
        if X_batch.shape[2] != self.expected_samples:
            X_batch = X_batch[:, :, :self.expected_samples]
            
        # Add channel dimension for Conv2D: shape (batch_size, 10, expected_samples, 1)
        X_batch = np.expand_dims(X_batch, axis=-1)
        return X_batch, y_batch

class MLflowCallback(tf.keras.callbacks.Callback):
    """
    Custom Keras callback to log metrics (loss, accuracy, val_loss, val_accuracy)
    to MLflow after each epoch, bypassing the need for TensorBoard.
    """
    def on_epoch_end(self, epoch, logs=None):
        if logs and mlflow is not None:
            # logs is a dictionary containing metrics from Keras fit()
            mlflow.log_metrics({k: float(v) for k, v in logs.items()}, step=epoch)


def compute_clinical_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Computes standard medical evaluation metrics.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "rmse": float(rmse)
    }

def print_clinical_metrics(dataset_name: str, metrics: dict) -> None:
    """
    Prints clinical metrics.
    """
    print(f"\n--- {dataset_name} CLINICAL METRICS ---")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}  (Seizure Sensitivity)")
    print(f"  F1-Score  : {metrics['f1']:.4f}")
    print(f"  RMSE      : {metrics['rmse']:.4f}")


def _load_config(config_path: str = "config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def train_model(config: dict):
    h5_path = config["paths"]["hdf5_database_path_2d"]

    target_onnx_path = config["paths"]["target_onnx_path"]
    
    batch_size = config["training"]["batch_size"]
    epochs = config["training"]["epochs"]
    learning_rate = config["training"]["learning_rate"]
    test_split = config["training"]["test_split"]
    
    input_channels = config["signal_processing"]["top_n_channels"]
    input_samples = int(
        config["signal_processing"]["target_hz"] * config["signal_processing"]["window_sec"]
    )
    
    print("=" * 60)
    print("1. LOADING FEATURE STORE DATASET")
    print("=" * 60)
    
    if not os.path.exists(h5_path):
        print(f"[!] Feature Store not found at {h5_path}; run preprocess.py first.")
        return
        
    if tf is None:
        raise ImportError("TensorFlow is required to train the model")
        
    with h5py.File(h5_path, "r") as h5f:
        y_key = 'raw_signals/y' if 'raw_signals/y' in h5f else 'y'
        y_full = np.array(h5f[y_key], dtype=np.int32)
        indices_full = np.arange(len(y_full))
        
    print(f"[*] Total samples loaded from Feature Store: {len(y_full)}")
    if len(y_full) == 0:
        print("[!] Dataset is empty; skipping training.")
        return
        
    # Check GPU availability
    gpu_devices = tf.config.list_physical_devices("GPU")
    if len(gpu_devices) > 0:
        print(f"✅ GPU DETECTED: {gpu_devices[0]}")
        tf.config.experimental.set_memory_growth(gpu_devices[0], True)
    else:
        print("[*] No GPU detected; falling back to CPU.")
        
    print("\n" + "=" * 60)
    print("2. DATA STRATIFICATION & IMBALANCE SCALING")
    print("=" * 60)
    
    # Stratified split to ensure equal seizure ratio in train and test sets
    idx_train, idx_val, y_train, _ = train_test_split(
        indices_full,
        y_full,
        test_size=test_split,
        random_state=42,
        stratify=y_full
    )
    
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weights = {classes[i]: float(weights[i]) for i in range(len(classes))}
    print(f"[*] Dataset split: {len(idx_train)} Train, {len(idx_val)} Validation")
    print(f"[*] Adjusted Class Weights: {class_weights}")
    
    train_gen = HDF5SignalGenerator(h5_path, idx_train, batch_size=batch_size)
    val_gen = HDF5SignalGenerator(h5_path, idx_val, batch_size=batch_size)
    
    # MLflow Setup
    if mlflow is not None:
        use_dagshub = config.get("mlflow", {}).get("use_dagshub", False)
        if use_dagshub:
            try:
                import dagshub
                repo_owner = config.get("mlflow", {}).get("repo_owner", "NeuroRoy26")
                repo_name = config.get("mlflow", {}).get("repo_name", "seizure-detection-real-time")
                print(f"[*] Initializing DAGsHub integration for {repo_owner}/{repo_name}...")
                dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
            except ImportError:
                print("[!] dagshub package not found. Run 'pip install dagshub' or set use_dagshub: false in config.yaml.")
        else:
            tracking_uri = config.get("mlflow", {}).get("tracking_uri", "sqlite:///mlflow.db")
            mlflow.set_tracking_uri(tracking_uri)
            
        experiment_name = config.get("mlflow", {}).get("experiment_name", "seizure-detection-2d-cnn")
        mlflow.set_experiment(experiment_name)


    print("\n" + "=" * 60)
    print("3. MODEL COMPILATION")
    print("=" * 60)
    
    run_ctx = mlflow.start_run() if mlflow is not None else None
    try:
        if mlflow is not None:
            print(f"[*] MLflow run initiated. Run ID: {run_ctx.info.run_id}")
            # Log config/signal processing params
            mlflow.log_params({
                "window_sec": config["signal_processing"]["window_sec"],
                "target_hz": config["signal_processing"]["target_hz"],
                "top_n_channels": config["signal_processing"]["top_n_channels"],
                "test_split": test_split,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "epochs": epochs
            })
            
        model = build_adapted_2d_cnn(input_shape=(input_channels, input_samples, 1))
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"]
        )
        model.summary()
        
        print(f"\n[*] Training for {epochs} epochs...")
        model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            class_weight=class_weights,
            verbose=1,
            callbacks=[MLflowCallback()] if mlflow is not None else []
        )
        
        print("\n" + "=" * 60)
        print("4. MODEL CLINICAL EVALUATION")
        print("=" * 60)
        
        raw_train_preds = model.predict(train_gen, verbose=0)
        y_train_pred = np.argmax(raw_train_preds, axis=1)
        
        raw_val_preds = model.predict(val_gen, verbose=0)
        y_val_pred = np.argmax(raw_val_preds, axis=1)
        
        # Align true labels from generators
        y_train_true = []
        for i in range(len(train_gen)):
            batch_indices = train_gen.indices[i * train_gen.batch_size : (i + 1) * train_gen.batch_size]
            y_train_true.extend(y_full[np.sort(batch_indices)])
        y_train_true = np.array(y_train_true)
        
        y_val_true = []
        for i in range(len(val_gen)):
            batch_indices = val_gen.indices[i * val_gen.batch_size : (i + 1) * val_gen.batch_size]
            y_val_true.extend(y_full[np.sort(batch_indices)])
        y_val_true = np.array(y_val_true)
        
        train_metrics = compute_clinical_metrics(y_train_true, y_train_pred)
        val_metrics = compute_clinical_metrics(y_val_true, y_val_pred)
        
        print_clinical_metrics("TRAINING DATASET", train_metrics)
        print_clinical_metrics("VALIDATION DATASET", val_metrics)
        
        if mlflow is not None:
            # Log final clinical evaluation metrics
            mlflow.log_metrics({
                "train_accuracy": train_metrics["accuracy"],
                "train_precision": train_metrics["precision"],
                "train_recall": train_metrics["recall"],
                "train_f1": train_metrics["f1"],
                "train_rmse": train_metrics["rmse"],
                "val_accuracy": val_metrics["accuracy"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "val_f1": val_metrics["f1"],
                "val_rmse": val_metrics["rmse"],
            })
            
        print("\n" + "=" * 60)
        print("5. EXPORTING TO PRODUCTION ONNX FORMAT")
        print("=" * 60)
        
        # Save as Keras SavedModel natively
        saved_model_dir = Path(tempfile.mkdtemp(prefix="saved_model_"))
        saved_model_dir.mkdir(parents=True, exist_ok=True)
        model.export(str(saved_model_dir))
        print(f"[*] Base graph exported to temporary path: {saved_model_dir}")
        
        # Convert SavedModel to ONNX
        onnx_cmd = [
            sys.executable,
            "-m",
            "tf2onnx.convert",
            "--saved-model",
            str(saved_model_dir),
            "--output",
            target_onnx_path
        ]
        
        try:
            print("[*] Compiling Keras Graph to ONNX representation...")
            subprocess.run(onnx_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(f"[SUCCESS] Exported production ONNX model to: {target_onnx_path}")
            
            if mlflow is not None and os.path.exists(target_onnx_path):
                mlflow.log_artifact(target_onnx_path, artifact_path="model")
                print("[*] ONNX model logged to MLflow artifacts repository.")
        except subprocess.CalledProcessError as e:
            print("[X] ONNX cross-compilation failed.")
            print(e.stderr.decode("utf-8"))
            
    finally:
        if mlflow is not None and run_ctx is not None:
            mlflow.end_run()
            print("[*] MLflow run finished and closed.")

def main():
    config = _load_config()
    train_model(config)

if __name__ == "__main__":
    main()

