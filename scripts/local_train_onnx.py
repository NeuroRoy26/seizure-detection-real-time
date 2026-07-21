"""
=============================================================================
  LOCAL BIG DATA TRAINER & DEPLOYMENT NODE
=============================================================================
This script streams the HDF5 Database locally into your 16GB of onboard RAM
using generator logic to flawlessly avoid memory overflows, and exports 
`latest.onnx` identically to the Github spec.
=============================================================================
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, mean_squared_error
import subprocess
import yaml
import tempfile
from pathlib import Path

# Optional heavy deps: allow importing this module for unit tests that
# don't need the full training / plotting pipeline.
try:
    import tensorflow as tf  # type: ignore
    from tensorflow.keras import layers, models  # type: ignore
    from tensorflow.keras.utils import Sequence  # type: ignore
except Exception:  # pragma: no cover
    tf = None
    layers = None
    models = None

    class Sequence:  # minimal fallback base class
        pass

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover
    plt = None

try:
    import seaborn as sns  # type: ignore
except Exception:  # pragma: no cover
    sns = None

# Keep module-level names defined so tests can patch/inspect them if needed.
HDF5_DATABASE_PATH = None
TARGET_ONNX_PATH = None
BATCH_SIZE = None
EPOCHS = None
LEARNING_RATE = None
TEST_SPLIT = None
INPUT_CHANNELS = None
INPUT_SAMPLES = None


def _load_config(config_path: str = "config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    if os.path.exists(".run_state.json"):
        try:
            import json
            with open(".run_state.json", "r") as f:
                state = json.load(f)
                if "best_indices" in state:
                    if "signal_processing" not in config:
                        config["signal_processing"] = {}
                    config["signal_processing"]["best_indices"] = state["best_indices"]
        except Exception:
            pass
    return config

class HDF5Generator(Sequence):
    def __init__(self, h5_path, indices, batch_size=64):
        self.h5_path = h5_path
        self.indices = indices
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.indices) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        sorted_idx = np.sort(batch_indices)
        
        with h5py.File(self.h5_path, 'r') as h5f:
            x_key = "raw_signals/X" if "raw_signals/X" in h5f else "X"
            y_key = "raw_signals/y" if "raw_signals/y" in h5f else "y"
            X_batch = h5f[x_key][sorted_idx]
            y_batch = h5f[y_key][sorted_idx]
            
        return X_batch, y_batch

def build_api_compliant_cnn(input_shape):
    if layers is None or models is None:
        raise ImportError("tensorflow is required for build_api_compliant_cnn()")

    if tf is None:
        raise ImportError("tensorflow is required for build_api_compliant_cnn()")

    # Data contract (do not change): API feeds ONNX with (1, 10, 256)
    inputs = layers.Input(shape=input_shape)

    # ---------------------------------------------------------------------
    # 1D -> 2D Bridge (strict requirement)
    # Immediately after Input: reshape to (10, 256, 1), then expand to 3ch.
    # ---------------------------------------------------------------------
    x = layers.Reshape((10, 256, 1), name="Bridge_Reshape_10x256x1")(inputs)
    x = layers.Conv2D(
        filters=3,
        kernel_size=(1, 1),
        padding="same",
        name="Bridge_ToRGB_1x1Conv",
    )(x)  # (10, 256, 3)

    # Optional: normalize after the bridge (kept for stable export + parity intent)
    x = layers.BatchNormalization(axis=-1, name="Input_Z_Scorer")(x)

    # Most ImageNet backbones expect >= 32x32 (often 224x224). We resize to a
    # canonical size while keeping the upstream contract unchanged.
    x = layers.Resizing(224, 224, interpolation="bilinear", name="Bridge_Resize_224")(x)

    # ---------------------------------------------------------------------
    # Transfer learning backbone (ImageNet), frozen.
    # ---------------------------------------------------------------------
    base = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3),
    )
    base.trainable = False

    # Forward through frozen backbone in inference mode.
    x = base(x, training=False)

    # ---------------------------------------------------------------------
    # Classification head
    # ---------------------------------------------------------------------
    x = layers.GlobalAveragePooling2D(name="Head_GAP")(x)
    x = layers.Dropout(0.25, name="Head_Dropout")(x)

    # IMPORTANT: api.py expects logits and applies softmax itself.
    outputs = layers.Dense(2, activation="linear", name="PyTorch_Logit_Mimic")(x)

    return models.Model(inputs=inputs, outputs=outputs, name="EEG_TL_MobileNetV2")

def print_metrics(dataset_name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"\n--- {dataset_name} PERFORMANCE ---")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  RMSE      : {rmse:.4f}")

def main():
    global HDF5_DATABASE_PATH
    global TARGET_ONNX_PATH
    global BATCH_SIZE
    global EPOCHS
    global LEARNING_RATE
    global TEST_SPLIT
    global INPUT_CHANNELS
    global INPUT_SAMPLES

    config = _load_config()

    HDF5_DATABASE_PATH = config["paths"]["hdf5_database_path"]
    TARGET_ONNX_PATH = config["paths"]["target_onnx_path"]
    BATCH_SIZE = config["training"]["batch_size"]
    EPOCHS = config["training"]["epochs"]
    LEARNING_RATE = config["training"]["learning_rate"]
    TEST_SPLIT = config["training"]["test_split"]
    INPUT_CHANNELS = config["signal_processing"]["top_n_channels"]
    INPUT_SAMPLES = int(
        config["signal_processing"]["target_hz"] * config["signal_processing"]["window_sec"]
    )

    print("=" * 60)
    print("1. HARDWARE NEGOTIATION")
    print("=" * 60)

    if tf is None:
        raise ImportError("tensorflow is required to run main()")

    physical_devices = tf.config.list_physical_devices("GPU")
    if len(physical_devices) > 0:
        print(f"[SUCCESS] GPU DETECTED: Utilizing NVIDIA GPU -> {physical_devices[0]}")
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    else:
        print("[X] WARNING: No GPU detected by TF. TensorFlow will fallback precisely to CPU.")

    print("\n" + "=" * 60)
    print("2. HDF5 LOCAL DATABASE STREAMING")
    print("=" * 60)

    with h5py.File(HDF5_DATABASE_PATH, "r") as h5f:
        print("[*] Extracting ONLY lightweight integer labels natively to preserve RAM...")
        y_key = "raw_signals/y" if "raw_signals/y" in h5f else "y"
        y_full = np.array(h5f[y_key], dtype=np.int32)
        indices_full = np.arange(len(y_full))

    print(f"  -> Total Seizure Labels Extracted : {len(y_full)}")

    print("\n" + "=" * 60)
    print("3. CLINICAL STRATIFICATION & LOSS BALANCING")
    print("=" * 60)

    idx_train, idx_test, y_train, _ = train_test_split(
        indices_full,
        y_full,
        test_size=TEST_SPLIT,
        random_state=42,
        stratify=y_full,
    )

    print(f"[*] Train Index Map: {len(idx_train)} | Test Index Map: {len(idx_test)}")

    classes = np.unique(y_train)
    weights_array = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weights = {classes[i]: float(weights_array[i]) for i in range(len(classes))}
    print(f"[*] Native Class Imbalance Scaler: {class_weights}")

    train_gen = HDF5Generator(HDF5_DATABASE_PATH, idx_train, batch_size=BATCH_SIZE)
    test_gen = HDF5Generator(HDF5_DATABASE_PATH, idx_test, batch_size=BATCH_SIZE)

    print("\n" + "=" * 60)
    print("4. 1D-CNN ARCHITECTURE (LOGIT COMPLIANCE)")
    print("=" * 60)

    model = build_api_compliant_cnn(input_shape=(INPUT_CHANNELS, INPUT_SAMPLES))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    print(f"[*] Launching Core Neural Network execution for {EPOCHS} Epochs...")
    history = model.fit(
        train_gen,
        validation_data=test_gen,
        epochs=EPOCHS,
        class_weight=class_weights,
        verbose=1,
    )

    print("\n" + "=" * 60)
    print("5. CLINICAL VALIDATION METRICS")
    print("=" * 60)

    print("\nEXTRACTING DEEP LEARNING LOGIT PREDICTIONS")

    raw_train_preds = model.predict(train_gen)
    y_train_pred = np.argmax(raw_train_preds, axis=1)

    raw_test_preds = model.predict(test_gen)
    y_test_pred = np.argmax(raw_test_preds, axis=1)

    # Retrieve perfectly aligned matching Truth arrays natively from the Generator loops
    y_train_true = []
    for i in range(len(train_gen)):
        batch_indices = train_gen.indices[
            i * train_gen.batch_size : (i + 1) * train_gen.batch_size
        ]
        y_train_true.extend(y_full[np.sort(batch_indices)])
    y_train_true = np.array(y_train_true)

    y_test_true = []
    for i in range(len(test_gen)):
        batch_indices = test_gen.indices[
            i * test_gen.batch_size : (i + 1) * test_gen.batch_size
        ]
        y_test_true.extend(y_full[np.sort(batch_indices)])
    y_test_true = np.array(y_test_true)

    print_metrics("TRAINING DATASET", y_train_true, y_train_pred)
    print_metrics("TESTING DATASET", y_test_true, y_test_pred)

    print("\n" + "=" * 60)
    print("6. ONNX DEPLOYMENT EXPORT")
    print("=" * 60)

    # Make sure we save as SavedModel natively so tf2onnx parses it easily
    SAVED_MODEL_DIR = Path(tempfile.mkdtemp(prefix="saved_model_"))
    SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.export(str(SAVED_MODEL_DIR))
    print(f"[*] Base Graph wrapper safely deployed to: {SAVED_MODEL_DIR}")

    import sys
    cmd = [
        sys.executable,
        "-m",
        "tf2onnx.convert",
        "--saved-model",
        str(SAVED_MODEL_DIR),
        "--output",
        TARGET_ONNX_PATH,
    ]

    try:
        print("[*] Triggering PyTorch ONNX cross-compiler execution...")
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        print(f"\n[SUCCESS] FULL DEPLOYMENT EXPORTED -> {TARGET_ONNX_PATH}")
        print("   The Streamlit Mock-API is perfectly mapped and ready to stream live EEG data!")
    except subprocess.CalledProcessError as e:
        print("\n[X] ONNX Conversion failed.")
        print(e.stderr.decode("utf-8"))


if __name__ == "__main__":
    main()
