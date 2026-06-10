"""
=============================================================================
  SAGEMAKER TRAINING NODE
=============================================================================
This script is executed inside a SageMaker TensorFlow training container.
It reads the preprocessed HDF5 database from the S3 mounted channel,
trains the seizure detection model, and exports the final model directly 
to ONNX format in the SageMaker model folder.
=============================================================================
"""

import os
import argparse
import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import Sequence
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error

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
            X_batch = h5f['X'][sorted_idx]
            y_batch = h5f['y'][sorted_idx]
            
        return X_batch, y_batch

def build_api_compliant_cnn(input_shape):
    # Data contract: API feeds ONNX with (1, 10, 256)
    inputs = layers.Input(shape=input_shape)

    # 1D -> 2D Bridge
    x = layers.Reshape((input_shape[0], input_shape[1], 1), name="Bridge_Reshape_10x256x1")(inputs)
    x = layers.Conv2D(
        filters=3,
        kernel_size=(1, 1),
        padding="same",
        name="Bridge_ToRGB_1x1Conv",
    )(x)  # (10, 256, 3)

    x = layers.BatchNormalization(axis=-1, name="Input_Z_Scorer")(x)
    x = layers.Resizing(224, 224, interpolation="bilinear", name="Bridge_Resize_224")(x)

    # Transfer learning backbone (ImageNet), frozen.
    base = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3),
    )
    base.trainable = False

    x = base(x, training=False)

    # Classification head
    x = layers.GlobalAveragePooling2D(name="Head_GAP")(x)
    x = layers.Dropout(0.25, name="Head_Dropout")(x)

    # API expects logits and applies softmax itself.
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
    parser = argparse.ArgumentParser()

    # SageMaker hyperparameters are passed as command-line arguments
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--top-n-channels', type=int, default=10)
    parser.add_argument('--target-hz', type=float, default=256.0)
    parser.add_argument('--window-sec', type=float, default=1.0)
    parser.add_argument('--test-split', type=float, default=0.2)

    # SageMaker specific environment variables / locations
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAINING', '/opt/ml/input/data/training'))

    args, _ = parser.parse_known_args()

    print("=" * 60)
    print("1. SYSTEM & HARDWARE INFO")
    print("=" * 60)
    print(f"TensorFlow Version: {tf.__version__}")
    
    physical_devices = tf.config.list_physical_devices("GPU")
    if len(physical_devices) > 0:
        print(f"GPU DETECTED: Utilizing NVIDIA GPU -> {physical_devices[0]}")
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    else:
        print("WARNING: No GPU detected. TensorFlow will run on CPU.")
        tf.config.threading.set_intra_op_parallelism_threads(2)
        tf.config.threading.set_inter_op_parallelism_threads(2)
        print("[*] Configured TensorFlow CPU threads: 2 (Intra) / 2 (Inter) to prevent OOM.")

    print("\n" + "=" * 60)
    print("2. DATABASE PATH RESOLUTION")
    print("=" * 60)
    h5_path = os.path.join(args.train, "train_database.h5")
    print(f"Looking for dataset at: {h5_path}")
    if not os.path.exists(h5_path):
        # Fallback if channel mounted differently
        print(f"Warning: {h5_path} not found. Checking contents of {args.train}...")
        files = os.listdir(args.train)
        print(f"Files in {args.train}: {files}")
        if len(files) == 1:
            h5_path = os.path.join(args.train, files[0])
            print(f"Using fallback dataset path: {h5_path}")
        else:
            raise FileNotFoundError(f"Could not locate training database .h5 file in {args.train}")

    print("\n" + "=" * 60)
    print("3. LOAD DATASET INTO MEMORY & STRATIFICATION")
    print("=" * 60)
    with h5py.File(h5_path, "r") as h5f:
        print("Loading signals and labels into memory...")
        X_full = np.array(h5f["X"], dtype=np.float32)
        y_full = np.array(h5f["y"], dtype=np.int32)

    print(f"Loaded dataset: X shape {X_full.shape} | y shape {y_full.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_full,
        y_full,
        test_size=args.test_split,
        random_state=42,
        stratify=y_full,
    )

    print(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")

    classes = np.unique(y_train)
    weights_array = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weights = {classes[i]: float(weights_array[i]) for i in range(len(classes))}
    print(f"Native Class Imbalance Scaler: {class_weights}")

    print("\n" + "=" * 60)
    print("4. MODEL COMPILATION")
    print("=" * 60)
    input_samples = int(args.target_hz * args.window_sec)
    model = build_api_compliant_cnn(input_shape=(args.top_n_channels, input_samples))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    model.summary()

    print("\n" + "=" * 60)
    print("5. MODEL TRAINING")
    print("=" * 60)
    model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weights,
        verbose=1,
    )

    print("\n" + "=" * 60)
    print("6. MODEL EVALUATION")
    print("=" * 60)
    raw_train_preds = model.predict(X_train, batch_size=args.batch_size)
    y_train_pred = np.argmax(raw_train_preds, axis=1)

    raw_test_preds = model.predict(X_test, batch_size=args.batch_size)
    y_test_pred = np.argmax(raw_test_preds, axis=1)

    print_metrics("TRAINING DATASET", y_train, y_train_pred)
    print_metrics("TESTING DATASET", y_test, y_test_pred)

    print("\n" + "=" * 60)
    print("7. ONNX EXPORT")
    print("=" * 60)
    
    # Import tf2onnx here to avoid forcing it to be installed if container has issues
    try:
        import tf2onnx
    except ImportError:
        # Install tf2onnx programmatically if not present in the container
        print("tf2onnx not found. Installing via pip...")
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tf2onnx", "onnx<1.16.0", "ml-dtypes==0.2.0"])
        import tf2onnx

    target_onnx_path = os.path.join(args.model_dir, "seizure_detector_mobilenetv2.onnx")
    print(f"Exporting model to ONNX: {target_onnx_path}")
    
    # Convert directly using python API (avoiding subprocess SavedModel step)
    spec = (tf.TensorSpec((None, args.top_n_channels, input_samples), tf.float32, name="eeg"),)
    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=17)
    
    os.makedirs(args.model_dir, exist_ok=True)
    with open(target_onnx_path, "wb") as f:
        f.write(model_proto.SerializeToString())
        
    print(f"[SUCCESS] Exported ONNX model to: {target_onnx_path}")

if __name__ == '__main__':
    main()
