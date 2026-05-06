"""
=============================================================================
  LOCAL BIG DATA TRAINER & DEPLOYMENT NODE
=============================================================================
This script streams the HDF5 Database locally into your 16GB of onboard RAM
using generator logic to flawlessly avoid memory overflows, and exports 
`latest.onnx` identically to the Github spec.
=============================================================================
"""

import os
import h5py
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, mean_squared_error
from tensorflow.keras import layers, models
from tensorflow.keras.utils import Sequence
import matplotlib.pyplot as plt
import seaborn as sns
import subprocess
import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

HDF5_DATABASE_PATH = config['paths']['hdf5_database_path']
TARGET_ONNX_PATH = config['paths']['target_onnx_path']
BATCH_SIZE = config['training']['batch_size']
EPOCHS = config['training']['epochs']
LEARNING_RATE = config['training']['learning_rate']
TEST_SPLIT = config['training']['test_split']
INPUT_CHANNELS = config['signal_processing']['top_n_channels']
INPUT_SAMPLES = int(config['signal_processing']['target_hz'] * config['signal_processing']['window_sec'])

print("="*60)
print("1. HARDWARE NEGOTIATION")
print("="*60)

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    print(f"✅ GPU DETECTED: Utilizing NVIDIA GPU -> {physical_devices[0]}")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    print("[X] WARNING: No GPU detected by TF. TensorFlow will fallback precisely to CPU.")


print("\n"+"="*60)
print("2. HDF5 LOCAL DATABASE STREAMING")
print("="*60)

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

with h5py.File(HDF5_DATABASE_PATH, 'r') as h5f:
    print(f"[*] Extracting ONLY lightweight integer labels natively to preserve RAM...")
    y_full = np.array(h5f['y'], dtype=np.int32)
    indices_full = np.arange(len(y_full))

print(f"  -> Total Seizure Labels Extracted : {len(y_full)}")


print("\n"+"="*60)
print("3. CLINICAL STRATIFICATION & LOSS BALANCING")
print("="*60)

# We split the numerical INDICES securely, keeping the massive Float arrays safely on your SSD!
idx_train, idx_test, y_train, y_test = train_test_split(
    indices_full, y_full, test_size=TEST_SPLIT, random_state=42, stratify=y_full
)

print(f"[*] Train Index Map: {len(idx_train)} | Test Index Map: {len(idx_test)}")

classes = np.unique(y_train)
weights_array = compute_class_weight('balanced', classes=classes, y=y_train)
class_weights = {classes[i]: float(weights_array[i]) for i in range(len(classes))}
print(f"[*] Native Class Imbalance Scaler: {class_weights}")

# Instantiating the streaming tunnels securely to your NVMe Drive
train_gen = HDF5Generator(HDF5_DATABASE_PATH, idx_train, batch_size=BATCH_SIZE)
test_gen = HDF5Generator(HDF5_DATABASE_PATH, idx_test, batch_size=BATCH_SIZE)


print("\n"+"="*60)
print("4. 1D-CNN ARCHITECTURE (LOGIT COMPLIANCE)")
print("="*60)

def build_api_compliant_cnn(input_shape):
    inputs = layers.Input(shape=input_shape)
    
    # Mathematical integration to bypass Streamlit API flaws
    x = layers.BatchNormalization(axis=-1, name="Input_Z_Scorer")(inputs)
    
    x = layers.Conv1D(filters=32, kernel_size=7, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    
    x = layers.Conv1D(filters=64, kernel_size=5, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    
    x = layers.Conv1D(filters=128, kernel_size=3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.GlobalAveragePooling1D()(x)
    
    outputs = layers.Dense(2, activation='linear', name="PyTorch_Logit_Mimic")(x)
    return models.Model(inputs=inputs, outputs=outputs)

model = build_api_compliant_cnn(input_shape=(INPUT_CHANNELS, INPUT_SAMPLES))
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

print(f"[*] Launching Core Neural Network execution for {EPOCHS} Epochs...")
history = model.fit(
    train_gen,
    validation_data=test_gen,
    epochs=EPOCHS,
    class_weight=class_weights,
    verbose=1
)


print("\n"+"="*60)
print("5. CLINICAL VALIDATION METRICS")
print("="*60)

print("\nEXTRACTING DEEP LEARNING LOGIT PREDICTIONS")

raw_train_preds = model.predict(train_gen)
y_train_pred = np.argmax(raw_train_preds, axis=1)

raw_test_preds = model.predict(test_gen)
y_test_pred = np.argmax(raw_test_preds, axis=1)

# Retrieve perfectly aligned matching Truth arrays natively from the Generator loops
y_train_true = []
for i in range(len(train_gen)):
    batch_indices = train_gen.indices[i * train_gen.batch_size : (i + 1) * train_gen.batch_size]
    y_train_true.extend(y_full[np.sort(batch_indices)])
y_train_true = np.array(y_train_true)

y_test_true = []
for i in range(len(test_gen)):
    batch_indices = test_gen.indices[i * test_gen.batch_size : (i + 1) * test_gen.batch_size]
    y_test_true.extend(y_full[np.sort(batch_indices)])
y_test_true = np.array(y_test_true)

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

print_metrics("TRAINING DATASET", y_train_true, y_train_pred)
print_metrics("TESTING DATASET", y_test_true, y_test_pred)


print("\n"+"="*60)
print("6. ONNX DEPLOYMENT EXPORT")
print("="*60)

# Make sure we save as SavedModel natively so tf2onnx parses it easily
SAVED_MODEL_DIR = r'D:\Antigravity\chb_seizure_pipeline\saved_model_dir'
model.export(SAVED_MODEL_DIR)
print(f"[*] Base Graph wrapper safely deployed to: {SAVED_MODEL_DIR}")

cmd = [
    "python", "-m", "tf2onnx.convert",
    "--saved-model", SAVED_MODEL_DIR,
    "--output", TARGET_ONNX_PATH,
    "--inputs-as-nchw", "Input_Z_Scorer" 
]

try:
    print(f"[*] Triggering PyTorch ONNX cross-compiler execution...")
    subprocess.run(" ".join(cmd), shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(f"\n[SUCCESS] FULL DEPLOYMENT EXPORTED -> {TARGET_ONNX_PATH}")
    print("   The Streamlit Mock-API is perfectly mapped and ready to stream live EEG data!")
except subprocess.CalledProcessError as e:
    print("\n[X] ONNX Conversion failed.")
    print(e.stderr.decode('utf-8'))
