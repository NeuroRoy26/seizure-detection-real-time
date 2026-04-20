"""
=============================================================================
  LOCAL BIG DATA PROCESSOR: HDF5 DATABASE EXTRACTION
=============================================================================
This script executes the entire Extract-Transform-Load (ETL) pipeline
natively on your local Windows PC, bypassing all cloud dependencies.
It evaluates stability via `channel_selection.py`, filters hardware noise, 
slices clinical data into 2-seconds, and blasts mathematically pure tensors 
directly into a local `train_database.h5` file using your NVMe SSD.
=============================================================================
"""

import os
import glob
import codecs
import re
import h5py
import mne
import numpy as np
import warnings
import yaml

# Suppress visual MNE duplication spam in Windows console
warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")

# Import the MATLAB-translated script we wrote earlier
from channel_selection import calculate_channel_stability

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# ── DYNAMIC CONFIGURATION ─────────────────────────────────────────────────
LOCAL_DATA_DIR = config['paths']['local_data_dir']
HDF5_DATABASE_PATH = config['paths']['hdf5_database_path']

WINDOW_SEC = config['signal_processing']['window_sec']
TARGET_HZ = config['signal_processing']['target_hz']
TOP_N_CHANNELS = config['signal_processing']['top_n_channels']

print("="*60)
print("1. DISCOVERING DATA & CALCULATING CHANNEL TOPOGRAPHY")
print("="*60)

# Recursively locate every EDF in the massive payload
all_edfs = sorted(glob.glob(os.path.join(LOCAL_DATA_DIR, '**', '*.edf'), recursive=True))
print(f"[*] Native Windows Scan discovered: {len(all_edfs)} EDFs.")

if len(all_edfs) == 0:
    raise FileNotFoundError("\n[X] Zero EDFs found! Please double-check LOCAL_DATA_DIR!")

# Grab 3 random patient files to run the stability check algorithm
discovery_files = all_edfs[:3]
sessions_data = []

sfreq_global = None
channel_names_global = None

for path in discovery_files:
    raw = mne.io.read_raw_edf(path, preload=True)
    sessions_data.append(raw.get_data())
    channel_names_global = raw.ch_names
    sfreq_global = raw.info['sfreq']

# Run the Scipy Mathematical Evaluation locally
discovery_results = calculate_channel_stability(sessions_data, sfreq_global, channel_names_global, top_n=TOP_N_CHANNELS)
BEST_INDICES = discovery_results['best_indices']


print("\n"+"="*60)
print("2. BUILDING LOCAL HDF5 COMPRESSION DATABASE")
print("="*60)

def parse_seizure_summary(summary_path):
    seizure_data = {}
    with codecs.open(summary_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    file_blocks = content.split("File Name: ")[1:]
    for block in file_blocks:
        lines = block.strip().split('\n')
        fname = lines[0].strip()
        seizures = []
        starts = re.findall(r'Start Time:\s*(\d+)\s*seconds', block, re.IGNORECASE)
        ends = re.findall(r'End Time:\s*(\d+)\s*seconds', block, re.IGNORECASE)
        for s, e in zip(starts, ends):
            seizures.append((int(s), int(e)))
        seizure_data[fname] = seizures
    return seizure_data

def preprocess_and_window(raw, seizure_times, best_indices, target_hz, win_sec):
    sfreq = raw.info['sfreq']
    
    raw.filter(1.0, 50.0, method='fir', verbose=False)
    raw.notch_filter(60.0, method='fir', verbose=False)
    if sfreq > target_hz:
        raw.resample(target_hz, verbose=False)
    
    data = raw.get_data()
    data = data[best_indices, :] # SHAVE TO TOP 10 CHANNELS
    
    win_samples = int(target_hz * win_sec)
    n_windows = data.shape[1] // win_samples
    X, y = [], []
    
    for i in range(n_windows):
        start_idx = i * win_samples
        end_idx = start_idx + win_samples
        chunk = data[:, start_idx:end_idx]
        
        start_sec = start_idx / target_hz
        end_sec = end_idx / target_hz
        label = 0
        for (sz_start, sz_end) in seizure_times:
            if not (end_sec <= sz_start or start_sec >= sz_end):
                label = 1
                break
        
        X.append(chunk)
        y.append(label)
        
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


# Boot up the HDF5 writer over the local NVMe logic
with h5py.File(HDF5_DATABASE_PATH, 'w') as h5f:
    max_dims = (None, len(BEST_INDICES), int(TARGET_HZ * WINDOW_SEC))
    ds_x = h5f.create_dataset('X', shape=(0, len(BEST_INDICES), 256), maxshape=max_dims, dtype='float32', chunks=True)
    ds_y = h5f.create_dataset('y', shape=(0,), maxshape=(None,), dtype='int32', chunks=True)
    
    # Locate all patient directories accurately
    patient_dirs = sorted(glob.glob(os.path.join(LOCAL_DATA_DIR, 'chb*')))
    
    for p_dir in patient_dirs:
        if not os.path.isdir(p_dir): continue
        summary_file = glob.glob(os.path.join(p_dir, '*summary.txt'))
        if not summary_file: continue
        
        seizure_map = parse_seizure_summary(summary_file[0])
        edfs = sorted(glob.glob(os.path.join(p_dir, '*.edf')))
        
        for edf_path in edfs:
            fname = os.path.basename(edf_path)
            try:
                raw_edf = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
                s_times = seizure_map.get(fname, [])
                X_chunk, y_chunk = preprocess_and_window(raw_edf, s_times, BEST_INDICES, TARGET_HZ, WINDOW_SEC)
                
                if len(X_chunk) > 0:
                    current_size = ds_x.shape[0]
                    ds_x.resize(current_size + len(X_chunk), axis=0)
                    ds_y.resize(current_size + len(X_chunk), axis=0)
                    ds_x[current_size:] = X_chunk
                    ds_y[current_size:] = y_chunk
                    
                print(f"  [OK] Extracted {len(X_chunk):>4} clinical windows -> {fname}")
                
            except Exception as e:
                print(f"  [X] Skipped locally corrupted file {fname} | Error: {e}")

print("\n=======================================================")
print(f"✅ LOCAL HDF5 PIPELINE COMPLETE! Database securely locked.")
print(f"   Database Path: {HDF5_DATABASE_PATH}")
print("=======================================================")
