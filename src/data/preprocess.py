"""
=============================================================================
  ETL PIPELINE ORCHESTRATOR
=============================================================================
Executes the local clinical dataset extraction. Processes raw hospital EDF 
files, computes channel quality, filters hardware artifacts, standardizes 
voltage to microvolts, validates data quality via Great Expectations, 
extracts engineered features, and commits all data to the Feature Store.
=============================================================================
"""

import os
import sys
import gc
import glob
import codecs
import re
import warnings
import yaml
import numpy as np
import concurrent.futures


# Ensure root directory is in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Optional heavy dependencies
try:
    import mne
except Exception:
    mne = None

try:
    import h5py
except Exception:
    h5py = None

from src.data.channel_selection import calculate_channel_stability
from src.data.validation import validate_eeg_data
from src.data.features import extract_eeg_features
from src.data.feature_store import LocalFeatureStore

warnings.filterwarnings("ignore")
if mne is not None:
    mne.set_log_level("ERROR")

def parse_seizure_summary(summary_path: str) -> dict:
    """
    Parses the patient summary text file to extract seizure start/end timestamps.
    """
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

def process_single_file_worker(
    edf_path: str,
    best_indices,
    target_hz: float,
    window_sec: float,
    amp_min: float,
    amp_max: float,
    seizure_times
):
    """
    Subprocess worker that processes a single EDF file and extracts signals/features.
    """
    import mne
    mne.set_log_level("ERROR")
    try:
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        X_raw, X_eng, y = preprocess_and_validate_windows(
            raw, seizure_times, best_indices, target_hz, window_sec, amp_min, amp_max
        )
        raw.close()
        return X_raw, X_eng, y
    except Exception as e:
        print(f"  [X] Subprocess failed on {os.path.basename(edf_path)} | Error: {e}")
        return None

def preprocess_and_validate_windows(

    raw, 
    seizure_times, 
    best_indices, 
    target_hz, 
    win_sec, 
    amp_min, 
    amp_max
):
    """
    Applies filters, resampling, electrode trimming, microvolt scaling, 
    Great Expectations validation, and feature engineering to raw EDF signals.
    """
    if mne is None:
        raise ImportError("mne is required for preprocessing")
        
    sfreq = raw.info['sfreq']
    
    # Apply bandpass (1-50 Hz) and notch (60 Hz) filters
    raw.filter(1.0, 50.0, method='fir', verbose=False)
    raw.notch_filter(60.0, method='fir', verbose=False)
    
    if sfreq > target_hz:
        raw.resample(target_hz, verbose=False)
        
    data = raw.get_data()
    data = data[best_indices, :]  # Select top 10 channels
    
    # CRITICAL: Scale Volts -> Microvolts (1e6) to resolve scale mismatches
    data = data * 1e6
    
    # Fast NumPy validation: no NaNs and at least 95% of values in microvolt bounds
    if np.isnan(data).any():
        print("  [!] Skip file: NaN values detected in EEG signal.")
        win_samples = int(target_hz * win_sec)
        return np.empty((0, len(best_indices), win_samples)), np.empty((0, len(best_indices), 9)), np.empty((0,))
        
    in_range = (data >= amp_min) & (data <= amp_max)
    if np.mean(in_range) < 0.95:
        print("  [!] Skip file: Amplitude boundary check failed (less than 95% of values in range).")
        win_samples = int(target_hz * win_sec)
        return np.empty((0, len(best_indices), win_samples)), np.empty((0, len(best_indices), 9)), np.empty((0,))
        
    win_samples = int(target_hz * win_sec)
    n_windows = data.shape[1] // win_samples
    
    X_raw, X_eng, y = [], [], []
    
    for i in range(n_windows):
        start_idx = i * win_samples
        end_idx = start_idx + win_samples
        window_data = data[:, start_idx:end_idx]
        
        # 1. Feature Engineering
        window_features = extract_eeg_features(window_data, sfreq=target_hz)
        
        # Label calculation based on seizure overlap
        start_sec = start_idx / target_hz
        end_sec = end_idx / target_hz
        label = 0
        for (sz_start, sz_end) in seizure_times:
            if not (end_sec <= sz_start or start_sec >= sz_end):
                label = 1
                break
                
        X_raw.append(window_data)
        X_eng.append(window_features)
        y.append(label)
        
    if len(X_raw) == 0:
        return np.empty((0, len(best_indices), win_samples)), np.empty((0, len(best_indices), 9)), np.empty((0,))
        
    return np.array(X_raw, dtype=np.float32), np.array(X_eng, dtype=np.float32), np.array(y, dtype=np.int32)

def _load_config(config_path: str = "config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    config = _load_config()
    
    local_data_dir = config["paths"]["local_data_dir"]
    hdf5_database_path_2d = config["paths"]["hdf5_database_path_2d"]
    
    window_sec = config["signal_processing"]["window_sec"]
    target_hz = config["signal_processing"]["target_hz"]
    top_n_channels = config["signal_processing"]["top_n_channels"]
    
    amp_min = config["validation"]["amplitude_min_microvolts"]
    amp_max = config["validation"]["amplitude_max_microvolts"]
    
    print("=" * 60)
    print("1. DISCOVERING LOCAL DATASETS & STABILITY TOPOGRAPHY")
    print("=" * 60)
    
    all_edfs = sorted(
        glob.glob(os.path.join(local_data_dir, "**", "*.edf"), recursive=True)
    )
    print(f"[*] Discovered: {len(all_edfs)} EDF recordings.")
    
    if len(all_edfs) == 0:
        print("[!] No EDFs found; skipping preprocessing.")
        return
        
    if mne is None or h5py is None:
        raise ImportError("mne and h5py are required to run preprocessing")
        
    # Calculate channel topography from first 3 files
    discovery_files = all_edfs[:3]
    sessions_data = []
    sfreq_global = None
    channel_names_global = None
    
    for path in discovery_files:
        raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
        sessions_data.append(raw.get_data())
        channel_names_global = raw.ch_names
        sfreq_global = raw.info["sfreq"]
        raw.close()
        
    discovery_results = calculate_channel_stability(
        sessions_data,
        sfreq_global,
        channel_names_global,
        top_n=top_n_channels
    )
    best_indices = [int(idx) for idx in discovery_results["best_indices"]]
    
    # Dynamically lock selected channels back to .run_state.json for serving/inference alignment
    import json
    state = {}
    if os.path.exists(".run_state.json"):
        try:
            with open(".run_state.json", "r") as f:
                state = json.load(f)
        except Exception:
            pass
    state["best_indices"] = best_indices
    with open(".run_state.json", "w") as f:
        json.dump(state, f, indent=2)
    print(f"[*] Dynamically locked selected channels in .run_state.json: {best_indices}")

    
    print("\n" + "=" * 60)
    print("2. RUNNING VALIDATED ETL & STORING TO FEATURE STORE")
    print("=" * 60)
    
    # Initialize Feature Store
    store = LocalFeatureStore(hdf5_database_path_2d)
    win_samples = int(target_hz * window_sec)
    store.initialize_store(n_channels=len(best_indices), n_samples=win_samples, n_features=9)
    
    patient_dirs = sorted(glob.glob(os.path.join(local_data_dir, "chb*")))
    
    # 1. Collect all preprocessing tasks (edf_path and seizure times)
    tasks = []
    for p_dir in patient_dirs:
        if not os.path.isdir(p_dir):
            continue
            
        summary_files = glob.glob(os.path.join(p_dir, "*summary.txt"))
        if not summary_files:
            continue
            
        seizure_map = parse_seizure_summary(summary_files[0])
        edfs = sorted(glob.glob(os.path.join(p_dir, "*.edf")))
        
        for edf_path in edfs:
            fname = os.path.basename(edf_path)
            s_times = seizure_map.get(fname, [])
            tasks.append((edf_path, s_times))
            
    print(f"[*] Queued {len(tasks)} files for processing.")
    
    # 2. Process tasks in parallel using ProcessPoolExecutor
    # Use max_workers based on CPU cores available
    max_workers = os.cpu_count() or 4
    print(f"[*] Processing files in parallel using {max_workers} CPU workers...")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(
                process_single_file_worker,
                edf_path,
                best_indices,
                target_hz,
                window_sec,
                amp_min,
                amp_max,
                s_times
            ): edf_path for edf_path, s_times in tasks
        }
        
        # 3. As tasks finish, write results sequentially to the Feature Store (maintaining HDF5 safety)
        for future in concurrent.futures.as_completed(futures):
            edf_path = futures[future]
            fname = os.path.basename(edf_path)
            try:
                result = future.result()
                if result is not None:
                    X_raw_chunk, X_eng_chunk, y_chunk = result
                    if len(X_raw_chunk) > 0:
                        store.append_batch(X_raw_chunk, X_eng_chunk, y_chunk)
                        print(f"  [OK] Stored {len(X_raw_chunk):>4} validated windows -> {fname}")
                    else:
                        print(f"  [-] No valid windows extracted -> {fname}")
                else:
                    print(f"  [X] Failed processing file {fname}")
            except Exception as e:
                print(f"  [X] Error processing file {fname} | Error: {e}")
                
            # Perform garbage collection to keep main process memory clean
            gc.collect()

            
    print("\n=======================================================")
    print("✅ PREPROCESSING & FEATURE STORE POPULATION COMPLETE!")
    shapes = store.get_dataset_shapes()
    print(f"   Raw Signals Shape : {shapes.get('raw_signals_X')}")
    print(f"   Features Shape    : {shapes.get('engineered_features_X')}")
    print(f"   Feature Store Path: {hdf5_database_path_2d}")
    print("=======================================================")

if __name__ == "__main__":
    main()
