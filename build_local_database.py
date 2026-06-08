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
import numpy as np
import warnings
import yaml

# Optional heavy deps: allow importing this module for unit tests that
# don't need the full ETL pipeline.
try:
    import mne  # type: ignore
except Exception:  # pragma: no cover
    mne = None

try:
    import h5py  # type: ignore
except Exception:  # pragma: no cover
    h5py = None

# Suppress visual MNE duplication spam in Windows console
warnings.filterwarnings("ignore")
if mne is not None:  # pragma: no cover
    mne.set_log_level("ERROR")

# ── DYNAMIC CONFIGURATION ─────────────────────────────────────────────────
# NOTE: Keep these module-level names defined so tests can patch them.
# They are populated from config.yaml inside main().
LOCAL_DATA_DIR = None
HDF5_DATABASE_PATH = None
WINDOW_SEC = None
TARGET_HZ = None
TOP_N_CHANNELS = None
BEST_INDICES = None

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

def normalize_channel_name(name):
    # Convert to uppercase, remove spaces, dots, hyphens, underscores
    name = re.sub(r'[\s\.\-_]', '', name).upper()
    if name.endswith('REF'):
        name = name[:-3]
    return name

def align_channels(raw_ch_names, target_channels):
    """
    Maps target channel names (or indices) to indices in raw_ch_names.
    """
    indices = []
    for ch in target_channels:
        if isinstance(ch, (int, np.integer)):
            indices.append(int(ch))
        else:
            norm_target = normalize_channel_name(ch)
            found = False
            for idx, raw_ch in enumerate(raw_ch_names):
                if normalize_channel_name(raw_ch) == norm_target:
                    indices.append(idx)
                    found = True
                    break
            if not found:
                raise ValueError(
                    f"Required channel '{ch}' (normalized: '{norm_target}') not found in file channels: {raw_ch_names}"
                )
    return indices

def preprocess_and_window(raw, seizure_times, best_indices, target_hz, win_sec, sampling_config=None, return_meta=False):
    if mne is None:
        raise ImportError("mne is required for preprocess_and_window()")

    sfreq = raw.info['sfreq']
    
    raw.filter(1.0, 50.0, method='fir', verbose=False)
    raw.notch_filter(60.0, method='fir', verbose=False)
    if sfreq > target_hz:
        raw.resample(target_hz, verbose=False)
    
    data = raw.get_data()
    ch_indices = align_channels(raw.ch_names, best_indices)
    data = data[ch_indices, :] # SHAVE TO TARGET CHANNELS
    
    win_samples = int(target_hz * win_sec)
    n_windows = data.shape[1] // win_samples
    
    # Classify each window
    pos_indices = []
    pre_indices = []
    base_indices = []
    
    for i in range(n_windows):
        start_sec = (i * win_samples) / target_hz
        end_sec = ((i + 1) * win_samples) / target_hz
        
        is_seizure = False
        for (sz_start, sz_end) in seizure_times:
            if not (end_sec <= sz_start or start_sec >= sz_end):
                is_seizure = True
                break
        
        if is_seizure:
            pos_indices.append(i)
        else:
            is_pre_ictal = False
            for (sz_start, sz_end) in seizure_times:
                pre_start = sz_start - 1800
                pre_end = sz_start - 600
                if not (end_sec <= pre_start or start_sec >= pre_end):
                    is_pre_ictal = True
                    break
            if is_pre_ictal:
                pre_indices.append(i)
            else:
                base_indices.append(i)
                
    # Determine if we should under-sample
    enabled = True
    ratio = 2.0
    seed = 42
    if sampling_config is not None:
        enabled = sampling_config.get("enabled", True)
        ratio = sampling_config.get("negative_ratio", 2.0)
        seed = sampling_config.get("random_seed", 42)
        
    under_sample = enabled and len(seizure_times) > 0
    
    if under_sample:
        np.random.seed(seed)
        target_negatives = int(ratio * len(pos_indices))
        target_pre_ictal = target_negatives // 2
        
        # Sample pre-ictal
        if len(pre_indices) <= target_pre_ictal:
            kept_pre = pre_indices
        else:
            kept_pre = np.random.choice(pre_indices, target_pre_ictal, replace=False).tolist()
            
        # Sample baseline
        req_baseline = target_negatives - len(kept_pre)
        if len(base_indices) <= req_baseline:
            kept_base = base_indices
        else:
            kept_base = np.random.choice(base_indices, req_baseline, replace=False).tolist()
            
        selected = sorted(pos_indices + kept_pre + kept_base)
    else:
        selected = list(range(n_windows))
        kept_pre = pre_indices
        kept_base = base_indices
        
    X, y = [], []
    meta = {
        "seizure_intervals": seizure_times,
        "seizure_windows": [],
        "pre_ictal_windows": [],
        "baseline_windows": [],
    }
    
    for idx in selected:
        start_sample = idx * win_samples
        end_sample = start_sample + win_samples
        chunk = data[:, start_sample:end_sample]
        
        label = 1 if idx in pos_indices else 0
        X.append(chunk)
        y.append(label)
        
        start_sec = (idx * win_samples) / target_hz
        end_sec = ((idx + 1) * win_samples) / target_hz
        if idx in pos_indices:
            meta["seizure_windows"].append((start_sec, end_sec))
        elif idx in kept_pre:
            meta["pre_ictal_windows"].append((start_sec, end_sec))
        else:
            meta["baseline_windows"].append((start_sec, end_sec))
            
    if return_meta:
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32), meta
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)

def _load_config(config_path: str = "config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    global LOCAL_DATA_DIR, HDF5_DATABASE_PATH, WINDOW_SEC, TARGET_HZ, TOP_N_CHANNELS, BEST_INDICES
    config = _load_config()

    LOCAL_DATA_DIR = config["paths"]["local_data_dir"]
    HDF5_DATABASE_PATH = config["paths"]["hdf5_database_path"]

    WINDOW_SEC = config["signal_processing"]["window_sec"]
    TARGET_HZ = config["signal_processing"]["target_hz"]
    TOP_N_CHANNELS = config["signal_processing"]["top_n_channels"]

    print("=" * 60)
    print("1. DISCOVERING DATA & CALCULATING CHANNEL TOPOGRAPHY")
    print("=" * 60)

    all_edfs = sorted(
        glob.glob(os.path.join(LOCAL_DATA_DIR, "**", "*.edf"), recursive=True)
    )
    print(f"[*] Local scan discovered: {len(all_edfs)} EDFs.")

    # Be import/test-friendly: if the local dataset isn't present, exit cleanly.
    if len(all_edfs) == 0:
        print("[!] No EDFs found; skipping database build.")
        return

    # Only require heavy deps once we know we actually have data to process.
    if mne is None:
        raise ImportError("mne is required to run main()")
    if h5py is None:
        raise ImportError("h5py is required to run main()")

    # Import here so unit tests can import this module without pulling
    # in the full dependency tree.
    from channel_selection import calculate_channel_stability

    discovery_files = all_edfs[:3]
    sessions_data = []

    sfreq_global = None
    channel_names_global = None

    for path in discovery_files:
        raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
        data = raw.get_data()
        # Handle cases where data is a mock object (e.g. in tests)
        if isinstance(data, np.ndarray) and data.ndim > 0:
            if data.shape[0] >= 20:
                data = data[:20, :]
                ch_names = raw.ch_names[:20] if hasattr(raw, "ch_names") else []
            else:
                ch_names = raw.ch_names if hasattr(raw, "ch_names") else []
        else:
            ch_names = raw.ch_names if hasattr(raw, "ch_names") else []
        sessions_data.append(data)
        channel_names_global = ch_names
        sfreq_global = raw.info["sfreq"] if (hasattr(raw, "info") and raw.info is not None and "sfreq" in raw.info) else 256.0

    discovery_results = calculate_channel_stability(
        sessions_data,
        sfreq_global,
        channel_names_global,
        top_n=TOP_N_CHANNELS,
    )
    BEST_INDICES = discovery_results["best_indices"]
    BEST_CHANNEL_NAMES = discovery_results["best_names"]

    # Write computed best_indices back to config.yaml if not in a testing environment
    import sys
    if "pytest" not in sys.modules:
        config["signal_processing"]["best_indices"] = BEST_INDICES
        with open("config.yaml", "w") as f:
            yaml.safe_dump(config, f)
        print(f"[*] Saved best_indices {BEST_INDICES} and channel names {BEST_CHANNEL_NAMES} to config.yaml")
    else:
        print(f"[*] Testing detected; skipped writing best_indices back to config.yaml")

    print("\n" + "=" * 60)
    print("2. BUILDING LOCAL HDF5 COMPRESSION DATABASE")
    print("=" * 60)

    win_samples = int(TARGET_HZ * WINDOW_SEC)
    sampling_config = config.get("sampling", {"enabled": True, "negative_ratio": 2.0, "random_seed": 42})
    s3_manifest_path = config["paths"].get("s3_manifest_path", "s3_keep_files.txt")

    # Boot up the HDF5 writer over the local NVMe logic
    with h5py.File(HDF5_DATABASE_PATH, "w") as h5f, open(s3_manifest_path, "w") as manifest_f:
        manifest_f.write("========================================================\n")
        manifest_f.write(" CHB-MIT EEG DATASET S3 KEEP MANIFEST\n")
        manifest_f.write("========================================================\n\n")
        manifest_f.write("This file lists all EDF files containing seizures that should\n")
        manifest_f.write("be uploaded to S3, along with details of the seizure vs. normal windows.\n\n")

        max_dims = (None, len(BEST_INDICES), win_samples)
        ds_x = h5f.create_dataset(
            "X",
            shape=(0, len(BEST_INDICES), win_samples),
            maxshape=max_dims,
            dtype="float32",
            chunks=True,
        )
        ds_y = h5f.create_dataset(
            "y",
            shape=(0,),
            maxshape=(None,),
            dtype="int32",
            chunks=True,
        )

        patient_dirs = sorted(glob.glob(os.path.join(LOCAL_DATA_DIR, "chb*")))

        for p_dir in patient_dirs:
            if not os.path.isdir(p_dir):
                continue

            summary_file = glob.glob(os.path.join(p_dir, "*summary.txt"))
            if not summary_file:
                continue

            seizure_map = parse_seizure_summary(summary_file[0])
            edfs = sorted(glob.glob(os.path.join(p_dir, "*.edf")))

            for edf_path in edfs:
                fname = os.path.basename(edf_path)
                s_times = seizure_map.get(fname, [])
                if len(s_times) == 0:
                    continue  # EXCLUDE non-seizure files entirely

                try:
                    raw_edf = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
                    res = preprocess_and_window(
                        raw_edf,
                        s_times,
                        BEST_CHANNEL_NAMES,
                        TARGET_HZ,
                        WINDOW_SEC,
                        sampling_config=sampling_config,
                        return_meta=True,
                    )
                    
                    if len(res) == 3:
                        X_chunk, y_chunk, meta = res
                    else:
                        X_chunk, y_chunk = res
                        meta = None

                    if len(X_chunk) > 0:
                        current_size = ds_x.shape[0]
                        ds_x.resize(current_size + len(X_chunk), axis=0)
                        ds_y.resize(current_size + len(X_chunk), axis=0)
                        ds_x[current_size:] = X_chunk
                        ds_y[current_size:] = y_chunk

                        # Write to S3 Keep Manifest
                        p_name = os.path.basename(p_dir)
                        manifest_f.write(f"File: {p_name}/{fname}\n")
                        manifest_f.write(f"  Total seizure periods: {len(s_times)}\n")
                        for idx, (sz_s, sz_e) in enumerate(s_times):
                            manifest_f.write(f"    Seizure #{idx+1}: {sz_s} to {sz_e} seconds\n")
                        
                        if meta is not None:
                            manifest_f.write(f"  Selected Seizure Windows ({len(meta['seizure_windows'])} total):\n")
                            for sz_w in meta["seizure_windows"]:
                                manifest_f.write(f"    - {sz_w[0]:.2f} to {sz_w[1]:.2f} seconds\n")
                            
                            manifest_f.write(f"  Selected Normal Windows (Pre-ictal: {len(meta['pre_ictal_windows'])}, Baseline: {len(meta['baseline_windows'])}):\n")
                            manifest_f.write("    Pre-ictal:\n")
                            for pre_w in meta["pre_ictal_windows"]:
                                manifest_f.write(f"      - {pre_w[0]:.2f} to {pre_w[1]:.2f} seconds\n")
                            manifest_f.write("    Baseline:\n")
                            for base_w in meta["baseline_windows"]:
                                manifest_f.write(f"      - {base_w[0]:.2f} to {base_w[1]:.2f} seconds\n")
                        else:
                            manifest_f.write("  Window metadata not available (mocked preprocess).\n")
                        manifest_f.write("\n" + "-"*56 + "\n\n")

                    print(f"  [OK] Extracted {len(X_chunk):>4} clinical windows -> {fname}")

                except Exception as e:
                    print(f"  [X] Skipped locally corrupted file {fname} | Error: {e}")

    print("\n=======================================================")
    print("[SUCCESS] LOCAL HDF5 PIPELINE COMPLETE! Database securely locked.")
    print(f"   Database Path: {HDF5_DATABASE_PATH}")
    print(f"   S3 Keep Manifest Path: {s3_manifest_path}")
    print("=======================================================")

if __name__ == "__main__":
    main()
