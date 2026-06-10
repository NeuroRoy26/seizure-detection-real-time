"""
=============================================================================
  DISTRIBUTED SPARK PREPROCESSING PIPELINE (ETL)
=============================================================================
Uses PySpark to parallelize EDF signal processing across multiple worker cores
or cluster nodes. Integrates Great Expectations validation, time-frequency
feature engineering, and saves results to a distributed Parquet Feature Store.
=============================================================================
"""

import os
import sys
import glob
import codecs
import re
import yaml
import numpy as np
import warnings

# Ensure root directory is in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from pyspark.sql import SparkSession
    from pyspark.sql.types import StructType, StructField, StringType, IntegerType, ArrayType, FloatType
except ImportError:
    print("[ERROR] PySpark is not installed in the active environment.")
    print("Please install it via: venv\\Scripts\\pip install pyspark")
    SparkSession = None

try:
    import mne
except ImportError:
    mne = None

from channel_selection import calculate_channel_stability
from src.validation import validate_eeg_data
from src.features import extract_eeg_features

warnings.filterwarnings("ignore")
if mne is not None:
    mne.set_log_level("ERROR")

def parse_seizure_summary(summary_path: str) -> dict:
    """Parses patient summary text file to extract seizure start/end timestamps."""
    seizure_data = {}
    with codecs.open(summary_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    file_blocks = content.split("File Name: ")[1:]
    for block in file_blocks:
        lines = block.strip().split('\n')
        if not lines:
            continue
        fname = lines[0].strip()
        seizures = []
        starts = re.findall(r'Start Time:\s*(\d+)\s*seconds', block, re.IGNORECASE)
        ends = re.findall(r'End Time:\s*(\d+)\s*seconds', block, re.IGNORECASE)
        for s, e in zip(starts, ends):
            seizures.append((int(s), int(e)))
        seizure_data[fname] = seizures
    return seizure_data

def process_file_in_spark_worker(edf_path: str, best_indices: list, target_hz: float, window_sec: float, amp_min: float, amp_max: float, seizure_times: list):
    """Processes a single EDF file inside a Spark worker and yields windowed rows."""
    import mne
    mne.set_log_level("ERROR")
    
    try:
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        sfreq = raw.info['sfreq']
        
        # Apply filters
        raw.filter(1.0, 50.0, method='fir', verbose=False)
        raw.notch_filter(60.0, method='fir', verbose=False)
        
        if sfreq > target_hz:
            raw.resample(target_hz, verbose=False)
            
        data = raw.get_data()
        data = data[best_indices, :]  # Select top 10 channels
        data = data * 1e6              # Volts -> Microvolts
        
        # Great Expectations validation
        if not validate_eeg_data(data, amp_min, amp_max):
            raw.close()
            return []
            
        win_samples = int(target_hz * window_sec)
        n_windows = data.shape[1] // win_samples
        
        results = []
        file_name = os.path.basename(edf_path)
        patient_id = os.path.basename(os.path.dirname(edf_path))
        
        for i in range(n_windows):
            start_idx = i * win_samples
            end_idx = start_idx + win_samples
            window_data = data[:, start_idx:end_idx]
            
            # Extract features (channels, 9)
            window_features = extract_eeg_features(window_data, sfreq=target_hz)
            
            # Label calculations
            start_sec = start_idx / target_hz
            end_sec = end_idx / target_hz
            label = 0
            for (sz_start, sz_end) in seizure_times:
                if not (end_sec <= sz_start or start_sec >= sz_end):
                    label = 1
                    break
            
            # Flatten arrays to list of floats for Spark schemas
            raw_flat = window_data.flatten().astype(float).tolist()
            features_flat = window_features.flatten().astype(float).tolist()
            
            results.append((
                patient_id,
                file_name,
                i,
                raw_flat,
                features_flat,
                label
            ))
            
        raw.close()
        return results
    except Exception as e:
        print(f"[X] Worker failed on {os.path.basename(edf_path)}: {e}")
        return []

def _load_config(config_path: str = "config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    if SparkSession is None:
        print("[ERROR] Cannot run preprocess_spark.py without PySpark package.")
        sys.exit(1)

    config = _load_config()
    
    local_data_dir = config["paths"]["local_data_dir"]
    parquet_dataset_path = config["paths"]["parquet_dataset_path"]
    
    window_sec = config["signal_processing"]["window_sec"]
    target_hz = config["signal_processing"]["target_hz"]
    top_n_channels = config["signal_processing"]["top_n_channels"]
    
    amp_min = config["validation"]["amplitude_min_microvolts"]
    amp_max = config["validation"]["amplitude_max_microvolts"]
    
    print("=" * 60)
    print("1. DISCOVERING LOCAL DATASETS & CHANNEL TOPOGRAPHY")
    print("=" * 60)
    
    all_edfs = sorted(glob.glob(os.path.join(local_data_dir, "**", "*.edf"), recursive=True))
    print(f"[*] Discovered {len(all_edfs)} EDF recordings.")
    
    if len(all_edfs) == 0:
        print("[!] No EDFs found; skipping spark preprocessing.")
        return
        
    # Standardize channel stability locks from first 3 files
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
    
    # Save stable locked indices to config.yaml
    config["signal_processing"]["best_indices"] = best_indices
    with open("config.yaml", "w") as f:
        yaml.safe_dump(config, f)
    print(f"[*] Locked channels in config.yaml: {best_indices}")

    print("\n" + "=" * 60)
    print("2. ESTABLISHING SPARK CLUSTER SESSION")
    print("=" * 60)
    
    spark = SparkSession.builder \
        .appName("EEG-Seizure-Detection-Preprocessing") \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .config("spark.sql.parquet.compression.codec", "snappy") \
        .getOrCreate()
        
    print(f"[*] Spark UI Web Interface: {spark.sparkContext.uiWebUrl}")

    # Build queue list of files
    patient_dirs = sorted(glob.glob(os.path.join(local_data_dir, "chb*")))
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
            # Store as strings/simple types to parallelize inside Spark
            tasks.append((edf_path, str(s_times)))

    print(f"[*] Processing {len(tasks)} files in Spark RDD partition queue...")
    
    # Parallelize file tasks
    rdd = spark.sparkContext.parallelize(tasks, numSlices=max(1, len(tasks) // 2))
    
    # Run maps on partition workers
    def map_partition_func(partition):
        for edf_path, s_times_str in partition:
            # Parse seizure times list from string
            s_times = eval(s_times_str)
            records = process_file_in_spark_worker(
                edf_path, best_indices, target_hz, window_sec, amp_min, amp_max, s_times
            )
            for r in records:
                yield r

    result_rdd = rdd.mapPartitions(map_partition_func)
    
    # Define DataFrame schema
    schema = StructType([
        StructField("patient_id", StringType(), False),
        StructField("file_name", StringType(), False),
        StructField("window_idx", IntegerType(), False),
        StructField("raw_signal", ArrayType(FloatType()), False),
        StructField("engineered_features", ArrayType(FloatType()), False),
        StructField("label", IntegerType(), False)
    ])
    
    df = spark.createDataFrame(result_rdd, schema)
    
    print(f"[*] Writing distributed Feature Store to: {parquet_dataset_path}...")
    df.write.mode("overwrite").parquet(parquet_dataset_path)
    
    print("\n=======================================================")
    print("✅ SPARK PREPROCESSING COMPLETE!")
    print(f"   Feature Store Parquet: {parquet_dataset_path}")
    print("=======================================================")
    
    spark.stop()

if __name__ == "__main__":
    main()
