import os
import h5py
import yaml
import numpy as np
import pandas as pd
import great_expectations as gx

def _load_config():
    config = {}
    if os.path.exists("config.yaml"):
        try:
            with open("config.yaml", "r") as f:
                config = yaml.safe_load(f)
        except Exception:
            pass
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

def main():
    config = _load_config()
    db_path = config.get("paths", {}).get("hdf5_database_path_2d", "train_database.h5")
    
    print("=" * 60)
    print("GREAT EXPECTATIONS DATABASE-WIDE QUALITY GATE")
    print("=" * 60)
    print(f"[*] Validating HDF5 Feature Store: {db_path}")
    
    if not os.path.exists(db_path):
        print(f"[ERROR] Database file not found at: {db_path}")
        return False
        
    with h5py.File(db_path, "r") as h5f:
        x_key = "raw_signals/X" if "raw_signals/X" in h5f else "X"
        y_key = "raw_signals/y" if "raw_signals/y" in h5f else "y"
        
        if x_key not in h5f or y_key not in h5f:
            print(f"[ERROR] Required datasets not found in HDF5 file. Keys present: {list(h5f.keys())}")
            return False
            
        # Get shape of datasets
        X_shape = h5f[x_key].shape
        y_shape = h5f[y_key].shape
        print(f"[*] Detected raw_signals/X shape: {X_shape}")
        print(f"[*] Detected raw_signals/y shape: {y_shape}")
        
        if len(X_shape) != 3:
            print(f"[ERROR] Expected 3D array (samples, channels, time) for X, got {len(X_shape)}D")
            return False
            
        n_samples, n_channels, n_time = X_shape
        if n_channels != 10:
            print(f"[ERROR] Expected exactly 10 channels, got {n_channels}")
            return False
            
        if X_shape[0] != y_shape[0]:
            print(f"[ERROR] Samples dimension mismatch: X has {X_shape[0]} but y has {y_shape[0]}")
            return False
            
        # Validate a representative random subset (e.g. up to 1000 samples)
        max_validation_samples = 1000
        if n_samples > max_validation_samples:
            print(f"[*] Database is large ({n_samples} samples). Validating a subset of {max_validation_samples} samples.")
            indices = np.random.choice(n_samples, max_validation_samples, replace=False)
            indices.sort()
            X_data = h5f[x_key][indices]
            y_data = h5f[y_key][indices]
        else:
            X_data = h5f[x_key][:]
            y_data = h5f[y_key][:]

    # Reshape X to 2D for Great Expectations (columns = channels, rows = samples * timepoints)
    samples_concat = X_data.transpose(1, 0, 2).reshape(10, -1).T
    df = pd.DataFrame(samples_concat, columns=[f"ch_{i}" for i in range(10)])
    # Add label column replicated across timepoints
    df["label"] = np.repeat(y_data, n_time)
    
    # Run Great Expectations validation
    context = gx.get_context()
    ds_name = "db_validation_datasource"
    try:
        datasource = context.data_sources.get(ds_name)
    except Exception:
        datasource = context.data_sources.add_pandas(ds_name)
        
    asset_name = "db_validation_asset"
    try:
        asset = datasource.get_asset(asset_name)
    except Exception:
        asset = datasource.add_dataframe_asset(asset_name)
        
    batch_def_name = "db_validation_batch_def"
    try:
        batch_definition = asset.get_batch_definition(batch_def_name)
    except Exception:
        batch_definition = asset.add_batch_definition_whole_dataframe(batch_def_name)
        
    suite_name = "db_validation_suite"
    try:
        suite = context.suites.get(suite_name)
    except Exception:
        suite = context.suites.add(gx.core.expectation_suite.ExpectationSuite(name=suite_name))
        
        # Define expectations
        amp_min = config.get("validation", {}).get("amplitude_min_microvolts", -1000.0)
        amp_max = config.get("validation", {}).get("amplitude_max_microvolts", 1000.0)
        
        for col in df.columns:
            if col == "label":
                suite.add_expectation(gx.expectations.ExpectColumnValuesToBeInSet(column=col, value_set=[0, 1]))
            else:
                suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column=col))
                suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(
                    column=col,
                    min_value=amp_min,
                    max_value=amp_max,
                    mostly=0.95
                ))

    batch = batch_definition.get_batch(batch_parameters={"dataframe": df})
    validation_result = batch.validate(suite)
    
    if validation_result.success:
        print("[SUCCESS] Great Expectations validation passed! Data quality meets clinical criteria.")
        return True
    else:
        print("[ERROR] Great Expectations validation failed. Please check the logs.")
        # Print summaries of failures
        for result in validation_result.results:
            if not result.success:
                print(f"  - Failed: {result.expectation_config.expectation_type} on column {result.expectation_config.kwargs.get('column')}")
        return False

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
