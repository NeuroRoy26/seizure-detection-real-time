"""
=============================================================================
  DATA VALIDATION ENGINE (GREAT EXPECTATIONS 1.x API)
=============================================================================
Defines data quality gates for windowed EEG arrays. Validates correct shape,
complete presence of data (no NaNs), and standard microvolt scaling boundaries.
=============================================================================
"""

import pandas as pd
import numpy as np
import great_expectations as gx

def validate_eeg_data(
    data: np.ndarray, 
    amp_min: float = -1000.0, 
    amp_max: float = 1000.0
) -> bool:
    """
    Validates the entire EEG recording array at once using Great Expectations 1.x.
    
    Parameters
    ----------
    data : np.ndarray
        EEG data array of shape (channels, samples).
    amp_min : float
        Minimum allowed amplitude in microvolts.
    amp_max : float
        Maximum allowed amplitude in microvolts.
        
    Returns
    -------
    bool
        True if all expectations are met, False otherwise.
    """
    if len(data.shape) != 2:
        print(f"[!] Validation failed: Expected 2D array (channels, samples), got shape {data.shape}")
        return False
        
    n_channels, n_samples = data.shape
    if n_channels != 10:
        print(f"[!] Validation failed: Expected exactly 10 channels, got {n_channels}")
        return False
        
    # Convert array to DataFrame: columns = channels, rows = time samples
    df = pd.DataFrame(data.T, columns=[f"ch_{i}" for i in range(n_channels)])
    
    try:
        # Initialize context (loads ephemeral context if offline)
        context = gx.get_context()
        
        # 1. Get or create datasource
        ds_name = "eeg_datasource"
        try:
            datasource = context.data_sources.get(ds_name)
        except Exception:
            datasource = context.data_sources.add_pandas(ds_name)
            
        # 2. Get or create asset
        asset_name = "eeg_asset"
        try:
            asset = datasource.get_asset(asset_name)
        except Exception:
            asset = datasource.add_dataframe_asset(asset_name)
            
        # 3. Get or create batch definition
        batch_def_name = "eeg_batch_def"
        try:
            batch_definition = asset.get_batch_definition(batch_def_name)
        except Exception:
            batch_definition = asset.add_batch_definition_whole_dataframe(batch_def_name)
            
        # 4. Get or create expectation suite
        suite_name = "eeg_validation_suite"
        try:
            suite = context.suites.get(suite_name)
        except Exception:
            suite = context.suites.add(gx.core.expectation_suite.ExpectationSuite(name=suite_name))
            
            # Add expectations for each channel
            for col in df.columns:
                suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column=col))
                suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(
                    column=col, 
                    min_value=amp_min, 
                    max_value=amp_max,
                    mostly=0.95
                ))



                
        # 5. Run validation on current window dataframe
        batch = batch_definition.get_batch(batch_parameters={"dataframe": df})
        validation_result = batch.validate(suite)
        
        return bool(validation_result.success)
        
    except Exception as e:
        print(f"[X] Internal error during Great Expectations validation: {e}")
        return False
