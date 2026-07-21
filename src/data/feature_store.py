"""
=============================================================================
  LOCAL OFFLINE FEATURE STORE
=============================================================================
Provides a structured, file-backed feature repository inside an HDF5 container.
Houses two distinct feature groups:
 1. `raw_signals`: Tensors of shape (N, channels, time) for CNN models.
 2. `engineered_features`: Vectors of shape (N, channels, features) for 
    classical machine learning baselines or statistics.
=============================================================================
"""

import os
import h5py
import numpy as np
from typing import Dict, Tuple

class LocalFeatureStore:
    """
    Local file-based Feature Store built on HDF5, optimized for low memory
    overhead and fast NVMe batch updates.
    """
    def __init__(self, h5_path: str):
        self.h5_path = h5_path

    def initialize_store(
        self, 
        n_channels: int = 10, 
        n_samples: int = 256, 
        n_features: int = 9
    ) -> None:
        """
        Initializes groups and empty chunked datasets in the HDF5 file.
        """
        os.makedirs(os.path.dirname(self.h5_path) or ".", exist_ok=True)
        
        with h5py.File(self.h5_path, "w") as h5f:
            # ── Raw Signals Feature Group ──
            g_raw = h5f.create_group("raw_signals")
            g_raw.create_dataset(
                "X",
                shape=(0, n_channels, n_samples),
                maxshape=(None, n_channels, n_samples),
                dtype="float32",
                chunks=True
            )
            g_raw.create_dataset(
                "y",
                shape=(0,),
                maxshape=(None,),
                dtype="int32",
                chunks=True
            )
            
            # ── Engineered Tabular Feature Group ──
            g_eng = h5f.create_group("engineered_features")
            g_eng.create_dataset(
                "X",
                shape=(0, n_channels, n_features),
                maxshape=(None, n_channels, n_features),
                dtype="float32",
                chunks=True
            )
            g_eng.create_dataset(
                "y",
                shape=(0,),
                maxshape=(None,),
                dtype="int32",
                chunks=True
            )
            
            print(f"[*] Initialized HDF5 local Feature Store at: {self.h5_path}")

    def append_batch(
        self, 
        X_raw: np.ndarray, 
        X_eng: np.ndarray, 
        y: np.ndarray
    ) -> None:
        """
        Appends a processed batch of raw signals and engineered features to the store.
        """
        if len(X_raw) == 0:
            return
            
        with h5py.File(self.h5_path, "a") as h5f:
            batch_size = len(X_raw)
            
            # 1. Store Raw Signals
            ds_raw_X = h5f["raw_signals/X"]
            ds_raw_y = h5f["raw_signals/y"]
            curr_raw_size = ds_raw_X.shape[0]
            
            ds_raw_X.resize(curr_raw_size + batch_size, axis=0)
            ds_raw_y.resize(curr_raw_size + batch_size, axis=0)
            
            ds_raw_X[curr_raw_size:] = X_raw
            ds_raw_y[curr_raw_size:] = y
            
            # 2. Store Engineered Features
            ds_eng_X = h5f["engineered_features/X"]
            ds_eng_y = h5f["engineered_features/y"]
            curr_eng_size = ds_eng_X.shape[0]
            
            ds_eng_X.resize(curr_eng_size + batch_size, axis=0)
            ds_eng_y.resize(curr_eng_size + batch_size, axis=0)
            
            ds_eng_X[curr_eng_size:] = X_eng
            ds_eng_y[curr_eng_size:] = y

    def get_dataset_shapes(self) -> Dict[str, Tuple[int, ...]]:
        """
        Returns the shapes of the datasets inside the feature store.
        """
        if not os.path.exists(self.h5_path):
            return {}
            
        with h5py.File(self.h5_path, "r") as h5f:
            return {
                "raw_signals_X": h5f["raw_signals/X"].shape,
                "raw_signals_y": h5f["raw_signals/y"].shape,
                "engineered_features_X": h5f["engineered_features/X"].shape,
                "engineered_features_y": h5f["engineered_features/y"].shape,
            }
