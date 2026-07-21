import unittest
from unittest.mock import patch, MagicMock, mock_open
import numpy as np
import os
import tempfile
import codecs

import src.data.preprocess as preprocessor

class TestPreprocessPipeline(unittest.TestCase):
    
    def test_parse_seizure_summary(self):
        summary_content = """
File Name: chb01_01.edf
Number of Seizures: 1
Seizure 1 Start Time: 2996 seconds
Seizure 1 End Time: 3036 seconds

File Name: chb01_02.edf
Number of Seizures: 0
"""
        with patch("codecs.open", mock_open(read_data=summary_content)):
            data = preprocessor.parse_seizure_summary("mock_summary.txt")
            self.assertEqual(data.get("chb01_01.edf"), [(2996, 3036)])
            self.assertEqual(data.get("chb01_02.edf"), [])

    @patch("src.data.preprocess.validate_eeg_data", return_value=True)
    def test_preprocess_and_validate_windows_success(self, mock_validate):
        # Create a mock raw object
        raw = MagicMock()
        raw.info = {'sfreq': 256.0}
        
        # 10 channels, 512 samples (2 seconds at 256 Hz)
        raw.get_data.return_value = np.random.normal(0, 1e-6, size=(10, 512))
        
        best_indices = list(range(10))
        seizure_times = [(1, 2)]
        
        X_raw, X_eng, y = preprocessor.preprocess_and_validate_windows(
            raw=raw,
            seizure_times=seizure_times,
            best_indices=best_indices,
            target_hz=256.0,
            win_sec=2.0,
            amp_min=-500.0,
            amp_max=500.0
        )
        
        self.assertEqual(X_raw.shape, (1, 10, 512))
        self.assertEqual(X_eng.shape, (1, 10, 9))
        self.assertEqual(y.shape, (1,))
        self.assertEqual(y[0], 1)  # Window spans 0-2s and seizure spans 1-2s, so they overlap

    def test_preprocess_and_validate_windows_validation_failed(self):
        raw = MagicMock()
        raw.info = {'sfreq': 256.0}
        # Inject NaNs to trigger fast validation check failure
        raw.get_data.return_value = np.full((10, 512), np.nan)
        
        X_raw, X_eng, y = preprocessor.preprocess_and_validate_windows(
            raw=raw,
            seizure_times=[],
            best_indices=list(range(10)),
            target_hz=256.0,
            win_sec=2.0,
            amp_min=-500.0,
            amp_max=500.0
        )
        self.assertEqual(len(X_raw), 0)
        self.assertEqual(len(X_eng), 0)
        self.assertEqual(len(y), 0)

    @patch("mne.io.read_raw_edf", side_effect=RuntimeError("EDF read error"), create=True)
    def test_process_single_file_worker_exception(self, mock_read):
        res = preprocessor.process_single_file_worker(
            edf_path="invalid_path.edf",
            best_indices=list(range(10)),
            target_hz=256.0,
            window_sec=2.0,
            amp_min=-500.0,
            amp_max=500.0,
            seizure_times=[]
        )
        self.assertIsNone(res)

    def test_load_config_default_fallback(self):
        # Call it with non-existent path and expect FileNotFoundError
        with self.assertRaises(FileNotFoundError):
            preprocessor._load_config("non_existent_config_path_xyz.yaml")

    @patch("src.data.preprocess._load_config")
    @patch("glob.glob", return_value=[])
    def test_main_no_edfs(self, mock_glob, mock_load):
        mock_load.return_value = {
            "paths": {
                "local_data_dir": "empty_dir",
                "hdf5_database_path_2d": "empty_store.h5"
            },
            "signal_processing": {
                "window_sec": 2.0,
                "target_hz": 256.0,
                "top_n_channels": 10
            },
            "validation": {
                "amplitude_min_microvolts": -500.0,
                "amplitude_max_microvolts": 500.0
            }
        }
        # This should print "No EDFs found" and return cleanly
        preprocessor.main()

    @patch("src.data.preprocess._load_config")
    @patch("glob.glob")
    @patch("mne.io.read_raw_edf", create=True)
    @patch("src.data.preprocess.calculate_channel_stability")
    @patch("builtins.open", new_callable=mock_open)
    def test_main_success(self, mock_open_file, mock_stability, mock_read, mock_glob, mock_load):
        mock_load.return_value = {
            "paths": {
                "local_data_dir": "data_dir",
                "hdf5_database_path_2d": "store.h5"
            },
            "signal_processing": {
                "window_sec": 2.0,
                "target_hz": 256.0,
                "top_n_channels": 10
            },
            "validation": {
                "amplitude_min_microvolts": -500.0,
                "amplitude_max_microvolts": 500.0
            }
        }
        mock_glob.side_effect = lambda path, recursive=False: (
            ["data_dir/chb01/chb01_01.edf"] if "*.edf" in path or "**" in path else []
        )
        
        mock_raw = MagicMock()
        mock_raw.ch_names = ["ch1", "ch2"]
        mock_raw.info = {"sfreq": 256.0}
        mock_raw.get_data.return_value = np.zeros((2, 512))
        mock_read.return_value = mock_raw
        
        mock_stability.return_value = {"best_indices": [0, 1]}
        
        preprocessor.main()
