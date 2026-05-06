"""
=============================================================================
  Universal Channel Selection / Stability Metric Calculation
=============================================================================
Translates rigid target-frequency MATLAB quality algorithms into a universally
agnostic Python function. Validates Signal-to-Noise Ratio (SNR), suppresses 
violent edge artifacts, and generates clinical real-world electrode recommendations.

Authoritative Output: List of highest quality N channels.
=============================================================================
"""

import numpy as np
from scipy.signal import butter, filtfilt, medfilt
from scipy.stats import median_abs_deviation

def calculate_channel_stability(sessions_data, sfreq, channel_names, top_n=10):
    """
    Computes optimal channels by validating hardware stability across multiple sessions.
    
    Parameters
    ----------
    sessions_data : list of numpy arrays
        Each array should be shape (channels, time).
    sfreq : float
        Sampling frequency in Hz (e.g., 256.0 for CHB-MIT).
    channel_names : list of str
        The string arrays representing clinical electrodes.
    top_n : int
        Number of best channels to return.
        
    Returns
    -------
    dict
        Contains raw indices, string names, and mathematical scores of the best channels.
    """
    channel_rms_all = []
    
    # Mathematical Epsilon to prevent division by zero in clean signals
    eps = np.finfo(float).eps
    
    # ── Universal Safety Constraints ───────────────────────────────────────────
    # Dynamically prevents user from filtering past the physical Nyquist limit
    nyquist = sfreq / 2.0
    low_cut = 1.0
    high_cut = min(50.0, nyquist - 1.0)
    
    b, a = butter(4, [low_cut / nyquist, high_cut / nyquist], btype='bandpass')

    print(f"[*] Processing {len(sessions_data)} sessions for channel stability...")
    
    for i, raw_signal in enumerate(sessions_data):
        n_channels, n_samples = raw_signal.shape
        
        # 1. Trim boundary impedance noise (first/last 1.5s)
        trim_samples = int(round(1.5 * sfreq))
        if n_samples > 2 * trim_samples:
            trimmed = raw_signal[:, trim_samples:-trim_samples]
        else:
            # Fallback if window is inherently too small to trim
            trimmed = raw_signal 
            
        # 2. Universal Bandpass Filtering (axis=1 means across time)
        filtered = filtfilt(b, a, trimmed, axis=1)
        
        # 3. Artifact Suppression (Median Absolute Deviation outlier rejection)
        z_thresh = 6.0
        medfilt_win = 101
        
        # We must iterate over channels to apply scipy's 1D median filter safely
        clean_filtered = np.zeros_like(filtered)
        
        for ch in range(n_channels):
            chan_data = filtered[ch, :]
            
            med_val = np.median(chan_data)
            # scale="normal" automatically adjusts MAD to approximate standard deviation
            mad_val = median_abs_deviation(chan_data, scale='normal')
            
            # Robust Z-Score mapping
            z_scores = (chan_data - med_val) / (mad_val + eps)
            spike_mask = np.abs(z_scores) > z_thresh
            
            # Sub in median filtered data strictly over violence spikes
            ch_medfilt = medfilt(chan_data, kernel_size=medfilt_win)
            
            chan_data_clean = chan_data.copy()
            chan_data_clean[spike_mask] = ch_medfilt[spike_mask]
            
            # Hard clip absolute anomalies
            clip_thresh = 5.0 * np.std(chan_data_clean)
            chan_data_clean = np.clip(chan_data_clean, -clip_thresh, clip_thresh)
            
            clean_filtered[ch, :] = chan_data_clean
        
        # 4. RMS computation for the active session
        rms = np.sqrt(np.mean(clean_filtered**2, axis=1))
        channel_rms_all.append(rms)

    # Convert to matrix: (sessions x channels)
    channel_rms_matrix = np.vstack(channel_rms_all)
    
    # ── Channel Quality Metric (Long-term Stability) ───────────────────────────
    mean_rms = np.mean(channel_rms_matrix, axis=0)
    std_rms = np.std(channel_rms_matrix, axis=0)
    
    cv_rms = std_rms / (mean_rms + eps)
    
    # Score = Power of Signal / Instability of Signal over time
    signal_score = mean_rms / (cv_rms + eps)
    
    # Sort descending securely
    sorted_idx = np.argsort(signal_score)[::-1]
    
    top_indices = sorted_idx[:top_n]
    top_channels = [channel_names[idx] for idx in top_indices]
    
    # Generate cleanly formatted console output
    print(f"[*] Top {top_n} Channels Automatically Selected:")
    for rank, idx in enumerate(top_indices):
        print(f"    #{rank+1} : {channel_names[idx]} \t| Score: {signal_score[idx]:.4f}")
        
    return {
        "best_indices": top_indices.tolist(),
        "best_names": top_channels,
        "scores": signal_score.tolist()
    }

