"""
=============================================================================
  FEATURE ENGINEERING ENGINE
=============================================================================
Computes mathematical and clinical features from windowed EEG time series.
Extracts 9 features per channel across time and frequency domains:
 - Time-domain: Variance, RMS, Line Length, Kurtosis
 - Frequency-domain: Relative power in Delta, Theta, Alpha, Beta, and Gamma
=============================================================================
"""

import numpy as np
from scipy.signal import welch
from scipy.stats import kurtosis

def extract_features_for_channel(x: np.ndarray, sfreq: float = 128.0) -> np.ndarray:
    """
    Computes time-domain and frequency-domain features for a single channel.
    
    Parameters
    ----------
    x : np.ndarray
        1D signal array of shape (samples,).
    sfreq : float
        Sampling frequency in Hz.
        
    Returns
    -------
    np.ndarray
        1D feature vector of shape (9,).
    """
    eps = np.finfo(float).eps
    
    # ── Time Domain Features ──
    var_feat = np.var(x)
    rms_feat = np.sqrt(np.mean(x**2))
    line_length_feat = np.sum(np.abs(np.diff(x)))
    
    # Kurtosis (excess kurtosis, normal distribution = 0.0)
    mean_val = np.mean(x)
    std_val = np.std(x)
    if std_val > eps:
        kurt_feat = np.mean((x - mean_val)**4) / (std_val**4) - 3.0
    else:
        kurt_feat = 0.0
        
    # ── Frequency Domain Features (Welch Periodogram) ──
    # nperseg should be smaller than sample length (e.g. 128 samples for 256 length window)
    nperseg = min(len(x), 128)
    freqs, psd = welch(x, fs=sfreq, nperseg=nperseg)
    
    total_power = np.sum(psd) + eps
    
    def get_relative_power(low: float, high: float) -> float:
        idx = (freqs >= low) & (freqs < high)
        return float(np.sum(psd[idx]) / total_power)
        
    # Standard EEG Bands
    delta_rel = get_relative_power(0.5, 4.0)   # Deep sleep/pathology
    theta_rel = get_relative_power(4.0, 8.0)   # Drowsiness/seizure slow wave
    alpha_rel = get_relative_power(8.0, 12.0)  # Relaxed wakefulness
    beta_rel  = get_relative_power(12.0, 30.0) # Active thinking/alert
    gamma_rel = get_relative_power(30.0, 45.0) # High cognitive processing
    
    return np.array([
        var_feat,
        rms_feat,
        line_length_feat,
        kurt_feat,
        delta_rel,
        theta_rel,
        alpha_rel,
        beta_rel,
        gamma_rel
    ], dtype=np.float32)

def extract_eeg_features(window: np.ndarray, sfreq: float = 128.0) -> np.ndarray:
    """
    Computes features for all channels in a given EEG window.
    
    Parameters
    ----------
    window : np.ndarray
        EEG data array of shape (channels, samples), e.g. (10, 256).
    sfreq : float
        Sampling frequency in Hz.
        
    Returns
    -------
    np.ndarray
        2D feature matrix of shape (channels, 9).
    """
    n_channels = window.shape[0]
    features = []
    
    for ch in range(n_channels):
        ch_features = extract_features_for_channel(window[ch, :], sfreq=sfreq)
        features.append(ch_features)
        
    return np.vstack(features) # Shape: (channels, 9)
