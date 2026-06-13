import numpy as np
import time
import random
import argparse
import os
import yaml
from typing import List, Optional

import requests

# Load config to get BEST_CHANNELS
BEST_CHANNELS = [0, 1, 5, 2, 13, 18, 14, 8, 19, 16] # fallback
if os.path.exists("config.yaml"):
    try:
        with open("config.yaml", "r") as f:
            cfg = yaml.safe_load(f)
            BEST_CHANNELS = cfg.get("signal_processing", {}).get("best_indices", BEST_CHANNELS)
    except Exception:
        pass

def generate_sample(
    t: float,
    num_channels: int,
    phase_offsets: np.ndarray,
    *,
    seizure_active: bool,
    # Backward compatibility arguments for unit tests
    frequency_hz: Optional[float] = None,
    amplitude: Optional[float] = None,
    seizure_multiplier: Optional[float] = None,
) -> List[float]:
    """
    Synthesizes a multi-channel EEG signal.
    Falls back to a simple sine wave if amplitude and frequency_hz are explicitly provided (for unit tests).
    """
    if amplitude is not None and frequency_hz is not None:
        seiz_mult = seizure_multiplier if seizure_multiplier is not None else 10.0
        amp = amplitude * (seiz_mult if seizure_active else 1.0)
        x = amp * np.sin(2 * np.pi * frequency_hz * t + phase_offsets)
        return x.astype(float).tolist()

    samples = []
    for ch in range(num_channels):
        ch_phase = phase_offsets[ch]
        noise = np.random.normal(0, 15.0)
        
        if not seizure_active:
            delta = 25.0 * np.sin(2 * np.pi * 1.5 * t + ch_phase)
            theta = 15.0 * np.sin(2 * np.pi * 5.0 * t + ch_phase * 1.3)
            alpha = 18.0 * np.sin(2 * np.pi * 10.0 * t + ch_phase * 0.7)
            beta = 8.0 * np.sin(2 * np.pi * 20.0 * t + ch_phase * 2.1)
            val = (delta + theta + alpha + beta + noise) * 0.8
        else:
            f_base = 2.5
            base_wave = 120.0 * np.sin(2 * np.pi * f_base * t + ch_phase)
            harmonic_1 = 35.0 * np.sin(2 * np.pi * (2 * f_base) * t + ch_phase * 1.5 + np.pi/4)
            harmonic_2 = 15.0 * np.sin(2 * np.pi * (3 * f_base) * t + ch_phase * 0.5 + np.pi/2)
            
            drift = 15.0 * np.sin(2 * np.pi * 1.0 * t + ch_phase * 0.9)
            background = 8.0 * np.sin(2 * np.pi * 10.0 * t + ch_phase * 0.7)
            val = base_wave + harmonic_1 + harmonic_2 + drift + background + noise
            
        samples.append(float(val))
    return samples

def post_with_retry(url: str, sample: List[float], timeout_s: float) -> Optional[int]:
    try:
        resp = requests.post(url, json={"data": sample}, timeout=timeout_s)
        return resp.status_code
    except requests.RequestException as e:
        print(f"[streamer] POST failed: {e}")
        return None

def fetch_simulator_state(state_url: str, timeout_s: float) -> str:
    try:
        resp = requests.get(state_url, timeout=timeout_s)
        if resp.status_code == 200:
            return resp.json().get("state", "normal")
    except Exception:
        pass
    return "normal"

def main() -> None:
    parser = argparse.ArgumentParser(description="Mock 23-channel EEG streamer -> FastAPI /ingest")
    parser.add_argument("--url", default="http://127.0.0.1:8000/ingest", help="FastAPI ingest URL")
    parser.add_argument("--channels", type=int, default=23, help="Number of EEG channels")
    parser.add_argument("--hz", type=float, default=128.0, help="Samples per second to send")
    parser.add_argument("--frequency", type=float, default=10.0, help="Sine frequency (Hz)")
    parser.add_argument("--amplitude", type=float, default=1.0, help="Baseline amplitude")
    parser.add_argument("--seizure-every", type=float, default=180.0, help="Seconds between mock seizures")
    parser.add_argument("--seizure-duration", type=float, default=10.0, help="Mock seizure duration (seconds)")
    parser.add_argument("--seizure-multiplier", type=float, default=10.0, help="Amplitude multiplier")
    parser.add_argument("--timeout", type=float, default=2.0, help="HTTP timeout seconds")
    parser.add_argument("--print-every", type=float, default=2.0, help="Print status every N seconds")
    args = parser.parse_args()

    dt = 1.0 / max(args.hz, 1e-6)
    t0 = time.time()
    next_print = t0
    seq = 0
    sim_state = "normal"
    seizure_active = False

    # ── Load Patient Dataset Cache ──
    patient_normal = []
    patient_seizure = []
    h5_db_path = "train_database.h5"
    has_patient_data = False
    
    if os.path.exists(h5_db_path):
        try:
            import h5py
            with h5py.File(h5_db_path, "r") as h5f:
                X_all = h5f["X"]
                y_all = h5f["y"][:]
                
                normal_idx = np.where(y_all == 0)[0]
                seizure_idx = np.where(y_all == 1)[0]
                
                # Cache up to 100 windows for performance
                for idx in normal_idx[:100]:
                    patient_normal.append(X_all[idx])
                for idx in seizure_idx[:100]:
                    patient_seizure.append(X_all[idx])
                    
            if len(patient_normal) > 0 and len(patient_seizure) > 0:
                has_patient_data = True
                print(f"[streamer] Loaded {len(patient_normal)} normal and {len(patient_seizure)} seizure clinical recordings from {h5_db_path}.")
        except Exception as e:
            print(f"[streamer] Error reading {h5_db_path}: {e}")

    state_url = args.url.replace("/ingest", "/simulator/state")
    phase_offsets = np.array([random.uniform(0, 2 * np.pi) for _ in range(args.channels)], dtype=float)

    # Patient stream pointers
    active_patient_set = None
    patient_sample_idx = 0
    patient_step = 0

    print(f"[streamer] Starting mock EEG stream to {args.url} at {args.hz}Hz...")
    print(f"[streamer] Checking simulator state from {state_url} every 64 samples...")

    while True:
        now = time.time()
        t = now - t0

        # Query state once every 64 samples
        if seq % 64 == 0:
            sim_state = fetch_simulator_state(state_url, timeout_s=args.timeout)
            seizure_active = (sim_state == "seizure")

        if sim_state in ["patient_normal", "patient_seizure"] and has_patient_data:
            # ── Stream Patient Data from HDF5 ──
            if sim_state == "patient_normal":
                active_patient_set = patient_normal
            else:
                active_patient_set = patient_seizure
                
            # If pointer exceeds active set size, wrap around
            if patient_sample_idx >= len(active_patient_set):
                patient_sample_idx = 0
                
            sample_data = active_patient_set[patient_sample_idx]
            
            # Map 10-channel signal to the 23-channel template
            sample = [random.uniform(-10.0, 10.0) for _ in range(args.channels)]
            
            # Check if signals are stored in Volts (scale of 1e-4) and scale to microvolts
            scale_mult = 1e6 if np.max(np.abs(sample_data)) < 0.1 else 1.0
            
            for i in range(10):
                if idx := BEST_CHANNELS[i] < args.channels:
                    sample[BEST_CHANNELS[i]] = float(sample_data[i, patient_step] * scale_mult)
            
            # Advance time-step pointer
            patient_step += 1
            if patient_step >= 256: # window sample size
                patient_step = 0
                patient_sample_idx += 1
        else:
            # ── Stream Synthesized Waveforms ──
            sample = generate_sample(
                t,
                args.channels,
                phase_offsets,
                seizure_active=seizure_active,
            )

        status = post_with_retry(args.url, sample, timeout_s=args.timeout)

        if now >= next_print:
            print(f"[streamer] seq={seq:<6} status={status} state={sim_state:<15}")
            next_print = now + max(args.print_every, 0.1)

        seq += 1
        time.sleep(dt)

if __name__ == "__main__":
    main()
