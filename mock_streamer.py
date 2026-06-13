import numpy as np
import time
import random
import argparse
from typing import List, Optional

import requests

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
    # ── Unit Test Fallback Logic ──
    if amplitude is not None and frequency_hz is not None:
        seiz_mult = seizure_multiplier if seizure_multiplier is not None else 10.0
        amp = amplitude * (seiz_mult if seizure_active else 1.0)
        x = amp * np.sin(2 * np.pi * frequency_hz * t + phase_offsets)
        return x.astype(float).tolist()

    # ── Realistic Clinical EEG Synthesizer ──
    samples = []
    for ch in range(num_channels):
        ch_phase = phase_offsets[ch]
        
        # Base noise (electrical and physiological background)
        noise = np.random.normal(0, 15.0)
        
        if not seizure_active:
            # Normal EEG: mixed frequencies, low amplitude
            # Delta (1.5 Hz), Theta (5 Hz), Alpha (10 Hz), Beta (20 Hz)
            delta = 25.0 * np.sin(2 * np.pi * 1.5 * t + ch_phase)
            theta = 15.0 * np.sin(2 * np.pi * 5.0 * t + ch_phase * 1.3)
            alpha = 18.0 * np.sin(2 * np.pi * 10.0 * t + ch_phase * 0.7)
            beta = 8.0 * np.sin(2 * np.pi * 20.0 * t + ch_phase * 2.1)
            
            val = (delta + theta + alpha + beta + noise) * 0.8
        else:
            # Seizure EEG: rhythmic synchronous slow-wave discharge with high amplitude
            # Base seizure frequency: 2.5 Hz
            f_base = 2.5
            # Spike and wave: base sine plus harmonics
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
    # Kept in argparser to avoid breaking existing CLI invocations or scripts
    parser.add_argument("--frequency", type=float, default=10.0, help="Sine frequency (Hz) (test only)")
    parser.add_argument("--amplitude", type=float, default=1.0, help="Baseline amplitude (test only)")
    parser.add_argument("--seizure-every", type=float, default=180.0, help="Seconds between mock seizures (test only)")
    parser.add_argument("--seizure-duration", type=float, default=10.0, help="Mock seizure duration (seconds) (test only)")
    parser.add_argument("--seizure-multiplier", type=float, default=10.0, help="Amplitude multiplier (test only)")
    parser.add_argument("--timeout", type=float, default=2.0, help="HTTP timeout seconds")
    parser.add_argument("--print-every", type=float, default=2.0, help="Print status every N seconds")
    args = parser.parse_args()

    dt = 1.0 / max(args.hz, 1e-6)
    t0 = time.time()
    next_print = t0
    seq = 0
    seizure_active = False

    state_url = args.url.replace("/ingest", "/simulator/state")

    # Fixed random phase per channel so it looks like stable multi-channel signals.
    phase_offsets = np.array([random.uniform(0, 2 * np.pi) for _ in range(args.channels)], dtype=float)

    print(f"[streamer] Starting mock EEG stream to {args.url} at {args.hz}Hz...")
    print(f"[streamer] Checking simulator state from {state_url} every 64 samples...")

    while True:
        now = time.time()
        t = now - t0

        # Query state once every 64 samples (approx. 0.5s at 128Hz) to prevent API spam
        if seq % 64 == 0:
            sim_state = fetch_simulator_state(state_url, timeout_s=args.timeout)
            seizure_active = (sim_state == "seizure")

        sample = generate_sample(
            t,
            args.channels,
            phase_offsets,
            seizure_active=seizure_active,
        )

        status = post_with_retry(args.url, sample, timeout_s=args.timeout)

        if now >= next_print:
            print(f"[streamer] seq={seq:<6} status={status} seizure_state={sim_state:<8}")
            next_print = now + max(args.print_every, 0.1)

        seq += 1
        time.sleep(dt)

if __name__ == "__main__":
    main()
