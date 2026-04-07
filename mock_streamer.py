import numpy as np
import time
import random
import argparse
from typing import List, Optional

import requests

def generate_sample(
    t: float,
    num_channels: int,
    frequency_hz: float,
    amplitude: float,
    phase_offsets: np.ndarray,
    *,
    seizure_active: bool,
    seizure_multiplier: float,
) -> List[float]:
    amp = amplitude * (seizure_multiplier if seizure_active else 1.0)
    x = amp * np.sin(2 * np.pi * frequency_hz * t + phase_offsets)
    return x.astype(float).tolist()


def post_with_retry(url: str, sample: List[float], timeout_s: float) -> Optional[int]:
    try:
        resp = requests.post(url, json={"data": sample}, timeout=timeout_s)
        return resp.status_code
    except requests.RequestException as e:
        print(f"[streamer] POST failed: {e}")
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Mock 23-channel EEG streamer -> FastAPI /ingest")
    parser.add_argument("--url", default="http://127.0.0.1:8000/ingest", help="FastAPI ingest URL")
    parser.add_argument("--channels", type=int, default=23, help="Number of EEG channels")
    parser.add_argument("--hz", type=float, default=100.0, help="Samples per second to send")
    parser.add_argument("--frequency", type=float, default=10.0, help="Sine frequency (Hz)")
    parser.add_argument("--amplitude", type=float, default=1.0, help="Baseline amplitude")
    parser.add_argument("--seizure-every", type=float, default=180.0, help="Seconds between mock seizures")
    parser.add_argument("--seizure-duration", type=float, default=10.0, help="Mock seizure duration (seconds)")
    parser.add_argument("--seizure-multiplier", type=float, default=10.0, help="Amplitude multiplier during seizure")
    parser.add_argument("--timeout", type=float, default=2.0, help="HTTP timeout seconds")
    parser.add_argument("--print-every", type=float, default=1.0, help="Print status every N seconds")
    args = parser.parse_args()

    dt = 1.0 / max(args.hz, 1e-6)
    t0 = time.time()
    next_print = t0
    seq = 0

    # Fixed random phase per channel so it looks like stable multi-channel signals.
    phase_offsets = np.array([random.uniform(0, 2 * np.pi) for _ in range(args.channels)], dtype=float)

    while True:
        now = time.time()
        t = now - t0
        cycle_pos = t % max(args.seizure_every, 1e-6)
        seizure_active = cycle_pos < max(args.seizure_duration, 0.0)

        sample = generate_sample(
            t,
            args.channels,
            args.frequency,
            args.amplitude,
            phase_offsets,
            seizure_active=seizure_active,
            seizure_multiplier=args.seizure_multiplier,
        )

        status = post_with_retry(args.url, sample, timeout_s=args.timeout)

        if now >= next_print:
            print(f"[streamer] seq={seq} status={status} seizure={seizure_active} channels={args.channels}")
            next_print = now + max(args.print_every, 0.1)

        seq += 1
        time.sleep(dt)


if __name__ == "__main__":
    main()
