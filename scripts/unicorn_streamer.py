"""
=============================================================================
  G.TEC UNICORN HYBRID BLACK REAL-TIME STREAMER
=============================================================================
Connects natively to a g.tec Unicorn Hybrid Black EEG headset via Bluetooth
using BrainFlow, downsamples the signal from 250Hz to 128Hz, pads the channels
from 8 to 10 (model-compliant shape), and streams live frames over WebSockets
to the FastAPI ingestion endpoint.
=============================================================================
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import argparse
import json
import numpy as np
import websockets

try:
    from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
    from brainflow.data_filter import DataFilter
    BRAINFLOW_AVAILABLE = True
except ImportError:
    BRAINFLOW_AVAILABLE = False


def parse_args():
    parser = argparse.ArgumentParser(description="g.tec Unicorn BrainFlow Streamer Client")
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1:8000",
        help="FastAPI host address (e.g. 127.0.0.1:8000 or wss://<hf-space-url>)"
    )
    parser.add_argument(
        "--serial",
        type=str,
        default="",
        help="Unicorn serial number (e.g. UN-2021.05.12). If omitted, runs in BrainFlow synthetic/simulation mode."
    )
    return parser.parse_args()


async def main():
    args = parse_args()

    if not BRAINFLOW_AVAILABLE:
        print("[ERROR] BrainFlow library is not installed. Please run:")
        print("  pip install brainflow")
        sys.exit(1)

    params = BrainFlowInputParams()
    
    if args.serial:
        print(f"[*] Initializing hardware connection to g.tec Unicorn: {args.serial}")
        params.mac_address = args.serial
        board_id = BoardIds.UNICORN_BOARD.value
    else:
        print("[*] No serial number provided. Running in BrainFlow Synthetic/Simulation mode.")
        board_id = BoardIds.SYNTHETIC_BOARD.value

    # Configure board session
    try:
        board = BoardShim(board_id, params)
        board.prepare_session()
        board.start_stream()
    except Exception as e:
        print(f"[ERROR] Failed to initialize board: {e}")
        print("Verify the headset is powered on, paired via Bluetooth, and not locked by another application.")
        sys.exit(1)

    sampling_rate = board.get_sampling_rate(board_id)
    eeg_channels = board.get_eeg_channels(board_id)
    
    print(f"[SUCCESS] BrainFlow session initialized. Headset sampling rate: {sampling_rate} Hz")
    print(f"[*] Available EEG channels index: {eeg_channels}")

    target_hz = 128
    window_sec = 2.0
    num_samples_needed = int(sampling_rate * window_sec) # 500 samples at 250Hz (Unicorn) or 200 samples at 200Hz (Synthetic)

    # Handle protocol schemas
    ws_protocol = "ws"
    if args.host.startswith("https://"):
        ws_protocol = "wss"
        host_clean = args.host.replace("https://", "")
    elif args.host.startswith("http://"):
        host_clean = args.host.replace("http://", "")
    elif args.host.startswith("wss://"):
        ws_protocol = "wss"
        host_clean = args.host.replace("wss://", "")
    elif args.host.startswith("ws://"):
        host_clean = args.host.replace("ws://", "")
    else:
        host_clean = args.host

    ws_url = f"{ws_protocol}://{host_clean}/ws/ingest"
    print(f"[*] Connecting to Ingestion WebSocket: {ws_url}")

    try:
        async with websockets.connect(ws_url) as websocket:
            print("[SUCCESS] Connected to server. Streaming live EEG channels...")
            
            while True:
                data = board.get_current_board_data(num_samples_needed)
                
                if data.shape[1] < num_samples_needed:
                    await asyncio.sleep(0.1)
                    continue
                
                # Extract EEG signal vectors
                raw_eeg = data[eeg_channels] # Shape: (8, samples) or (16, samples) for synthetic
                
                # Check channel dimensions
                n_chans = raw_eeg.shape[0]
                
                # Downsample to model-compliant 128Hz
                resampled_eeg = []
                for ch in range(min(n_chans, 8)):  # Keep at most 8 channels
                    resampled_chan = DataFilter.perform_resampling(raw_eeg[ch], target_hz, sampling_rate)
                    resampled_eeg.append(resampled_chan)
                
                resampled_eeg = np.array(resampled_eeg) # Shape: (8, 256)
                
                # Pad to 10 channels for model compliance
                padded_eeg = np.zeros((10, 256))
                
                # Copy 8 resampled channels
                padded_eeg[0:min(len(resampled_eeg), 8), :] = resampled_eeg[0:min(len(resampled_eeg), 8)]
                
                # Duplicate channels to fill the remaining 2 slots
                if len(resampled_eeg) >= 8:
                    padded_eeg[8] = resampled_eeg[2] # Duplicate Cz
                    padded_eeg[9] = resampled_eeg[6] # Duplicate Oz
                
                # Send frame payload to FastAPI Ingestion route
                payload = {
                    "channels": padded_eeg.tolist(),
                    "state": "patient_resting"
                }
                
                await websocket.send(json.dumps(payload))
                
                # Fetch interval: 250ms matches model window stride
                await asyncio.sleep(0.25)
                
    except websockets.exceptions.ConnectionClosed:
        print("[ERROR] Connection to WebSocket server closed.")
    except Exception as e:
        print(f"[ERROR] Stream interrupted: {e}")
    finally:
        try:
            board.stop_stream()
            board.release_session()
            print("[*] BrainFlow session released cleanly.")
        except Exception:
            pass


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[*] Exiting streamer client.")
        sys.exit(0)
