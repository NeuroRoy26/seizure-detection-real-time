"""
=============================================================================
  GENERAL BRAINFLOW REAL-TIME EEG STREAMER CLIENT
=============================================================================
Connects to any EEG headset supported by BrainFlow (OpenBCI, Muse, Unicorn, etc.),
reads raw signal buffers, downsamples them to 128Hz, dynamically pads or 
truncates channels to match the trained model's expected shape (10 channels),
and streams live telemetry over WebSockets to the FastAPI ingestion server.
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
    parser = argparse.ArgumentParser(description="Universal BrainFlow EEG Streamer Client")
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1:8000",
        help="FastAPI host address (e.g. 127.0.0.1:8000 or NeuroRoy26-seizure-detection-real-time.hf.space)"
    )
    parser.add_argument(
        "--board-id",
        type=int,
        default=-1,  # Default to Synthetic/Simulation Board
        help="BrainFlow Board ID (e.g., -1 for Synthetic, 0 for Cyton, 13 for Unicorn, 38 for Muse 2). See BrainFlow docs."
    )
    parser.add_argument(
        "--serial-number",
        type=str,
        default="",
        help="Headset serial number or identifier (MAC address / Bluetooth serial)."
    )
    parser.add_argument(
        "--serial-port",
        type=str,
        default="",
        help="Serial port for USB dongles (e.g., COM3 on Windows or /dev/ttyUSB0 on Linux)."
    )
    parser.add_argument(
        "--channels-expected",
        type=int,
        default=10,
        help="Number of channels the trained model expects (default 10)."
    )
    return parser.parse_args()


async def main():
    args = parse_args()

    if not BRAINFLOW_AVAILABLE:
        print("[ERROR] BrainFlow library is not installed. Please run:")
        print("  pip install brainflow")
        sys.exit(1)

    params = BrainFlowInputParams()
    if args.serial_number:
        params.mac_address = args.serial_number
    if args.serial_port:
        params.serial_port = args.serial_port

    board_id = args.board_id
    if board_id == -1:
        print("[*] Utilizing BrainFlow Synthetic/Simulation Board.")
        board_id = BoardIds.SYNTHETIC_BOARD.value
    else:
        print(f"[*] Initializing connection for Board ID: {board_id}")

    try:
        board = BoardShim(board_id, params)
        board.prepare_session()
        board.start_stream()
    except Exception as e:
        print(f"[ERROR] Failed to initialize BrainFlow board session: {e}")
        print("Verify your device is powered on, paired, and parameters (port/serial) are correct.")
        sys.exit(1)

    sampling_rate = board.get_sampling_rate(board_id)
    eeg_channels = board.get_eeg_channels(board_id)
    
    print(f"[SUCCESS] BrainFlow session active. Device sampling rate: {sampling_rate} Hz")
    print(f"[*] Native EEG channels indexed by BrainFlow: {eeg_channels} (Total: {len(eeg_channels)} channels)")

    target_hz = 128
    window_sec = 2.0
    num_samples_needed = int(sampling_rate * window_sec) # 2 seconds of buffer

    # Parse and construct the WebSocket target URL
    host_clean = args.host
    for prefix in ["https://", "http://", "wss://", "ws://"]:
        if host_clean.startswith(prefix):
            host_clean = host_clean.replace(prefix, "")

    ws_protocol = "wss" if "hf.space" in host_clean or "https" in args.host or "wss" in args.host else "ws"
    ws_url = f"{ws_protocol}://{host_clean}/ws/ingest"
    print(f"[*] Streaming live telemetry to: {ws_url}")

    try:
        async with websockets.connect(ws_url) as websocket:
            print("[SUCCESS] Connected to FastAPI ingestion route. Streaming...")
            
            while True:
                data = board.get_current_board_data(num_samples_needed)
                
                if data.shape[1] < num_samples_needed:
                    await asyncio.sleep(0.1)
                    continue
                
                # Extract EEG signal arrays
                raw_eeg = data[eeg_channels] # Shape: (native_channels, samples)
                n_native_chans = raw_eeg.shape[0]
                
                # 1. Downsample all channels to 128Hz
                resampled_eeg = []
                for ch in range(n_native_chans):
                    resampled_chan = DataFilter.perform_resampling(raw_eeg[ch], target_hz, sampling_rate)
                    resampled_eeg.append(resampled_chan)
                
                resampled_eeg = np.array(resampled_eeg) # Shape: (n_native_chans, 256)
                
                # 2. Match channel counts (Pad or Truncate)
                model_channels = args.channels_expected
                aligned_eeg = np.zeros((model_channels, 256))
                
                if n_native_chans >= model_channels:
                    # Truncate: Use the first N channels
                    aligned_eeg = resampled_eeg[0:model_channels, :]
                else:
                    # Pad: Copy native channels, fill remainder by duplicating the last channel
                    aligned_eeg[0:n_native_chans, :] = resampled_eeg
                    for ch in range(n_native_chans, model_channels):
                        aligned_eeg[ch] = resampled_eeg[-1] # Duplicate the last channel
                
                payload = {
                    "channels": aligned_eeg.tolist(),
                    "state": "patient_resting"
                }
                
                await websocket.send(json.dumps(payload))
                
                # Sleep interval matching model window stride
                await asyncio.sleep(0.25)
                
    except websockets.exceptions.ConnectionClosed:
        print("[ERROR] Connection to WebSocket server closed.")
    except Exception as e:
        print(f"[ERROR] Streaming interrupted: {e}")
    finally:
        try:
            board.stop_stream()
            board.release_session()
            print("[*] Released BrainFlow session cleanly.")
        except Exception:
            pass


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[*] Streamer client stopped.")
        sys.exit(0)
