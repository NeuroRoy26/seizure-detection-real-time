import subprocess
import sys
import time
import os
import signal

def main():
    print("=" * 60)
    print("ML-DRIVEN REAL-TIME EEG CLASSIFICATION LAUNCHER")
    print("=" * 60)
    
    # Track processes
    processes = []
    
    # Get active Python executable path
    py_exec = sys.executable
    print(f"[*] Python interpreter detected: {py_exec}")
    
    # 1. Start FastAPI backend (on local port 8000)
    print("[*] Launching FastAPI Backend on 127.0.0.1:8000...")
    backend = subprocess.Popen([
        py_exec, "-m", "uvicorn", "api:app",
        "--host", "127.0.0.1",
        "--port", "8000"
    ])
    processes.append(("FastAPI Backend", backend))
    
    # Give FastAPI a moment to start
    time.sleep(2)
    
    # 2. Start Mock EEG Streamer (submitting samples to FastAPI backend)
    print("[*] Launching Background EEG Streamer...")
    streamer = subprocess.Popen([
        py_exec, "mock_streamer.py",
        "--url", "http://127.0.0.1:8000/ingest",
        "--hz", "128.0"
    ])
    processes.append(("EEG Streamer", streamer))
    
    # 3. Start Streamlit Frontend (listening on port 7860 for Hugging Face Spaces)
    port = os.environ.get("PORT", "7860")
    print(f"[*] Launching Streamlit Dashboard on 0.0.0.0:{port}...")
    frontend = subprocess.Popen([
        py_exec, "-m", "streamlit", "run", "dashboard.py",
        "--server.port", port,
        "--server.address", "0.0.0.0",
        "--server.headless", "true"
    ])
    processes.append(("Streamlit Frontend", frontend))
    
    print("\n[SUCCESS] All services started. Monitoring subprocesses...\n")
    
    # Clean termination handler
    def terminate_all(signum=None, frame=None):
        print("\n[*] Terminating all services...")
        for name, proc in processes:
            if proc.poll() is None:
                print(f"  [-] Terminating {name} (PID: {proc.pid})...")
                proc.terminate()
        print("[*] Clean exit completed.")
        sys.exit(0)
        
    # Register termination signals
    signal.signal(signal.SIGINT, terminate_all)
    signal.signal(signal.SIGTERM, terminate_all)
    
    # Monitor loop
    try:
        while True:
            # Check if any process has exited unexpectedly
            for name, proc in processes:
                exit_code = proc.poll()
                if exit_code is not None:
                    print(f"\n[!] Service '{name}' exited unexpectedly with code {exit_code}.")
                    terminate_all()
            time.sleep(1.0)
    except (KeyboardInterrupt, SystemExit):
        terminate_all()

if __name__ == "__main__":
    main()
