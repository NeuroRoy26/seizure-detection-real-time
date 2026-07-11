import streamlit as st
import requests
import time
import os
import matplotlib.pyplot as plt
import numpy as np

# Premium styling configurations
st.set_page_config(
    page_title="Real-Time EEG Seizure Detector",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for the recruiter-facing medical dashboard
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=JetBrains+Mono:wght@400;700&display=swap');
    
    html, body, .main, [data-testid="stMain"], .stApp {
        overflow-y: scroll !important;
    }
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    
    /* Custom status badge styling */
    .status-badge {
        padding: 6px 12px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.85rem;
        text-transform: uppercase;
        display: inline-block;
    }
    .status-connected {
        background-color: #09ab3b22;
        color: #09ab3b;
        border: 1px solid #09ab3b44;
    }
    .status-disconnected {
        background-color: #ff4b4b22;
        color: #ff4b4b;
        border: 1px solid #ff4b4b44;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Use Matplotlib dark mode for professional contrast
plt.style.use('dark_background')

API_URL_DEFAULT = os.environ.get("API_URL", "http://127.0.0.1:8000/latest")
CHANNELS = 23

import yaml
# Load master config to check LLM enablement
config = {}
if os.path.exists("config.yaml"):
    try:
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
    except Exception:
        pass

llm_enabled = config.get("llm", {}).get("enabled", False)
BEST_CHANNELS = config.get("signal_processing", {}).get("best_indices", [0, 1, 5, 2, 13, 18, 14, 8, 19, 16])

def _init_state() -> None:
    st.session_state.setdefault("running", False)
    st.session_state.setdefault("buffer", [])  # list[list[float]]: 2D array of (time, channels)
    st.session_state.setdefault("prob_history", [])  # list[float]: history of model prediction probabilities
    st.session_state.setdefault("connection_status", "Disconnected")
    st.session_state.setdefault("last_update_time", None)
    st.session_state.setdefault("seizure_triggered_at", None)
    st.session_state.setdefault("seizure_active", False)
    st.session_state.setdefault("chat_history", [])

def _fetch_latest(url: str, timeout_s: float) -> dict:
    resp = requests.get(url, timeout=timeout_s)
    resp.raise_for_status()
    return resp.json()

def _trim_buffer(max_len: int) -> None:
    if max_len <= 0:
        st.session_state.buffer = []
        return
    if len(st.session_state.buffer) > max_len:
        st.session_state.buffer = st.session_state.buffer[-max_len:]

def _plot_eeg(buffer: list, channel_mode: str) -> None:
    st.subheader("Live EEG Waveforms")
    if not buffer:
        st.info("Start the stream to view incoming EEG signals.")
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#1e222b')
    ax.grid(True, color='#2e3440', linestyle='--', alpha=0.5)

    if isinstance(buffer[0], dict):
        raw_data = [x["data"] for x in buffer]
    else:
        raw_data = buffer

    buffer_np = np.array(raw_data)

    if channel_mode == "All channels (overlaid)":
        for i in range(CHANNELS):
            ax.plot(buffer_np[:, i], linewidth=1, alpha=0.7, label=f"Ch {i+1}")
        ax.legend(ncols=6, fontsize=6, loc='upper right')
    else:
        idx = int(channel_mode.split()[-1]) - 1
        ax.plot(buffer_np[:, idx], linewidth=1.8, color="#00d4ff", label=channel_mode)
        ax.legend(loc='upper right')

    ax.set_xlabel("Time Step (Rolling Window)", fontsize=9, color="#888888")
    ax.set_ylabel("Amplitude (μV)", fontsize=9, color="#888888")
    ax.tick_params(colors='#888888', labelsize=8)
    ax.set_ylim(-220, 220)
    fig.subplots_adjust(left=0.12, right=0.95, top=0.9, bottom=0.18)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

def _plot_prob(prob_history: list) -> None:
    st.subheader("Model Inference Probability")
    if not prob_history:
        st.info("Start the stream to track seizure probability.")
        return

    fig, ax = plt.subplots(figsize=(10, 2))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#1e222b')
    ax.grid(True, color='#2e3440', linestyle='--', alpha=0.5)

    if isinstance(prob_history[0], dict):
        probs = [x["seizure_probability"] for x in prob_history]
    else:
        probs = prob_history
    
    line_color = "#09ab3b" # green
    if probs[-1] > 0.8:
        line_color = "#ff4b4b" # red
    elif probs[-1] > 0.2:
        line_color = "#ffa500" # orange

    ax.plot(probs, linewidth=2.5, color=line_color)
    ax.fill_between(range(len(probs)), probs, color=line_color, alpha=0.15)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Time Step", fontsize=8, color="#888888")
    ax.set_ylabel("Seizure Probability", fontsize=8, color="#888888")
    ax.tick_params(colors='#888888', labelsize=8)
    fig.subplots_adjust(left=0.12, right=0.95, top=0.9, bottom=0.25)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

_init_state()

# Title and subtitle blocks
st.title("ML-Driven Real-Time EEG Classification for Seizure Detection")
st.markdown(
    """
    <div style="font-size: 0.88rem; color: #a3b3c9; margin-bottom: 15px; line-height: 1.6;">
        <b>Developer:</b> Sindhujit Roy &nbsp;|&nbsp; 
        <b>Inference Stack:</b> Real-Time ONNX Engine &nbsp;|&nbsp; 
        <b>MLOps Pipeline:</b> HDF5 Feature Store &bull; DVC Versioning &bull; Great Expectations Validation &bull; DAGsHub & MLflow Tracking &bull; AWS SageMaker &bull; Clinical EEG Playback
    </div>
    """,
    unsafe_allow_html=True
)

with st.sidebar:
    st.header("Settings")
    api_url = st.text_input("FastAPI `/latest` URL", value=API_URL_DEFAULT)
    polling_rate = st.slider("Poll interval (seconds)", min_value=0.05, max_value=2.0, value=0.1, step=0.05)
    window_seconds = st.slider("Rolling window (seconds)", min_value=1, max_value=60, value=15, step=1)
    timeout_s = st.slider("HTTP timeout (seconds)", min_value=0.5, max_value=10.0, value=2.0, step=0.5)
    
    channel_options = ["Channel 1"] + [f"Channel {i}" for i in range(2, CHANNELS + 1)] + ["All channels (overlaid)"]
    channel_mode = st.selectbox(
        "EEG Electrode Mode",
        options=channel_options,
        index=len(channel_options) - 1,
    )
    
    if st.button("Clear Local Buffer"):
        st.session_state.buffer = []
        
    st.markdown("---")
    st.header("EEG Waveform Simulator")
    
    state_url = api_url.replace("/latest", "/simulator/state")
    
    def get_sim_state() -> str:
        try:
            r = requests.get(state_url, timeout=timeout_s)
            if r.status_code == 200:
                return r.json().get("state", "normal")
        except Exception:
            pass
        return "disconnected"

    def set_sim_state(state: str):
        try:
            requests.post(state_url, json={"state": state}, timeout=timeout_s)
        except Exception as e:
            st.sidebar.error(f"Failed to communicate with simulator: {e}")
    current_state = get_sim_state()
    
    default_mode_idx = 1  # Default to Real Patient Recording (index 1)
    if current_state in ["normal", "seizure"]:
        default_mode_idx = 0
         
    mode_choice = st.sidebar.radio(
        "EEG Source Mode",
        options=["Synthesized Waves", "Real Patient Recording"],
        index=default_mode_idx,
        key="eeg_source_mode"
    )
    
    # Handle Source Mode change when no seizure is active
    if not st.session_state.get("seizure_active", False):
        if mode_choice == "Synthesized Waves" and current_state != "normal" and current_state != "disconnected":
            set_sim_state("normal")
            st.rerun()
        elif mode_choice == "Real Patient Recording" and current_state != "patient_normal" and current_state != "disconnected":
            set_sim_state("patient_normal")
            st.rerun()

    # Seizure trigger with 10s cooldown and countdown
    cooldown_remaining = 0
    if st.session_state.get("seizure_triggered_at") is not None:
        elapsed = time.time() - st.session_state.seizure_triggered_at
        if elapsed < 10.0:
            cooldown_remaining = int(10.0 - elapsed)
        else:
            st.session_state.seizure_triggered_at = None
            st.session_state.seizure_active = False
            if mode_choice == "Synthesized Waves":
                set_sim_state("normal")
            else:
                set_sim_state("patient_normal")
            st.rerun()

    if cooldown_remaining > 0:
        st.sidebar.button(f"⚡ Trigger Seizure ({cooldown_remaining}s)", disabled=True, use_container_width=True, key="btn_trigger_seizure_disabled")
    else:
        if st.sidebar.button("⚡ Trigger Seizure", use_container_width=True, key="btn_trigger_seizure"):
            st.session_state.seizure_triggered_at = time.time()
            st.session_state.seizure_active = True
            if mode_choice == "Synthesized Waves":
                set_sim_state("seizure")
            else:
                set_sim_state("patient_seizure")
            st.rerun()
        
    if current_state in ["normal", "patient_normal"]:
        st.sidebar.success(f"✅ State: Normal EEG active ({current_state})")
    elif current_state in ["seizure", "patient_seizure"]:
        st.sidebar.error(f"🚨 State: Seizure active ({current_state})")
    else:
        st.sidebar.warning("⚠️ State: Simulator disconnected")

    st.sidebar.markdown(
        """
        <small style='color: #888888;'>
        Choose <b>Real Patient Recording</b> to stream historical clinical signals from the CHB-MIT dataset. 
        Click <b>Trigger Seizure</b> to inject a 10-second seizure window anomaly and observe model outputs live!
        </small>
        <hr style='margin: 15px 0; border: none; border-top: 1px solid #444444;'>
        <div style='text-align: center;'>
            <a href='https://github.com/NeuroRoy26/seizure-detection-real-time' target='_blank' style='text-decoration: none;'>
                <img src='https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github&logoColor=white' alt='GitHub Repository' style='height: 28px;'>
            </a>
        </div>
        """,
        unsafe_allow_html=True
    )

# Restructure main interface into Tabs
if llm_enabled:
    tab_stream, tab_explorer, tab_llm = st.tabs(["Real-Time Stream", "Clinical Dataset Explorer", "AI Assistant & Reporting"])
else:
    tab_stream, tab_explorer = st.tabs(["Real-Time Stream", "Clinical Dataset Explorer"])

with tab_stream:
    # Execution Controls
    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([1, 1, 4])
    with col_ctrl1:
        if st.button("Start Stream", use_container_width=True, disabled=st.session_state.running, key="btn_start"):
            st.session_state.running = True
            st.rerun()
    with col_ctrl2:
        if st.button("Stop Stream", use_container_width=True, disabled=not st.session_state.running, key="btn_stop"):
            st.session_state.running = False
            st.rerun()
    with col_ctrl3:
        status_class = "status-connected" if "Connected" in st.session_state.connection_status else "status-disconnected"
        st.markdown(
            f"""
            <div style="margin-top: 5px;">
                <b>Connection status</b>: 
                <span class="status-badge {status_class}">{st.session_state.connection_status}</span>
                <span style="margin-left: 20px; color:#888;">Last updated: {st.session_state.last_update_time or "N/A"}</span>
            </div>
            """, 
            unsafe_allow_html=True
        )

    st.markdown("---")

    max_len = int(window_seconds / max(polling_rate, 1e-6))

    if st.session_state.running:
        try:
            payload = _fetch_latest(api_url, timeout_s=timeout_s)
            if "data" in payload and "seizure_probability" in payload:
                st.session_state.buffer = payload["data"]
                st.session_state.prob_history.append(payload["seizure_probability"])
                if len(st.session_state.prob_history) > max_len:
                    st.session_state.prob_history = st.session_state.prob_history[-max_len:]
                st.session_state.connection_status = "Connected"
                st.session_state.last_update_time = time.strftime("%H:%M:%S")
            else:
                st.session_state.connection_status = "Connected (no data yet)"
        except Exception as e:
            st.session_state.connection_status = f"Disconnected ({type(e).__name__})"
            st.error(f"Failed to fetch data: {e}")

    # Dynamic Alert Metric Card
    latest_prob = st.session_state.prob_history[-1] if st.session_state.prob_history else 0.0
    prob_pct = latest_prob * 100

    col_metric, col_plot = st.columns([1.2, 2.5])

    with col_metric:
        st.subheader("Clinical Assessment")
        if prob_pct > 80:
            st.markdown(
                f"""
                <div style="background-color: #ff4b4b18; border: 2px solid #ff4b4b; border-radius: 12px; padding: 25px; text-align: center; margin-bottom: 20px; box-shadow: 0 4px 15px rgba(255, 75, 75, 0.25);">
                    <span style="color: #ff4b4b; font-size: 1.15rem; font-weight: bold; text-transform: uppercase; letter-spacing: 1.5px; display: block; margin-bottom: 10px;">SEIZURE DETECTED</span>
                    <h1 style="color: #ff4b4b; margin: 0; font-size: 4rem; font-family: 'JetBrains Mono', monospace; font-weight: 700;">{prob_pct:.1f}%</h1>
                    <p style="color: #ffffffdd; margin: 15px 0 0 0; font-size: 0.95rem; line-height: 1.4;">
                        High-power rhythmic synchronous delta-band discharges (2.5Hz) detected. 
                        <b>Immediate clinical intervention recommended.</b>
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
        elif prob_pct > 20:
            st.markdown(
                f"""
                <div style="background-color: #ffa50018; border: 2px solid #ffa500; border-radius: 12px; padding: 25px; text-align: center; margin-bottom: 20px; box-shadow: 0 4px 15px rgba(255, 165, 0, 0.25);">
                    <span style="color: #ffa500; font-size: 1.15rem; font-weight: bold; text-transform: uppercase; letter-spacing: 1.5px; display: block; margin-bottom: 10px;">⚠️ ELEVATED ACTIVITY</span>
                    <h1 style="color: #ffa500; margin: 0; font-size: 4rem; font-family: 'JetBrains Mono', monospace; font-weight: 700;">{prob_pct:.1f}%</h1>
                    <p style="color: #ffffffdd; margin: 15px 0 0 0; font-size: 0.95rem; line-height: 1.4;">
                        Elevated amplitude waves and slow oscillatory components present. 
                        Continuous monitoring active.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div style="background-color: #09ab3b18; border: 2px solid #09ab3b; border-radius: 12px; padding: 25px; text-align: center; margin-bottom: 20px; box-shadow: 0 4px 15px rgba(9, 171, 59, 0.25);">
                    <span style="color: #09ab3b; font-size: 1.15rem; font-weight: bold; text-transform: uppercase; letter-spacing: 1.5px; display: block; margin-bottom: 10px;">✅ BRAIN STATUS: NORMAL</span>
                    <h1 style="color: #09ab3b; margin: 0; font-size: 4rem; font-family: 'JetBrains Mono', monospace; font-weight: 700;">{prob_pct:.1f}%</h1>
                    <p style="color: #ffffffdd; margin: 15px 0 0 0; font-size: 0.95rem; line-height: 1.4;">
                        Normal low-amplitude baseline rhythms (Delta, Theta, Alpha, Beta mix). 
                        Patient signals stable.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        st.markdown(
            f"""
            <div style="background-color: #1e222b; border-radius: 12px; padding: 20px; border: 1px solid #2e3440;">
                <h5 style="margin-top:0; color:#00d4ff;">Model Properties</h5>
                <small style="color: #ffffffcc; line-height: 1.5; display:block;">
                    • <b>Backbone</b>: MobileNetV2 (Transfer Learning)<br/>
                    • <b>Input Tensor</b>: 10 Channels x 256 samples (2.0s @ 128Hz)<br/>
                    • <b>Engine</b>: ONNX Runtime (CPU Execution Provider)<br/>
                    • <b>Standardization</b>: Voltage scaled to microvolts (μV)
                </small>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div style="background-color: #1e222b; border-radius: 12px; padding: 20px; border: 1px solid #2e3440; margin-top: 15px;">
                <h5 style="margin-top:0; color:#9b51e0;"> Model Training Logs and Metrics</h5>
                <small style="color: #ffffffcc; line-height: 1.5; display:block; margin-bottom: 12px;">
                    Explore hyperparameters, evaluation metrics, and model runs logged publicly.
                </small>
                <div style="display: flex; gap: 10px;">
                    <a href="https://dagshub.com/NeuroRoy26/seizure-detection-real-time/experiments" target="_blank" style="flex: 1; text-align: center; background-color: #00d4ff1f; color: #00d4ff; border: 1px solid #00d4ff44; padding: 6px 12px; border-radius: 8px; font-size: 0.8rem; font-weight: bold; text-decoration: none;">DAGsHub Experiment Logs (Free Login Required)</a>
                    <a href="https://dagshub.com/NeuroRoy26/seizure-detection-real-time.mlflow/#/experiments" target="_blank" style="flex: 1; text-align: center; background-color: #9b51e01f; color: #9b51e0; border: 1px solid #9b51e044; padding: 6px 12px; border-radius: 8px; font-size: 0.8rem; font-weight: bold; text-decoration: none;">MLflow Current Model Log</a>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    with col_plot:
        _plot_eeg(st.session_state.buffer, channel_mode=channel_mode)
        _plot_prob(st.session_state.prob_history)


with tab_explorer:
    st.header("Clinical Dataset Explorer")
    st.markdown(
        """
        Explore real clinical recordings from the **CHB-MIT EEG Database** stored in the local HDF5 Feature Store. 
        Select specific patient signal windows, inspect the true medical diagnosis, and classify them using the ONNX model.
        """
    )
    
    h5_db_path = "train_database.h5"
    if not os.path.exists(h5_db_path):
        st.warning(f"Feature database `train_database.h5` not found in workspace root. Run preprocess.py or dvc pull first.")
    else:
        try:
            import h5py
            
            with h5py.File(h5_db_path, "r") as h5f:
                X_all = h5f["X"]
                y_all = h5f["y"][:]
                
                n_total = len(y_all)
                n_seizures = int(np.sum(y_all == 1))
                n_normal = n_total - n_seizures
                
                # Visual stats panel
                col_s1, col_s2, col_s3 = st.columns(3)
                col_s1.metric("Total Windows", f"{n_total:,}")
                col_s2.metric("Normal Windows (Label 0)", f"{n_normal:,}")
                col_s3.metric("Seizure Windows (Label 1)", f"{n_seizures:,}")
                
                st.markdown("---")
                
                # Filter selection
                col_f1, col_f2 = st.columns(2)
                with col_f1:
                    class_choice = st.selectbox(
                        "Target EEG Signal Class",
                        options=["Normal EEG Baseline", "Active Seizure Segment"]
                    )
                    target_label = 1 if class_choice == "Active Seizure Segment" else 0
                    
                # Get indices matching the label
                matching_indices = np.where(y_all == target_label)[0]
                
                with col_f2:
                    sample_selector = st.slider(
                        "Sample Index Selector",
                        min_value=0,
                        max_value=len(matching_indices) - 1,
                        value=0,
                        help="Browse index of matching samples in HDF5 Feature Store"
                    )
                
                actual_idx = matching_indices[sample_selector]
                
                # Load selected sample (10 channels x 256 samples)
                sample_data = X_all[actual_idx]
                
                # Plot sample data
                st.subheader(f"Real Recording Waveform (HDF5 Index: {actual_idx})")
                
                col_ch, col_ctrl = st.columns([1, 4])
                with col_ch:
                    explorer_ch = st.selectbox(
                        "Electrode Focus",
                        options=["All 10 Channels"] + [f"Channel {i+1}" for i in range(10)]
                    )
                
                fig, ax = plt.subplots(figsize=(10, 3.5))
                fig.patch.set_facecolor('#0e1117')
                ax.set_facecolor('#1e222b')
                ax.grid(True, color='#2e3440', linestyle='--', alpha=0.5)
                
                # Database signals might be stored in Volts (e.g. scale of 1e-4)
                if np.max(np.abs(sample_data)) < 0.1:
                    sample_scaled = sample_data * 1e6
                else:
                    sample_scaled = sample_data
                
                if explorer_ch == "All 10 Channels":
                    for i in range(10):
                        ax.plot(sample_scaled[i], linewidth=0.8, alpha=0.7, label=f"Ch {i+1}")
                    ax.legend(ncols=5, fontsize=6, loc='upper right')
                else:
                    ch_idx = int(explorer_ch.split()[-1]) - 1
                    ax.plot(sample_scaled[ch_idx], linewidth=1.5, color="#ff00d4" if target_label == 1 else "#00ff66", label=explorer_ch)
                    ax.legend(loc='upper right')
                    
                ax.set_xlabel("Sample index (2s window @ 128Hz)", fontsize=8, color="#888888")
                ax.set_ylabel("Amplitude (μV)", fontsize=8, color="#888888")
                ax.tick_params(colors='#888888', labelsize=8)
                st.pyplot(fig)
                plt.close(fig)
                
                # Classification trigger
                col_btn, col_res = st.columns([1.2, 2.5])
                with col_btn:
                    st.write("") # spacing
                    st.write("")
                    analyze_clicked = st.button("Analyze Clinical Signal", use_container_width=True, key="btn_analyze")
                
                with col_res:
                    if analyze_clicked:
                        classify_url = api_url.replace("/latest", "/classify_window")
                        try:
                            payload_data = sample_data.tolist()
                            
                            with st.spinner("Analyzing with ONNX Runtime model..."):
                                resp = requests.post(classify_url, json={"data": payload_data}, timeout=timeout_s)
                                
                            if resp.status_code == 200:
                                pred_prob = resp.json().get("seizure_probability", 0.0)
                                pred_pct = pred_prob * 100
                                
                                if target_label == 1:
                                    true_diag = "ACTIVE SEIZURE"
                                    true_color = "#ff4b4b"
                                else:
                                    true_diag = "NORMAL (BASELINE)"
                                    true_color = "#09ab3b"
                                    
                                pred_color = "#09ab3b"
                                if pred_prob > 0.8:
                                    pred_color = "#ff4b4b"
                                elif pred_prob > 0.2:
                                    pred_color = "#ffa500"
                                    
                                st.markdown(
                                    f"""
                                    <div style="background-color: #1e222b; border-radius: 10px; padding: 15px; border: 1px solid #2e3440; display: flex; justify-content: space-around; align-items: center; text-align: center; margin-top:10px;">
                                        <div>
                                            <span style="color: #888888; font-size: 0.8rem; text-transform: uppercase;">True Label</span>
                                            <h4 style="color: {true_color}; margin: 5px 0 0 0; font-weight: bold;">{true_diag}</h4>
                                        </div>
                                        <div style="border-left: 1px solid #2e3440; height: 40px;"></div>
                                        <div>
                                            <span style="color: #888888; font-size: 0.8rem; text-transform: uppercase;">Model Prediction</span>
                                            <h4 style="color: {pred_color}; margin: 5px 0 0 0; font-weight: bold; font-family: 'JetBrains Mono', monospace;">{pred_pct:.2f}%</h4>
                                        </div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )
                            else:
                                st.error(f"Inference endpoint error: {resp.text}")
                        except Exception as e:
                            st.error(f"Failed to communicate with classification API: {e}")
        except Exception as e:
            st.error(f"Failed to load dataset: {e}")

if llm_enabled:
    with tab_llm:
        st.header("AI Clinical Assistant & Reporting")
        st.markdown(
            """
            This module integrates **Hugging Face Cloud Inference API** serving open-source Large Language Models (like `Qwen2.5-7B-Instruct`).
            Use it to generate clinical SOAP reports, explain EEG features, or chat with the AI assistant.
            """
        )
        
        # Check LLM connection status
        llm_health_url = api_url.replace("/latest", "/llm/health")
        llm_report_url = api_url.replace("/latest", "/llm/report")
        llm_explain_url = api_url.replace("/latest", "/llm/explain")
        
        try:
            health_resp = requests.get(llm_health_url, timeout=timeout_s)
            if health_resp.status_code == 200:
                health_data = health_resp.json()
                status = health_data.get("status", "unknown")
                msg = health_data.get("message", "")
            else:
                status = "error"
                msg = f"HTTP Error {health_resp.status_code}"
        except Exception as e:
            status = "error"
            msg = f"Failed to connect to FastAPI LLM route: {str(e)}"
            
        if status == "disabled":
            st.warning("⚠️ LLM features are disabled in config.yaml.")
            st.info("Enable `llm.enabled` in `config.yaml` to run.")
        elif status == "missing_token":
            st.error("❌ Hugging Face Access Token (HF_TOKEN) is not set.")
            st.markdown(
                """
                Please set the **`HF_TOKEN`** environment variable in your terminal before running:
                ```powershell
                $env:HF_TOKEN="your_hugging_face_token_here"
                python start.py
                ```
                *Note: You can generate a free token in your Hugging Face account under Settings > Access Tokens.*
                """
            )
        elif status == "loading":
            st.warning(f"🔄 Model Booting: {msg}")
            if st.button("Refresh Connection Status"):
                st.rerun()
        elif status == "error":
            st.error(f"❌ Connection Error: {msg}")
        elif status == "ok":
            st.success(f"✅ Connected to Hugging Face Cloud LLM ({config.get('llm', {}).get('model_id', 'Qwen/Qwen2.5-7B-Instruct')})")
            
            # Divide into columns for different tasks
            col_l, col_r = st.columns(2)
            
            with col_l:
                st.subheader("Generate Clinical SOAP Report")
                st.markdown("Compile current telemetry readings, seizure alert statuses, and channel features into a clinical report draft.")
                
                # Check if buffer has data
                if not st.session_state.buffer:
                    st.info("No active EEG data stream to report. Start the stream first.")
                else:
                    if st.button("Draft Clinical SOAP Report", use_container_width=True, key="btn_draft_report"):
                        # Extract latest features
                        try:
                            from src.features import extract_eeg_features
                            buffer_np = np.array(st.session_state.buffer)
                            
                            # Standardize shape to (channels, samples)
                            if len(buffer_np) >= 256:
                                window_raw = buffer_np[-256:, :].T # (time, channels) -> (channels, time)
                                # Clamp to first 20 channels then best 10 channels
                                if window_raw.shape[0] >= 20:
                                    window_raw = window_raw[:20, :]
                                    best_channels_data = window_raw[BEST_CHANNELS, :] # (10, 256)
                                    feats = extract_eeg_features(best_channels_data, sfreq=128.0) # (10, 9)
                                    
                                    # Create list of dict features
                                    feats_list = []
                                    for ch_idx, ch_feats in enumerate(feats):
                                        feats_list.append({
                                            "channel": str(BEST_CHANNELS[ch_idx] + 1),
                                            "variance": float(ch_feats[0]),
                                            "rms": float(ch_feats[1]),
                                            "delta": float(ch_feats[4]),
                                            "theta": float(ch_feats[5])
                                        })
                                else:
                                    feats_list = None
                            else:
                                feats_list = None
                        except Exception as e:
                            st.error(f"Error computing EEG features: {e}")
                            feats_list = None
                            
                        # Send request
                        latest_prob = st.session_state.prob_history[-1] if st.session_state.prob_history else 0.0
                        
                        # Get current simulator state
                        sim_state = "patient_normal"
                        try:
                            sim_resp = requests.get(api_url.replace("/latest", "/simulator/state"), timeout=timeout_s)
                            if sim_resp.status_code == 200:
                                sim_state = sim_resp.json().get("state", "patient_normal")
                        except Exception:
                            pass
                            
                        with st.spinner("LLM generating clinical draft..."):
                            try:
                                payload = {
                                    "seizure_probability": latest_prob,
                                    "active_state": sim_state,
                                    "features": feats_list
                                }
                                report_resp = requests.post(llm_report_url, json=payload, timeout=30.0)
                                if report_resp.status_code == 200:
                                    st.session_state.latest_report = report_resp.json().get("report", "No report text returned.")
                                else:
                                    st.error(f"Failed to generate report: {report_resp.text}")
                            except Exception as ex:
                                st.error(f"API request failed: {ex}")
                                
                if "latest_report" in st.session_state:
                    st.markdown("### Generated Report Draft")
                    st.markdown(
                        f"""
                        <div style="background-color: #1e222b; border-radius: 8px; padding: 15px; border: 1px solid #2e3440; max-height: 400px; overflow-y: auto;">
                            {st.session_state.latest_report}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    st.download_button(
                        label="Download Report as Text",
                        data=st.session_state.latest_report,
                        file_name=f"clinical_eeg_report_{time.strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown"
                    )
                    
            with col_r:
                st.subheader("Explain Channel Metrics")
                st.markdown("Select an active electrode to interpret its current mathematical features in clinical neurophysiology terms.")
                
                # Check if buffer has data
                if not st.session_state.buffer:
                    st.info("No active EEG data stream. Start the stream first.")
                else:
                    explain_ch_name = st.selectbox(
                        "Electrode to Explain",
                        options=[f"Channel {idx+1} (Best index: {val})" for idx, val in enumerate(BEST_CHANNELS)]
                    )
                    explain_idx = int(explain_ch_name.split()[1]) - 1 # 0-indexed best channel select
                    
                    if st.button("Interpret Channel Features", use_container_width=True, key="btn_explain_channel"):
                        # Extract latest features for this specific channel
                        try:
                            from src.features import extract_features_for_channel
                            buffer_np = np.array(st.session_state.buffer)
                            if len(buffer_np) >= 256:
                                window_raw = buffer_np[-256:, :].T # (time, channels) -> (channels, time)
                                # Grab original channel index
                                orig_ch_idx = BEST_CHANNELS[explain_idx]
                                ch_data = window_raw[orig_ch_idx, :]
                                ch_feats = extract_features_for_channel(ch_data, sfreq=128.0)
                                
                                feat_dict = {
                                    "variance": float(ch_feats[0]),
                                    "rms": float(ch_feats[1]),
                                    "line_length": float(ch_feats[2]),
                                    "kurtosis": float(ch_feats[3]),
                                    "delta_relative": float(ch_feats[4]),
                                    "theta_relative": float(ch_feats[5]),
                                    "alpha_relative": float(ch_feats[6]),
                                    "beta_relative": float(ch_feats[7]),
                                    "gamma_relative": float(ch_feats[8]),
                                }
                                
                                with st.spinner("AI explaining mathematical metrics..."):
                                    explain_payload = {
                                        "channel_idx": orig_ch_idx,
                                        "features": feat_dict
                                    }
                                    explain_resp = requests.post(llm_explain_url, json=explain_payload, timeout=25.0)
                                    if explain_resp.status_code == 200:
                                        st.session_state.latest_explanation = explain_resp.json().get("explanation", "No explanation text returned.")
                                    else:
                                        st.error(f"Explanation API failed: {explain_resp.text}")
                            else:
                                st.warning("Not enough samples in buffer yet (needs at least 256 samples / 2 seconds).")
                        except Exception as e:
                            st.error(f"Failed to calculate channel features: {e}")
                            
                if "latest_explanation" in st.session_state:
                    st.markdown("### AI Physiological Explanation")
                    st.info(st.session_state.latest_explanation)
            
            st.markdown("---")
            st.subheader("Conversational EEG Assistant")
            st.markdown("Ask general questions about the patient's alert history, model configurations, or general clinical neurophysiology of epilepsy.")
            
            # Display chat messages
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    
            if chat_prompt := st.chat_input("Ask the AI EEG Assistant a question..."):
                st.session_state.chat_history.append({"role": "user", "content": chat_prompt})
                with st.chat_message("user"):
                    st.markdown(chat_prompt)
                    
                with st.spinner("Thinking..."):
                    try:
                        chat_url = api_url.replace("/latest", "/llm/chat")
                        chat_resp = requests.post(chat_url, json={"prompt": chat_prompt}, timeout=25.0)
                        if chat_resp.status_code == 200:
                            response_text = chat_resp.json().get("response", "No response text returned.")
                        else:
                            response_text = f"API error ({chat_resp.status_code}): {chat_resp.text}"
                    except Exception as ex:
                        response_text = f"Failed to send request to backend: {ex}"
                        
                st.session_state.chat_history.append({"role": "assistant", "content": response_text})
                with st.chat_message("assistant"):
                    st.markdown(response_text)

# Streamlit-native loop: sleep and rerun if stream is running
if st.session_state.running:
    time.sleep(polling_rate)
    st.rerun()
