import streamlit as st
import requests
import time
import matplotlib.pyplot as plt

API_URL_DEFAULT = "http://127.0.0.1:8000/latest"
CHANNELS = 23


def _init_state() -> None:
    st.session_state.setdefault("running", False)
    st.session_state.setdefault("buffer", [])  # list[dict]: {"data": [...], "seizure_probability": float}
    st.session_state.setdefault("connection_status", "Disconnected")
    st.session_state.setdefault("last_update_time", None)


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


def _plot_eeg(buffer: list[dict], channel_mode: str) -> None:
    st.subheader("EEG Data")
    if not buffer:
        st.info("No samples buffered yet.")
        return

    fig, ax = plt.subplots()
    if channel_mode == "All channels (overlaid)":
        for i in range(CHANNELS):
            ax.plot([x["data"][i] for x in buffer], linewidth=1, alpha=0.8, label=f"Ch {i+1}")
        ax.legend(ncols=4, fontsize=7)
    else:
        # channel_mode looks like "Channel 1" ... "Channel 23"
        idx = int(channel_mode.split()[-1]) - 1
        ax.plot([x["data"][idx] for x in buffer], linewidth=2, label=channel_mode)
        ax.legend()
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Amplitude")
    st.pyplot(fig)
    plt.close(fig)


def _plot_prob(buffer: list[dict]) -> None:
    st.subheader("Seizure Probability")
    if not buffer:
        st.info("No samples buffered yet.")
        return

    fig, ax = plt.subplots()
    ax.plot([x["seizure_probability"] for x in buffer], linewidth=2)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Probability")
    st.pyplot(fig)
    plt.close(fig)


_init_state()

st.title("EEG Dashboard")

with st.sidebar:
    st.header("Settings")
    api_url = st.text_input("FastAPI `/latest` URL", value=API_URL_DEFAULT)
    polling_rate = st.slider("Poll interval (seconds)", min_value=0.05, max_value=2.0, value=0.1, step=0.05)
    window_seconds = st.slider("Rolling window (seconds)", min_value=1, max_value=60, value=10, step=1)
    timeout_s = st.slider("HTTP timeout (seconds)", min_value=0.5, max_value=10.0, value=2.0, step=0.5)
    channel_mode = st.selectbox(
        "EEG plot",
        options=["Channel 1"] + [f"Channel {i}" for i in range(2, CHANNELS + 1)] + ["All channels (overlaid)"],
        index=0,
    )
    if st.button("Clear buffer"):
        st.session_state.buffer = []

col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    if st.button("Start", disabled=st.session_state.running):
        st.session_state.running = True
with col2:
    if st.button("Stop", disabled=not st.session_state.running):
        st.session_state.running = False
with col3:
    st.write(f"**Status**: {st.session_state.connection_status}")
    st.write(f"**Last update**: {st.session_state.last_update_time}")

max_len = int(window_seconds / max(polling_rate, 1e-6))

if st.session_state.running:
    try:
        payload = _fetch_latest(api_url, timeout_s=timeout_s)
        if "data" in payload and "seizure_probability" in payload:
            st.session_state.buffer.append(payload)
            _trim_buffer(max_len=max_len)
            st.session_state.connection_status = "Connected"
            st.session_state.last_update_time = time.strftime("%Y-%m-%d %H:%M:%S")
        else:
            # e.g. {"message": "No data available"}
            st.session_state.connection_status = "Connected (no data yet)"
    except Exception as e:
        st.session_state.connection_status = f"Disconnected ({type(e).__name__})"
        st.error(f"Failed to fetch data: {e}")

_plot_eeg(st.session_state.buffer, channel_mode=channel_mode)
_plot_prob(st.session_state.buffer)

# Streamlit-native "loop": sleep then rerun (no infinite while loop).
if st.session_state.running:
    time.sleep(polling_rate)
    st.rerun()
