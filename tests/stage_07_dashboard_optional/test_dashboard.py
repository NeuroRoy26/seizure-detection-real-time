import importlib
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

CHANNEL_COUNT = 23


class _Ctx(MagicMock):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _SessionState(dict):
    def __getattr__(self, item):
        return self[item]

    def __setattr__(self, key, value):
        self[key] = value


def _fake_streamlit():
    st = SimpleNamespace()
    st.session_state = _SessionState()
    
    sidebar_mock = _Ctx()
    sidebar_mock.columns = lambda *args, **kwargs: [_Ctx(), _Ctx()]
    st.sidebar = sidebar_mock
    # Use MagicMock to safely intercept calls without side effects or recursion.
    st.set_page_config = MagicMock()
    st.markdown = MagicMock()
    st.success = MagicMock()
    st.warning = MagicMock()
    st.title = MagicMock()
    st.subheader = MagicMock()
    st.info = MagicMock()
    st.pyplot = MagicMock()
    st.error = MagicMock()
    st.write = MagicMock()
    st.header = MagicMock()
    st.text_input = lambda *args, **kwargs: "http://127.0.0.1:8000/latest"
    st.slider = lambda *args, **kwargs: kwargs.get("value")
    st.selectbox = lambda *args, **kwargs: "Channel 1"
    st.button = lambda *args, **kwargs: False
    st.tabs = lambda specs, *args, **kwargs: [_Ctx() for _ in range(len(specs))]
    st.spinner = lambda *args, **kwargs: _Ctx()
    def mock_columns(spec, *args, **kwargs):
        n = spec if isinstance(spec, int) else len(spec)
        return [_Ctx() for _ in range(n)]
    st.columns = mock_columns
    st.rerun = lambda: None
    return st


def _import_dashboard(monkeypatch):
    monkeypatch.setitem(sys.modules, "streamlit", _fake_streamlit())
    # Ensure we don't reuse a previously-imported dashboard module that may have
    # imported the real streamlit (or carries mutated session_state).
    sys.modules.pop("dashboard", None)
    dashboard = importlib.import_module("dashboard")
    return dashboard


def test_trim_buffer_keeps_only_last_max_len(monkeypatch):
    dashboard = _import_dashboard(monkeypatch)
    dashboard.st.session_state.buffer = [{"v": i} for i in range(5)]
    dashboard._trim_buffer(max_len=2)
    assert dashboard.st.session_state.buffer == [{"v": 3}, {"v": 4}]


def test_fetch_latest_returns_json(monkeypatch):
    dashboard = _import_dashboard(monkeypatch)

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"data": [1.0] * CHANNEL_COUNT, "seizure_probability": 0.2}

    monkeypatch.setattr(dashboard.requests, "get", lambda *args, **kwargs: FakeResponse())
    payload = dashboard._fetch_latest("http://example.test/latest", timeout_s=1.0)
    assert payload["seizure_probability"] == 0.2


def test_init_state_sets_defaults(monkeypatch):
    dashboard = _import_dashboard(monkeypatch)
    dashboard._init_state()
    assert dashboard.st.session_state["running"] is False
    assert dashboard.st.session_state["buffer"] == []
    assert dashboard.st.session_state["connection_status"] == "Disconnected"


def test_trim_buffer_zero_max_len_clears(monkeypatch):
    dashboard = _import_dashboard(monkeypatch)
    dashboard.st.session_state.buffer = [{"v": 1}, {"v": 2}]
    dashboard._trim_buffer(max_len=0)
    assert dashboard.st.session_state.buffer == []


def test_plot_eeg_empty_buffer_does_not_raise(monkeypatch):
    dashboard = _import_dashboard(monkeypatch)
    dashboard._plot_eeg([], channel_mode="Channel 1")


def test_plot_eeg_single_channel(monkeypatch):
    dashboard = _import_dashboard(monkeypatch)
    # Avoid any matplotlib backend/UI interactions; just ensure no recursion/errors.
    dashboard.plt.subplots = MagicMock(return_value=(MagicMock(), MagicMock()))
    dashboard.plt.close = MagicMock()
    buffer = [{"data": [float(i)] * 23, "seizure_probability": 0.1} for i in range(5)]
    dashboard._plot_eeg(buffer, channel_mode="Channel 3")
