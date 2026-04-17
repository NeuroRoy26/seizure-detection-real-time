import importlib
import sys
from types import SimpleNamespace

CHANNEL_COUNT = 23


class _Ctx:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        del exc_type, exc, tb
        return False


class _SessionState(dict):
    def __getattr__(self, item):
        return self[item]

    def __setattr__(self, key, value):
        self[key] = value


def _fake_streamlit():
    st = SimpleNamespace()
    st.session_state = _SessionState()
    st.sidebar = _Ctx()
    st.title = lambda *args, **kwargs: None
    st.subheader = lambda *args, **kwargs: None
    st.info = lambda *args, **kwargs: None
    st.pyplot = lambda *args, **kwargs: None
    st.error = lambda *args, **kwargs: None
    st.write = lambda *args, **kwargs: None
    st.header = lambda *args, **kwargs: None
    st.text_input = lambda *args, **kwargs: "http://127.0.0.1:8000/latest"
    st.slider = lambda *args, **kwargs: kwargs.get("value")
    st.selectbox = lambda *args, **kwargs: "Channel 1"
    st.button = lambda *args, **kwargs: False
    st.columns = lambda *args, **kwargs: [_Ctx(), _Ctx(), _Ctx()]
    st.rerun = lambda: None
    return st


def test_trim_buffer_keeps_only_last_max_len(monkeypatch):
    monkeypatch.setitem(sys.modules, "streamlit", _fake_streamlit())
    dashboard = importlib.import_module("dashboard")
    dashboard.st.session_state.buffer = [{"v": i} for i in range(5)]
    dashboard._trim_buffer(max_len=2)
    assert dashboard.st.session_state.buffer == [{"v": 3}, {"v": 4}]


def test_fetch_latest_returns_json(monkeypatch):
    monkeypatch.setitem(sys.modules, "streamlit", _fake_streamlit())
    dashboard = importlib.import_module("dashboard")

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"data": [1.0] * CHANNEL_COUNT, "seizure_probability": 0.2}

    monkeypatch.setattr(dashboard.requests, "get", lambda *args, **kwargs: FakeResponse())
    payload = dashboard._fetch_latest("http://example.test/latest", timeout_s=1.0)
    assert payload["seizure_probability"] == 0.2
