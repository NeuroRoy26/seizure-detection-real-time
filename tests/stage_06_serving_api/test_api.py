from fastapi.testclient import TestClient


def test_latest_returns_no_data_message(api_module):
    client = TestClient(api_module.app)
    response = client.get("/latest")
    assert response.status_code == 200
    assert response.json() == {"message": "No data available"}


def test_ingest_then_latest_returns_payload_and_probability(api_module, eeg_payload, monkeypatch):
    client = TestClient(api_module.app)
    monkeypatch.setattr(api_module, "_predict_seizure_probability_from_buffer", lambda: 0.42)
    api_module._sample_buffer.clear()

    ingest_response = client.post("/ingest", json=eeg_payload)
    latest_response = client.get("/latest")

    assert ingest_response.status_code == 200
    assert ingest_response.json()["message"] == "Data ingested successfully"
    assert latest_response.status_code == 200
    latest_json = latest_response.json()
    assert latest_json["data"] == [eeg_payload["data"]]
    assert latest_json["seizure_probability"] == 0.42


def test_ingest_rejects_invalid_payload(api_module):
    client = TestClient(api_module.app)
    invalid_payload = [1, 2, 3]
    response = client.post("/ingest", json=invalid_payload)
    assert response.status_code == 422


def test_classify_window_endpoint(api_module, monkeypatch):
    client = TestClient(api_module.app)
    
    class FakeInput:
        name = "input_0"
    
    class FakeSession:
        def get_inputs(self):
            return [FakeInput()]
        def run(self, output_names, input_feed):
            # Return logits where normal class (idx 0) has higher value
            return [[[2.0, -2.0]]]
            
    monkeypatch.setattr(api_module, "_onnx_session", FakeSession())
    
    valid_window = [[0.0] * 256 for _ in range(10)]
    response = client.post("/classify_window", json={"data": valid_window})
    
    assert response.status_code == 200
    json_data = response.json()
    assert "seizure_probability" in json_data
    assert json_data["seizure_probability"] < 0.05


def test_api_startup_event(api_module, monkeypatch):
    import time
    called = []
    def fake_refresh():
        called.append(True)
    monkeypatch.setattr(api_module, "_maybe_refresh_model", fake_refresh)
    
    # Run startup manually
    import asyncio
    asyncio.run(api_module._startup())
    
    assert len(called) == 1
    assert len(api_module._sample_buffer) == api_module.WINDOW_SAMPLES
    assert api_module.latest_data is not None


def test_simulator_state_endpoints(api_module):
    client = TestClient(api_module.app)
    
    # Test GET
    response = client.get("/simulator/state")
    assert response.status_code == 200
    assert "state" in response.json()
    
    # Test POST valid
    response = client.post("/simulator/state", json={"state": "patient_seizure"})
    assert response.status_code == 200
    assert response.json()["state"] == "patient_seizure"
    
    # Test POST invalid
    response = client.post("/simulator/state", json={"state": "invalid_state_name"})
    assert response.status_code == 200
    assert "error" in response.json()


def test_classify_window_missing_model(api_module):
    client = TestClient(api_module.app)
    api_module._onnx_session = None
    
    valid_window = [[0.0] * 256 for _ in range(10)]
    response = client.post("/classify_window", json={"data": valid_window})
    assert response.status_code == 200
    assert "error" in response.json()
    assert response.json()["error"] == "Model not loaded"


def test_disabled_llm_endpoints(api_module):
    client = TestClient(api_module.app)
    api_module.llm_client.enabled = False
    
    # Test report
    response = client.post("/llm/report", json={"seizure_probability": 0.5, "active_state": "normal"})
    assert response.status_code == 200
    assert "error" in response.json()
    
    # Test explain
    response = client.post("/llm/explain", json={"channel_idx": 0, "features": {}})
    assert response.status_code == 200
    assert "error" in response.json()
    
    # Test chat
    response = client.post("/llm/chat", json={"prompt": "hello"})
    assert response.status_code == 200
    assert "error" in response.json()


def test_latest_periodic_refresh(api_module, monkeypatch):
    client = TestClient(api_module.app)
    api_module.latest_data = [1.0] * 23
    
    # Configure variables for refresh
    api_module.MODEL_URL = "http://fake-model-url"
    api_module.MODEL_REFRESH_SECONDS = 1
    api_module._model_last_loaded_at = None
    
    called = []
    monkeypatch.setattr(api_module, "_maybe_refresh_model", lambda: called.append(True))
    monkeypatch.setattr(api_module, "_predict_seizure_probability_from_buffer", lambda: 0.1)
    
    response = client.get("/latest")
    assert response.status_code == 200
    assert len(called) == 1
