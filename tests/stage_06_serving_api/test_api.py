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

