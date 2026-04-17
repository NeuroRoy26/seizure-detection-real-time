from fastapi.testclient import TestClient


def test_latest_returns_no_data_message(api_module):
    client = TestClient(api_module.app)
    response = client.get("/latest")
    assert response.status_code == 200
    assert response.json() == {"message": "No data available"}


def test_ingest_then_latest_returns_payload_and_probability(api_module, eeg_payload, monkeypatch):
    client = TestClient(api_module.app)
    monkeypatch.setattr(api_module, "_predict_seizure_probability_from_buffer", lambda: 0.42)

    ingest_response = client.post("/ingest", json=eeg_payload)
    latest_response = client.get("/latest")

    assert ingest_response.status_code == 200
    assert ingest_response.json()["message"] == "Data ingested successfully"
    assert latest_response.status_code == 200
    latest_json = latest_response.json()
    assert latest_json["data"] == eeg_payload["data"]
    assert latest_json["seizure_probability"] == 0.42


def test_ingest_rejects_invalid_payload(api_module):
    client = TestClient(api_module.app)
    response = client.post("/ingest", json=[1, 2, 3])
    assert response.status_code == 422
