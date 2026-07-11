import unittest
from unittest.mock import patch, MagicMock, mock_open
from fastapi.testclient import TestClient
import os
import json
import time

from src.llm_client import LLMClient
import api

class TestLLMClient(unittest.TestCase):
    def setUp(self):
        self.mock_config = {
            "llm": {
                "enabled": True,
                "model_id": "test-model",
                "api_url": "https://api-inference.huggingface.co/models"
            }
        }
        
    @patch("builtins.open", new_callable=unittest.mock.mock_open, read_data="llm:\n  enabled: true\n  model_id: 'test-model'\n")
    @patch("os.path.exists", return_value=True)
    def test_load_config(self, mock_exists, mock_open):
        llm = LLMClient()
        self.assertTrue(llm.enabled)
        self.assertEqual(llm.model_id, "test-model")

    def test_health_check_ok(self):
        # Test
        llm = LLMClient()
        llm.enabled = True
        llm.hf_token = "fake-token"
        
        health = llm.check_health()
        self.assertEqual(health["status"], "ok")
        self.assertIn("Successfully connected", health["message"])

    def test_health_check_missing_token(self):
        llm = LLMClient()
        llm.enabled = True
        llm.hf_token = ""
        llm.api_key = ""
        
        health = llm.check_health()
        self.assertEqual(health["status"], "missing_token")

    @patch("requests.post")
    def test_generate_report(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"choices": [{"message": {"content": "Clinical EEG Report Draft"}}]}
        mock_post.return_value = mock_response

        llm = LLMClient()
        llm.enabled = True
        llm.hf_token = "fake-token"

        report = llm.generate_report(
            seizure_probability=0.85,
            active_state="patient_seizure",
            features_list=[{"channel": "1", "variance": 10.0, "rms": 5.0, "delta": 0.5, "theta": 0.2}]
        )
        self.assertEqual(report, "Clinical EEG Report Draft")

    @patch("requests.post")
    def test_explain_features(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"choices": [{"message": {"content": "EEG Feature Explanation"}}]}
        mock_post.return_value = mock_response

        llm = LLMClient()
        llm.enabled = True
        llm.hf_token = "fake-token"

        explanation = llm.explain_features(
            channel_idx=0,
            features={"variance": 10.0, "rms": 5.0}
        )
        self.assertEqual(explanation, "EEG Feature Explanation")

    @patch("time.sleep")
    @patch("requests.post")
    def test_retry_on_rate_limit(self, mock_post, mock_sleep):
        # 1st attempt: 429 rate limit
        mock_resp_429 = MagicMock()
        mock_resp_429.status_code = 429
        # 2nd attempt: 200 success
        mock_resp_200 = MagicMock()
        mock_resp_200.status_code = 200
        mock_resp_200.json.return_value = {"choices": [{"message": {"content": "Retry Success Response"}}]}
        
        mock_post.side_effect = [mock_resp_429, mock_resp_200]
        
        llm = LLMClient()
        llm.enabled = True
        llm.hf_token = "fake-token"
        
        response = llm._query_api("Hello", max_tokens=10)
        self.assertEqual(response, "Retry Success Response")
        self.assertEqual(mock_post.call_count, 2)
        mock_sleep.assert_called_once_with(2.0)

    @patch("builtins.open", new_callable=mock_open, read_data="GROQ_API_KEY=test-env-key-123\n")
    @patch("os.path.exists", return_value=True)
    def test_env_loading(self, mock_exists, mock_open_file):
        orig_env = dict(os.environ)
        if "GROQ_API_KEY" in os.environ:
            del os.environ["GROQ_API_KEY"]
            
        try:
            llm = LLMClient()
            self.assertEqual(llm.api_key, "test-env-key-123")
        finally:
            os.environ.clear()
            os.environ.update(orig_env)


class TestFastAPILLMEndpoints(unittest.TestCase):
    def setUp(self):
        # Force LLM configuration on the API's client for testing
        api.llm_client.enabled = True
        api.llm_client.hf_token = "fake-token"
        self.client = TestClient(api.app)

    def tearDown(self):
        # Restore configuration
        api.llm_client.load_config()
        api.llm_client.hf_token = os.environ.get("HF_TOKEN", "")

    @patch("src.llm_client.LLMClient.check_health")
    def test_api_health_endpoint(self, mock_health):
        mock_health.return_value = {"status": "ok", "message": "Connected"}
        response = self.client.get("/llm/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    @patch("src.llm_client.LLMClient._query_api")
    def test_api_report_endpoint(self, mock_query):
        mock_query.return_value = "Mock Report Content"
        payload = {
            "seizure_probability": 0.90,
            "active_state": "patient_seizure",
            "features": [{"channel": "1", "variance": 1.2}]
        }
        response = self.client.post("/llm/report", json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["report"], "Mock Report Content")

    @patch("src.llm_client.LLMClient._query_api")
    def test_api_explain_endpoint(self, mock_query):
        mock_query.return_value = "Mock Explanation Content"
        payload = {
            "channel_idx": 4,
            "features": {"variance": 2.5}
        }
        response = self.client.post("/llm/explain", json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["explanation"], "Mock Explanation Content")

    @patch("src.llm_client.LLMClient._query_api")
    def test_api_chat_endpoint(self, mock_query):
        mock_query.return_value = "Mock Chat Response"
        payload = {
            "prompt": "What is delta activity?"
        }
        response = self.client.post("/llm/chat", json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["response"], "Mock Chat Response")
