import unittest
from unittest.mock import patch, MagicMock, mock_open
from fastapi.testclient import TestClient
import os
import json
import time
import requests

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
        self.patcher = patch("src.rag_retriever.RAGRetriever.retrieve_literature", return_value=[])
        self.mock_retrieve = self.patcher.start()

    def tearDown(self):
        self.patcher.stop()
        
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
        self.assertTrue(report.startswith("Clinical EEG Report Draft"))
        self.assertTrue("Disclaimer" in report)

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
        self.assertTrue(explanation.startswith("EEG Feature Explanation"))
        self.assertTrue("Disclaimer" in explanation)

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
        self.assertTrue(response.startswith("Retry Success Response"))
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

    @patch("requests.post")
    def test_explain_features_caching(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"choices": [{"message": {"content": "Explanation Content"}}]}
        mock_post.return_value = mock_response

        llm = LLMClient()
        llm.enabled = True
        llm.hf_token = "fake-token"

        # 1st call
        res1 = llm.explain_features(channel_idx=2, features={"variance": 1.0})
        # 2nd call (should hit cache)
        res2 = llm.explain_features(channel_idx=2, features={"variance": 1.0})

        self.assertEqual(res1, res2)
        self.assertEqual(mock_post.call_count, 1)  # Only 1 HTTP call made!

    def test_disabled_client_behavior(self):
        llm = LLMClient()
        llm.enabled = False
        health = llm.check_health()
        self.assertEqual(health["status"], "disabled")
        resp = llm._query_api("test prompt")
        self.assertEqual(resp, "LLM features are currently disabled in configuration.")

    def test_missing_key_behavior(self):
        llm = LLMClient()
        llm.enabled = True
        llm.api_key = ""
        llm.hf_token = ""
        resp = llm._query_api("test prompt")
        self.assertTrue("Missing API key" in resp)

    @patch("builtins.open", side_effect=IOError("corrupted config"))
    def test_load_config_error(self, mock_open_file):
        llm = LLMClient()
        llm.config_path = "some_config.yaml"
        llm.load_config()  # should handle exception internally

    @patch("builtins.open", side_effect=IOError("permission denied"))
    @patch("os.path.exists", return_value=True)
    def test_env_loading_error(self, mock_exists, mock_open_file):
        llm = LLMClient()  # should handle exception internally

    @patch("time.sleep")
    @patch("requests.post")
    def test_booting_delay_handling(self, mock_post, mock_sleep):
        # 503 error
        mock_resp_503 = MagicMock()
        mock_resp_503.status_code = 503
        mock_resp_503.json.return_value = {"estimated_time": 25.0}

        mock_post.return_value = mock_resp_503

        llm = LLMClient()
        llm.enabled = True
        llm.hf_token = "fake-token"

        resp = llm._query_api("test", max_tokens=10)
        self.assertTrue("booting up" in resp)
        self.assertTrue("25 seconds" in resp)

        # JSON decode error case
        mock_resp_503.json.side_effect = ValueError("invalid json")
        resp = llm._query_api("test", max_tokens=10)
        self.assertTrue("booting up" in resp)

    @patch("time.sleep")
    @patch("requests.post")
    def test_timeout_handling(self, mock_post, mock_sleep):
        mock_post.side_effect = requests.exceptions.Timeout("timeout")

        llm = LLMClient()
        llm.enabled = True
        llm.hf_token = "fake-token"

        resp = llm._query_api("test", max_tokens=10)
        self.assertTrue("timed out" in resp)

    @patch("time.sleep")
    @patch("requests.post")
    def test_general_exception_handling(self, mock_post, mock_sleep):
        mock_post.side_effect = RuntimeError("other error")

        llm = LLMClient()
        llm.enabled = True
        llm.hf_token = "fake-token"

        resp = llm._query_api("test", max_tokens=10)
        self.assertTrue("other error" in resp)

    @patch("requests.post")
    def test_http_error_handling(self, mock_post):
        mock_resp_500 = MagicMock()
        mock_resp_500.status_code = 500
        mock_resp_500.text = "Internal Server Error"
        mock_post.return_value = mock_resp_500

        llm = LLMClient()
        llm.enabled = True
        llm.hf_token = "fake-token"

        resp = llm._query_api("test", max_tokens=10)
        self.assertTrue("LLM API error (500)" in resp)
        self.assertTrue("Internal Server Error" in resp)


class TestFastAPILLMEndpoints(unittest.TestCase):
    def setUp(self):
        # Force LLM configuration on the API's client for testing
        api.llm_client.enabled = True
        api.llm_client.hf_token = "fake-token"
        self.client = TestClient(api.app)
        self.patcher = patch("src.rag_retriever.RAGRetriever.retrieve_literature", return_value=[])
        self.mock_retrieve = self.patcher.start()

    def tearDown(self):
        # Restore configuration
        api.llm_client.load_config()
        api.llm_client.hf_token = os.environ.get("HF_TOKEN", "")
        self.patcher.stop()

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


class TestRAGRetriever(unittest.TestCase):
    def setUp(self):
        self.mock_config_content = """
llm:
  enabled: true
  rag:
    enabled: true
    num_articles: 2
    pubmed_script_path: "fake/path/pubmed_api.py"
"""
        self.temp_config_path = "temp_test_config.yaml"
        with open(self.temp_config_path, "w") as f:
            f.write(self.mock_config_content)

    def tearDown(self):
        if os.path.exists(self.temp_config_path):
            os.remove(self.temp_config_path)

    @patch("os.path.exists")
    def test_load_config(self, mock_exists):
        # Setup mock_exists to return True for our fake config path
        mock_exists.side_effect = lambda p: p in [self.temp_config_path, "fake/path/pubmed_api.py"]
        from src.rag_retriever import RAGRetriever
        retriever = RAGRetriever(self.temp_config_path)
        self.assertTrue(retriever.enabled)
        self.assertEqual(retriever.num_articles, 2)
        self.assertEqual(retriever.pubmed_script_path, "fake/path/pubmed_api.py")

    @patch("src.rag_retriever.RAGRetriever._run_pubmed_api")
    @patch("os.path.exists", return_value=True)
    def test_retrieve_literature_success(self, mock_exists, mock_run):
        from src.rag_retriever import RAGRetriever
        retriever = RAGRetriever(self.temp_config_path)
        
        # Mock search_pubmed to return PMIDs
        # Mock fetch_article_abstracts to return list of articles
        mock_run.side_effect = [
            ["12345", "67890"],  # search_pubmed result
            [                    # fetch_article_abstracts result
                {
                    "pmid": "12345",
                    "title": "EEG Analysis",
                    "authors": ["Author A"],
                    "journal": "J Neuro",
                    "pubdate": "2023",
                    "abstract": "Abstract A"
                },
                {
                    "pmid": "67890",
                    "title": "Seizure Classification",
                    "authors": ["Author B"],
                    "journal": "J Epilepsy",
                    "pubdate": "2024",
                    "abstract": "Abstract B"
                }
            ]
        ]
        
        articles = retriever.retrieve_literature(0.8, "patient_seizure")
        self.assertEqual(len(articles), 2)
        self.assertEqual(articles[0]["title"], "EEG Analysis")
        self.assertEqual(articles[1]["title"], "Seizure Classification")

    @patch("src.rag_retriever.RAGRetriever._run_pubmed_api", return_value=None)
    @patch("os.path.exists", return_value=True)
    def test_retrieve_literature_failure(self, mock_exists, mock_run):
        from src.rag_retriever import RAGRetriever
        retriever = RAGRetriever(self.temp_config_path)
        articles = retriever.retrieve_literature(0.1, "normal")
        self.assertEqual(len(articles), 0)


class TestRAGLLMIntegration(unittest.TestCase):
    def setUp(self):
        self.mock_config_content = """
llm:
  enabled: true
  rag:
    enabled: true
    num_articles: 2
    pubmed_script_path: "fake/path/pubmed_api.py"
"""
        self.temp_config_path = "temp_test_config.yaml"
        with open(self.temp_config_path, "w") as f:
            f.write(self.mock_config_content)

    def tearDown(self):
        if os.path.exists(self.temp_config_path):
            os.remove(self.temp_config_path)

    @patch("src.rag_retriever.RAGRetriever._run_pubmed_api")
    @patch("requests.post")
    @patch("os.path.exists", return_value=True)
    def test_generate_report_with_rag(self, mock_exists, mock_post, mock_run):
        # Mock PubMed script execution
        mock_run.side_effect = [
            ["12345"],
            [
                {
                    "pmid": "12345",
                    "title": "EEG Analysis Study",
                    "authors": ["Dr. Smith", "Dr. Jones"],
                    "journal": "J Clinical Neuro",
                    "pubdate": "2022",
                    "abstract": "Seizure activity shows distinct patterns."
                }
            ]
        ]
        
        # Mock LLM API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"choices": [{"message": {"content": "Clinical Report content reflecting study findings."}}]}
        mock_post.return_value = mock_response

        # Instantiate LLMClient with the test config
        llm = LLMClient(self.temp_config_path)
        llm.enabled = True
        llm.hf_token = "fake-token"

        report = llm.generate_report(0.85, "patient_seizure")
        
        # Check that prompt was grounded and final report contains pubmed link
        self.assertIn("EEG Analysis Study", mock_post.call_args[1]["json"]["messages"][0]["content"])
        self.assertIn("Clinical Report content reflecting study findings.", report)
        self.assertIn("https://pubmed.ncbi.nlm.nih.gov/12345/", report)
        self.assertIn("Disclaimer", report)
