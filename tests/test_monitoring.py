import os
import json
import unittest
from unittest.mock import patch, MagicMock
from src.monitoring import log_llm_transaction

class TestMonitoringAudit(unittest.TestCase):
    def setUp(self):
        self.log_dir = "metrics"
        self.log_file = os.path.join(self.log_dir, "llm_audit.jsonl")
        # Ensure clean slate
        if os.path.exists(self.log_file):
            try:
                os.remove(self.log_file)
            except Exception:
                pass

    def tearDown(self):
        # Cleanup
        if os.path.exists(self.log_file):
            try:
                os.remove(self.log_file)
            except Exception:
                pass

    @patch("src.monitoring.audit_logger")
    def test_log_llm_transaction_success(self, mock_logger):
        req_data = {
            "seizure_probability": 0.85,
            "active_state": "seizure",
            "features_summary": "Channel 1 high variance"
        }
        resp_data = {
            "report": "Patient is exhibiting epileptiform patterns."
        }
        latency = 1200.5
        cited = ["12345"]
        uncited = ["67890"]
        coverage = 0.5

        # Call function
        log_llm_transaction(
            request_data=req_data,
            response_data=resp_data,
            latency_ms=latency,
            cited_pmids=cited,
            uncited_pmids=uncited,
            citation_coverage=coverage
        )

        # 1. Assert local audit file was created and contains valid entry
        self.assertTrue(os.path.exists(self.log_file))
        
        with open(self.log_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        self.assertEqual(len(lines), 1)
        entry = json.loads(lines[0])
        
        self.assertIn("timestamp", entry)
        self.assertEqual(entry["user"], "Guest")
        self.assertEqual(entry["latency_ms"], latency)
        self.assertEqual(entry["citation_coverage"], coverage)
        self.assertEqual(entry["cited_pmids"], cited)
        self.assertEqual(entry["uncited_pmids"], uncited)
        self.assertEqual(entry["request"], req_data)
        self.assertEqual(entry["response"], resp_data)

        # 2. Assert Python logger was called
        mock_logger.info.assert_called_once()
        logger_arg = mock_logger.info.call_args[0][0]
        logger_entry = json.loads(logger_arg)
        self.assertEqual(logger_entry["latency_ms"], latency)

    @patch("src.monitoring.st")
    @patch("src.monitoring.audit_logger")
    def test_log_llm_transaction_streamlit_headers(self, mock_logger, mock_st):
        # Mock st.context.headers to simulate user name extraction on HF Spaces
        mock_st.context = MagicMock()
        mock_st.context.headers = {
            "x-hf-user-username": "NeuroClinician99"
        }

        req_data = {}
        resp_data = {}

        log_llm_transaction(
            request_data=req_data,
            response_data=resp_data,
            latency_ms=100.0,
            cited_pmids=[],
            uncited_pmids=[],
            citation_coverage=1.0
        )

        with open(self.log_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        entry = json.loads(lines[0])
        self.assertEqual(entry["user"], "NeuroClinician99")
