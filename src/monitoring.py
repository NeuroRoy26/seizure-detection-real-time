import os
import json
import logging
from datetime import datetime
import streamlit as st

def track_visit():
    """
    Tracks unique visits by logging connection details (IP, locale, timezone, user-agent)
    to a local JSON file inside the metrics/ directory.
    """
    try:
        # Get client details from Streamlit context
        ip = st.context.ip_address or "Localhost"
        locale = st.context.locale or "Unknown"
        timezone = st.context.timezone or "Unknown"
        
        # Access request headers
        headers = st.context.headers
        user_agent = headers.get("user-agent", "Unknown")
        
        # Hugging Face Spaces specific headers
        # x-hf-user-username is set by Hugging Face when user is logged in
        hf_user = headers.get("x-hf-user-username") or headers.get("x-forwarded-user") or "Guest"

        log_dir = "metrics"
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "unique_visitors.json")

        # Load existing metrics
        if os.path.exists(log_file):
            try:
                with open(log_file, "r") as f:
                    data = json.load(f)
            except Exception:
                data = {"visitors": {}, "total_views": 0}
        else:
            data = {"visitors": {}, "total_views": 0}

        # Unique visitor identifier (combines IP and HF User)
        visitor_id = f"{ip}_{hf_user}"
        now_str = datetime.now().isoformat()

        if visitor_id not in data["visitors"]:
            data["visitors"][visitor_id] = {
                "first_visit": now_str,
                "last_visit": now_str,
                "ip": ip,
                "hf_user": hf_user,
                "locale": locale,
                "timezone": timezone,
                "user_agent": user_agent,
                "visit_count": 1
            }
        else:
            data["visitors"][visitor_id]["last_visit"] = now_str
            data["visitors"][visitor_id]["visit_count"] += 1

        data["total_views"] += 1

        with open(log_file, "w") as f:
            json.dump(data, f, indent=2)
            
    except Exception as e:
        # Fallback to console print to ensure zero user-facing disruption
        print(f"[Monitoring] Error logging visitor details: {e}")


# Configure logger for LLM auditing as a backup/alternative path for external logging agents
audit_logger = logging.getLogger("llm_audit")

def log_llm_transaction(
    request_data: dict,
    response_data: dict,
    latency_ms: float,
    cited_pmids: list,
    uncited_pmids: list,
    citation_coverage: float
) -> None:
    """
    Logs LLM transactions (seizure metadata, prompt parameters, generated output, 
    citation validation coverage, and latency) to metrics/llm_audit.jsonl and 
    emits it to Python system logger.
    """
    try:
        # Try to retrieve context info if running in a Streamlit session
        hf_user = "Guest"
        try:
            if hasattr(st, "context") and st.context:
                headers = st.context.headers
                hf_user = headers.get("x-hf-user-username") or headers.get("x-forwarded-user") or "Guest"
        except Exception:
            pass

        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "user": hf_user,
            "latency_ms": latency_ms,
            "citation_coverage": citation_coverage,
            "cited_pmids": cited_pmids,
            "uncited_pmids": uncited_pmids,
            "request": request_data,
            "response": response_data
        }

        # Log via Python logging framework
        audit_logger.info(json.dumps(log_entry))

        # Write to local file store
        log_dir = "metrics"
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "llm_audit.jsonl")

        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
            
    except Exception as e:
        print(f"[Monitoring] Failed to log LLM transaction: {e}")

