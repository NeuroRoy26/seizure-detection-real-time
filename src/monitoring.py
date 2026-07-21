import os
import json
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
