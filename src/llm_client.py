import os
import requests
import yaml
from typing import Optional, Dict, Any, List

class LLMClient:
    """
    Client for querying open-source LLMs hosted on Groq or Hugging Face.
    Does not require heavy local installs, is completely free, and queries remotely.
    """
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        
        # Load local .env file if it exists (for local development/testing)
        if os.path.exists(".env"):
            try:
                with open(".env", "r") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            os.environ[k.strip()] = v.strip()
            except Exception:
                pass

        self.enabled = False
        self.model_id = "llama-3.1-8b-instant"
        self.api_url_base = "https://api.groq.com/openai/v1"
        self.api_key = os.environ.get("GROQ_API_KEY", "").strip()
        self.hf_token = os.environ.get("HF_TOKEN", "").strip()
        
        self.load_config()

    def load_config(self) -> None:
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r") as f:
                    config = yaml.safe_load(f)
                    llm_config = config.get("llm", {})
                    self.enabled = llm_config.get("enabled", False)
                    self.model_id = llm_config.get("model_id", self.model_id)
                    self.api_url_base = llm_config.get("api_url", self.api_url_base)
            except Exception:
                pass

    @property
    def api_url(self) -> str:
        return f"{self.api_url_base.rstrip('/')}/chat/completions"

    def check_health(self) -> Dict[str, Any]:
        """
        Checks if the LLM client is configured, the required key is present, and the API is reachable.
        """
        if not self.enabled:
            return {"status": "disabled", "message": "LLM feature flag is disabled in config.yaml."}
        if not self.api_key and not self.hf_token:
            return {"status": "missing_token", "message": "Neither GROQ_API_KEY nor HF_TOKEN is set."}
            
        try:
            # Send a minimal test request to verify authentication & model availability
            token = self.api_key if self.api_key else self.hf_token
            headers = {"Authorization": f"Bearer {token}"}
            payload = {
                "model": self.model_id,
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 5
            }
            # Set a low timeout for the health check
            resp = requests.post(self.api_url, json=payload, headers=headers, timeout=5.0)
            if resp.status_code == 200:
                provider = "Groq" if self.api_key else "Hugging Face"
                return {"status": "ok", "message": f"Successfully connected to {provider} model: {self.model_id}."}
            elif resp.status_code == 503:
                return {"status": "loading", "message": "Model is currently loading on the server. Try again in a few seconds."}
            else:
                return {"status": "error", "message": f"API responded with status code {resp.status_code}: {resp.text}"}
        except requests.exceptions.Timeout:
            return {"status": "timeout", "message": "Health check timed out trying to reach LLM API."}
        except Exception as e:
            return {"status": "error", "message": f"Connection failed: {str(e)}"}

    def _query_api(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
        """
        Queries the LLM API using chat completions format.
        """
        if not self.enabled:
            return "LLM features are currently disabled in configuration."
        
        token = self.api_key if self.api_key else self.hf_token
        if not token:
            return "Missing API key. Please configure GROQ_API_KEY or HF_TOKEN to enable AI features."
            
        headers = {"Authorization": f"Bearer {token}"}
        payload = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        try:
            resp = requests.post(self.api_url, json=payload, headers=headers, timeout=25.0)
            if resp.status_code == 200:
                data = resp.json()
                if "choices" in data and len(data["choices"]) > 0:
                    message = data["choices"][0].get("message", {})
                    content = message.get("content", "")
                    if content:
                        return content.strip()
                return "Error: Unexpected response format from LLM API."
            elif resp.status_code == 503:
                try:
                    est_time = resp.json().get("estimated_time", 20.0)
                    return f"The model ({self.model_id}) is currently booting up. Please wait about {int(est_time)} seconds and try again."
                except Exception:
                    return f"The model ({self.model_id}) is booting up. Please retry in a few moments."
            else:
                return f"LLM API error ({resp.status_code}): {resp.text}"
        except requests.exceptions.Timeout:
            return "The request to the LLM API timed out. Please try again."
        except Exception as e:
            return f"Failed to generate content: {str(e)}"

    def generate_report(self, seizure_probability: float, active_state: str, features_list: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Generates a professional clinical EEG monitoring report.
        """
        # Format the features list for the prompt
        features_summary = ""
        if features_list:
            features_summary += "\nChannel Specific Feature Metrics:\n"
            for f in features_list:
                ch = f.get("channel", "Unknown")
                var = f.get("variance", 0.0)
                rms = f.get("rms", 0.0)
                delta = f.get("delta", 0.0)
                theta = f.get("theta", 0.0)
                features_summary += f"- Channel {ch}: Variance={var:.2f}, RMS={rms:.2f}, Delta Power={delta:.2f}, Theta Power={theta:.2f}\n"

        prompt = f"""<system>
You are an expert clinical neurophysiologist and epileptologist assistant. 
Your goal is to write a highly professional, clinical EEG report summary based on quantitative parameters generated by an real-time AI seizure detection system.
Always write in a clinical, objective, and precise tone.
Do not include any chat conversational filler like "Here is your report". Start immediately with the report.
</system>

User:
Generate an EEG monitoring report based on the following automated detection data:
- AI Seizure Probability: {seizure_probability * 100:.1f}%
- Reported Patient Status: {active_state.upper().replace('_', ' ')}
{features_summary}
Please format your response in standard Markdown using these sections:
1. **CLINICAL STATUS & ALERT SUMMARY** (State status, probability, and whether alert criteria were met)
2. **SIGNAL QUANTITATIVE METRICS** (Highlight significant variances, slow-wave delta/theta power increases, and anomalous channels)
3. **CLINICAL IMPLICATIONS & RECOMMENDATION** (Provide assessment and recommended next steps)

Assistant:"""
        
        return self._query_api(prompt, max_tokens=600, temperature=0.3)

    def explain_features(self, channel_idx: int, features: Dict[str, float]) -> str:
        """
        Explains why a specific channel is exhibiting anomalous features in physiological terms.
        """
        feats_str = "\n".join([f"- {k}: {v:.4f}" for k, v in features.items()])
        prompt = f"""<system>
You are a medical AI explainer designed to bridge the gap between machine learning features and clinical neurophysiology.
Explain the mathematical EEG feature properties of a specific channel in natural, physiological terms for a medical practitioner. Keep it concise (under 200 words).
Do not include conversational preambles.
</system>

User:
Explain the following features computed from a 2-second window on Channel {channel_idx + 1}:
{feats_str}

Please explain:
1. What these mathematical metrics mean biologically/physiologically.
2. Whether this pattern points to normal brain activity, motion artifacts, or potential epileptiform/ictal discharges.

Assistant:"""
        return self._query_api(prompt, max_tokens=300, temperature=0.3)

