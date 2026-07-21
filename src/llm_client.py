import os
import requests
import yaml
import time
from typing import Optional, Dict, Any, List
from src.serving.rag_retriever import RAGRetriever
from src.monitoring import log_llm_transaction


class LLMClient:
    """
    Client for querying open-source LLMs hosted on Groq or Hugging Face.
    Does not require heavy local installs, is completely free, and queries remotely.
    """
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.retriever = RAGRetriever(self.config_path)
        
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
        self._explanation_cache: Dict[str, str] = {}

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
        if hasattr(self, 'retriever'):
            self.retriever.load_config()

    @property
    def api_url(self) -> str:
        return f"{self.api_url_base.rstrip('/')}/chat/completions"

    def check_health(self) -> Dict[str, Any]:
        """
        Checks if the LLM client is configured, and the required key is present.
        """
        if not self.enabled:
            return {"status": "disabled", "message": "LLM feature flag is disabled in config.yaml."}
        if not self.api_key and not self.hf_token:
            return {"status": "missing_token", "message": "Neither GROQ_API_KEY nor HF_TOKEN is set."}
            
        provider = "Groq" if self.api_key else "Hugging Face"
        return {"status": "ok", "message": f"Successfully connected configuration to {provider} with model: {self.model_id}."}

    def _query_api(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
        """
        Queries the LLM API using chat completions format with exponential backoff on rate limits.
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
        
        max_retries = 3
        backoff = 2.0
        
        start_time = time.time()
        for attempt in range(max_retries):
            try:
                resp = requests.post(self.api_url, json=payload, headers=headers, timeout=25.0)
                if resp.status_code == 200:
                    data = resp.json()
                    if "choices" in data and len(data["choices"]) > 0:
                        message = data["choices"][0].get("message", {})
                        content = message.get("content", "")
                        if content:
                            content_str = content.strip()
                            # Append safety disclaimer guardrail
                            disclaimer = "\n\n---\n*⚠️ **Disclaimer**: This is an AI-generated clinical report assistant tool. All telemetry insights and notes are suggestions only and MUST be reviewed and signed off by a qualified neurologist before making clinical decisions.*"
                            if not content_str.endswith(disclaimer):
                                content_str += disclaimer
                            
                            latency = (time.time() - start_time) * 1000
                            print(f"[LLM LOG] Query Success | Model: {self.model_id} | Latency: {latency:.2f}ms | Length: {len(content_str)} chars")
                            return content_str
                    return "Error: Unexpected response format from LLM API."
                elif resp.status_code == 429:
                    # Rate limit hit, wait and retry
                    if attempt < max_retries - 1:
                        time.sleep(backoff)
                        backoff *= 1.5
                        continue
                    return "Groq free tier rate limit exceeded. Please wait a moment and try again."
                elif resp.status_code == 503:
                    # Model loading / temporary unavailable
                    if attempt < max_retries - 1:
                        time.sleep(backoff)
                        backoff *= 1.5
                        continue
                    try:
                        est_time = resp.json().get("estimated_time", 20.0)
                        return f"The model ({self.model_id}) is currently booting up. Please wait about {int(est_time)} seconds and try again."
                    except Exception:
                        return f"The model ({self.model_id}) is booting up. Please retry in a few moments."
                else:
                    return f"LLM API error ({resp.status_code}): {resp.text}"
            except requests.exceptions.Timeout:
                if attempt == max_retries - 1:
                    return "The request to the LLM API timed out. Please try again."
                time.sleep(backoff)
            except Exception as e:
                if attempt == max_retries - 1:
                    return f"Failed to generate content: {str(e)}"
                time.sleep(backoff)

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

        # Fetch relevant literature context using the RAG retriever if enabled
        articles = []
        literature_context = ""
        if self.retriever.enabled:
            try:
                articles = self.retriever.retrieve_literature(seizure_probability, active_state, features_list)
                if articles:
                    literature_context = "\nRelevant Scientific Literature Context (from PubMed):\n"
                    for idx, art in enumerate(articles, 1):
                        authors = ", ".join(art.get("authors", [])[:3])
                        if len(art.get("authors", [])) > 3:
                            authors += " et al."
                        title = art.get("title", "Unknown Title")
                        journal = art.get("journal", "Unknown Journal")
                        pubdate = art.get("pubdate", "Unknown Date")
                        pmid = art.get("pmid", "Unknown PMID")
                        abstract_snippet = (art.get("abstract", "") or "")[:400]
                        literature_context += f"Reference [{idx}] (PMID: {pmid}): {authors}. '{title}' {journal} ({pubdate}). Abstract: {abstract_snippet}...\n\n"
            except Exception as e:
                print(f"[RAG] Error fetching literature: {e}")

        prompt = f"""<system>
You are an expert clinical neurophysiologist and epileptologist assistant. 
Your goal is to write a highly professional, clinical EEG report summary based on quantitative parameters generated by an real-time AI seizure detection system.
Always write in a clinical, objective, and precise tone.
Do not include any chat conversational filler like "Here is your report". Start immediately with the report.
"""
        if literature_context:
            prompt += """
Incorporate the provided clinical literature/studies to ground your assessment, EEG signals interpretation, and next-step recommendations.
Cite the references using their [PMID: XXX] or Reference Index (e.g., [1]) when discussing relevant findings.
"""
        prompt += """</system>

User:
Generate an EEG monitoring report based on the following automated detection data:
"""
        if literature_context:
            prompt += f"{literature_context}\n"

        prompt += f"""- AI Seizure Probability: {seizure_probability * 100:.1f}%
- Reported Patient Status: {active_state.upper().replace('_', ' ')}
{features_summary}
Please format your response in standard Markdown using these sections:
1. **CLINICAL STATUS & ALERT SUMMARY** (State status, probability, and whether alert criteria were met)
2. **SIGNAL QUANTITATIVE METRICS** (Highlight significant variances, slow-wave delta/theta power increases, and anomalous channels)
3. **CLINICAL IMPLICATIONS & RECOMMENDATION** (Provide assessment and recommended next steps)
"""
        if literature_context:
            prompt += "4. **REFERENCES & EVIDENCE GROUNDING** (Provide a list of PubMed papers referenced with their PMIDs and details)\n"
        
        prompt += "\nAssistant:"
        
        t_start = time.perf_counter()
        report = self._query_api(prompt, max_tokens=750, temperature=0.3)
        latency_ms = (time.perf_counter() - t_start) * 1000

        # Calculate citation coverage
        cited_pmids = []
        uncited_pmids = []
        citation_coverage = 1.0
        if articles:
            cited_count = 0
            for idx, art in enumerate(articles, 1):
                pmid = art.get("pmid")
                if pmid:
                    if pmid in report or f"[{idx}]" in report or f"Reference [{idx}]" in report:
                        cited_count += 1
                        cited_pmids.append(pmid)
                    else:
                        uncited_pmids.append(pmid)
            citation_coverage = cited_count / len(articles) if articles else 1.0

        # Append explicit PubMed URLs to ensure compliance with the literature grounding rules
        if articles:
            urls_section = "\n\n---\n**PubMed Grounding Sources:**\n"
            for art in articles:
                pmid = art.get("pmid")
                if pmid:
                    url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                    urls_section += f"- [{art.get('title', 'PubMed Article')}]({url}) (PMID: {pmid})\n"
            
            # Append citation coverage badge
            if citation_coverage == 1.0:
                urls_section += f"\n✅ **Citation Grounding**: 100% ({len(cited_pmids)}/{len(articles)} sources verified)\n"
            else:
                urls_section += f"\n⚠️ **Citation Grounding**: {citation_coverage*100:.0f}% ({len(cited_pmids)}/{len(articles)} sources verified, {len(uncited_pmids)} uncited: {', '.join(uncited_pmids)})\n"

            # Find the safety disclaimer and insert the URLs before it
            disclaimer = "\n\n---\n*⚠️ **Disclaimer**: This is an AI-generated clinical report assistant tool. All telemetry insights and notes are suggestions only and MUST be reviewed and signed off by a qualified neurologist before making clinical decisions.*"
            if report.endswith(disclaimer):
                report = report[:-len(disclaimer)] + urls_section + disclaimer
            else:
                report = report + urls_section + disclaimer
        
        # Log the full transaction for security and clinical audits
        try:
            req_data = {
                "seizure_probability": seizure_probability,
                "active_state": active_state,
                "features_summary": features_summary
            }
            resp_data = {
                "report": report
            }
            log_llm_transaction(
                request_data=req_data,
                response_data=resp_data,
                latency_ms=latency_ms,
                cited_pmids=cited_pmids,
                uncited_pmids=uncited_pmids,
                citation_coverage=citation_coverage
            )
        except Exception as e:
            print(f"[LLM] Error calling log_llm_transaction: {e}")
        
        return report

    def explain_features(self, channel_idx: int, features: Dict[str, float]) -> str:
        """
        Explains why a specific channel is exhibiting anomalous features in physiological terms.
        Uses in-memory caching to avoid redundant API queries.
        """
        cache_key = f"{channel_idx}:{sorted(features.items())}"
        if cache_key in self._explanation_cache:
            print(f"[LLM CACHE] Hit for key: {cache_key}")
            return self._explanation_cache[cache_key]

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
        resp = self._query_api(prompt, max_tokens=300, temperature=0.3)
        if resp and not resp.startswith("Error") and "failed" not in resp.lower() and "disabled" not in resp.lower() and "missing" not in resp.lower():
            self._explanation_cache[cache_key] = resp
        return resp

