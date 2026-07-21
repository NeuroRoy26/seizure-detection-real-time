import os
import yaml
import requests
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional

class RAGRetriever:
    """
    Retriever class that implements Retrieval-Augmented Generation (RAG) by
    querying PubMed via direct HTTP requests to NCBI E-utilities, returning
    scientific evidence relevant to the patient's current EEG signal parameters.
    """
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.enabled = False
        self.num_articles = 3
        self.pubmed_script_path = None
        self.load_config()

    def load_config(self) -> None:
        if os.path.exists(self.config_path):
            try:
                import yaml
                with open(self.config_path, "r") as f:
                    config = yaml.safe_load(f)
                    llm_config = config.get("llm", {})
                    rag_config = llm_config.get("rag", {})
                    self.enabled = rag_config.get("enabled", False)
                    self.num_articles = rag_config.get("num_articles", 3)
                    self.pubmed_script_path = rag_config.get("pubmed_script_path", None)
            except Exception:
                pass

    def _search_pubmed(self, query: str) -> List[str]:
        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": self.num_articles,
            "sort": "relevance",
            "retmode": "json"
        }
        try:
            resp = requests.get(url, params=params, timeout=10.0)
            if resp.status_code == 200:
                data = resp.json()
                return data.get("esearchresult", {}).get("idlist", [])
            else:
                print(f"[RAG] PubMed search failed with status {resp.status_code}")
        except Exception as e:
            print(f"[RAG] Error searching PubMed: {e}")
        return []

    def _fetch_abstracts(self, pmids: List[str]) -> List[Dict[str, Any]]:
        if not pmids:
            return []
        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "rettype": "abstract",
            "retmode": "xml"
        }
        try:
            resp = requests.get(url, params=params, timeout=15.0)
            if resp.status_code != 200:
                print(f"[RAG] PubMed fetch failed with status {resp.status_code}")
                return []
            
            root = ET.fromstring(resp.content)
            results = []
            for article in root.iter("PubmedArticle"):
                pmid_elem = article.find(".//PMID")
                if pmid_elem is None:
                    continue

                art = article.find(".//Article")
                if art is None:
                    continue

                authors = []
                for author in art.findall(".//AuthorList/Author"):
                    last = author.findtext("LastName") or ""
                    init = author.findtext("Initials") or ""
                    name = f"{last} {init}".strip() if last else author.findtext("CollectiveName") or ""
                    if name:
                        authors.append(name)

                abstract_parts = []
                for at in art.findall(".//Abstract/AbstractText"):
                    label = at.get("Label")
                    text = "".join(at.itertext())
                    if label:
                        abstract_parts.append(f"{label}: {text}")
                    else:
                        abstract_parts.append(text)
                abstract = "\n".join(abstract_parts) if abstract_parts else None

                doi = None
                for eid in art.findall("ELocationID"):
                    if eid.get("EIdType") == "doi":
                        doi = eid.text
                        break

                journal_elem = art.find(".//Journal")
                journal = None
                pubdate = None
                if journal_elem is not None:
                    journal = journal_elem.findtext("Title")
                    pd = journal_elem.find(".//PubDate")
                    if pd is not None:
                        year = pd.findtext("Year") or ""
                        month = pd.findtext("Month") or ""
                        day = pd.findtext("Day") or ""
                        medline = pd.findtext("MedlineDate") or ""
                        pubdate = f"{year} {month} {day}".strip() if year else medline

                results.append({
                    "pmid": pmid_elem.text,
                    "title": art.findtext("ArticleTitle"),
                    "authors": authors,
                    "journal": journal,
                    "pubdate": pubdate,
                    "doi": doi,
                    "abstract": abstract,
                })
            return results
        except Exception as e:
            print(f"[RAG] Error fetching abstracts: {e}")
        return []

    def _run_pubmed_api(self, action: str, query_or_pmids: Any) -> Any:
        """
        Internal wrapper maintaining compatibility with tests that mock this method.
        Directly calls the native HTTP utilities under the hood.
        """
        if action == "search":
            return self._search_pubmed(query_or_pmids)
        elif action == "fetch":
            return self._fetch_abstracts(query_or_pmids)
        return None

    def retrieve_literature(
        self,
        seizure_probability: float,
        active_state: str,
        features_list: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Formulates queries based on EEG telemetry, searches PubMed, fetches abstracts,
        and returns structured article details.
        """
        if not self.enabled:
            return []

        # 1. Formulate search queries in order of specificity
        queries = []
        
        # Specific query based on probability and active state
        if seizure_probability > 0.5 or "seizure" in active_state.lower():
            queries.append('"EEG seizure detection" OR "epileptiform discharge"')
            queries.append('EEG seizure detection features')
        else:
            # Feature-specific queries
            has_high_power = False
            if features_list:
                avg_delta = sum(f.get("delta", 0.0) for f in features_list) / len(features_list)
                avg_theta = sum(f.get("theta", 0.0) for f in features_list) / len(features_list)
                if avg_delta > 1.5 or avg_theta > 1.5:
                    queries.append('"EEG delta theta power" epilepsy')
                    has_high_power = True
            
            if not has_high_power:
                queries.append('"quantitative EEG features" epilepsy')
        
        # General fallback queries
        queries.append('EEG seizure detection')
        queries.append('EEG epilepsy')

        pmids = []
        # Try queries until we get at least 1 PMID
        for query in queries:
            pmids = self._run_pubmed_api("search", query)
            if pmids:
                break
        
        if not pmids:
            print("[RAG] No literature found for any query.")
            return []

        # 2. Fetch article abstracts
        return self._run_pubmed_api("fetch", pmids)
