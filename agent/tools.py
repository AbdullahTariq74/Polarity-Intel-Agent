import logging
import os
import time
from typing import List, Optional

from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()

logger = logging.getLogger(__name__)

_client: Optional[TavilyClient] = None

MAX_RETRIES = 3
BASE_DELAY = 1.5


def _get_client() -> TavilyClient:
    global _client
    if _client is None:
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            raise ValueError("TAVILY_API_KEY not set in environment")
        _client = TavilyClient(api_key=api_key)
    return _client


def web_search(query: str, max_results: int = 8) -> List[dict]:
    """
    Execute a web search using the Tavily API.
    Implements exponential backoff retry on transient failures.
    Returns empty list on final failure - never raises (graceful degradation).
    """
    client = _get_client()

    for attempt in range(MAX_RETRIES):
        try:
            response = client.search(
                query=query,
                search_depth="advanced",
                max_results=max_results,
                include_raw_content=True
            )
            results = response.get("results", [])
            logger.debug("Search '%s' returned %d results", query, len(results))
            return [
                {
                    "url": r.get("url", ""),
                    "title": r.get("title", ""),
                    "content": r.get("content", ""),
                    "raw_content": r.get("raw_content", ""),
                    "relevance_score": r.get("score", 0.0)
                }
                for r in results
            ]
        except Exception as exc:
            wait = BASE_DELAY * (2 ** attempt)
            logger.warning("Search attempt %d failed: %s. Retrying in %.1fs", attempt + 1, exc, wait)
            if attempt < MAX_RETRIES - 1:
                time.sleep(wait)

    logger.error("All retry attempts exhausted for query: '%s'. Returning empty.", query)
    return []


def enrich_entity(entity_name: str) -> dict:
    """
    Attempt targeted enrichment of a named investor entity.
    Falls back to empty dict on failure - caller handles absence gracefully.
    """
    try:
        results = web_search(
            f'"{entity_name}" family office investment mandate AUM portfolio',
            max_results=3
        )
        return {"entity": entity_name, "enrichment": results}
    except Exception as exc:
        logger.warning("Enrichment failed for %s: %s", entity_name, exc)
        return {"entity": entity_name, "enrichment": []}
