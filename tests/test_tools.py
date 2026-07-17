from unittest.mock import MagicMock, patch

from agent import tools


def _mock_tavily_response():
    return {
        "results": [
            {
                "url": "https://example.com/investor",
                "title": "Example Capital",
                "content": "Example Capital is a family office.",
                "raw_content": "Example Capital is a family office investing in growth equity.",
                "score": 0.87
            }
        ]
    }


def test_web_search_returns_normalized_results():
    with patch.object(tools, "_get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.search.return_value = _mock_tavily_response()
        mock_get_client.return_value = mock_client

        results = tools.web_search("example family office", max_results=5)

        assert len(results) == 1
        assert results[0]["url"] == "https://example.com/investor"
        assert results[0]["title"] == "Example Capital"
        assert results[0]["relevance_score"] == 0.87
        mock_client.search.assert_called_once()


def test_web_search_returns_empty_list_after_exhausted_retries():
    with patch.object(tools, "_get_client") as mock_get_client, \
         patch.object(tools.time, "sleep") as mock_sleep:
        mock_client = MagicMock()
        mock_client.search.side_effect = RuntimeError("transient network error")
        mock_get_client.return_value = mock_client

        results = tools.web_search("failing query", max_results=5)

        assert results == []
        assert mock_client.search.call_count == tools.MAX_RETRIES
        assert mock_sleep.call_count == tools.MAX_RETRIES - 1


def test_enrich_entity_falls_back_to_empty_on_failure():
    with patch.object(tools, "web_search", side_effect=RuntimeError("boom")):
        result = tools.enrich_entity("Example Capital")

        assert result["entity"] == "Example Capital"
        assert result["enrichment"] == []
