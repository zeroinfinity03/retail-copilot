"""Web research agent tests — extraction gate, decline path, error shape.

The web agent runs in two stages: a gateway LLM extracts a focused web query
(or declines with search_query=None), then Perplexity Sonar Pro researches it.
Both calls cost money and need network, so we mock make_web_query (the
extraction) and the Perplexity client (_get_client) to test the dict shapes.
"""

from unittest.mock import MagicMock, patch

from agents.web_research_agent import run, WebQuery


def _fake_response(answer="External market context.", citations=None, num_queries=3):
    """Build a fake Sonar Pro response with the shape `model_dump` would
    return — content + citations + usage.num_search_queries."""
    resp = MagicMock()
    resp.model_dump.return_value = {
        "choices": [{"message": {"content": answer}}],
        "citations": citations or ["https://example.com/article-1"],
        "usage": {"num_search_queries": num_queries},
    }
    return resp


def test_run_returns_consistent_shape_on_success():
    with patch("agents.web_research_agent.make_web_query",
               return_value=WebQuery(search_query="2026 fashion retail trends")), \
         patch("agents.web_research_agent._get_client") as get_client:
        get_client.return_value.chat.completions.create.return_value = _fake_response()
        out = run("2026 fashion retail trends")

    assert set(out.keys()) >= {"answer", "citations", "search_queries_run", "model", "error"}
    assert out["error"] is None
    assert out["answer"].startswith("External")
    assert out["citations"] == ["https://example.com/article-1"]
    assert out["search_queries_run"] == 3


def test_run_declines_when_no_web_query_needed():
    """When the extractor returns search_query=None (pure internal data or a
    forecast), the agent returns an empty result WITHOUT calling Perplexity."""
    with patch("agents.web_research_agent.make_web_query",
               return_value=WebQuery(search_query=None)), \
         patch("agents.web_research_agent._get_client") as get_client:
        out = run("Forecast knitwear revenue for the next quarter.")
        get_client.assert_not_called()   # Perplexity is never touched

    assert out["answer"] is None
    assert out["citations"] == []
    assert out["search_queries_run"] == 0
    assert out["error"] is None


def test_run_returns_error_dict_when_api_fails():
    """A Perplexity network/API failure must NOT crash — it returns a dict
    with error set (same keys as the happy path)."""
    with patch("agents.web_research_agent.make_web_query",
               return_value=WebQuery(search_query="zara denim pricing 2026")), \
         patch("agents.web_research_agent._get_client") as get_client:
        get_client.return_value.chat.completions.create.side_effect = RuntimeError("network down")
        out = run("How is Zara pricing denim in 2026?")

    assert out["error"] is not None
    assert "network down" in out["error"]
    assert out["answer"] is None
    assert out["citations"] == []
    assert out["search_queries_run"] == 0


def test_run_handles_missing_citations_gracefully():
    """If Perplexity returns no citations, the agent shouldn't break."""
    resp = MagicMock()
    resp.model_dump.return_value = {
        "choices": [{"message": {"content": "answer"}}],
        # no "citations" key
        "usage": {"num_search_queries": 1},
    }
    with patch("agents.web_research_agent.make_web_query",
               return_value=WebQuery(search_query="some focused query")), \
         patch("agents.web_research_agent._get_client") as get_client:
        get_client.return_value.chat.completions.create.return_value = resp
        out = run("anything")

    assert out["citations"] == []
    assert out["error"] is None
