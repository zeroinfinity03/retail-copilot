"""Web research agent tests — error path shape, model_dump parsing.

The web agent calls Perplexity's Sonar Pro API. Real calls cost money
and require network, so we mock the OpenAI client to test the dict
shapes the agent returns (both happy path and failure path).
"""

from unittest.mock import MagicMock, patch

from agents.web_research_agent import run


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
    with patch("agents.web_research_agent._get_client") as get_client:
        get_client.return_value.chat.completions.create.return_value = _fake_response()
        out = run("2026 fashion retail trends")

    assert set(out.keys()) >= {"answer", "citations", "search_queries_run", "model", "error"}
    assert out["error"] is None
    assert out["answer"].startswith("External")
    assert out["citations"] == ["https://example.com/article-1"]
    assert out["search_queries_run"] == 3


def test_run_returns_error_dict_when_api_fails():
    """Network or API failure must NOT crash — it returns a dict with error set."""
    with patch("agents.web_research_agent._get_client") as get_client:
        get_client.return_value.chat.completions.create.side_effect = RuntimeError("network down")
        out = run("anything")

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
    with patch("agents.web_research_agent._get_client") as get_client:
        get_client.return_value.chat.completions.create.return_value = resp
        out = run("anything")

    assert out["citations"] == []
    assert out["error"] is None
