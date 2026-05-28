"""Synthesizer tests — should_skip_synthesis logic and format helpers.

Pure tests: don't hit the LLM. They verify the skip decision and the
per-specialist formatting that's sent to the LLM.
"""

from agents.synthesizer_agent import (
    _format_chart_block,
    _format_sql_block,
    _format_web_block,
    should_skip_synthesis,
)


# ============================================================
# should_skip_synthesis
# ============================================================

def test_skip_when_single_row_sql_only():
    """A bare 'count' query with one row doesn't need a narrative wrapper."""
    state = {
        "sql_results": {"row_count": 1, "rows": [{"x": 1}]},
        "web_results": None,
        "chart_results": None,
        "forecast_results": None,
    }
    assert should_skip_synthesis(state) is True


def test_never_skip_when_forecast_present():
    """Forecast always needs narrative — even with single-row SQL."""
    state = {
        "sql_results": {"row_count": 1, "rows": [{"x": 1}]},
        "web_results": None,
        "chart_results": None,
        "forecast_results": {"ensemble_forecast": [{"yhat": 1.0}]},
    }
    assert should_skip_synthesis(state) is False


def test_never_skip_when_sql_has_multiple_rows():
    state = {
        "sql_results": {"row_count": 5, "rows": [{"x": i} for i in range(5)]},
        "web_results": None,
        "chart_results": None,
        "forecast_results": None,
    }
    assert should_skip_synthesis(state) is False


def test_never_skip_when_web_present():
    state = {
        "sql_results": None,
        "web_results": {"answer": "external context"},
        "chart_results": None,
        "forecast_results": None,
    }
    assert should_skip_synthesis(state) is False


# ============================================================
# Format helpers — these slim each specialist's output for the LLM
# ============================================================

def test_format_sql_block_includes_explanation_and_row_count():
    sql = {
        "rows": [{"x": 1}, {"x": 2}],
        "row_count": 2,
        "explanation": "Top 2 items",
    }
    text = _format_sql_block(sql)
    assert "Top 2 items" in text
    assert "Row count: 2" in text


def test_format_sql_block_reports_error():
    text = _format_sql_block({"error": "syntax error near GROUP"})
    assert "syntax error" in text


def test_format_sql_block_handles_none():
    assert _format_sql_block(None) == "(none)"


def test_format_web_block_caps_citations():
    """Web citations are capped at 6 to avoid token bloat."""
    web = {
        "answer": "market context",
        "citations": [f"https://example.com/{i}" for i in range(20)],
    }
    text = _format_web_block(web)
    assert "market context" in text
    # the 7th-onward citation should NOT appear (cap is 6)
    assert "https://example.com/6" not in text


def test_format_chart_block_returns_title_and_caption():
    chart = {"title": "Top sellers", "caption": "Pluto led."}
    text = _format_chart_block(chart)
    assert "Top sellers" in text
    assert "Pluto led." in text
