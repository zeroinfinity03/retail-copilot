"""End-to-end pipeline test — drives a real query through the whole stack.

This test calls real LLMs (OpenAI + Perplexity) and queries the real DuckDB
warehouse. It costs a few cents per run and takes ~10-30 seconds.

Skipped by default (marked `slow`). Run explicitly with:
    uv run pytest tests/test_e2e.py -m slow

Requires:
  - backend/.env with OPENAI_API_KEY and PERPLEXITY_API_KEY
  - backend/data/db/hm.duckdb (built once via `uv run python scripts/load_data.py`)
"""

import os
from pathlib import Path

import pytest

from agents.supervisor_P import run as run_pipeline

BACKEND_DIR = Path(__file__).parent.parent
DB_PATH = BACKEND_DIR / "data" / "db" / "hm.duckdb"


pytestmark = pytest.mark.skipif(
    not DB_PATH.exists() or not os.getenv("OPENAI_API_KEY"),
    reason="e2e test requires built DuckDB warehouse + OPENAI_API_KEY in .env",
)


@pytest.mark.slow
def test_pipeline_runs_simple_count_query():
    """Smoke test: a single-step SQL query should complete end-to-end."""
    result = run_pipeline("How many active club members are there?")

    assert "plan" in result
    assert result["plan"]["steps"], "supervisor should produce at least one step"
    assert any(s["agent"] == "sql" for s in result["plan"]["steps"])

    sql = result["sql_results"] or {}
    assert sql.get("error") is None, f"SQL agent failed: {sql.get('error')}"
    assert sql.get("row_count") == 1, "expected exactly one row for a count query"


@pytest.mark.slow
def test_pipeline_runs_top_n_with_chart():
    """Top-N query should trigger sql + chart and produce a Plotly figure."""
    result = run_pipeline("Top 3 best-selling Ladieswear articles last quarter as a chart")

    plan_agents = {s["agent"] for s in result["plan"]["steps"]}
    assert "sql" in plan_agents and "chart" in plan_agents

    sql = result["sql_results"] or {}
    chart = result["chart_results"] or {}
    assert sql.get("error") is None
    assert sql.get("row_count", 0) >= 1
    assert chart.get("chart_type") in {"bar", "horizontal_bar"}
    assert chart.get("color_column") is None, (
        "color_column must stay null per the prompt rule"
    )
    assert chart.get("figure_html"), "chart agent must return a self-contained HTML figure"
