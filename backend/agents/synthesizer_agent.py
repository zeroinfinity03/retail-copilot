"""
Synthesizer — composes a single narrative report from the specialist
agents' outputs.

Pipeline:
  1. Read state["user_query"] + sliced outputs from SQL / Web / Chart.
  2. Build a structured user message (USER QUESTION / SQL DATA / WEB RESEARCH / CHART).
  3. One LLM call (gpt-5-mini) returns the briefing as markdown text.
  4. Return {final_report, error}.

Notes:
  - The LLM gets only the slices it needs (sample rows, web answer text, chart title/caption).
  - Full SQL row list, figure_html, figure_json are NOT sent — keeps tokens bounded.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

BACKEND_DIR = Path(__file__).parent.parent
PROMPT_PATH = BACKEND_DIR / "prompts" / "synthesizer.txt"

load_dotenv(BACKEND_DIR / ".env")

MODEL = "gpt-5-mini"
SAMPLE_ROW_LIMIT = 5
WEB_CITATION_LIMIT = 6

_client: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


# ============================================================
# Helpers
# ============================================================

def load_system_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def _format_sql_block(sql: Optional[dict]) -> str:
    if not sql or sql.get("error"):
        if sql and sql.get("error"):
            return f"(error: {sql['error']})"
        return "(none)"
    if not sql.get("rows"):
        return "(none)"
    parts = []
    if sql.get("explanation"):
        parts.append(sql["explanation"])
    parts.append(f"Row count: {sql.get('row_count', 0)}")
    parts.append(f"Sample rows: {sql['rows'][:SAMPLE_ROW_LIMIT]}")
    return "\n".join(parts)


def _format_web_block(web: Optional[dict]) -> str:
    if not web or web.get("error"):
        if web and web.get("error"):
            return f"(error: {web['error']})"
        return "(none)"
    answer = web.get("answer")
    if not answer:
        return "(none)"
    cites = (web.get("citations") or [])[:WEB_CITATION_LIMIT]
    parts = [answer]
    if cites:
        parts.append(f"Citations: {cites}")
    return "\n".join(parts)


def _format_chart_block(chart: Optional[dict]) -> str:
    if not chart or chart.get("error"):
        if chart and chart.get("error"):
            return f"(error: {chart['error']})"
        return "(none)"
    if not chart.get("title"):
        return "(none)"
    parts = [f"Title: {chart['title']}"]
    if chart.get("caption"):
        parts.append(f"Caption: {chart['caption']}")
    return "\n".join(parts)


def _format_forecast_block(fc: Optional[dict]) -> str:
    """Compact forecast summary for the LLM.

    Sends: series label, horizon, holdout MAPEs, last historical value,
    and a 3-row sample (start / mid / end) of the forecast values table.
    Skips full historical and forecast arrays (token-bloated).
    """
    if not fc or fc.get("error"):
        if fc and fc.get("error"):
            return f"(error: {fc['error']})"
        return "(none)"

    series  = fc.get("series_label") or "(unlabeled)"
    horizon = fc.get("horizon_days") or 0
    metrics = fc.get("metrics") or {}
    parts = [f"Series: {series}", f"Horizon: {horizon} days"]

    p_mape = metrics.get("prophet_mape")
    s_mape = metrics.get("sarima_mape")
    winner = metrics.get("winner_by_mape")
    mape_bits = []
    if p_mape is not None:
        mape_bits.append(f"Prophet {p_mape:.2f}%")
    if s_mape is not None:
        mape_bits.append(f"SARIMA {s_mape:.2f}%")
    if mape_bits:
        line = "Holdout MAPE: " + ", ".join(mape_bits)
        if winner:
            line += f" (lower = {winner})"
        parts.append(line)

    historical = fc.get("historical") or []
    if historical:
        last = historical[-1]
        last_ds = last.get("ds")
        last_y  = last.get("y")
        if last_ds is not None and last_y is not None:
            parts.append(f"Last historical ({last_ds}): {float(last_y):.2f}")

    # Sample of the values table (downsampled by the forecast agent; pick first/mid/last)
    table = fc.get("values_table") or []
    if table:
        n = len(table)
        sample_idx = sorted({0, n // 2, n - 1})
        sample = [table[i] for i in sample_idx if 0 <= i < n]
        parts.append("Sample forecast rows (date | prophet | sarima | ensemble):")
        for row in sample:
            ds = row.get("ds")
            p  = row.get("prophet")
            s  = row.get("sarima")
            e  = row.get("ensemble")
            cells = []
            if p is not None: cells.append(f"prophet={p}")
            if s is not None: cells.append(f"sarima={s}")
            if e is not None: cells.append(f"ensemble={e}")
            parts.append(f"  {ds}: " + " | ".join(cells))

    explanation = fc.get("explanation")
    if explanation:
        parts.append(f"Note: {explanation}")

    return "\n".join(parts)


def build_user_message(state: dict) -> str:
    return (
        f"USER QUESTION: {state.get('user_query', '')}\n\n"
        f"SQL DATA:\n{_format_sql_block(state.get('sql_results'))}\n\n"
        f"WEB RESEARCH:\n{_format_web_block(state.get('web_results'))}\n\n"
        f"FORECAST:\n{_format_forecast_block(state.get('forecast_results'))}\n\n"
        f"CHART:\n{_format_chart_block(state.get('chart_results'))}\n"
    )


# ============================================================
# Smart skip
# ============================================================

def should_skip_synthesis(state: dict) -> bool:
    """Skip synthesizer for trivial queries where the section dump is enough."""
    sql = state.get("sql_results") or {}
    web = state.get("web_results") or {}
    chart = state.get("chart_results") or {}
    forecast = state.get("forecast_results") or {}

    has_sql = bool(sql.get("rows"))
    has_web = bool(web.get("answer"))
    has_chart = bool(chart.get("figure_html"))
    has_forecast = bool(forecast.get("ensemble_forecast") or forecast.get("prophet_forecast") or forecast.get("sarima_forecast"))

    # Forecast always needs narrative — never skip when present
    if has_forecast:
        return False

    # Trivial: only a single-row SQL count
    if has_sql and not has_web and not has_chart:
        if sql.get("row_count", 0) <= 1:
            return True

    # Trivial: chart only (no web context)
    if has_chart and not has_web:
        return False  # still synthesize — chart with a narrative reads better

    return False


# ============================================================
# Main entry
# ============================================================

def run(state: dict) -> dict:
    """Compose the final report. Returns {final_report, error, skipped}."""
    # --- Step 1: Smart-skip check — trivial queries don't need synthesis ---
    if should_skip_synthesis(state):
        return {"final_report": None, "error": None, "skipped": True}

    # --- Step 2: Build the user message (format each specialist's output as a text block) ---
    user_message = build_user_message(state)

    # --- Step 3: LLM call — returns markdown narrative ---
    try:
        response = _get_client().chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": load_system_prompt()},
                {"role": "user",   "content": user_message},
            ],
        )
    except Exception as e:
        return {"final_report": None, "error": str(e), "skipped": False}

    # --- Step 4: Pack result and return to supervisor ---
    return {
        "final_report": response.choices[0].message.content,
        "error":        None,
        "skipped":      False,
    }


def run_stream(state: dict):
    """Streaming variant of run(). Yields events as the LLM emits token chunks.

    Events:
      {"type": "skip"}                              — synthesis skipped (trivial query)
      {"type": "text",  "content": "<token chunk>"} — incremental token chunk
      {"type": "done",  "final_report": "<text>"}   — full accumulated report
      {"type": "error", "error": "<message>"}       — fatal API error

    Used by main.py's /api/chat streaming endpoint. Same skip / format logic
    as run(); only the LLM call is switched to stream=True.
    """
    if should_skip_synthesis(state):
        yield {"type": "skip"}
        return

    user_message = build_user_message(state)

    accumulated: list[str] = []
    try:
        stream = _get_client().chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": load_system_prompt()},
                {"role": "user",   "content": user_message},
            ],
            stream=True,
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                accumulated.append(delta)
                yield {"type": "text", "content": delta}
        yield {"type": "done", "final_report": "".join(accumulated)}
    except Exception as e:
        yield {"type": "error", "error": str(e)}


# ============================================================
# Smoke test (mock state)
# ============================================================

if __name__ == "__main__":
    fake_state = {
        "user_query": "Show me total revenue by index_group last quarter",
        "sql_results": {
            "explanation": "Total normalized revenue by index_group for Q3 2020.",
            "row_count": 5,
            "rows": [
                {"index_group": "Ladieswear",    "revenue": 61294},
                {"index_group": "Divided",       "revenue": 19477},
                {"index_group": "Menswear",      "revenue": 5129},
                {"index_group": "Sport",         "revenue": 5125},
                {"index_group": "Baby/Children", "revenue": 1272},
            ],
        },
        "web_results": None,
        "chart_results": {
            "title": "Total revenue by index_group (last quarter)",
            "caption": "Ladieswear was the top driver, generating roughly 3x the next-largest group.",
            "figure_html": "<plotly html>",  # not sent to LLM
        },
    }

    result = run(fake_state)
    print("Skipped:", result["skipped"])
    print("Error:", result["error"])
    print("---")
    print(result["final_report"])
