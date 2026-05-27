"""
Supervisor (Plain Python) — orchestrates specialist agents without LangGraph.

Flow:
  1. LLM call (make_plan): decompose user query into a structured Plan
     (rationale + ordered list of (agent, task) steps).
  2. Initialise a shared state dict (one slot per specialist).
  3. Stage 1 — parallel batch: dispatch sql / web / forecast steps
     concurrently on a ThreadPoolExecutor, collect results via as_completed.
  4. Stage 2 — sequential chart: if the plan includes a chart step AND
     SQL produced rows, call the chart agent with those rows.
  5. Synthesizer: pass the populated state to the synthesizer for the
     final narrative report.
  6. Return the populated state dict.

This file is the "if we did not use LangGraph at all" reference — the
LangGraph version (supervisor_L.py) has identical behaviour, just expressed
as a StateGraph with conditional edges.
"""

from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Literal, Optional

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

# Make `from agents.X import ...` work whether you run from backend/ or elsewhere
BACKEND_DIR = Path(__file__).parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from agents.sql_agent import run as run_sql
from agents.web_research_agent import run as run_web
from agents.chart_agent import run as run_chart
from agents.forecasting_agent import run as run_forecast
from agents.synthesizer_agent import run as run_synthesizer

PROMPT_PATH = BACKEND_DIR / "prompts" / "supervisor.txt"
load_dotenv(BACKEND_DIR / ".env")

MODEL = "gpt-5-mini"
_client: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


# ============================================================
# Schemas
# ============================================================

class PlanStep(BaseModel):
    agent: Literal["sql", "web", "forecast", "chart"]
    task: str


class Plan(BaseModel):
    rationale: str
    steps: list[PlanStep]


# ============================================================
# Planning
# ============================================================

def make_plan(user_query: str) -> Plan:
    """One LLM call: NL question -> ordered list of (agent, task)."""
    response = _get_client().chat.completions.parse(
        model=MODEL,
        messages=[
            {"role": "system", "content": PROMPT_PATH.read_text(encoding="utf-8")},
            {"role": "user", "content": user_query},
        ],
        response_format=Plan,
    )
    return response.choices[0].message.parsed


# ============================================================
# Main entry
# ============================================================

def run(user_query: str, verbose: bool = False, skip_synthesizer: bool = False) -> dict:
    """Run the full multi-agent pipeline on a user query.

    If skip_synthesizer=True, the synthesizer step is omitted (caller will
    stream the synthesizer separately via synthesizer_agent.run_stream).

    Returns a state dict with all agents' outputs.
    """
    plan = make_plan(user_query)

    if verbose:
        print(f"📋 Plan: {plan.rationale}")
        for s in plan.steps:
            print(f"   - {s.agent}: {s.task}")
        print()

    state: dict = {
        "user_query": user_query,
        "plan": plan.model_dump(),
        "sql_results": None,
        "web_results": None,
        "forecast_results": None,
        "chart_results": None,
        "final_report": None,
    }

    # --- Stage 1: SQL, Web, Forecast all run in PARALLEL (independent) -----
    # The forecast agent calls sql_agent internally for its own history,
    # so it doesn't depend on the supervisor's top-level sql step.
    parallel_jobs: dict = {}
    for step in plan.steps:
        if step.agent in ("sql", "web", "forecast"):
            parallel_jobs[step.agent] = step.task

    if parallel_jobs:
        if verbose:
            print(f"⚡ Running in parallel: {list(parallel_jobs.keys())}")
        t0 = time.perf_counter()

        runners = {"sql": run_sql, "web": run_web, "forecast": run_forecast}
        with ThreadPoolExecutor(max_workers=len(parallel_jobs)) as pool:
            futures = {
                pool.submit(runners[agent], task): agent
                for agent, task in parallel_jobs.items()
            }
            for fut in as_completed(futures):
                agent = futures[fut]
                state[f"{agent}_results"] = fut.result()
                if verbose:
                    print(f"   ✓ {agent} done at +{time.perf_counter() - t0:.1f}s")

    # --- Stage 2: Chart runs AFTER SQL (depends on SQL rows) ---------------
    # The supervisor prompt advises the LLM not to emit a chart step when
    # forecast is in the plan (forecast renders its own combined chart);
    # nothing here actively skips chart based on forecast presence.
    needs_chart = any(s.agent == "chart" for s in plan.steps)
    if needs_chart:
        if verbose: print(f"📊 Chart agent ...")
        sql_out = state.get("sql_results") or {}
        if not sql_out.get("rows"):
            state["chart_results"] = {
                "error": "Chart requested but no SQL rows available."
            }
        else:
            state["chart_results"] = run_chart(
                columns=sql_out["columns"],
                rows=sql_out["rows"],
                user_question=user_query,
            )

    # Synthesizer composes the final narrative report from all agent outputs.
    # Skipped when the caller plans to stream the synthesizer separately.
    if not skip_synthesizer:
        if verbose: print(f"🧶 Synthesizer ...")
        synth_out = run_synthesizer(state)
        state["final_report"] = synth_out["final_report"]
        state["synthesizer_skipped"] = synth_out.get("skipped", False)
        state["synthesizer_error"] = synth_out.get("error")

    return state


# ============================================================
# Smoke test
# ============================================================

if __name__ == "__main__":
    import json

    tests = [
        "Show me the top 5 best-selling Ladieswear articles last quarter as a chart",
        "What are 2026 fashion retail trends?",
        "How many active club members are there?",
        "Forecast our total daily revenue for the next 90 days",
    ]

    for q in tests:
        print("=" * 72)
        print(f"USER: {q}")
        print("-" * 72)
        result = run(q, verbose=True)

        sql      = result.get("sql_results")      or {}
        web      = result.get("web_results")      or {}
        chart    = result.get("chart_results")    or {}
        forecast = result.get("forecast_results") or {}
        brief = {
            "plan_rationale": result["plan"]["rationale"],
            "sql":      {"sql": (sql.get("sql") or "")[:120], "row_count": sql.get("row_count")}                          if sql      else None,
            "web":      {"answer_len": len(web.get("answer") or ""), "citations": len(web.get("citations") or [])}        if web      else None,
            "chart":    {"chart_type": chart.get("chart_type"), "title": chart.get("title"), "caption": chart.get("caption")} if chart    else None,
            "forecast": {"series_label": forecast.get("series_label"), "horizon_days": forecast.get("horizon_days"),
                         "metrics": forecast.get("metrics")}                                                              if forecast else None,
        }
        print(json.dumps(brief, indent=2, default=str))
        print()
