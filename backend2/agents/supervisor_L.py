"""
Supervisor (LangGraph) — proper parallel version using superstep fan-out.

Graph topology (parallel-aware):

    [entry] → supervisor
                 │
                 │  (conditional edge returns LIST of agent names →
                 │   LangGraph fans them out into ONE superstep,
                 │   nodes run concurrently in the default thread-pool
                 │   executor via langgraph's run_in_executor wrapper)
                 ▼
        ┌────────┼────────┐
        │        │        │
       sql     web     forecast
        │        │        │
        └────────┼────────┘
                 ▼
          post_parallel (barrier — waits for everyone before it fires)
                 │
                 │ (conditional edge → chart OR straight to synthesizer)
                 ▼
              chart? ───┐
                 │      │
                 ▼      ▼
            synthesizer
                 │
                 ▼
                END

Why this works:
  - When add_conditional_edges' routing function returns a LIST, every
    destination runs in parallel as one superstep.
  - When multiple nodes have an edge into the SAME destination, that
    destination acts as a barrier — it only fires after ALL inbound
    branches complete.
  - Sync node functions are wrapped automatically by langgraph with
    `run_in_executor(None, func)` — which submits them to asyncio's
    default ThreadPoolExecutor. So the parallel branches actually run
    on separate threads.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal, Optional, TypedDict

from pydantic import BaseModel

BACKEND_DIR = Path(__file__).parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from agents.sql_agent import run as run_sql
from agents.web_research_agent import run as run_web
from agents.chart_agent import run as run_chart
from agents.forecasting_agent import run as run_forecast
from agents.synthesizer_agent import run as run_synthesizer
from agents._llm import structured       # ← DeepSeek tool-strict-mode helper

try:
    from langgraph.graph import StateGraph, END
except ImportError as e:
    raise SystemExit(
        "langgraph not installed. From backend/ run: uv add langgraph"
    ) from e


PROMPT_PATH = BACKEND_DIR / "prompts" / "supervisor.txt"


# ============================================================
# Schemas
# ============================================================

class PlanStep(BaseModel):
    agent: Literal["sql", "web", "forecast", "chart"]
    task: str


class Plan(BaseModel):
    rationale: str
    steps: list[PlanStep]


class AgentState(TypedDict, total=False):
    user_query: str
    plan: Optional[dict]
    sql_results: Optional[dict]
    web_results: Optional[dict]
    forecast_results: Optional[dict]
    chart_results: Optional[dict]
    final_report: Optional[str]
    synthesizer_skipped: Optional[bool]
    synthesizer_error: Optional[str]
    skip_synthesizer: Optional[bool]


# ============================================================
# Planning
# ============================================================

def make_plan(user_query: str) -> Plan:
    return structured(
        messages=[
            {"role": "system", "content": PROMPT_PATH.read_text(encoding="utf-8")},
            {"role": "user", "content": user_query},
        ],
        schema=Plan,
    )


# ============================================================
# Helpers
# ============================================================

def _task_for(state: AgentState, agent: str) -> str:
    plan = state.get("plan") or {"steps": []}
    for s in plan["steps"]:
        if s["agent"] == agent:
            return s["task"]
    return state.get("user_query", "")


def _agents_in_plan(state: AgentState) -> set[str]:
    plan = state.get("plan") or {"steps": []}
    return {s["agent"] for s in plan["steps"]}


# ============================================================
# Nodes
# ============================================================

def supervisor_node(state: AgentState) -> dict:
    """First and only call: make the plan."""
    if state.get("plan") is not None:
        return {}   # plan already exists, no-op
    plan = make_plan(state["user_query"])
    return {"plan": plan.model_dump()}


def sql_node(state: AgentState) -> dict:
    return {"sql_results": run_sql(_task_for(state, "sql"))}


def web_node(state: AgentState) -> dict:
    return {"web_results": run_web(_task_for(state, "web"))}


def forecast_node(state: AgentState) -> dict:
    return {"forecast_results": run_forecast(_task_for(state, "forecast"))}


def chart_node(state: AgentState) -> dict:
    sql_out = state.get("sql_results") or {}
    if not sql_out.get("rows"):
        return {
            "chart_results": {
                "error": "Chart requested but no SQL rows available."
            }
        }
    return {
        "chart_results": run_chart(
            columns=sql_out["columns"],
            rows=sql_out["rows"],
            user_question=state["user_query"],
        )
    }


def post_parallel_node(state: AgentState) -> dict:
    """Barrier node — fires once SQL and Web (whichever ran) both complete.

    LangGraph runs it only after every inbound branch finishes. It's a no-op
    pass-through; it exists so the conditional edge that follows has a clean
    place to live.
    """
    return {}


def synthesizer_node(state: AgentState) -> dict:
    # Caller (e.g., streaming endpoint) sets skip_synthesizer when it plans to
    # run the synthesizer separately in streaming mode. The node becomes a no-op.
    if state.get("skip_synthesizer"):
        return {}
    out = run_synthesizer(state)
    return {
        "final_report": out["final_report"],
        "synthesizer_skipped": out.get("skipped", False),
        "synthesizer_error": out.get("error"),
    }


# ============================================================
# Routing (conditional edges)
# ============================================================

def fanout_from_supervisor(state: AgentState) -> list[str]:
    """Return the LIST of nodes to run in parallel after the supervisor.

    LangGraph treats a list return as a superstep fan-out — every name in
    the list fires concurrently in the next step.

    `sql`, `web`, and `forecast` are independent (forecast calls sql_agent
    internally, separately from any top-level sql step) so they all fan out
    together. `chart` runs sequentially after the barrier.
    """
    agents = _agents_in_plan(state)

    parallel = [a for a in ("sql", "web", "forecast") if a in agents]
    if parallel:
        return parallel

    # No parallel-eligible agents. Skip directly to chart if present, else synthesizer.
    if "chart" in agents:
        return ["chart"]
    return ["synthesizer"]


def after_parallel(state: AgentState) -> str:
    """After the parallel batch, do we still need the chart or go to synth?

    Route through `chart` whenever the plan asked for one — even if SQL
    returned no rows. `chart_node` itself writes the "no SQL rows" error into
    `chart_results`, matching supervisor_P.py exactly.
    """
    if "chart" in _agents_in_plan(state):
        return "chart"
    return "synthesizer"


# ============================================================
# Graph
# ============================================================

def build_graph():
    g = StateGraph(AgentState)
    g.add_node("supervisor",    supervisor_node)
    g.add_node("sql",           sql_node)
    g.add_node("web",           web_node)
    g.add_node("forecast",      forecast_node)
    g.add_node("post_parallel", post_parallel_node)
    g.add_node("chart",         chart_node)
    g.add_node("synthesizer",   synthesizer_node)

    g.set_entry_point("supervisor")

    # Supervisor → fan out to sql / web / forecast (parallel) — OR straight to chart / synth
    g.add_conditional_edges(
        "supervisor",
        fanout_from_supervisor,
        ["sql", "web", "forecast", "chart", "synthesizer"],
    )

    # SQL, Web, and Forecast all flow into the barrier
    g.add_edge("sql",           "post_parallel")
    g.add_edge("web",           "post_parallel")
    g.add_edge("forecast",      "post_parallel")

    # Barrier → either chart or synthesizer
    g.add_conditional_edges(
        "post_parallel",
        after_parallel,
        ["chart", "synthesizer"],
    )

    # Chart → synthesizer
    g.add_edge("chart",         "synthesizer")

    # Synthesizer is terminal
    g.add_edge("synthesizer",   END)

    return g.compile()


def run(user_query: str, skip_synthesizer: bool = False) -> dict:
    """Run the full multi-agent graph on a user query.

    If skip_synthesizer=True, the synthesizer node short-circuits (the caller
    will stream the synthesizer separately via synthesizer_agent.run_stream).
    """
    app = build_graph()
    initial: AgentState = {
        "user_query":       user_query,
        "plan":             None,
        "sql_results":      None,
        "web_results":      None,
        "forecast_results": None,
        "chart_results":    None,
        "final_report":     None,
        "skip_synthesizer": skip_synthesizer,
    }
    return app.invoke(initial)


# ============================================================
# Smoke test
# ============================================================

if __name__ == "__main__":
    import json
    import time

    q = "Why are jacket sales dropping?"
    print(f"USER: {q}")
    t0 = time.perf_counter()
    result = run(q)
    print(f"Total time: {time.perf_counter() - t0:.1f}s")

    sql      = result.get("sql_results")      or {}
    web      = result.get("web_results")      or {}
    chart    = result.get("chart_results")    or {}
    forecast = result.get("forecast_results") or {}
    print(json.dumps({
        "plan":     result.get("plan"),
        "sql":      {"sql": (sql.get("sql") or "")[:120], "row_count": sql.get("row_count")}                          if sql      else None,
        "web":      {"answer_len": len(web.get("answer") or ""), "citations": len(web.get("citations") or [])}        if web      else None,
        "chart":    {"chart_type": chart.get("chart_type"), "title": chart.get("title")}                              if chart    else None,
        "forecast": {"series_label": forecast.get("series_label"), "horizon_days": forecast.get("horizon_days"),
                     "metrics": forecast.get("metrics")}                                                              if forecast else None,
        "final_report_preview": (result.get("final_report") or "")[:200],
    }, indent=2, default=str))
