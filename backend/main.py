"""
FastAPI server — bridges the frontend chat UI to the supervisor agent.

Endpoint:
  POST /api/chat
    Body: { "messages": [{ "role": "user" | "assistant", "content": str }, ...] }
    Response: text/event-stream (Server-Sent Events) — newline-delimited events:
      data: {"type": "text", "content": "<token chunk>"}
      data: {"type": "text", "content": "<token chunk>"}
      ...
      data: {"type": "complete", "chart_html": ..., "chart_title": ..., "chart_caption": ...}

Specialist agents (sql / web / forecast / chart) run blocking on a worker
thread because they don't stream. Only the synthesizer's narrative output
is streamed token-by-token. The chart payload (if any) ships as the final
"complete" SSE event so the frontend can embed it after the prose finishes.

Run:
  cd backend
  uv run fastapi dev main.py

If the DuckDB warehouse doesn't exist yet, the server auto-builds it
from CSVs in backend/raw_data/ on first startup (~2 min, one-time).

Frontend (Vite) proxies /api → http://localhost:8000.
"""

from __future__ import annotations

import json
import subprocess
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import asyncio
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Make `from agents.X import ...` work
BACKEND_DIR = Path(__file__).parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# Plain Python supervisor (faster to test). Swap to supervisor_L for LangGraph version.
# from agents.supervisor_P import run as run_pipeline
from agents.supervisor_L import run as run_pipeline
from agents.synthesizer_agent import run_stream as run_synthesizer_stream


DB_PATH = BACKEND_DIR / "data" / "db" / "hm.duckdb"
LOAD_SCRIPT = BACKEND_DIR / "scripts" / "load_data.py"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """One-time DB build on first startup if the warehouse is missing."""
    if not DB_PATH.exists():
        print(f"⚠️  DuckDB warehouse not found at {DB_PATH}")
        print(f"→ Building it from CSVs in backend/raw_data/ (this takes ~2 min, one-time)\n")
        result = subprocess.run([sys.executable, str(LOAD_SCRIPT)], cwd=str(BACKEND_DIR))
        if result.returncode != 0:
            print(
                "\n❌ DB build failed. Make sure the H&M CSVs are in backend/raw_data/:\n"
                "   articles.csv, customers.csv, transactions_train.csv"
            )
            raise SystemExit(1)
        print("\n✅ DB ready. Server starting...\n")
    yield


app = FastAPI(title="H&M Retail Insights Backend", lifespan=lifespan)


# ============================================================
# Request / response schemas
# ============================================================

class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]


# ============================================================
# Formatting — turn the supervisor's state dict into Markdown
# ============================================================

def format_table(columns: list[str], rows: list[dict], max_rows: int = 10) -> str:
    if not rows:
        return ""
    rows = rows[:max_rows]
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = "\n".join(
        "| " + " | ".join(str(r.get(c, ""))[:60] for c in columns) + " |"
        for r in rows
    )
    return f"{header}\n{sep}\n{body}"


def format_state_as_markdown(state: dict) -> str:
    parts: list[str] = []

    plan = state.get("plan") or {}
    if plan.get("rationale"):
        parts.append(f"**Plan:** {plan['rationale']}\n")

    sql = state.get("sql_results") or {}
    if sql and (sql.get("rows") or sql.get("error") or sql.get("explanation")):
        parts.append("### 🗄  Internal data\n")
        if sql.get("error"):
            parts.append(f"_Error:_ {sql['error']}\n")
        else:
            if sql.get("explanation"):
                parts.append(f"{sql['explanation']}\n")
            if sql.get("rows") and sql.get("columns"):
                parts.append(format_table(sql["columns"], sql["rows"]))
                if sql.get("row_count", 0) > 10:
                    parts.append(f"_…{sql['row_count'] - 10} more rows_\n")
            parts.append("")

    web = state.get("web_results") or {}
    if web and web.get("answer"):
        parts.append("### 🌐 Market research\n")
        parts.append(web["answer"])
        parts.append("")
        if web.get("citations"):
            parts.append("**Sources:**")
            for c in web["citations"][:6]:
                parts.append(f"- {c}")
            parts.append("")

    fc = state.get("forecast_results") or {}
    if fc and (fc.get("series_label") or fc.get("error")):
        parts.append("### 📈 Forecast\n")
        if fc.get("error"):
            parts.append(f"_Forecast could not be produced:_ {fc['error']}\n")
        else:
            label = fc.get("series_label") or "Forecast"
            horizon = fc.get("horizon_days") or 0
            parts.append(f"**{label} — {horizon}-day projection**")
            m = fc.get("metrics") or {}
            mape_parts = []
            if m.get("prophet_mape") is not None:
                mape_parts.append(f"Prophet {m['prophet_mape']:.2f}%")
            if m.get("sarima_mape") is not None:
                mape_parts.append(f"SARIMA {m['sarima_mape']:.2f}%")
            if mape_parts:
                parts.append(f"_Holdout MAPE — {', '.join(mape_parts)}_\n")
            table = fc.get("values_table") or []
            if table:
                cols = [k for k in ("ds", "prophet", "sarima", "ensemble") if k in table[0]]
                parts.append(format_table(cols, table, max_rows=10))
            # Forecast chart HTML is returned separately in the JSON response.

    chart = state.get("chart_results") or {}
    if chart and (chart.get("title") or chart.get("error")):
        parts.append("### 📊 Chart\n")
        if chart.get("error"):
            parts.append(f"_Chart could not be rendered:_ {chart['error']}\n")
        else:
            parts.append(f"**{chart.get('title', 'Chart')}**")
            if chart.get("caption"):
                parts.append(f"_{chart['caption']}_\n")
            # Note: the actual chart HTML is returned separately in the JSON
            # response so the frontend can embed it in an iframe.

    final = state.get("final_report")
    if final and not final.startswith("(synthesizer step pending"):
        parts.append("---\n")
        parts.append("### 📝 Summary\n")
        parts.append(final)

    if not parts:
        return "I couldn't complete this query — the supervisor returned no usable output."

    return "\n".join(parts)


# ============================================================
# Routes
# ============================================================

@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


def _sse(event: dict) -> str:
    """Format a single Server-Sent Event line."""
    return f"data: {json.dumps(event)}\n\n"


def _pick_chart_payload(state: dict) -> dict:
    """Choose forecast chart over chart-agent chart when both exist."""
    forecast = state.get("forecast_results") or {}
    chart = state.get("chart_results") or {}
    if forecast.get("chart_html"):
        return {
            "chart_html":    forecast["chart_html"],
            "chart_title":   f"{forecast.get('series_label', 'Forecast')} — {forecast.get('horizon_days', 0)}-day projection",
            "chart_caption": forecast.get("explanation"),
        }
    return {
        "chart_html":    chart.get("figure_html"),
        "chart_title":   chart.get("title"),
        "chart_caption": chart.get("caption"),
    }


@app.post("/api/chat")
async def chat(req: ChatRequest):
    """Streaming endpoint. Runs all specialists blocking, streams synthesizer.

    SSE events:
      {"type": "text",     "content": "<chunk>"}   — incremental synthesizer tokens
      {"type": "complete", "chart_html": ..., "chart_title": ..., "chart_caption": ...}
      {"type": "error",    "error": "<message>"}
    """
    user_msgs = [m for m in req.messages if m.role == "user"]
    if not user_msgs:
        async def err():
            yield _sse({"type": "error", "error": "No user message provided."})
        return StreamingResponse(err(), media_type="text/event-stream")

    user_query = user_msgs[-1].content

    # Stage 1: Run the full specialist pipeline (sql / web / forecast / chart) on a
    # worker thread. skip_synthesizer=True tells the supervisor to NOT call the
    # synthesizer node — we'll stream it ourselves below.
    state = await asyncio.to_thread(run_pipeline, user_query, True)

    chart_payload = _pick_chart_payload(state)

    def event_generator():
        # Stage 2: Stream the synthesizer's narrative token-by-token.
        try:
            for event in run_synthesizer_stream(state):
                if event["type"] == "text":
                    yield _sse(event)
                elif event["type"] == "skip":
                    # Trivial query — synthesizer chose to skip. Fall back to the
                    # section-dump format and send the whole thing as one chunk.
                    fallback = format_state_as_markdown(state)
                    yield _sse({"type": "text", "content": fallback})
                elif event["type"] == "error":
                    # Synthesizer LLM call failed — send section dump as fallback.
                    fallback = format_state_as_markdown(state)
                    yield _sse({"type": "text", "content": fallback})
                # "done" event (full accumulated text) — we don't forward it;
                # the frontend has been accumulating from the text chunks already.
        except Exception as e:
            yield _sse({"type": "error", "error": str(e)})
            return

        # Stage 3: Final event — chart payload + completion signal.
        yield _sse({"type": "complete", **chart_payload})

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ============================================================
# OPTIONAL: Enable session-level memory
# ============================================================
#
# By default each query is isolated — the supervisor only sees the latest
# user message, so follow-up references like "now break that down by
# category" don't resolve.
#
# To enable session memory (memory that lasts as long as the browser tab
# is open), REPLACE this section inside chat():
#
#     user_query = user_msgs[-1].content
#
# WITH this section:
#
#     history = "\n".join(
#         f"{m.role.upper()}: {m.content}" for m in req.messages[:-1]
#     )
#     current = req.messages[-1].content
#     user_query = (
#         f"Conversation so far:\n{history}\n\nCurrent question: {current}"
#         if history else current
#     )
#
# Nothing else changes — no agent code is touched. The supervisor receives
# the full conversation as context and can resolve references against
# previous turns. Closing/refreshing the page clears the memory.
# ============================================================
