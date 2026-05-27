"""
FastAPI server — bridges the frontend chat UI to the supervisor agent.

Endpoint:
  POST /api/chat
    Body: { "messages": [{ "role": "user" | "assistant", "content": str }, ...] }
    Response: JSON — { text: markdown, chart_html?, chart_title?, chart_caption? }

Run:
  cd backend
  uv run fastapi dev main.py

If the DuckDB warehouse doesn't exist yet, the server auto-builds it
from CSVs in backend/raw_data/ on first startup (~2 min, one-time).

Frontend (Vite) proxies /api → http://localhost:8000.
"""

from __future__ import annotations

import subprocess
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import asyncio
from fastapi import FastAPI
from pydantic import BaseModel

# Make `from agents.X import ...` work
BACKEND_DIR = Path(__file__).parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# Plain Python supervisor (faster to test). Swap to supervisor_L for LangGraph version.
from agents.supervisor_P import run as run_pipeline


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
# Response schema
# ============================================================

class ChatResponse(BaseModel):
    text: str
    chart_html: Optional[str] = None
    chart_title: Optional[str] = None
    chart_caption: Optional[str] = None


# ============================================================
# Routes
# ============================================================

@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    user_msgs = [m for m in req.messages if m.role == "user"]
    if not user_msgs:
        return ChatResponse(text="No user message provided.")

    user_query = user_msgs[-1].content

    # Run pipeline off the event loop (LangGraph / specialist calls are sync)
    state = await asyncio.to_thread(run_pipeline, user_query)

    # Prefer the synthesizer's narrative report. Fall back to the section
    # dump if the synthesizer was skipped or errored.
    text = state.get("final_report") or format_state_as_markdown(state)

    # Prefer the forecast chart (richer: history + 2 forecasts + ensemble).
    # If no forecast is in this run, fall back to the regular chart agent's output.
    forecast = state.get("forecast_results") or {}
    chart = state.get("chart_results") or {}
    if forecast.get("chart_html"):
        return ChatResponse(
            text=text,
            chart_html=forecast["chart_html"],
            chart_title=f"{forecast.get('series_label', 'Forecast')} — {forecast.get('horizon_days', 0)}-day projection",
            chart_caption=forecast.get("explanation"),
        )
    return ChatResponse(
        text=text,
        chart_html=chart.get("figure_html"),
        chart_title=chart.get("title"),
        chart_caption=chart.get("caption"),
    )
