"""
Web Research Agent — decides what a question needs from the web, then uses
Perplexity Sonar Pro to research it.

Pipeline:
  1. Extraction (gateway LLM): turn the full user question into ONE focused
     web-search query, or null if nothing needs the web (H&M internal data
     or a pure forecast).
  2. If null: decline (no Perplexity call, no off-target research).
  3. Else: send the FOCUSED query to Perplexity Sonar Pro. Perplexity
     expands sub-queries, searches, reads pages, synthesizes, and attaches
     source URLs.
  4. Return a dict to the supervisor.

Notes:
  - Extraction goes through the llm.py gateway (deepseek-v4-flash).
  - Perplexity is OpenAI-compatible; same SDK, base_url swapped.
  - Citations come as a top-level `citations` field on the response, not
    inline. We access via model_dump() because the OpenAI SDK's typed model
    doesn't include Perplexity-specific fields.
  - Requires PERPLEXITY_API_KEY in backend/.env.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

from agents.llm import structured

# ============================================================
# Setup
# ============================================================

BACKEND_DIR = Path(__file__).parent.parent
PROMPT_PATH = BACKEND_DIR / "prompts" / "web_research_agent.txt"

load_dotenv(BACKEND_DIR / ".env")

PERPLEXITY_BASE_URL = "https://api.perplexity.ai"
MODEL = "sonar-pro"

# Search controls — keep retail-relevant
SEARCH_RECENCY = "year"        # "hour" | "day" | "week" | "month" | "year"
SEARCH_CONTEXT_SIZE = "high"   # "low" | "medium" | "high"

# Perplexity answer style. The "what to search" decision is already made by
# the extraction step; this only governs how Perplexity writes the answer.
PERPLEXITY_SYSTEM = (
    "You are H&M's market research analyst. Answer the query using current "
    "web results. Lead with the bottom-line answer in the first sentence. "
    "Use specific numbers when the sources have them, and compare competitors "
    "when relevant. Keep it tight (200-400 words). Write clean prose with no "
    "inline citation markers like [1], and no em dashes."
)

_client: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    """Lazy singleton pointed at Perplexity."""
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=os.getenv("PERPLEXITY_API_KEY"),
            base_url=PERPLEXITY_BASE_URL,
        )
    return _client


def load_system_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


# ============================================================
# Step 1 — extraction schema + call
# ============================================================

class WebQuery(BaseModel):
    """What the extraction LLM must return."""
    search_query: Optional[str] = Field(
        default=None,
        description=(
            "ONE focused web-search query for the part of the question that "
            "needs external web information, or null if nothing does (H&M "
            "internal data or a pure forecast)."
        ),
    )


def make_web_query(task: str) -> WebQuery:
    """One gateway LLM call: full user question -> focused web query (or null)."""
    return structured(
        messages=[
            {"role": "system", "content": load_system_prompt()},
            {"role": "user", "content": task},
        ],
        schema=WebQuery,
    )


def _empty_result(error: Optional[str] = None) -> dict:
    """Fixed-shape result for the decline and error paths."""
    return {
        "answer": None,
        "citations": [],
        "search_queries_run": 0,
        "model": MODEL,
        "error": error,
    }


# ============================================================
# Main entry point
# ============================================================

def run(task: str) -> dict:
    """Run the web research agent on the full user question.

    Returns:
        {
            "answer":             str | None,
            "citations":          list[str],
            "search_queries_run": int,
            "model":              str,
            "error":              str | None,
        }
    """
    # --- Step 1: Extract a focused web-search query (or decline) ---
    spec = make_web_query(task)
    if not spec.search_query:
        # Nothing here needs the web (internal data or a pure forecast).
        return _empty_result()

    # --- Step 2: Call Perplexity Sonar Pro on the FOCUSED query ---
    messages = [
        {"role": "system", "content": PERPLEXITY_SYSTEM},
        {"role": "user", "content": spec.search_query},
    ]
    try:
        response = _get_client().chat.completions.create(
            model=MODEL,
            messages=messages,
            extra_body={
                "web_search_options": {"search_context_size": SEARCH_CONTEXT_SIZE},
                "search_recency_filter": SEARCH_RECENCY,
            },
        )
    except Exception as e:
        return _empty_result(error=str(e))

    # --- Step 3: Parse response — model_dump for Perplexity-specific fields ---
    data = response.model_dump()
    answer = data["choices"][0]["message"]["content"]
    citations = data.get("citations") or []
    if not citations:
        # Some Perplexity responses populate only `search_results` (a list of
        # {title, url, ...}) and leave the top-level `citations` empty. Fall
        # back to those URLs so the sources don't silently disappear.
        citations = [r.get("url") for r in (data.get("search_results") or []) if r.get("url")]
    num_queries = data.get("usage", {}).get("num_search_queries", 0)

    # --- Step 4: Return result dict to supervisor ---
    return {
        "answer": answer,
        "citations": citations,
        "search_queries_run": num_queries,
        "model": MODEL,
        "error": None,
    }


# ============================================================
# Smoke test
# ============================================================

if __name__ == "__main__":
    import json

    test_tasks = [
        "How is Zara pricing their denim in 2026?",                   # market -> query
        "Forecast knitwear revenue for the next quarter.",            # forecast -> null
        "What were our top 5 selling articles last quarter?",         # internal -> null
        "Which fashion retailers reported over 20M profit in 2025?",  # general -> query
    ]
    for task in test_tasks:
        print("=" * 70)
        print(f"TASK: {task}")
        result = run(task)
        brief = {**result, "answer": (result["answer"] or "")[:120]}
        print(json.dumps(brief, indent=2, default=str))
        print()
