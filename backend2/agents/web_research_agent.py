"""
Web Research Agent — uses Perplexity Sonar Pro for external market research.

Pipeline:
  1. Build messages: system prompt (web_research_agent.txt) + user sub-task.
  2. Call Perplexity Sonar Pro via the OpenAI-compatible client.
     Perplexity internally: expands the query into sub-queries, searches the
     web, reads pages, synthesizes the answer, and attaches source URLs.
  3. Extract the answer text + citation URLs.
  4. Return a dict to LangGraph state.

Notes:
  - Sonar Pro is OpenAI-compatible — same SDK, just `base_url` swapped.
  - Citations come as a top-level `citations` field on the response (not
    inline in the text). We access via `model_dump()` because the OpenAI
    SDK's typed model doesn't include Perplexity-specific fields.
  - Requires PERPLEXITY_API_KEY in backend/.env.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

# ============================================================
# Setup
# ============================================================

BACKEND_DIR = Path(__file__).parent.parent
PROMPT_PATH = BACKEND_DIR / "prompts" / "web_research_agent.txt"

load_dotenv(BACKEND_DIR / ".env")

PERPLEXITY_BASE_URL = "https://api.perplexity.ai"
MODEL = "sonar-pro"

# Search controls — keep retail-relevant
SEARCH_RECENCY = "year"   # "hour" | "day" | "week" | "month" | "year"
SEARCH_CONTEXT_SIZE = "high"   # "low" | "medium" | "high"

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


# ============================================================
# Helpers
# ============================================================

def load_system_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


# ============================================================
# Main entry point
# ============================================================

def run(task: str) -> dict:
    """Run the web research agent on a natural-language sub-task.

    Returns:
        {
            "answer":             str | None,
            "citations":          list[str],
            "search_queries_run": int,
            "model":              str,
            "error":              str | None,
        }
    """
    # --- Step 1: Build messages (system prompt + user task) ---
    messages = [
        {"role": "system", "content": load_system_prompt()},
        {"role": "user", "content": task},
    ]

    # --- Step 2: Call Perplexity Sonar Pro API (try/except for error capture) ---
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
        return {
            "answer": None,
            "citations": [],
            "search_queries_run": 0,
            "model": MODEL,
            "error": str(e),
        }

    # --- Step 3: Parse response — model_dump for Perplexity-specific fields ---
    # Perplexity returns extra fields (citations, search_results, etc.) that
    # the OpenAI SDK's typed model doesn't know about. Dump to dict to access.
    data = response.model_dump()
    answer = data["choices"][0]["message"]["content"]
    citations = data.get("citations") or []
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
        "How is Zara pricing their denim in 2026?",
        "Top fashion retail trends for 2026",
    ]

    for task in test_tasks:
        print("=" * 70)
        print(f"TASK: {task}")
        print("-" * 70)
        result = run(task)
        print(json.dumps(result, indent=2, default=str))
        print()
