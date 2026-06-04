"""
SQL Agent — generates DuckDB SQL for retail analytics sub-tasks.

Pipeline:
  1. LLM call: (system prompt + sub-task) -> structured JSON {sql, explanation}
  2. Python: allowlist (only a single SELECT/WITH) + read-only DuckDB execution
  3. On SQL error: one retry, feeding the error message back to the LLM
  4. Return: {sql, explanation, columns, rows, row_count, error}

Notes:
  - The LLM never touches the database. Python is the only executor.
  - Schema lives in backend/prompts/sql_agent.txt (kept in the system prompt).
  - Requires provider credentials in backend/.env.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import duckdb
from pydantic import BaseModel, Field

from agents.llm import structured

# ============================================================
# Setup
# ============================================================

BACKEND_DIR = Path(__file__).parent.parent
PROMPT_PATH = BACKEND_DIR / "prompts" / "sql_agent.txt"
DB_PATH = BACKEND_DIR / "data" / "db" / "hm.duckdb"

# Sandbox configuration
# Allowlist (default-deny): only a single SELECT/WITH read query is permitted.
# Everything else (DROP, DELETE, INSERT, UPDATE, ALTER, TRUNCATE, CREATE,
# ATTACH, plus COPY, EXPORT, INSTALL, LOAD, PRAGMA, SET, ...) is rejected
# before it reaches the engine.
ALLOWED_STARTS = ("SELECT", "WITH")
MAX_ROWS_RETURNED = 1000
MAX_RETRIES = 1


# ============================================================
# Structured output schema
# ============================================================

class SQLOutput(BaseModel):
    """What the LLM must return."""
    sql: Optional[str] = Field(
        description="DuckDB read query starting with SELECT or WITH, or null if the task is out of scope."
    )
    explanation: str = Field(
        description="One-sentence plain English summary of the query intent (20 words max)."
    )


# ============================================================
# Helpers
# ============================================================

def load_system_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def read_only_guard(sql: str) -> Optional[str]:
    """Allowlist gate: permit only a single SELECT or WITH read query.

    Default-deny. Returns an error message if the query is not allowed, else
    None. This rejects writes (DROP/DELETE/INSERT/UPDATE/ALTER/TRUNCATE/
    CREATE/ATTACH) and side-effect statements (COPY/EXPORT/INSTALL/LOAD/
    PRAGMA/SET/...) up front. read_only=True is the connection-level backstop.
    """
    # Drop a single trailing ';' and surrounding whitespace.
    cleaned = sql.strip().rstrip(";").strip()

    # A leftover ';' means a second statement was chained in (e.g. injection).
    # (A ';' inside a string literal would also trip this, but the analytics
    #  queries here never put one there, so it is a safe trade-off.)
    if ";" in cleaned:
        return "only a single statement is allowed (no ';' chaining)"

    # Must be a read query: SELECT or WITH (CTE). startswith accepts a tuple.
    if not cleaned.upper().startswith(ALLOWED_STARTS):
        return "only SELECT or WITH (read-only) queries are allowed"

    return None


def generate_sql(task: str, error_context: Optional[str] = None) -> SQLOutput:
    """One LLM call: NL task -> {sql, explanation}.

    If error_context is provided, it is appended to the user message so the LLM
    can correct a previous failed attempt.
    """
    system_prompt = load_system_prompt()
    user_message = task
    if error_context:
        user_message = (
            f"{task}\n\n"
            f"Previous SQL attempt failed with this error — correct it:\n"
            f"{error_context}"
        )

    return structured(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        schema=SQLOutput,
    )


def execute_sql(sql: str) -> dict:
    """Run SQL in a sandboxed read-only DuckDB connection.

    Layers of safety:
      1. Allowlist (only a single SELECT/WITH read query passes; default-deny)
      2. read_only=True at connection level (engine-level backstop)
      3. Row count cap on the returned data (MAX_ROWS_RETURNED)

    Note: DuckDB lacks a built-in statement timeout. For v1 we rely on
    the read-only mode + row cap. If long-query control becomes needed,
    wrap execution in a thread + threading.Timer that calls con.interrupt().
    """
    violation = read_only_guard(sql)
    if violation:
        return {
            "columns": [],
            "rows": [],
            "row_count": 0,
            "error": f"Blocked: {violation} (read-only agent).",
        }

    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        result_df = con.execute(sql).fetchdf()
        return {
            "columns": list(result_df.columns),
            "rows": result_df.head(MAX_ROWS_RETURNED).to_dict(orient="records"),
            "row_count": int(len(result_df)),
            "error": None,
        }
    except Exception as e:
        return {
            "columns": [],
            "rows": [],
            "row_count": 0,
            "error": str(e),
        }
    finally:
        con.close()


# ============================================================
# Main entry point
# ============================================================

def run(task: str) -> dict:
    """Run the SQL agent on a natural-language sub-task.

    Returns:
        {
            "sql":         str | None,
            "explanation": str,
            "columns":     list[str],
            "rows":        list[dict],
            "row_count":   int,
            "error":       str | None,
        }
    """
    # --- Step 1: Generate SQL via LLM ---
    llm_output = generate_sql(task)

    # --- Step 2: LLM declined (out-of-scope) → early return ---
    if llm_output.sql is None:
        return {
            "sql": None,
            "explanation": llm_output.explanation,
            "columns": [],
            "rows": [],
            "row_count": 0,
            "error": None,
        }

    # --- Step 3: Safety check + DuckDB execute (inside execute_sql) ---
    exec_result = execute_sql(llm_output.sql)

    # --- Step 4: Retry once if execution failed ---
    if exec_result["error"] and MAX_RETRIES > 0:
        llm_output_retry = generate_sql(task, error_context=exec_result["error"])
        if llm_output_retry.sql:
            exec_result = execute_sql(llm_output_retry.sql)
            llm_output = llm_output_retry  # keep the corrected SQL in the return

    # --- Step 5: Pack result and return to supervisor ---
    return {
        "sql": llm_output.sql,
        "explanation": llm_output.explanation,
        **exec_result,
    }


# ============================================================
# Smoke test
# ============================================================

if __name__ == "__main__":
    import json

    test_tasks = [
        "Top 5 best-selling articles in Ladieswear last quarter by revenue",
        "How many active club members are there?",
        "What is the current inventory level for SKU 0108775015?",  # fallback case
    ]

    for task in test_tasks:
        print("=" * 70)
        print(f"TASK: {task}")
        print("-" * 70)
        result = run(task)
        # Strip the heavy 'rows' field for readability, but show row_count
        result_brief = {**result, "rows": f"<{len(result['rows'])} rows>"}
        print(json.dumps(result_brief, default=str, indent=2))
        print()
