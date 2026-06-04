"""SQL agent tests — read-only allowlist, output schema, sandbox execution.

Pure tests: don't hit the LLM. They verify the safety layers and the
SQLOutput schema.
"""

import pytest
from pydantic import ValidationError

from agents.sql_agent import (
    SQLOutput,
    read_only_guard,
    execute_sql,
)


# ============================================================
# Read-only allowlist (only a single SELECT / WITH query passes)
# ============================================================

def test_allows_pure_select():
    assert read_only_guard("SELECT * FROM customers LIMIT 10") is None


def test_allows_with_cte():
    assert read_only_guard("WITH x AS (SELECT 1 AS n) SELECT n FROM x") is None


def test_allows_trailing_semicolon():
    assert read_only_guard("SELECT 1;") is None


def test_allows_keyword_as_substring_in_column_name():
    """Columns like updated_at / created_by are fine: the allowlist only
    looks at the leading keyword, not substrings."""
    assert read_only_guard("SELECT updated_at FROM transactions") is None
    assert read_only_guard("SELECT created_by FROM customers") is None


def test_blocks_drop_statement():
    assert read_only_guard("DROP TABLE customers") is not None


def test_blocks_lowercase_delete():
    assert read_only_guard("delete from articles") is not None


def test_blocks_side_effect_statements():
    """COPY / PRAGMA / INSTALL etc. are not SELECT/WITH, so they are rejected."""
    assert read_only_guard("COPY (SELECT * FROM articles) TO 'out.csv'") is not None
    assert read_only_guard("PRAGMA database_list") is not None
    assert read_only_guard("INSTALL httpfs") is not None


def test_blocks_chained_second_statement():
    """A second statement smuggled in via ';' is rejected even though it
    starts with SELECT."""
    assert read_only_guard("SELECT 1; DROP TABLE customers") is not None


def test_blocks_prefix_lookalike():
    """A word that merely starts with SELECT (e.g. SELECTED) is not a real
    SELECT statement: the first-token check rejects it."""
    assert read_only_guard("SELECTED FROM x") is not None


def test_blocks_empty_query():
    assert read_only_guard("   ") is not None
    assert read_only_guard(";") is not None


# ============================================================
# SQLOutput schema
# ============================================================

def test_sqloutput_accepts_valid_select():
    out = SQLOutput(sql="SELECT COUNT(*) FROM customers", explanation="row count")
    assert out.sql.startswith("SELECT")
    assert out.explanation == "row count"


def test_sqloutput_accepts_null_sql_for_out_of_scope():
    """When the task is out of scope, the LLM is allowed to return sql=None."""
    out = SQLOutput(sql=None, explanation="No inventory data in warehouse")
    assert out.sql is None


def test_sqloutput_requires_explanation():
    with pytest.raises(ValidationError):
        SQLOutput(sql="SELECT 1")   # missing explanation


# ============================================================
# execute_sql sandbox
# ============================================================

def test_execute_sql_blocks_non_select():
    """execute_sql must reject a DROP statement BEFORE hitting DuckDB."""
    result = execute_sql("DROP TABLE customers")
    assert result["rows"] == []
    assert result["error"] is not None
    assert "Blocked" in result["error"]
