"""SQL agent tests — keyword blocklist, output schema, sandbox execution.

Pure tests: don't hit the LLM. They verify the safety layers and the
SQLOutput schema.
"""

import pytest
from pydantic import ValidationError

from agents.sql_agent import (
    SQLOutput,
    contains_banned_keyword,
    execute_sql,
)


# ============================================================
# Keyword blocklist
# ============================================================

def test_blocks_drop_statement():
    assert contains_banned_keyword("DROP TABLE customers") == "DROP"


def test_blocks_lowercase_delete():
    assert contains_banned_keyword("delete from articles") == "DELETE"


def test_allows_pure_select():
    assert contains_banned_keyword("SELECT * FROM customers LIMIT 10") is None


def test_allows_keyword_as_substring_in_column_name():
    """UPDATED_AT contains 'UPDATE' as substring but isn't an UPDATE statement.
    The padded-matching algorithm should let it through."""
    assert contains_banned_keyword("SELECT updated_at FROM transactions") is None


def test_allows_created_by_column():
    assert contains_banned_keyword("SELECT created_by FROM customers") is None


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

def test_execute_sql_blocks_banned_keyword():
    """execute_sql must reject a DROP statement BEFORE hitting DuckDB."""
    result = execute_sql("DROP TABLE customers")
    assert result["rows"] == []
    assert result["error"] is not None
    assert "Blocked" in result["error"]
    assert "DROP" in result["error"]
