"""Chart agent tests — ChartSpec schema + validate_spec column existence checks.

Pure tests: don't hit the LLM. They verify the spec validator catches
hallucinated column names and missing required fields per chart type.
"""

import pytest
from pydantic import ValidationError

from agents.chart_agent import ChartSpec, validate_spec


# ============================================================
# ChartSpec schema
# ============================================================

def test_chartspec_accepts_valid_bar():
    spec = ChartSpec(
        chart_type="bar", x_column="channel", y_column="revenue",
        title="Revenue by channel", caption="Online led.",
    )
    assert spec.chart_type == "bar"


def test_chartspec_rejects_unknown_chart_type():
    with pytest.raises(ValidationError):
        ChartSpec(
            chart_type="bubble_chart_3000",   # not in the Literal
            x_column="x", title="t", caption="c",
        )


def test_chartspec_requires_title_and_caption():
    with pytest.raises(ValidationError):
        ChartSpec(chart_type="bar", x_column="x")   # missing title + caption


# ============================================================
# validate_spec — column existence
# ============================================================

def test_validate_spec_catches_hallucinated_x_column():
    spec = ChartSpec(
        chart_type="bar", x_column="not_in_data", y_column="revenue",
        title="t", caption="c",
    )
    err = validate_spec(spec, columns=["channel", "revenue"])
    assert err is not None
    assert "not_in_data" in err


def test_validate_spec_allows_existing_columns():
    spec = ChartSpec(
        chart_type="bar", x_column="channel", y_column="revenue",
        title="t", caption="c",
    )
    assert validate_spec(spec, columns=["channel", "revenue"]) is None


# ============================================================
# validate_spec — required fields per chart type
# ============================================================

def test_treemap_requires_path_columns_and_values_column():
    spec = ChartSpec(
        chart_type="treemap", x_column="index_group_name",
        title="t", caption="c",
    )   # missing path_columns AND values_column
    err = validate_spec(spec, columns=["index_group_name", "product_group_name", "revenue"])
    assert err is not None
    assert "path_columns" in err
    assert "values_column" in err


def test_scatter_3d_requires_z_column():
    spec = ChartSpec(
        chart_type="scatter_3d", x_column="a", y_column="b",
        title="t", caption="c",
    )   # missing z_column
    err = validate_spec(spec, columns=["a", "b", "c"])
    assert err is not None
    assert "z_column" in err


def test_histogram_only_needs_x():
    spec = ChartSpec(
        chart_type="histogram", x_column="age",
        title="t", caption="c",
    )
    assert validate_spec(spec, columns=["age"]) is None
