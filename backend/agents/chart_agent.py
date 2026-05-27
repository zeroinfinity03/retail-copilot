"""
Chart Agent — converts SQL agent rows into a Plotly figure.

Pipeline:
  1. Read columns + sample rows + user question from inputs.
  2. LLM call: returns a structured chart spec
     {chart_type, x_column, y_column, color_column, title, caption}.
  3. Validate spec — columns must actually exist in the data.
  4. Python builds the Plotly figure from the spec.
  5. Return dict with figure_html + caption to LangGraph state.

Notes:
  - Uses Plotly Express (one-liner per chart type).
  - figure_html is a self-contained HTML string the synthesizer can embed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import pandas as pd
import plotly.express as px
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

# ============================================================
# Setup
# ============================================================

BACKEND_DIR = Path(__file__).parent.parent
PROMPT_PATH = BACKEND_DIR / "prompts" / "chart_agent.txt"

load_dotenv(BACKEND_DIR / ".env")

MODEL = "gpt-5-mini"
MAX_RETRIES = 1

_client: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


# ============================================================
# Structured output schema
# ============================================================

ChartType = Literal[
    "bar",
    "horizontal_bar",
    "line",
    "area",
    "scatter",
    "scatter_3d",
    "density_heatmap",
    "pie",
    "box",
    "histogram",
    "treemap",
    "funnel",
]


class ChartSpec(BaseModel):
    chart_type: ChartType
    x_column: str = Field(description="Exact column name to use on the x-axis (for histogram, the column to count).")
    y_column: Optional[str] = Field(
        default=None,
        description="Exact column name for y-axis. Null for histogram (only x needed). Null for treemap (use values_column instead).",
    )
    color_column: Optional[str] = Field(
        default=None,
        description="Optional grouping column; null if no grouping.",
    )
    path_columns: Optional[list[str]] = Field(
        default=None,
        description="Hierarchy levels for treemap (e.g., ['index_group_name','product_group_name']). Null for other chart types.",
    )
    values_column: Optional[str] = Field(
        default=None,
        description="Size/value column for treemap. Null for other chart types.",
    )
    z_column: Optional[str] = Field(
        default=None,
        description=(
            "Exact column name for the z-axis (third numeric dimension). "
            "Required for scatter_3d. Optional for density_heatmap (when provided, "
            "the heatmap aggregates z by x,y bins instead of counting). Null otherwise."
        ),
    )
    is_time_series: bool = Field(
        default=False,
        description="True if x-axis is a date/time column AND chart_type is line/area/bar. Adds a range slider.",
    )
    title: str
    caption: str = Field(description="One-line takeaway under the chart.")


# ============================================================
# Helpers
# ============================================================

def load_system_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def build_user_message(
    columns: list[str],
    rows: list[dict],
    user_question: str,
) -> str:
    sample = rows[:3]
    return (
        f"User question:\n{user_question}\n\n"
        f"SQL result columns: {columns}\n\n"
        f"Sample rows (first 3):\n{sample}"
    )


def get_chart_spec(
    columns: list[str],
    rows: list[dict],
    user_question: str,
    error_context: Optional[str] = None,
) -> ChartSpec:
    user_msg = build_user_message(columns, rows, user_question)
    if error_context:
        user_msg += (
            f"\n\nPrevious spec failed with this error — correct it:\n{error_context}"
        )

    response = _get_client().chat.completions.parse(
        model=MODEL,
        messages=[
            {"role": "system", "content": load_system_prompt()},
            {"role": "user", "content": user_msg},
        ],
        response_format=ChartSpec,
    )
    return response.choices[0].message.parsed


def validate_spec(spec: ChartSpec, columns: list[str]) -> Optional[str]:
    """Return an error message if the spec references columns that don't exist
    or omits required fields for the chosen chart type."""
    bad: list[str] = []

    # Single-column references
    for label, col in [
        ("x_column", spec.x_column),
        ("y_column", spec.y_column),
        ("color_column", spec.color_column),
        ("values_column", spec.values_column),
        ("z_column", spec.z_column),
    ]:
        if col is not None and col not in columns:
            bad.append(f"{label}={col!r} not in columns {columns}")

    # path_columns is a list — check every entry
    if spec.path_columns:
        for col in spec.path_columns:
            if col not in columns:
                bad.append(f"path_columns entry {col!r} not in columns {columns}")

    # Required-field checks per chart type
    if spec.chart_type in {"bar", "horizontal_bar", "line", "area", "scatter", "box", "funnel"}:
        if not spec.y_column:
            bad.append(f"{spec.chart_type} requires y_column")
    if spec.chart_type == "pie" and not spec.y_column:
        bad.append("pie requires y_column (the values)")
    if spec.chart_type == "treemap":
        if not spec.path_columns:
            bad.append("treemap requires path_columns")
        if not spec.values_column:
            bad.append("treemap requires values_column")
    if spec.chart_type == "scatter_3d":
        if not spec.y_column:
            bad.append("scatter_3d requires y_column")
        if not spec.z_column:
            bad.append("scatter_3d requires z_column")
    if spec.chart_type == "density_heatmap":
        if not spec.y_column:
            bad.append("density_heatmap requires y_column")

    return "; ".join(bad) if bad else None


def render_figure(spec: ChartSpec, rows: list[dict]) -> dict:
    """Build the Plotly figure from the spec. Returns
    {figure_html, figure_json, error}."""
    if not rows:
        return {
            "figure_html": None,
            "figure_json": None,
            "error": "No rows to plot.",
        }

    df = pd.DataFrame(rows)

    common = dict(
        data_frame=df,
        x=spec.x_column,
        y=spec.y_column,
        title=spec.title,
    )
    if spec.color_column:
        common["color"] = spec.color_column

    try:
        if spec.chart_type == "bar":
            fig = px.bar(**common)
            fig.update_yaxes(rangemode="tozero")

        elif spec.chart_type == "horizontal_bar":
            fig = px.bar(orientation="h", **common)
            fig.update_xaxes(rangemode="tozero")
            fig.update_yaxes(categoryorder="total ascending")

        elif spec.chart_type == "line":
            fig = px.line(**common)

        elif spec.chart_type == "area":
            fig = px.area(**common)

        elif spec.chart_type == "scatter":
            fig = px.scatter(**common)

        elif spec.chart_type == "scatter_3d":
            scatter3d_kwargs = dict(
                data_frame=df,
                x=spec.x_column,
                y=spec.y_column,
                z=spec.z_column,
                title=spec.title,
            )
            if spec.color_column:
                scatter3d_kwargs["color"] = spec.color_column
            fig = px.scatter_3d(**scatter3d_kwargs)

        elif spec.chart_type == "density_heatmap":
            heatmap_kwargs = dict(
                data_frame=df,
                x=spec.x_column,
                y=spec.y_column,
                title=spec.title,
            )
            if spec.z_column:
                heatmap_kwargs["z"] = spec.z_column
                heatmap_kwargs["histfunc"] = "avg"
            fig = px.density_heatmap(**heatmap_kwargs)

        elif spec.chart_type == "box":
            fig = px.box(**common)

        elif spec.chart_type == "histogram":
            # Histogram needs only x — drop y from common
            hist_kwargs = {"data_frame": df, "x": spec.x_column, "title": spec.title}
            if spec.color_column:
                hist_kwargs["color"] = spec.color_column
            fig = px.histogram(**hist_kwargs)

        elif spec.chart_type == "pie":
            fig = px.pie(
                data_frame=df,
                names=spec.x_column,
                values=spec.y_column,
                title=spec.title,
            )

        elif spec.chart_type == "treemap":
            if not spec.path_columns or not spec.values_column:
                return {
                    "figure_html": None,
                    "figure_json": None,
                    "error": "treemap requires path_columns and values_column",
                }
            fig = px.treemap(
                data_frame=df,
                path=[px.Constant("All")] + spec.path_columns,
                values=spec.values_column,
                title=spec.title,
            )

        elif spec.chart_type == "funnel":
            fig = px.funnel(**common)

        else:
            return {
                "figure_html": None,
                "figure_json": None,
                "error": f"Unknown chart type: {spec.chart_type}",
            }

        # Range slider — only meaningful for line / area / bar on time-axis
        if spec.is_time_series and spec.chart_type in {"line", "area", "bar"}:
            fig.update_xaxes(rangeslider_visible=True)

    except Exception as e:
        return {"figure_html": None, "figure_json": None, "error": str(e)}

    return {
        "figure_html": fig.to_html(include_plotlyjs="cdn", full_html=False),
        "figure_json": fig.to_json(),
        "error": None,
    }


# ============================================================
# Main entry point
# ============================================================

def run(
    columns: list[str],
    rows: list[dict],
    user_question: str,
) -> dict:
    """Generate a Plotly chart from SQL result data.

    Returns:
        {
            "chart_type":  str | None,
            "title":       str | None,
            "caption":     str | None,
            "figure_html": str | None,
            "figure_json": str | None,
            "error":       str | None,
        }
    """
    # --- Step 1: Empty-rows early return ---
    if not rows:
        return {
            "chart_type": None,
            "title": None,
            "caption": None,
            "figure_html": None,
            "figure_json": None,
            "error": "No data to plot — upstream SQL agent returned no rows.",
        }

    # --- Step 2: LLM call → ChartSpec ---
    spec = get_chart_spec(columns, rows, user_question)

    # --- Step 3: Validate spec (column existence + required fields) + retry once on error ---
    err = validate_spec(spec, columns)
    if err and MAX_RETRIES > 0:
        spec = get_chart_spec(columns, rows, user_question, error_context=err)
        err = validate_spec(spec, columns)
    if err:
        return {
            "chart_type": spec.chart_type,
            "title": spec.title,
            "caption": spec.caption,
            "figure_html": None,
            "figure_json": None,
            "error": err,
        }

    # --- Step 4: Render Plotly figure via render_figure() ---
    render = render_figure(spec, rows)

    # --- Step 5: Pack result and return to supervisor ---
    return {
        "chart_type": spec.chart_type,
        "title": spec.title,
        "caption": spec.caption,
        "figure_html": render["figure_html"],
        "figure_json": render["figure_json"],
        "error": render["error"],
    }


# ============================================================
# Smoke test
# ============================================================

if __name__ == "__main__":
    import json

    # Simulate SQL agent output: top 5 articles by revenue
    test_columns = ["prod_name", "revenue"]
    test_rows = [
        {"prod_name": "Pluto RW slacks (1)", "revenue": 553.98},
        {"prod_name": "Lilly long shacket",   "revenue": 284.01},
        {"prod_name": "Mariette Blazer",      "revenue": 260.31},
        {"prod_name": "Lucy blouse",          "revenue": 258.08},
        {"prod_name": "Primo slacks",         "revenue": 254.26},
    ]
    test_question = "Show me the top 5 best-selling Ladieswear articles last quarter."

    result = run(test_columns, test_rows, test_question)
    # Strip out the huge HTML/JSON for readability
    brief = {**result,
             "figure_html": f"<{len(result['figure_html'] or '')} chars>",
             "figure_json": f"<{len(result['figure_json'] or '')} chars>"}
    print(json.dumps(brief, indent=2, default=str))
