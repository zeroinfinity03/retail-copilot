"""Forecasting agent tests — schema, MAPE math, helper functions.

Pure tests: don't hit the LLM or Prophet/SARIMA fits (too slow for unit
tests). They verify ForecastSpec validation and the small helper utilities.
"""

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from agents.forecasting_agent import (
    ForecastSpec,
    _downsample_table,
    _mape,
    _rows_to_dataframe,
)


# ============================================================
# ForecastSpec schema
# ============================================================

def test_forecastspec_accepts_valid_input():
    spec = ForecastSpec(
        sql_task="Daily Ladieswear revenue",
        horizon_days=90,
        granularity="day",
        country_holidays="SE",
        series_label="Ladieswear daily revenue",
        explanation="90-day projection.",
    )
    assert spec.horizon_days == 90


def test_forecastspec_caps_horizon_at_180():
    with pytest.raises(ValidationError):
        ForecastSpec(
            sql_task="x", horizon_days=365,    # exceeds le=180
            granularity="day", series_label="x", explanation="x",
        )


def test_forecastspec_rejects_horizon_below_7():
    with pytest.raises(ValidationError):
        ForecastSpec(
            sql_task="x", horizon_days=3,      # below ge=7
            granularity="day", series_label="x", explanation="x",
        )


# ============================================================
# MAPE math
# ============================================================

def test_mape_zero_when_predictions_exact():
    actual = np.array([100.0, 200.0, 300.0])
    pred = np.array([100.0, 200.0, 300.0])
    assert _mape(actual, pred) == 0.0


def test_mape_computes_percentage_error():
    actual = np.array([100.0, 100.0])
    pred = np.array([90.0, 110.0])     # 10% off each, both directions
    assert _mape(actual, pred) == pytest.approx(10.0)


def test_mape_skips_zero_actuals():
    """Division by zero would blow up — _mape masks zeros out."""
    actual = np.array([0.0, 100.0])
    pred = np.array([50.0, 110.0])     # only the second contributes
    assert _mape(actual, pred) == pytest.approx(10.0)


# ============================================================
# _rows_to_dataframe
# ============================================================

def test_rows_to_dataframe_picks_date_and_numeric_columns():
    sql_result = {
        "rows": [
            {"week": "2020-01-01", "units": 100},
            {"week": "2020-01-08", "units": 150},
        ],
    }
    df = _rows_to_dataframe(sql_result)
    assert list(df.columns) == ["ds", "y"]
    assert len(df) == 2
    assert df["y"].iloc[0] == 100


def test_rows_to_dataframe_raises_on_empty():
    with pytest.raises(ValueError):
        _rows_to_dataframe({"rows": []})


def test_rows_to_dataframe_raises_when_too_few_columns():
    with pytest.raises(ValueError):
        _rows_to_dataframe({"rows": [{"only_one_col": 1}]})


# ============================================================
# _downsample_table
# ============================================================

def test_downsample_returns_all_rows_when_short():
    """If a series has ≤3k rows, return everything (no down-sampling)."""
    df = pd.DataFrame({"ds": range(5), "yhat": range(5)})
    out = _downsample_table(df, k=10)
    assert len(out) == 5


def test_downsample_picks_first_mid_last_when_long():
    df = pd.DataFrame({"ds": range(100), "yhat": range(100)})
    out = _downsample_table(df, k=10)
    assert len(out) == 30   # 10 + 10 + 10
