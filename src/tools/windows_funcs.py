# src/tools/windows_funcs.py
from __future__ import annotations

import pandas as pd
import streamlit as st
from src.tools.util import resolve_column


def rank_within(group_col: str, order_col: str, method: str = "dense", ascending: bool = False):
    """
    Rank rows within each group (SQL-style: row_number/rank/dense_rank-ish).
    - method: 'average' | 'min' | 'max' | 'first' | 'dense' | 'ordinal'
    - ascending: False means highest values get rank 1.
    Returns (DataFrame, error:str|None)
    """
    df = st.session_state.df
    if df is None:
        return None, "No dataset loaded."

    g = resolve_column(group_col, df.columns)
    o = resolve_column(order_col, df.columns)
    if g is None or o is None:
        return None, f"Missing column(s). Available: {list(df.columns)}"

    out = df.copy()
    # Coerce order column to numeric if it looks numeric; otherwise rank on raw
    series = pd.to_numeric(out[o], errors="ignore")
    out["rank"] = out.groupby(g)[series.name if hasattr(series, "name") else o].rank(
        method=method, ascending=ascending
    )
    return out, None


def cumulative_sum(group_col: str, metric: str):
    """
    Cumulative sum of `metric` within each `group_col` partition.
    """
    df = st.session_state.df
    if df is None:
        return None, "No dataset loaded."

    g = resolve_column(group_col, df.columns)
    m = resolve_column(metric, df.columns)
    if g is None or m is None:
        return None, "Missing group or metric column"

    out = df.copy()
    out[m] = pd.to_numeric(out[m], errors="coerce")
    out["cumsum"] = out.groupby(g)[m].cumsum()
    return out, None


def rolling_mean(time_col: str, metric: str, window: int = 3):
    """
    Rolling mean over `metric`, ordered by `time_col`.
    - window: number of rows in the rolling window (not calendar-aware).
    """
    df = st.session_state.df
    if df is None:
        return None, "No dataset loaded."

    t = resolve_column(time_col, df.columns)
    m = resolve_column(metric, df.columns)
    if t is None or m is None:
        return None, "Missing columns"

    out = df.copy()
    # Try to parse time column to datetime if possible
    try:
        out[t] = pd.to_datetime(out[t], errors="coerce")
    except Exception:
        pass
    out = out.sort_values(t)
    out[m] = pd.to_numeric(out[m], errors="coerce")
    out[f"rolling_{window}"] = out[m].rolling(window=window, min_periods=1).mean()
    return out, None


def lag_lead(time_col: str, metric: str, shift: int = 1):
    """
    Create a lag/lead column for `metric`, ordered by `time_col`.
    - shift > 0  → lag (previous rows)
    - shift < 0  → lead (next rows)
    """
    df = st.session_state.df
    if df is None:
        return None, "No dataset loaded."

    t = resolve_column(time_col, df.columns)
    m = resolve_column(metric, df.columns)
    if t is None or m is None:
        return None, "Missing columns"

    out = df.copy()
    try:
        out[t] = pd.to_datetime(out[t], errors="coerce")
    except Exception:
        pass
    out = out.sort_values(t)
    out[m] = pd.to_numeric(out[m], errors="coerce")
    out[f"shift_{shift}"] = out[m].shift(shift)
    return out, None
