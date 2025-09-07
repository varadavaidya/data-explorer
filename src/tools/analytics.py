# src/tools/analytics.py
import pandas as pd
import streamlit as st

def list_columns() -> str:
    if st.session_state.df is None:
        return "No dataset loaded. Please upload a CSV from the sidebar."
    return "Columns (with dtypes):\n" + "\n".join(
        [f"- **{c}**: `{st.session_state.schema[c]}`" for c in st.session_state.df.columns]
    )

def top_k_by_group(k: int, group_col: str, metric: str):
    df = st.session_state.df
    if df is None:
        return None, "No dataset loaded."
    if group_col not in df.columns:
        return None, f"Column `{group_col}` not found. Available: {list(df.columns)}"
    if metric not in df.columns:
        nums = list(df.select_dtypes(include="number").columns)
        return None, f"Metric column `{metric}` not found. Try one of: {nums}"

    mseries = pd.to_numeric(
        df[metric].astype(str).str.replace(r"[^\d\.\-]", "", regex=True),
        errors="coerce",
    )
    tmp = df.copy()
    tmp[metric] = mseries

    grouped = tmp.groupby(group_col)[metric].sum(min_count=1).sort_values(ascending=False).head(k)
    return grouped.reset_index(), None

def list_columns() -> str:
    if st.session_state.df is None:
        return "No dataset loaded. Please upload a CSV from the sidebar."
    return "Columns (with dtypes):\n" + "\n".join(
        [f"- **{c}**: `{st.session_state.schema[c]}`" for c in st.session_state.df.columns]
    )


def top_k_by_group(k: int, group_col: str, metric: str):
    df = st.session_state.df
    if df is None:
        return None, "No dataset loaded."
    if group_col not in df.columns:
        return None, f"Column `{group_col}` not found. Available: {list(df.columns)}"
    if metric not in df.columns:
        nums = list(df.select_dtypes(include="number").columns)
        return None, f"Metric column `{metric}` not found. Try one of: {nums}"

    # robust numeric coercion (handles "₹1,200", "1,200.50", etc.)
    mseries = pd.to_numeric(
        df[metric].astype(str).str.replace(r"[^\d\.\-]", "", regex=True),
        errors="coerce",
    )
    tmp = df.copy()
    tmp[metric] = mseries

    grouped = tmp.groupby(group_col)[metric].sum(min_count=1).sort_values(ascending=False).head(k)
    return grouped.reset_index(), None


def aggregate_by_group(group_col: str, metric: str, agg: str = "sum"):
    """Return a 2-col dataframe grouped by `group_col` with the chosen aggregation over `metric`."""
    df = st.session_state.df
    if df is None:
        return None, "No dataset loaded."
    if group_col not in df.columns:
        return None, f"Column `{group_col}` not found. Available: {list(df.columns)}"
    if agg.lower() not in {"count"} and metric not in df.columns:
        nums = list(df.select_dtypes(include="number").columns)
        return None, f"Metric column `{metric}` not found. Try one of: {nums}"

    tmp = df.copy()

    if agg.lower() == "count":
        # count rows per group (ignores `metric`)
        out = tmp.groupby(group_col).size().reset_index(name="count").sort_values("count", ascending=False)
        return out, None

    # robust numeric coercion for metric (₹1,200 → 1200)
    mseries = pd.to_numeric(
        tmp[metric].astype(str).str.replace(r"[^\d\.\-]", "", regex=True),
        errors="coerce",
    )
    tmp[metric] = mseries

    agg_map = {
        "sum": "sum",
        "mean": "mean",
        "avg": "mean",
        "average": "mean",
        "min": "min",
        "max": "max",
        "median": "median",
    }
    a = agg_map.get(agg.lower(), "sum")
    grouped = getattr(tmp.groupby(group_col)[metric], a)().reset_index()
    grouped = grouped.sort_values(grouped.columns[-1], ascending=False)
    return grouped, None