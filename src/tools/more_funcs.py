# src/tools/more_funcs.py
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

ARTIFACT_DIR = Path("artifacts/plots")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

def describe_numeric():
    df = st.session_state.df
    if df is None:
        return None, "No dataset loaded."
    desc = df.select_dtypes(include="number").describe().T.reset_index().rename(columns={"index": "column"})
    return desc, None

def missing_report():
    df = st.session_state.df
    if df is None:
        return None, "No dataset loaded."
    miss = df.isna().sum()
    total = len(df)
    out = pd.DataFrame({
        "column": miss.index,
        "missing": miss.values,
        "pct_missing": (miss.values / max(total, 1)) * 100.0
    }).sort_values("pct_missing", ascending=False).reset_index(drop=True)
    return out, None

def value_counts(col: str, top: int = 20):
    df = st.session_state.df
    if df is None:
        return None, "No dataset loaded."
    if col not in df.columns:
        return None, f"Column `{col}` not found. Available: {list(df.columns)}"
    vc = df[col].astype(str).value_counts(dropna=False).head(top).reset_index()
    vc.columns = [col, "count"]
    return vc, None

def correlation_matrix(save_name: str = "corr_matrix"):
    df = st.session_state.df
    if df is None:
        return None, None, "No dataset loaded."
    num = df.select_dtypes(include="number")
    if num.empty:
        return None, None, "No numeric columns to correlate."
    corr = num.corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    # Matplotlib heatmap
    cax = ax.imshow(corr.values, interpolation="nearest")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticks(range(len(corr.columns)))
    ax.set_yticklabels(corr.columns)
    fig.colorbar(cax)
    fig.tight_layout()
    out = ARTIFACT_DIR / f"{save_name}.png"
    fig.savefig(out)
    plt.close(fig)
    return corr.reset_index().rename(columns={"index": "column"}), str(out), None

def histogram(col: str, bins: int = 30, save_name: str = "hist"):
    df = st.session_state.df
    if df is None:
        return None, "No dataset loaded.", None
    if col not in df.columns:
        return None, f"Column `{col}` not found.", None
    series = pd.to_numeric(df[col], errors="coerce")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(series.dropna().values, bins=bins)
    ax.set_xlabel(col); ax.set_ylabel("frequency")
    fig.tight_layout()
    out = ARTIFACT_DIR / f"{save_name}_{col}.png"
    fig.savefig(out)
    plt.close(fig)
    return pd.DataFrame({col: series}), None, str(out)

def boxplot(y: str, by: str | None = None, save_name: str = "box"):
    df = st.session_state.df
    if df is None:
        return None, "No dataset loaded.", None
    if y not in df.columns:
        return None, f"Column `{y}` not found.", None
    fig, ax = plt.subplots(figsize=(7, 4))
    if by and by in df.columns:
        groups = [g.dropna().values for _, g in df.groupby(by)[y]]
        ax.boxplot(groups, labels=[str(k) for k in df.groupby(by).groups.keys()], vert=True)
        ax.set_xlabel(by)
    else:
        ax.boxplot(pd.to_numeric(df[y], errors="coerce").dropna().values, vert=True)
    ax.set_ylabel(y)
    fig.tight_layout()
    out = ARTIFACT_DIR / f"{save_name}_{y}{'_'+by if by else ''}.png"
    fig.savefig(out)
    plt.close(fig)
    # return small summary table (median, q1, q3)
    return None, None, str(out)

def pivot_table(index: str, columns: str, values: str, agg: str = "sum"):
    df = st.session_state.df
    if df is None:
        return None, "No dataset loaded."
    for c in [index, columns, values]:
        if c not in df.columns:
            return None, f"Column `{c}` not found."
    # coerce values numeric if possible
    vals = pd.to_numeric(df[values].astype(str).str.replace(r"[^\d\.\-]", "", regex=True), errors="coerce")
    tmp = df.copy()
    tmp[values] = vals
    agg_map = {"sum":"sum", "mean":"mean", "avg":"mean", "average":"mean", "min":"min", "max":"max", "median":"median", "count":"count"}
    a = agg_map.get(agg.lower(), "sum")
    pt = pd.pivot_table(tmp, index=index, columns=columns, values=values,
                        aggfunc=(len if a=="count" else a), fill_value=0).reset_index()
    # flatten columns if multi-index
    if isinstance(pt.columns, pd.MultiIndex):
        pt.columns = ["_".join([str(c) for c in tup if c!=""]) for tup in pt.columns.values]
    return pt, None

def outliers_zscore(col: str, threshold: float = 3.0):
    df = st.session_state.df
    if df is None:
        return None, "No dataset loaded."
    if col not in df.columns:
        return None, f"Column `{col}` not found."
    series = pd.to_numeric(df[col], errors="coerce")
    mu, sigma = series.mean(), series.std(ddof=0)
    if np.isnan(mu) or np.isnan(sigma) or sigma == 0:
        return None, "Cannot compute z-scores for this column."
    z = (series - mu) / sigma
    mask = z.abs() >= threshold
    out = df.loc[mask].copy()
    out["zscore"] = z[mask]
    return out, None
