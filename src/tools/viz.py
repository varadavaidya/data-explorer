# src/tools/viz.py
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import streamlit as st
from src.tools.util import resolve_column


ARTIFACT_DIR = Path("artifacts/plots")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

def plot_group_sum(x_col: str, y_col: str, hue: str | None = None, fname_prefix: str = "chart"):
    df = st.session_state.df
    if df is None:
        return None, "No dataset loaded."

    x = resolve_column(x_col, df.columns) if x_col else None
    y = resolve_column(y_col, df.columns) if y_col else None
    h = resolve_column(hue, df.columns) if hue else None

    for col_in, name in [(x, x_col), (y, y_col), (h, hue)]:
        if col_in is None and name:
            return None, f"Column `{name}` not in dataset. Available: {list(df.columns)}"

    fig, ax = plt.subplots(figsize=(9, 5))
    if h:
        pivot = df.pivot_table(index=x, columns=h, values=y, aggfunc="sum")
        pivot.plot(kind="bar", ax=ax)
    else:
        df.groupby(x)[y].sum().plot(kind="bar", ax=ax)
    ax.set_xlabel(x or ""); ax.set_ylabel(y or "")
    fig.tight_layout()

    out = ARTIFACT_DIR / f"{fname_prefix}_{x}_{y}{'_'+h if h else ''}.png"
    fig.savefig(out); plt.close(fig)
    return str(out), None
