# src/tools/viz.py
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import streamlit as st

ARTIFACT_DIR = Path("artifacts/plots")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

def plot_group_sum(x_col: str, y_col: str, hue: str | None = None, fname_prefix: str = "chart"):
    df = st.session_state.df
    if df is None:
        return None, "No dataset loaded."
    for col in [c for c in [x_col, y_col, hue] if c]:
        if col not in df.columns:
            return None, f"Column `{col}` not in dataset."

    fig, ax = plt.subplots(figsize=(9, 5))
    if hue:
        pivot = df.pivot_table(index=x_col, columns=hue, values=y_col, aggfunc="sum")
        pivot.plot(kind="bar", ax=ax)
    else:
        df.groupby(x_col)[y_col].sum().plot(kind="bar", ax=ax)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    fig.tight_layout()

    out = ARTIFACT_DIR / f"{fname_prefix}_{x_col}_{y_col}{'_'+hue if hue else ''}.png"
    fig.savefig(out)
    plt.close(fig)
    return str(out), None
