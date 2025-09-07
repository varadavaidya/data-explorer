# streamlit_app.py — Data Explorer (Streamlit)
from __future__ import annotations
import uuid
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# --- DB init & session id
from src.db import init_db, db

init_db()  # ensure artifacts/app.db & tables exist

st.set_page_config(page_title="Data Explorer", layout="wide")

# session id per browser tab
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
db.ensure_session(st.session_state.session_id, title="Streamlit Session")

# --- Session state defaults (kept in memory during tab lifetime)
if "messages" not in st.session_state:
    st.session_state.messages = []  # [{"role","type",...}]
if "df" not in st.session_state:
    st.session_state.df = None
if "schema" not in st.session_state:
    st.session_state.schema = {}
if "defaults" not in st.session_state:
    st.session_state.defaults = {"metric": None, "time_col": None}

# Load persisted chat history once per browser tab refresh
if "loaded_from_db" not in st.session_state:
    st.session_state.messages = db.load_messages(st.session_state.session_id)
    st.session_state.loaded_from_db = True

# --- Artifacts dir for charts
ARTIFACT_DIR = Path("artifacts/plots")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# Helpers: messages & render
# =========================
def add_message(role: str, content: str):
    st.session_state.messages.append({"role": role, "type": "text", "content": content})
    db.add_text(st.session_state.session_id, role, content)


def add_table(df: pd.DataFrame, caption: str | None = None):
    msg = {
        "role": "assistant",
        "type": "table",
        "columns": list(df.columns),
        "data": df.to_dict(orient="records"),
        "caption": caption,
    }
    st.session_state.messages.append(msg)
    db.add_table(st.session_state.session_id, msg["columns"], msg["data"], caption=caption)


def add_image(path: str, caption: str | None = None, spec: dict | None = None):
    st.session_state.messages.append({"role": "assistant", "type": "image", "path": path, "caption": caption})
    db.add_image(st.session_state.session_id, path, caption=caption)
    db.add_plot(st.session_state.session_id, path, spec or {})


def render_messages():
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            mtype = m.get("type", "text")
            if mtype == "text":
                st.markdown(m.get("content", ""))
            elif mtype == "table":
                _df = pd.DataFrame(m["data"], columns=m["columns"])
                st.dataframe(_df, use_container_width=True)
                if m.get("caption"):
                    st.caption(m["caption"])
            elif mtype == "image":
                st.image(m["path"], caption=m.get("caption"))


# =========================
# Analytics & plotting
# =========================
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


# =========================
# Tiny router (rule-based)
# =========================
def simple_router(user_text: str):
    s = user_text.lower()

    # schema questions
    if any(w in s for w in ["column", "columns", "schema", "what columns", "show columns"]):
        return "schema_probe", {}

    # top-k: "top 5 category by revenue"
    if "top" in s and "by" in s:
        import re
        m = re.search(r"top\s+(\d+)\s+(\w+)\s+by\s+(\w+)", s)
        if m:
            k = int(m.group(1))
            group_col = m.group(2)
            metric = m.group(3)
            return "qa_numeric", {"k": k, "group": group_col, "metric": metric}

    # plotting: "plot revenue by month [split by region]"
    if any(w in s for w in ["plot", "chart", "graph"]):
        words = s.replace(",", " ").split()
        y = None
        x = None
        hue = None
        if "plot" in words:
            try:
                idx = words.index("plot")
                y = words[idx + 1]
            except Exception:
                pass
        if "by" in words:
            try:
                idx = words.index("by")
                x = words[idx + 1]
            except Exception:
                pass
        if "split" in words and "by" in words:
            try:
                idx = words.index("split")
                if words[idx + 1] == "by":
                    hue = words[idx + 2]
            except Exception:
                pass
        # fallbacks to inferred defaults
        y = y or st.session_state.defaults.get("metric")
        x = x or st.session_state.defaults.get("time_col")
        return "plot", {"x": x, "y": y, "hue": hue}

    # default help
    return "help", {}


# =========================
# Sidebar: dataset & danger zone
# =========================
st.sidebar.header("Dataset")
uploaded = st.sidebar.file_uploader("Upload a CSV", type=["csv"])
if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.session_state.df = df
    st.session_state.schema = {c: str(df[c].dtype) for c in df.columns}
    # infer defaults
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    time_candidates = [c for c in df.columns if any(k in c.lower() for k in ["date", "time", "month", "year"])]
    st.session_state.defaults["metric"] = st.session_state.defaults["metric"] or (num_cols[0] if num_cols else None)
    st.session_state.defaults["time_col"] = st.session_state.defaults["time_col"] or (
        time_candidates[0] if time_candidates else None
    )

if st.session_state.df is not None:
    with st.sidebar.expander("Schema & Preview", expanded=False):
        st.write(pd.DataFrame({"column": list(st.session_state.schema.keys()), "dtype": list(st.session_state.schema.values())}))
        st.dataframe(st.session_state.df.head(10), use_container_width=True)
        st.caption(f"Rows: {len(st.session_state.df):,}")
else:
    st.sidebar.info("Upload a CSV to get started.")

# Danger Zone: resets
with st.sidebar.expander("⚠️ Danger zone", expanded=False):
    st.caption("These actions permanently delete data.")
    if st.button("Clear current session (chat + DB)"):
        # remove this session's plot files
        try:
            for p in db.get_plot_paths(st.session_state.session_id):
                Path(p).unlink(missing_ok=True)
        except Exception as e:
            st.toast(f"Note: Could not remove some plot files: {e}")
        # clear DB rows for this session
        db.clear_session(st.session_state.session_id)
        # reset Streamlit state with a fresh session id
        st.session_state.messages = []
        st.session_state.df = None
        st.session_state.schema = {}
        st.session_state.defaults = {"metric": None, "time_col": None}
        st.session_state.session_id = str(uuid.uuid4())
        db.ensure_session(st.session_state.session_id, title="Streamlit Session")
        st.success("Current session cleared.")
        st.experimental_rerun()

    confirm = st.checkbox("I understand this will delete ALL sessions")
    if st.button("Wipe entire database") and confirm:
        try:
            for img in Path("artifacts/plots").glob("*.png"):
                img.unlink(missing_ok=True)
        except Exception as e:
            st.toast(f"Note: Could not remove some plot files: {e}")
        db.wipe_all()
        st.session_state.messages = []
        st.session_state.df = None
        st.session_state.schema = {}
        st.session_state.defaults = {"metric": None, "time_col": None}
        st.session_state.session_id = str(uuid.uuid4())
        db.ensure_session(st.session_state.session_id, title="Streamlit Session")
        st.success("Entire database wiped.")
        st.experimental_rerun()


# =========================
# Main UI
# =========================
st.title("💬 Data Explorer")
st.caption(
    "Upload a CSV in the sidebar, then ask things like "
    "‘what columns do i have’, ‘top 5 category by revenue’, "
    "or ‘plot marks by subject, split by name’."
)

render_messages()

prompt = st.chat_input("Ask a question…")
if prompt:
    add_message("user", prompt)
    intent, args = simple_router(prompt)

    if intent == "schema_probe":
        add_message("assistant", list_columns())

    elif intent == "qa_numeric":
        df_out, err = top_k_by_group(args.get("k", 5), args.get("group"), args.get("metric"))
        if err:
            add_message("assistant", f"❌ {err}")
        else:
            add_message("assistant", f"Here are the top {args.get('k', 5)} {args.get('group')} by {args.get('metric')}:")
            add_table(
                df_out,
                caption="Tip: say ‘plot them by <time_col>’ or ‘plot marks by subject, split by name’.",
            )

    elif intent == "plot":
        x = args.get("x")
        y = args.get("y")
        hue = args.get("hue")
        if not y:
            add_message("assistant", "I couldn't infer the metric to plot. Try: ‘plot revenue by month’.")
        else:
            path, err = plot_group_sum(x, y, hue=hue, fname_prefix="user")
            if err:
                add_message("assistant", f"❌ {err}")
            else:
                spec = {"x": x, "y": y, "hue": hue, "agg": "sum"}
                add_message("assistant", f"Saved chart to `{path}`.")
                add_image(path, caption=path, spec=spec)

    else:
        default_metric = st.session_state.defaults.get("metric") or "<metric>"
        default_time = st.session_state.defaults.get("time_col") or "<time_col>"
        add_message(
            "assistant",
            (
                "Try these:\n"
                "- **Columns?** → ‘what columns do i have?’\n"
                f"- **Top-k** → ‘top 5 category by {default_metric}’\n"
                f"- **Plot** → ‘plot {default_metric} by {default_time}’ or "
                f"‘plot {default_metric} by {default_time}, split by region’\n"
            ),
        )

    render_messages()
