# streamlit_app.py — Data Explorer (Streamlit)
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(page_title="Data Explorer", layout="wide")

# --- Session state (why): keeps memory across chat turns within the same browser tab
if "messages" not in st.session_state:
    st.session_state.messages = []  # [{"role": "user"/"assistant", "content": "..."}]
if "df" not in st.session_state:
    st.session_state.df = None
if "schema" not in st.session_state:
    st.session_state.schema = {}
if "defaults" not in st.session_state:
    st.session_state.defaults = {"metric": None, "time_col": None}

ARTIFACT_DIR = Path("artifacts/plots")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)  # ensure folder exists

# --- Sidebar: CSV upload + preview (why): the dataset is the core of exploration
st.sidebar.header("Dataset")
uploaded = st.sidebar.file_uploader("Upload a CSV", type=["csv"])
if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.session_state.df = df
    st.session_state.schema = {c: str(df[c].dtype) for c in df.columns}
    # infer defaults: pick first numeric as metric; pick date-like col as time_col
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    time_candidates = [c for c in df.columns if any(k in c.lower() for k in ["date", "time", "month", "year"])]
    st.session_state.defaults["metric"] = st.session_state.defaults["metric"] or (num_cols[0] if num_cols else None)
    st.session_state.defaults["time_col"] = st.session_state.defaults["time_col"] or (time_candidates[0] if time_candidates else None)

if st.session_state.df is not None:
    with st.sidebar.expander("Schema & Preview", expanded=False):
        st.write(
            pd.DataFrame(
                {"column": list(st.session_state.schema.keys()), "dtype": list(st.session_state.schema.values())}
            )
        )
        st.dataframe(st.session_state.df.head(10), use_container_width=True)
        st.caption(f"Rows: {len(st.session_state.df):,}")
else:
    st.sidebar.info("Upload a CSV to get started.")

# --- Helpers (why): keep UI clean and reusable)
def add_message(role: str, content: str):
    st.session_state.messages.append({"role": role, "content": content})

def render_messages():
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

def list_columns():
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
    grouped = df.groupby(group_col)[metric].sum().sort_values(ascending=False).head(k)
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
    ax.set_xlabel(x_col); ax.set_ylabel(y_col)
    fig.tight_layout()

    out = ARTIFACT_DIR / f"{fname_prefix}_{x_col}_{y_col}{'_'+hue if hue else ''}.png"
    fig.savefig(out)
    plt.close(fig)
    return str(out), None

def simple_router(user_text: str):
    """Tiny rule-based intent router (why): fast to ship; later you can swap with LangGraph nodes."""
    s = user_text.lower()

    # schema questions
    if any(w in s for w in ["column", "columns", "schema", "what columns", "show columns"]):
        return "schema_probe", {}

    # top-k pattern: "top 5 category by revenue"
    if "top" in s and "by" in s:
        import re
        m = re.search(r"top\s+(\d+)\s+(\w+)\s+by\s+(\w+)", s)
        if m:
            k = int(m.group(1)); group_col = m.group(2); metric = m.group(3)
            return "qa_numeric", {"k": k, "group": group_col, "metric": metric}

    # plots: "plot revenue by month [split by region]"
    if any(w in s for w in ["plot", "chart", "graph"]):
        words = s.replace(",", " ").split()
        y = None; x = None; hue = None
        if "plot" in words:
            try:
                idx = words.index("plot")
                y = words[idx+1]
            except Exception:
                pass
        if "by" in words:
            try:
                idx = words.index("by")
                x = words[idx+1]
            except Exception:
                pass
        if "split" in words and "by" in words:
            try:
                idx = words.index("split")
                if words[idx+1] == "by":
                    hue = words[idx+2]
            except Exception:
                pass
        # fallbacks from defaults
        y = y or st.session_state.defaults.get("metric")
        x = x or st.session_state.defaults.get("time_col")
        return "plot", {"x": x, "y": y, "hue": hue}

    # default help
    return "help", {}

# --- Main UI
st.title("💬 Data Explorer")
st.caption("Upload a CSV from the sidebar, then ask: ‘top 5 category by revenue’, ‘plot revenue by month’, or ‘what columns do I have?’")

render_messages()
prompt = st.chat_input("Ask a question…")

if prompt:
    add_message("user", prompt)
    intent, args = simple_router(prompt)

    if intent == "schema_probe":
        reply = list_columns()
        add_message("assistant", reply)

    elif intent == "qa_numeric":
        df_out, err = top_k_by_group(args.get("k", 5), args.get("group"), args.get("metric"))
        if err:
            add_message("assistant", f"❌ {err}")
        else:
            add_message("assistant", f"Here are the top {args.get('k', 5)} {args.get('group')} by {args.get('metric')}:")
            with st.chat_message("assistant"):
                st.dataframe(df_out, use_container_width=True)
                st.caption("Tip: Say ‘plot them by month’ or ‘plot revenue by month, split by region’.")
            # remember last table if you want (future)

    elif intent == "plot":
        x = args.get("x"); y = args.get("y"); hue = args.get("hue")
        if not y:
            add_message("assistant", "I couldn't infer the metric to plot. Try: ‘plot revenue by month’.")
        else:
            path, err = plot_group_sum(x, y, hue=hue, fname_prefix="user")
            if err:
                add_message("assistant", f"❌ {err}")
            else:
                add_message("assistant", f"Saved chart to `{path}`.")
                with st.chat_message("assistant"):
                    st.image(path, caption=path)

    else:
        default_metric = st.session_state.defaults.get("metric") or "<metric>"
        default_time = st.session_state.defaults.get("time_col") or "<time_col>"
        add_message(
            "assistant",
            (
                "Try these:\n"
                "- **Columns?** → ‘what columns do I have?’\n"
                f"- **Top-k** → ‘top 5 category by {default_metric}’\n"
                f"- **Plot** → ‘plot {default_metric} by {default_time}’ or ‘plot {default_metric} by {default_time}, split by region’\n"
            ),
        )

    render_messages()
