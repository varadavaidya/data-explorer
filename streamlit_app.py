# streamlit_app.py — Data Explorer (Streamlit)
from __future__ import annotations
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st

# --- DB init & session id
from src.db import init_db, db

# Tools (logic lives in src/tools/*)
from src.tools.analytics import list_columns, top_k_by_group, aggregate_by_group
from src.tools.viz import plot_group_sum
from src.tools.windows_funcs import rank_within, cumulative_sum, rolling_mean, lag_lead
from src.tools.more_funcs import (
    describe_numeric, missing_report, value_counts, correlation_matrix,
    histogram, boxplot, pivot_table, outliers_zscore
)

# ------------------------------------------------------------------
# App bootstrap
# ------------------------------------------------------------------
st.set_page_config(page_title="Data Explorer", layout="wide")
init_db()  # ensure artifacts/app.db & tables exist

# session id per browser tab
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
db.ensure_session(st.session_state.session_id, title="Streamlit Session")

# --- Session state defaults (kept in memory during tab lifetime)
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of dicts: {"role","type",...}
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

# ------------------------------------------------------------------
# Helpers: artifacts, messages, render
# ------------------------------------------------------------------
ARTIFACT_DIR = Path("artifacts/plots")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


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
        "uid": str(uuid.uuid4()),  # unique id for download button keys
    }
    st.session_state.messages.append(msg)
    db.add_table(
        st.session_state.session_id,
        msg["columns"],
        msg["data"],
        caption=caption,
    )


def add_image(path: str, caption: str | None = None, spec: dict | None = None):
    st.session_state.messages.append({"role": "assistant", "type": "image", "path": path, "caption": caption})
    db.add_image(st.session_state.session_id, path, caption=caption)
    db.add_plot(st.session_state.session_id, path, spec or {})


def render_messages():
    for i, m in enumerate(st.session_state.messages):
        with st.chat_message(m["role"]):
            mtype = m.get("type", "text")
            if mtype == "text":
                st.markdown(m.get("content", ""))
            elif mtype == "table":
                _df = pd.DataFrame(m["data"], columns=m["columns"])
                st.dataframe(_df, use_container_width=True)
                if m.get("caption"):
                    st.caption(m["caption"])
                # Download CSV button (unique key)
                csv_bytes = _df.to_csv(index=False).encode("utf-8")
                dl_key = f"dl-{m.get('uid') or f'{i}-{st.session_state.session_id}'}"
                st.download_button(
                    "Download CSV",
                    data=csv_bytes,
                    file_name=f"table_{i}.csv",
                    mime="text/csv",
                    key=dl_key,
                )
            elif mtype == "image":
                st.image(m["path"], caption=m.get("caption"))


# ------------------------------------------------------------------
# Router (rules today; LLM later). IMPORTANT: preserve original casing.
# ------------------------------------------------------------------
def simple_router(user_text: str):
    s = user_text.lower().strip()

    # schema questions
    if "column" in s and ("what" in s or "show" in s or "schema" in s):
        return "schema_probe", {}

    # top-k: "top 5 category by revenue"
    import re
    m = re.search(r"top\s+(\d+)\s+(\w+)\s+by\s+(\w+)", user_text, re.I)
    if m:
        k = int(m.group(1))
        group_col = m.group(2).strip()
        metric = m.group(3).strip()
        return "qa_numeric", {"k": k, "group": group_col, "metric": metric}

    # aggregations: "average marks by subject", "sum revenue by region", "count by subject"
    import re
    m = re.search(
    r"(sum|average|avg|mean|count|min|max|median)\s+(?:of\s+)?([A-Za-z0-9_\-\s]+?)?\s*by\s+([A-Za-z0-9_\-\s]+)",
    user_text,
    re.I,
)
    if m:
        agg = m.group(1).lower()
        metric = (m.group(2) or "").strip()   # may be empty for "count by X"
        group_col = m.group(3).strip()
        return "agg", {"agg": agg, "metric": metric, "group": group_col}


    # ranking: "rank students by marks within subject"
    if "rank" in s and "by" in s and "within" in s:
        words = user_text.split()
        try:
            idx_by = [w.lower() for w in words].index("by")
            idx_within = [w.lower() for w in words].index("within")
            order_col = words[idx_by + 1]
            group_col = words[idx_within + 1]
            return "rank_within", {"group": group_col, "order": order_col}
        except Exception:
            pass

    # cumulative sum: "cumulative sum of revenue by month"
    m = re.search(r"cumulative\s+sum\s+of\s+(\w+)\s+by\s+(\w+)", user_text, re.I)
    if m:
        metric, group_col = m.group(1), m.group(2)
        return "cumsum", {"group": group_col, "metric": metric}

    # rolling mean: "rolling 3 month average of sales"
    m = re.search(r"rolling\s+(\d+)\s+\w+\s+average\s+of\s+(\w+)", user_text, re.I)
    if m:
        window, metric = int(m.group(1)), m.group(2)
        return "rolling_mean", {
            "time_col": st.session_state.defaults.get("time_col"),
            "metric": metric,
            "window": window,
        }

    # lag: "lag marks by 1"
    m = re.search(r"lag\s+(\w+)\s+by\s+(\d+)", user_text, re.I)
    if m:
        metric, shift = m.group(1), int(m.group(2))
        return "lag", {"time_col": st.session_state.defaults.get("time_col"), "metric": metric, "shift": shift}

    # value counts: "value counts of subject" or "top 10 values for subject"
    m = re.search(r"(?:value\s+counts\s+(?:of|for)|top\s+(\d+)\s+values\s+for)\s+(\w+)", user_text, re.I)
    if m:
        top = int(m.group(1)) if m.group(1) else 20
        col = m.group(2)
        return "value_counts", {"col": col, "top": top}

    # correlation matrix
    if "correlation" in s:
        return "corr", {}

    # histogram: "histogram of marks [bins 20]"
    m = re.search(r"histogram\s+of\s+(\w+)(?:.*bins\s+(\d+))?", user_text, re.I)
    if m:
        col = m.group(1)
        bins = int(m.group(2)) if m.group(2) else 30
        return "hist", {"col": col, "bins": bins}

    # boxplot: "boxplot marks [by subject]"
    m = re.search(r"boxplot\s+(\w+)(?:\s+by\s+(\w+))?", user_text, re.I)
    if m:
        y = m.group(1)
        by = m.group(2)
        return "box", {"y": y, "by": by}

    # pivot table: "pivot values revenue by region and month [agg mean]"
    m = re.search(r"pivot\s+(?:values\s+)?(\w+)\s+by\s+(\w+)\s+and\s+(\w+)(?:.*agg\s+(\w+))?", user_text, re.I)
    if m:
        values, index, columns, agg = m.group(1), m.group(2), m.group(3), (m.group(4) or "sum")
        return "pivot", {"index": index, "columns": columns, "values": values, "agg": agg}

    # outliers: "outliers in marks [z 3]"
    m = re.search(r"outliers\s+in\s+(\w+)(?:.*z\s+([\d\.]+))?", user_text, re.I)
    if m:
        col = m.group(1)
        thr = float(m.group(2)) if m.group(2) else 3.0
        return "outliers", {"col": col, "z": thr}

    # plots: "plot revenue by month [split by region]"
    # plots: "plot revenue by month [split by region]"
    words = user_text.replace(",", " ").split()
    low = [w.lower() for w in words]
    if "plot" in low:
        y = x = hue = None
        i = low.index("plot")
        if i + 1 < len(words):
            y = words[i + 1]  # metric
        if "by" in low:
            j = low.index("by")
            if j + 1 < len(words):
                x = words[j + 1]  # x axis
        if "split" in low:
            k = low.index("split")
            if k + 2 < len(words) and low[k + 1] == "by":
                hue = words[k + 2]  # series
        # fallback only when user asked to plot
        y = y or st.session_state.defaults.get("metric")
        x = x or st.session_state.defaults.get("time_col")
        return "plot", {"x": x, "y": y, "hue": hue}

    # default help
    return "help", {}


# ------------------------------------------------------------------
# Sidebar: dataset (upload) + schema + danger zone
# ------------------------------------------------------------------
st.sidebar.header("Dataset")

# Upload a CSV (no examples)
uploaded = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.session_state.df = df
    st.session_state.schema = {c: str(df[c].dtype) for c in df.columns}

    # Infer sensible defaults
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    time_candidates = [c for c in df.columns if any(k in c.lower() for k in ["date", "time", "month", "year"])]
    st.session_state.defaults["metric"] = st.session_state.defaults.get("metric") or (num_cols[0] if num_cols else None)
    st.session_state.defaults["time_col"] = st.session_state.defaults.get("time_col") or (
        time_candidates[0] if time_candidates else None
    )

# Schema preview (always visible if a DF is loaded)
with st.sidebar.expander("📑 Schema", expanded=True):
    if st.session_state.df is None:
        st.info("Upload a CSV to see columns and types.")
    else:
        cols = list(st.session_state.df.columns)
        dtypes = [str(st.session_state.df[c].dtype) for c in cols]
        schema_df = pd.DataFrame({"column": cols, "dtype": dtypes})
        st.dataframe(schema_df, use_container_width=True, hide_index=True)
        # Optional: quick sample
        if st.checkbox("Show first 10 rows", key="show_head"):
            st.dataframe(st.session_state.df.head(10), use_container_width=True)

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

    confirm = st.checkbox("I understand this will delete ALL sessions", key="wipe_all_confirm")
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

# ------------------------------------------------------------------
# Main UI
# ------------------------------------------------------------------
st.title("💬 Data Explorer")
st.caption(
    "Upload a CSV and ask things like: "
    "• ‘what columns do i have’ • ‘top 5 category by revenue’ • "
    "‘average marks by subject’ • ‘correlation matrix’ • "
    "‘histogram of marks bins 20’ • ‘boxplot marks by subject’ • "
    "‘pivot values revenue by region and month agg mean’ • "
    "‘rank students by marks within subject’ • "
    "‘cumulative sum of revenue by month’ • ‘lag marks by 1’ • "
    "‘plot revenue by month, split by region’"
)

with st.expander("📒 Cheat sheet (what can I ask?)", expanded=False):
    st.markdown("""
- **Schema**: `what columns do i have`
- **Top-k**: `top 5 category by revenue`
- **Aggregations**: `average marks by subject`, `sum revenue by region`, `count by subject`
- **Window-like**: `rank students by marks within subject`, `cumulative sum of revenue by month`, `rolling 3 month average of sales`, `lag marks by 1`
- **Exploration**: `missing values report`, `value counts of subject`, `correlation matrix`
- **Visuals**: `plot revenue by month, split by region`, `histogram of marks bins 20`, `boxplot marks by subject`
- **Pivot**: `pivot values revenue by region and month agg mean`
""")

prompt = st.chat_input("Ask a question…")

if prompt:
    # record the user message
    add_message("user", prompt)

    # route to an intent (rules today; LLM later)
    intent, args = simple_router(prompt)

    # ---- schema probe
    if intent == "schema_probe":
        add_message("assistant", list_columns())

    # ---- top-k numeric
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

    # ---- general aggregations (sum/average/count/min/max/median)
    elif intent == "agg":
        df_out, err = aggregate_by_group(args.get("group"), args.get("metric", ""), agg=args.get("agg", "sum"))
        if err:
            add_message("assistant", f"❌ {err}")
        else:
            pretty_agg = {"avg": "average"}.get(args.get("agg", "sum"), args.get("agg", "sum"))
            title = (
                f"{pretty_agg.capitalize()} {args.get('metric')} by {args.get('group')}"
                if args.get("agg") != "count"
                else f"Count by {args.get('group')}"
            )
            add_message("assistant", title)
            add_table(df_out, caption="Use ‘top 5 ... by ...’ for ranking, or ‘plot ... by ...’ to visualize.")

    # ---- window-like funcs
    elif intent == "rank_within":
        df_out, err = rank_within(args["group"], args["order"])
        add_message("assistant", f"Rank of {args['order']} within {args['group']}" if not err else f"❌ {err}")
        if not err:
            add_table(df_out)

    elif intent == "cumsum":
        df_out, err = cumulative_sum(args["group"], args["metric"])
        add_message("assistant", f"Cumulative sum of {args['metric']} by {args['group']}" if not err else f"❌ {err}")
        if not err:
            add_table(df_out)

    elif intent == "rolling_mean":
        df_out, err = rolling_mean(args["time_col"], args["metric"], args.get("window", 3))
        add_message("assistant", f"Rolling {args.get('window',3)}-period mean of {args['metric']}" if not err else f"❌ {err}")
        if not err:
            add_table(df_out)

    elif intent == "lag":
        df_out, err = lag_lead(args["time_col"], args["metric"], args.get("shift", 1))
        add_message("assistant", f"Lagged {args['metric']} by {args.get('shift',1)}" if not err else f"❌ {err}")
        if not err:
            add_table(df_out)

    # ---- extra analysis tools
    elif intent == "describe":
        df_out, err = describe_numeric()
        add_message("assistant", "Summary statistics (numeric columns):" if not err else f"❌ {err}")
        if not err:
            add_table(df_out)

    elif intent == "missing":
        df_out, err = missing_report()
        add_message("assistant", "Missing values report:" if not err else f"❌ {err}")
        if not err:
            add_table(df_out)

    elif intent == "value_counts":
        df_out, err = value_counts(args["col"], top=args.get("top", 20))
        add_message("assistant", f"Value counts for `{args['col']}`:" if not err else f"❌ {err}")
        if not err:
            add_table(df_out)

    elif intent == "corr":
        df_out, path, err = correlation_matrix()
        if err:
            add_message("assistant", f"❌ {err}")
        else:
            add_message("assistant", "Correlation matrix (numeric columns):")
            add_table(df_out, caption="Heatmap saved below.")
            add_image(path, caption=path, spec={"type": "corr"})

    elif intent == "hist":
        df_out, err, path = histogram(args["col"], bins=args.get("bins", 30))
        if err:
            add_message("assistant", f"❌ {err}")
        else:
            add_message("assistant", f"Histogram of `{args['col']}`:")
            add_image(path, caption=path, spec={"type": "hist", "col": args["col"], "bins": args.get("bins", 30)})

    elif intent == "box":
        df_out, err, path = boxplot(args["y"], by=args.get("by"))
        if err:
            add_message("assistant", f"❌ {err}")
        else:
            title = f"Boxplot of {args['y']}" + (f" by {args.get('by')}" if args.get("by") else "")
            add_message("assistant", title)
            add_image(path, caption=path, spec={"type": "box", "y": args["y"], "by": args.get("by")})

    elif intent == "pivot":
        df_out, err = pivot_table(args["index"], args["columns"], args["values"], agg=args.get("agg", "sum"))
        add_message(
            "assistant",
            f"Pivot: {args['index']} × {args['columns']} → {args['values']} ({args.get('agg','sum')})"
            if not err
            else f"❌ {err}",
        )
        if not err:
            add_table(df_out)

    # ---- plotting (bar / grouped bar)
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

    # ---- default help
    else:
        default_metric = st.session_state.defaults.get("metric") or "<metric>"
        default_time = st.session_state.defaults.get("time_col") or "<time_col>"
        add_message(
            "assistant",
            (
                "Try these:\n"
                "- **Schema** → ‘what columns do i have?’\n"
                f"- **Top-k** → ‘top 5 category by {default_metric}’\n"
                f"- **Aggregate** → ‘average {default_metric} by category’, ‘count by subject’\n"
                "- **Window** → ‘rank students by marks within subject’, ‘cumulative sum of revenue by month’\n"
                "- **Explore** → ‘missing values report’, ‘value counts of subject’, ‘correlation matrix’\n"
                f"- **Visualize** → ‘plot {default_metric} by {default_time}’, ‘histogram of {default_metric} bins 20’, ‘boxplot {default_metric} by category’\n"
                "- **Pivot** → ‘pivot values revenue by region and month agg mean’\n"
            ),
        )

# Re-render history (single call → no duplicate keys)
render_messages()
