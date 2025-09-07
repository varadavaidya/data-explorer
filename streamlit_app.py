# streamlit_app.py — Data Explorer (Streamlit)
from __future__ import annotations
import uuid
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from src.tools.analytics import list_columns, top_k_by_group, aggregate_by_group
from src.tools.viz import plot_group_sum 
from src.tools.windows_funcs import rank_within, cumulative_sum, rolling_mean, lag_lead
from src.tools.more_funcs import (
    describe_numeric, missing_report, value_counts, correlation_matrix,
    histogram, boxplot, pivot_table, outliers_zscore
)

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
                # Download CSV button
                csv_bytes = _df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download CSV",
                    data=csv_bytes,
                    file_name=f"table_{i}.csv",
                    mime="text/csv",
                    key=f"dl-{i}",
                )
            elif mtype == "image":
                st.image(m["path"], caption=m.get("caption"))




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
    
        # aggregations: "average marks by subject", "sum revenue by region", "count by subject"
    if " by " in s and any(w in s for w in ["sum", "average", "avg", "mean", "count", "min", "max", "median"]):
        import re
        m = re.search(r"(sum|average|avg|mean|count|min|max|median)\s+(\w+)?\s*by\s+(\w+)", s)
        if m:
            agg = m.group(1)
            metric = m.group(2) or ""  # metric can be empty for "count by subject"
            group_col = m.group(3)
            return "agg", {"agg": agg, "metric": metric, "group": group_col}
        
        # ranking: "rank students by marks within subject"
    if "rank" in s and "by" in s and "within" in s:
        words = s.split()
        try:
            idx_by = words.index("by")
            idx_within = words.index("within")
            order_col = words[idx_by+1]
            group_col = words[idx_within+1]
            return "rank_within", {"group": group_col, "order": order_col}
        except Exception:
            pass

    # cumulative sum: "cumulative sum of revenue by month"
    if "cumulative sum" in s:
        import re
        m = re.search(r"cumulative sum of (\w+) by (\w+)", s)
        if m:
            metric, group_col = m.group(1), m.group(2)
            return "cumsum", {"group": group_col, "metric": metric}

    # rolling mean: "rolling 3 month average of sales"
    if "rolling" in s and "average" in s:
        import re
        m = re.search(r"rolling\s+(\d+)\s+\w+\s+average of (\w+)", s)
        if m:
            window, metric = int(m.group(1)), m.group(2)
            # assume time_col is already in defaults
            return "rolling_mean", {"time_col": st.session_state.defaults.get("time_col"), "metric": metric, "window": window}

    # lag: "lag marks by 1"
    if s.startswith("lag"):
        import re
        m = re.search(r"lag (\w+) by (\d+)", s)
        if m:
            metric, shift = m.group(1), int(m.group(2))
            return "lag", {"time_col": st.session_state.defaults.get("time_col"), "metric": metric, "shift": shift}
    
        # describe / summary
    if "describe" in s or "summary" in s:
        return "describe", {}

    # missing values report
    if "missing" in s and ("values" in s or "report" in s):
        return "missing", {}

    # value counts: "value counts of subject" or "top 10 values for subject"
    if "value counts" in s or ("top" in s and "values" in s):
        import re
        m = re.search(r"(?:value counts of|value counts for|top\s+(\d+)\s+values\s+for)\s+(\w+)", s)
        if m:
            top = int(m.group(1)) if m.group(1) else 20
            col = m.group(2)
            return "value_counts", {"col": col, "top": top}

    # correlation matrix
    if "correlation matrix" in s or "correlation" in s:
        return "corr", {}

    # histogram: "histogram of marks [bins 20]"
    if "histogram" in s:
        import re
        m = re.search(r"histogram of (\w+)(?:.*bins\s+(\d+))?", s)
        if m:
            col = m.group(1); bins = int(m.group(2)) if m.group(2) else 30
            return "hist", {"col": col, "bins": bins}

    # boxplot: "boxplot marks [by subject]"
    if "boxplot" in s:
        import re
        m = re.search(r"boxplot\s+(\w+)(?:\s+by\s+(\w+))?", s)
        if m:
            y = m.group(1); by = m.group(2)
            return "box", {"y": y, "by": by}

    # pivot table: "pivot values revenue by region and month [agg mean]"
    if s.startswith("pivot") or "pivot table" in s:
        import re
        m = re.search(r"pivot (?:values\s+)?(\w+)\s+by\s+(\w+)\s+and\s+(\w+)(?:.*agg\s+(\w+))?", s)
        if m:
            values, index, columns, agg = m.group(1), m.group(2), m.group(3), (m.group(4) or "sum")
            return "pivot", {"index": index, "columns": columns, "values": values, "agg": agg}

    # outliers: "outliers in marks [z 3]"
    if "outlier" in s:
        import re
        m = re.search(r"outliers in (\w+)(?:.*z\s+([\d\.]+))?", s)
        if m:
            col = m.group(1); thr = float(m.group(2)) if m.group(2) else 3.0
            return "outliers", {"col": col, "z": thr}




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
    
    elif intent == "agg":
        agg = args.get("agg", "sum")
        metric = args.get("metric", "")
        group = args.get("group")
        df_out, err = aggregate_by_group(group, metric, agg=agg)
        if err:
            add_message("assistant", f"❌ {err}")
        else:
            pretty_agg = {"avg":"average"}.get(agg, agg)
            title = f"{pretty_agg.capitalize()} {metric} by {group}" if agg != "count" else f"Count by {group}"
            add_message("assistant", title)
            add_table(df_out, caption="Use ‘top 5 ... by ...’ for ranking, or ‘plot ... by ...’ to visualize.")


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
    

    elif intent == "rank_within":
        df_out, err = rank_within(args["group"], args["order"])
        if err: add_message("assistant", f"❌ {err}")
        else: add_table(df_out, caption=f"Rank of {args['order']} within {args['group']}")

    elif intent == "cumsum":
        df_out, err = cumulative_sum(args["group"], args["metric"])
        if err: add_message("assistant", f"❌ {err}")
        else: add_table(df_out, caption=f"Cumulative sum of {args['metric']} by {args['group']}")

    elif intent == "rolling_mean":
        df_out, err = rolling_mean(args["time_col"], args["metric"], args["window"])
        if err: add_message("assistant", f"❌ {err}")
        else: add_table(df_out, caption=f"Rolling {args['window']}-period mean of {args['metric']}")

    elif intent == "lag":
        df_out, err = lag_lead(args["time_col"], args["metric"], args["shift"])
        if err: add_message("assistant", f"❌ {err}")
        else: add_table(df_out, caption=f"Lagged {args['metric']} by {args['shift']}")
    
        elif intent == "describe":
        df_out, err = describe_numeric()
        add_message("assistant", "Summary statistics (numeric columns):" if not err else f"❌ {err}")
        if not err: add_table(df_out)

    elif intent == "missing":
        df_out, err = missing_report()
        add_message("assistant", "Missing values report:" if not err else f"❌ {err}")
        if not err: add_table(df_out)

    elif intent == "value_counts":
        df_out, err = value_counts(args["col"], top=args.get("top", 20))
        add_message("assistant", f"Value counts for `{args['col']}`:" if not err else f"❌ {err}")
        if not err: add_table(df_out)

    elif intent == "corr":
        df_out, path, err = correlation_matrix()
        if err: add_message("assistant", f"❌ {err}")
        else:
            add_message("assistant", "Correlation matrix (numeric columns):")
            add_table(df_out, caption="Heatmap saved below.")
            add_image(path, caption=path, spec={"type": "corr"})

    elif intent == "hist":
        df_out, err, path = histogram(args["col"], bins=args.get("bins", 30))
        if err: add_message("assistant", f"❌ {err}")
        else:
            add_message("assistant", f"Histogram of `{args['col']}`:")
            add_image(path, caption=path, spec={"type": "hist", "col": args["col"], "bins": args.get("bins", 30)})

    elif intent == "box":
        df_out, err, path = boxplot(args["y"], by=args.get("by"))
        if err: add_message("assistant", f"❌ {err}")
        else:
            title = f"Boxplot of {args['y']}" + (f" by {args['by']}" if args.get("by") else "")
            add_message("assistant", title)
            add_image(path, caption=path, spec={"type": "box", "y": args["y"], "by": args.get("by")})

    elif intent == "pivot":
        df_out, err = pivot_table(args["index"], args["columns"], args["values"], agg=args.get("agg", "sum"))
        add_message("assistant", f"Pivot: {args['index']} × {args['columns']} → {args['values']} ({args.get('agg','sum')})"
                     if not err else f"❌ {err}")
        if not err: add_table(df_out)

    elif intent == "outliers":
        df_out, err = outliers_zscore(args["col"], threshold=args.get("z", 3.0))
        if err:
            add_message("assistant", f"❌ {err}")
        else:
            add_message("assistant", f"Outliers in `{args['col']}` (|z| ≥ {args.get('z',3.0)}):")
            if df_out.empty:
                add_message("assistant", "No outliers found.")
            else:
                add_table(df_out)

    



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
