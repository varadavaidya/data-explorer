# streamlit_app.py — RELIABLE DROP-IN
from __future__ import annotations
import os, sys, uuid
from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd
import streamlit as st
from src.llm.router_graph import run_router_graph

# ------------------------------------------------------------------
# Make sure we can import from ./src even if not a package
# ------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent / "src"))
# --- Imports: router(s), tools, db
try:
    from src.router import simple_router
except Exception:
    # Minimal fallback router if import fails
    def simple_router(txt: str):
        low = (txt or "").lower().strip()
        if low in {"what columns do i have", "columns", "schema", "list columns"}:
            return "schema_probe", {}
        if low.startswith("mean ") or "mean of" in low or "average" in low:
            # naive overall mean: "mean of X"
            parts = low.split()
            col = parts[-1] if parts else ""
            return "agg_overall", {"agg": "mean", "metric": col}
        return "help", {}

from src.tools.util import resolve_column
import src.tools.analytics as analytics
import src.tools.more_funcs as more
import src.tools.viz as viz
import src.tools.windows_funcs as win
import src.db as db

# Creative router (optional)
try:
    from src.llm.router_graph import run_router_graph
    HAVE_LLM = True
except Exception:
    HAVE_LLM = False

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
ART_DIR = Path("artifacts")
PLOTS_DIR = ART_DIR / "plots"
ART_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# Session bootstrap
# ------------------------------------------------------------------
def _init_state():
    st.session_state.setdefault("session_id", str(uuid.uuid4()))
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("df", None)
    st.session_state.setdefault("schema", {})
    st.session_state.setdefault("defaults", {"metric": None, "time_col": None})
    st.session_state.setdefault("use_creative", False)
    st.session_state.setdefault("last_router_status", None)
    st.session_state.setdefault("last_uploaded", None)
    try:
        db.ensure_session(st.session_state.session_id, title="Streamlit Session")
    except Exception:
        pass

_init_state()

# ------------------------------------------------------------------
# Chat helpers
# ------------------------------------------------------------------
def _uid(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"

def add_message(role: str, text: str):
    st.session_state.messages.append({"role": role, "type": "text", "text": text})
    try:
        db.add_message(st.session_state.session_id, role, text)
    except Exception:
        pass

def add_table(df: pd.DataFrame, tag: str = "TABLE"):
    uid = _uid("tbl")
    st.session_state.messages.append({"role": "assistant", "type": "table", "df": df, "tag": tag, "uid": uid})
    try:
        db.add_table(st.session_state.session_id, df, tag=tag)
    except Exception:
        pass

def add_image(path: str, caption: Optional[str] = None, spec: Optional[Dict[str, Any]] = None):
    uid = _uid("img")
    st.session_state.messages.append({"role": "assistant", "type": "image", "path": path, "caption": caption, "uid": uid})
    try:
        db.add_image(st.session_state.session_id, path, caption=caption, spec=spec or {})
    except Exception:
        pass

def render_messages():
    for m in st.session_state.messages:
        if m["type"] == "text":
            with st.chat_message("user" if m["role"] == "user" else "assistant"):
                st.write(m["text"])
        elif m["type"] == "table":
            with st.chat_message("assistant"):
                if m.get("tag"):
                    st.caption(f"🧮 {m['tag']}")
                st.dataframe(m["df"], use_container_width=True)
                st.download_button(
                    "Download CSV",
                    m["df"].to_csv(index=False).encode("utf-8"),
                    file_name=f"{m['uid']}.csv",
                    mime="text/csv",
                    key=f"dl-{m['uid']}",
                )
        elif m["type"] == "image":
            with st.chat_message("assistant"):
                st.image(m["path"], caption=m.get("caption") or m["path"], use_container_width=True)

# ------------------------------------------------------------------
# Page
# ------------------------------------------------------------------
st.set_page_config(page_title="📊 Data Explorer", page_icon="📊", layout="wide")
st.title("📊 Data Explorer")
st.caption(
    "Upload a CSV and ask things like: "
    "• ‘what columns do i have’ • ‘top 5 Vehicle_Type’ • "
    "‘average Range_km by Region’ • ‘mean of Year’ • "
    "‘histogram of marks bins 20’ • ‘boxplot marks by subject’ • "
    "‘pivot values revenue by region and month agg mean’ • "
    "‘rank students by marks within subject’ • "
    "‘cumulative sum of revenue by month’ • ‘lag marks by 1’ • "
    "‘plot average Range_km by Year, split by Region’"
)

# ------------------------------------------------------------------
# Sidebar: upload + schema + creative toggle + danger + debug
# ------------------------------------------------------------------
st.sidebar.header("Dataset")

uploaded = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
if uploaded is not None and st.session_state.get("last_uploaded") != uploaded.name:
    df = pd.read_csv(uploaded)
    st.session_state.df = df
    st.session_state.schema = {c: str(df[c].dtype) for c in df.columns}

    # infer defaults
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    time_candidates = [c for c in df.columns if any(k in c.lower() for k in ["date", "time", "month", "year"])]
    st.session_state.defaults["metric"] = st.session_state.defaults.get("metric") or (num_cols[0] if num_cols else None)
    st.session_state.defaults["time_col"] = st.session_state.defaults.get("time_col") or (time_candidates[0] if time_candidates else None)

    add_message("assistant", f"📥 Loaded **{uploaded.name}** with {len(df)} rows and {len(df.columns)} columns.")
    st.session_state["last_uploaded"] = uploaded.name

# Schema
with st.sidebar.expander("📑 Schema", expanded=True):
    if st.session_state.df is None:
        st.info("Upload a CSV to see columns and types.")
    else:
        df = st.session_state.df
        schema_df = pd.DataFrame({"column": df.columns, "dtype": [str(df[c].dtype) for c in df.columns]})
        st.dataframe(schema_df, use_container_width=True, hide_index=True)
        if st.checkbox("Show first 10 rows", key="show_head"):
            st.dataframe(df.head(10), use_container_width=True)

# Creative toggle
label = "✨ Creative Mode (LLM routing + narration)"
if not HAVE_LLM:
    st.sidebar.checkbox(label + " — (Ollama/LLM not available)", value=False, key="use_creative", disabled=True)
else:
    st.sidebar.checkbox(label, value=False, key="use_creative")

# Danger zone
with st.sidebar.expander("⚠️ Danger zone", expanded=False):
    st.caption("These actions permanently delete data.")
    if st.button("Clear current session (chat + DB)"):
        try:
            for p in db.get_plot_paths(st.session_state.session_id):
                Path(p).unlink(missing_ok=True)
        except Exception:
            pass
        try:
            db.clear_session(st.session_state.session_id)
        except Exception:
            pass
        st.session_state.update(
            messages=[],
            df=None,
            schema={},
            defaults={"metric": None, "time_col": None},
            session_id=str(uuid.uuid4()),
            last_uploaded=None,
        )
        try:
            db.ensure_session(st.session_state.session_id, title="Streamlit Session")
        except Exception:
            pass
        st.success("Current session cleared.")
        st.rerun()

    confirm = st.checkbox("I understand this will delete ALL sessions", key="wipe_all_confirm")
    if st.button("Wipe entire database") and confirm:
        try:
            for img in PLOTS_DIR.glob("*.png"):
                img.unlink(missing_ok=True)
        except Exception:
            pass
        try:
            db.wipe_all()
        except Exception:
            pass
        st.session_state.update(
            messages=[],
            df=None,
            schema={},
            defaults={"metric": None, "time_col": None},
            session_id=str(uuid.uuid4()),
            last_uploaded=None,
        )
        try:
            db.ensure_session(st.session_state.session_id, title="Streamlit Session")
        except Exception:
            pass
        st.success("Entire database wiped.")
        st.rerun()

# Debug
with st.sidebar.expander("🔍 Debug", expanded=False):
    st.write("Router mode:", "Creative" if st.session_state.use_creative else "Rules")
    st.json(st.session_state.get("last_router_status") or {"note": "no router calls yet"})

# ------------------------------------------------------------------
# Render existing chat, then input
# ------------------------------------------------------------------
creative_mode = st.sidebar.toggle("Creative Mode", value=False)
prompt = st.chat_input("Ask a question…")
from src.llm.router_graph import run_router_graph

if prompt:
    if creative_mode:
        result, narrative, error = run_router_graph(prompt)
        router_used = "creative"
    else:
        result, narrative, error = run_rule_based_router(prompt)
        router_used = "rules"

    # Show user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Show assistant message (result, narrative, or error)
    with st.chat_message("assistant"):
        if result:
            kind = result.get("kind")
            payload = result.get("payload")

            if kind == "text":
                st.markdown(payload)
            elif kind == "table":
                st.dataframe(payload, use_container_width=True)
            elif kind == "image":
                st.image(payload["path"])
            elif kind == "table+image":
                st.dataframe(payload["table"], use_container_width=True)
                st.image(payload["path"])
        if narrative:
            st.info(narrative)
        if error:
            st.error(error)

# ------------------------------------------------------------------
# Intent handlers (Rules path and fallback use these)
# ------------------------------------------------------------------
def handle_intent(intent: str, args: Dict[str, Any]):
    df = st.session_state.df
    if df is None:
        add_message("assistant", "❌ No dataset loaded.")
        return

    # Common agg normalization
    def _norm_agg(a: str) -> str:
        return {"avg": "mean", "average": "mean", "minimum": "min", "maximum": "max"}.get((a or "").lower(), (a or "").lower())

    try:
        if intent == "schema_probe":
            add_message("assistant", "Columns:\n- " + "\n- ".join(df.columns))

        elif intent == "qa_numeric":  # top k by sum(metric)
            k = int(args.get("k", 5))
            group = resolve_column(args.get("group"), df.columns) or args.get("group")
            metric = resolve_column(args.get("metric"), df.columns) or args.get("metric")
            table, err = analytics.top_k_by_group(k, group, metric)
            if err: add_message("assistant", f"❌ {err}")
            else: add_table(table, tag=f"TOP-{k} by {metric}")

        elif intent == "agg":  # grouped
            agg = _norm_agg(args.get("agg", "mean"))
            group = resolve_column(args.get("group"), df.columns) or args.get("group")
            metric = resolve_column(args.get("metric"), df.columns) or args.get("metric")
            table, err = analytics.aggregate_by_group(group, metric, agg=agg)
            if err: add_message("assistant", f"❌ {err}")
            else: add_table(table, tag=f"{agg.upper()} of {metric} by {group}")

        elif intent == "agg_overall":  # overall single number
            agg = _norm_agg(args.get("agg", "mean"))
            metric = resolve_column(args.get("metric"), df.columns) or args.get("metric") or ""
            if not metric:
                add_message("assistant", "❌ No column specified for aggregation.")
                return
            series = pd.to_numeric(df[metric], errors="coerce")
            funcs = {
                "mean": series.mean, "sum": series.sum, "count": series.count,
                "min": series.min, "max": series.max, "median": series.median,
            }
            if agg not in funcs:
                add_message("assistant", f"❌ Unsupported aggregation `{agg}`."); return
            val = funcs[agg]()
            add_table(pd.DataFrame({"metric": [metric], agg: [val]}), tag=f"{agg} of {metric}")

        elif intent == "plot":
            x = args.get("x") or st.session_state.defaults.get("time_col")
            y = args.get("y") or st.session_state.defaults.get("metric")
            hue = args.get("hue")
            path, err = viz.plot_group_sum(x, y, hue=hue, fname_prefix="graph")
            if err: add_message("assistant", f"❌ {err}")
            else: add_image(path, caption=path)

        elif intent == "value_counts":
            col = resolve_column(args.get("col"), df.columns) or args.get("col")
            top = int(args.get("top", 20))
            table, err = more.value_counts(col, top=top)
            if err: add_message("assistant", f"❌ {err}")
            else: add_table(table, tag=f"TOP-{top} {col}")

        elif intent == "describe":
            table, err = more.describe_numeric()
            if err: add_message("assistant", f"❌ {err}")
            else: add_table(table, tag="DESCRIBE")

        elif intent == "missing":
            table, err = more.missing_report()
            if err: add_message("assistant", f"❌ {err}")
            else: add_table(table, tag="MISSING REPORT")

        elif intent == "corr":
            table, path, err = more.correlation_matrix()
            if err: add_message("assistant", f"❌ {err}")
            else:
                add_table(table, tag="CORRELATION")
                add_image(path, caption=path)

        elif intent == "hist":
            col = resolve_column(args.get("col"), df.columns) or args.get("col")
            bins = int(args.get("bins", 30))
            _, err, path = more.histogram(col, bins=bins)
            if err: add_message("assistant", f"❌ {err}")
            else: add_image(path, caption=path)

        elif intent == "box":
            y = resolve_column(args.get("y"), df.columns) or args.get("y")
            by = resolve_column(args.get("by"), df.columns) or args.get("by")
            _, err, path = more.boxplot(y, by=by)
            if err: add_message("assistant", f"❌ {err}")
            else: add_image(path, caption=path)

        elif intent == "pivot":
            idx = resolve_column(args.get("index"), df.columns) or args.get("index")
            cols = resolve_column(args.get("columns"), df.columns) or args.get("columns")
            vals = resolve_column(args.get("values"), df.columns) or args.get("values")
            agg = _norm_agg(args.get("agg", "sum"))
            table, err = more.pivot_table(idx, cols, vals, agg=agg)
            if err: add_message("assistant", f"❌ {err}")
            else: add_table(table, tag=f"PIVOT {agg}")

        elif intent == "rank_within":
            group = resolve_column(args.get("group"), df.columns) or args.get("group")
            order = resolve_column(args.get("order"), df.columns) or args.get("order")
            table, err = win.rank_within(group, order)
            if err: add_message("assistant", f"❌ {err}")
            else: add_table(table, tag="RANK WITHIN")

        elif intent == "cumsum":
            group = resolve_column(args.get("group"), df.columns) or args.get("group")
            metric = resolve_column(args.get("metric"), df.columns) or args.get("metric")
            table, err = win.cumulative_sum(group, metric)
            if err: add_message("assistant", f"❌ {err}")
            else: add_table(table, tag="CUMSUM")

        elif intent == "rolling_mean":
            time_col = resolve_column(args.get("time_col"), df.columns) or args.get("time_col") or st.session_state.defaults.get("time_col")
            metric = resolve_column(args.get("metric"), df.columns) or args.get("metric")
            window = int(args.get("window", 3))
            table, err = win.rolling_mean(time_col, metric, window=window)
            if err: add_message("assistant", f"❌ {err}")
            else: add_table(table, tag=f"ROLLING MEAN w={window}")

        elif intent == "lag":
            time_col = resolve_column(args.get("time_col"), df.columns) or args.get("time_col") or st.session_state.defaults.get("time_col")
            metric = resolve_column(args.get("metric"), df.columns) or args.get("metric")
            shift = int(args.get("shift", 1))
            table, err = win.lag_lead(time_col, metric, shift=shift)
            if err: add_message("assistant", f"❌ {err}")
            else: add_table(table, tag=f"LAG {shift}")

        elif intent == "outliers":
            col = resolve_column(args.get("col"), df.columns) or args.get("col")
            z = float(args.get("z", 3.0))
            table, err = more.outliers_zscore(col, threshold=z)
            if err: add_message("assistant", f"❌ {err}")
            else: add_table(table, tag=f"OUTLIERS z>{z}")

        else:  # help / unknown
            add_message("assistant",
                "I can help with schema, top-k, aggregates (overall or by group), plots, hist/box, "
                "pivot, rolling/lag, and outliers.\n\nTry:\n"
                "• `what columns do i have`\n"
                "• `top 5 Vehicle_Type`\n"
                "• `mean of Year`\n"
                "• `average Range_km by Region`\n"
                "• `plot average Range_km by Year, split by Region`"
            )

    except Exception as e:
        add_message("assistant", f"❌ Error: {e}")

# ------------------------------------------------------------------
# Handle chat: Creative → fallback → Rules
# ------------------------------------------------------------------
if prompt:
    add_message("user", prompt)

    if st.session_state.use_creative and HAVE_LLM:
        result, narrative, err = run_router_graph(prompt)
        st.session_state["last_router_status"] = {"input": prompt, "error": (str(err) if err else None), "has_result": bool(result)}
        if err or not result:
            add_message("assistant", "🤖 (Router had trouble — switching to safe mode.)")
            intent, args = simple_router(prompt)
            handle_intent(intent, args)
        else:
            kind, payload = result["kind"], result["payload"]
            if kind == "text":
                add_message("assistant", payload)
            elif kind == "table":
                add_table(payload)
            elif kind == "table+image":
                add_table(payload["table"])
                add_image(payload["path"], caption=payload["path"])
            elif kind == "image":
                add_image(payload["path"], caption=payload.get("caption") or payload["path"])
            if narrative:
                add_message("assistant", narrative)
    else:
        intent, args = simple_router(prompt)
        st.session_state["last_router_status"] = {"input": prompt, "router": "rules", "intent": intent, "args": args}
        handle_intent(intent, args)



render_messages()