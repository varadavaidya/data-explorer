# src/llm/router_graph.py
from __future__ import annotations
from typing import Optional, Literal, Dict, Any, Tuple
from pydantic import BaseModel, Field, ValidationError
from langgraph.graph import StateGraph, END
from langchain_community.chat_models import ChatOllama
import os
from langchain_core.messages import SystemMessage, HumanMessage
import pandas as pd
import streamlit as st

# ---- import your existing tools
from src.tools.analytics import list_columns, top_k_by_group, aggregate_by_group
from src.tools.viz import plot_group_sum
from src.tools.more_funcs import (
    describe_numeric, missing_report, value_counts, correlation_matrix,
    histogram, boxplot, pivot_table, outliers_zscore
)
from src.tools.windows_funcs import rank_within, cumulative_sum, rolling_mean, lag_lead
from src.tools.util import resolve_column

# --------- Pydantic schemas describing allowed tool calls ----------
class CallAgg(BaseModel):
    tool: Literal["aggregate_by_group"] = "aggregate_by_group"
    agg: Literal["sum","mean","average","avg","count","min","minimum","max","maximum","median"]
    metric: Optional[str] = None
    group: Optional[str] = None

class CallTopK(BaseModel):
    tool: Literal["top_k_by_group"] = "top_k_by_group"
    k: int = 5
    group: str
    metric: str

class CallPlot(BaseModel):
    tool: Literal["plot_group_sum"] = "plot_group_sum"
    x: Optional[str] = None
    y: Optional[str] = None
    hue: Optional[str] = None
    agg: Optional[str] = "sum"

class CallValueCounts(BaseModel):
    tool: Literal["value_counts"] = "value_counts"
    col: str
    top: int = 20

class CallSimple(BaseModel):
    tool: Literal["list_columns","describe","missing","corr","hist","box","pivot","rank_within","cumsum","rolling_mean","lag","outliers","help"]
    args: Dict[str, Any] = Field(default_factory=dict)

Intent = CallAgg | CallTopK | CallPlot | CallValueCounts | CallSimple

# --------- LangGraph state ---------
class S(BaseModel):
    query: str
    intent: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None  # {"kind": "table|image|text", "payload": ...}
    narrative: Optional[str] = None

# --------- LLMs ---------
MODEL = os.getenv("OLLAMA_MODEL", "llama3")  # change default if you pulled a different tag
router_llm = ChatOllama(model=MODEL, temperature=0.2)
narrator_llm = ChatOllama(model=MODEL, temperature=0.7)


SYSTEM_ROUTE = (
    "You are a routing assistant. Return ONE JSON object only, no prose. "
    "Choose one tool and arguments that best answer the user's request. "
    "Use ONLY the provided column names. If unsure, use {\"tool\":\"help\",\"args\":{}}."
)

def _route_prompt(query: str, columns: list[str]) -> str:
    return (
        "Available tools:\n"
        "1) list_columns\n"
        "2) top_k_by_group {k:int, group:str, metric:str}\n"
        "3) aggregate_by_group {agg:[sum,mean,average,avg,count,min,minimum,max,maximum,median], metric?:str, group?:str}\n"
        "4) plot_group_sum {x?:str, y?:str, hue?:str, agg?:str}\n"
        "5) value_counts {col:str, top?:int}\n"
        "6) describe | missing | corr | hist | box | pivot | rank_within | cumsum | rolling_mean | lag | outliers | help\n\n"
        f"Columns: {columns}\n\n"
        f"User: {query}\n"
        "Return ONLY JSON for ONE tool."
    )

def node_route(state: S) -> S:
    cols = list(st.session_state.df.columns) if st.session_state.df is not None else []
    msgs = [
        SystemMessage(SYSTEM_ROUTE),
        HumanMessage(_route_prompt(state.query, cols)),
    ]
    try:
        out = router_llm.invoke(msgs).content.strip()
    except Exception as e:
        # Graceful failure (don’t crash the app)
        state.error = f"LLM router unavailable: {e}"
        return state

    # strip possible code fences
    if out.startswith("```"):
        out = out.strip("`").replace("json", "", 1).strip()

    import json
    try:
        data = json.loads(out)
        intent = Intent.model_validate(data).model_dump()
        state.intent = intent
    except Exception as e:
        state.error = f"Router parse/validation failed: {e}"
    return state

# normalize agg synonyms
def _norm_agg(agg: str) -> str:
    return {
        "avg": "mean",
        "average": "mean",
        "minimum": "min",
        "maximum": "max",
    }.get(agg, agg)

# Execute your existing tools; always apply resolve_column before using columns
def node_execute(state: S) -> S:
    if state.error or not state.intent:
        return state
    intent = state.intent
    df = st.session_state.df
    if df is None:
        state.error = "No dataset loaded."
        return state

    tool = intent.get("tool")
    args = intent.get("args", {}) if "args" in intent else {k:v for k,v in intent.items() if k != "tool"}
    kind = "text"
    payload = None

    try:
        if tool == "list_columns":
            payload = list_columns()
            state.result = {"kind":"text","payload":payload}

        elif tool == "top_k_by_group":
            g = resolve_column(args["group"], df.columns)
            m = resolve_column(args["metric"], df.columns)
            table, err = top_k_by_group(args.get("k",5), g or args["group"], m or args["metric"])
            if err: raise ValueError(err)
            state.result = {"kind":"table","payload":table}

        elif tool == "aggregate_by_group":
            agg = _norm_agg(args.get("agg","mean"))
            group = args.get("group")
            metric = args.get("metric","")
            if group:
                g = resolve_column(group, df.columns) or group
                table, err = aggregate_by_group(g, metric, agg=agg)
                if err: raise ValueError(err)
                state.result = {"kind":"table","payload":table}
            else:
                # overall aggregation
                col = resolve_column(metric, df.columns) or metric
                if col not in df.columns: raise ValueError(f"Column `{metric}` not found.")
                series = pd.to_numeric(df[col], errors="coerce")
                funcs = {
                    "mean": series.mean, "sum": series.sum, "count": series.count,
                    "min": series.min, "max": series.max, "median": series.median
                }
                val = funcs[agg]() if agg in funcs else None
                if val is None: raise ValueError(f"Unsupported agg `{agg}`.")
                t = pd.DataFrame({"metric":[col], agg:[val]})
                state.result = {"kind":"table","payload":t}

        elif tool == "plot_group_sum":
            x = args.get("x")
            y = args.get("y")
            hue = args.get("hue")
            path, err = plot_group_sum(x, y, hue=hue, fname_prefix="graph")
            if err: raise ValueError(err)
            state.result = {"kind":"image","payload":{"path": path, "caption": path}}

        elif tool == "value_counts":
            c = args["col"]
            table, err = value_counts(c, top=args.get("top",20))
            if err: raise ValueError(err)
            state.result = {"kind":"table","payload":table}

        elif tool == "describe":
            table, err = describe_numeric()
            if err: raise ValueError(err)
            state.result = {"kind":"table","payload":table}

        elif tool == "missing":
            table, err = missing_report()
            if err: raise ValueError(err)
            state.result = {"kind":"table","payload":table}

        elif tool == "corr":
            table, path, err = correlation_matrix()
            if err: raise ValueError(err)
            state.result = {"kind":"table+image","payload":{"table":table, "path":path}}

        elif tool == "hist":
            col = args.get("col") or args.get("column")
            bins = int(args.get("bins",30))
            _, err, path = histogram(col, bins=bins)
            if err: raise ValueError(err)
            state.result = {"kind":"image","payload":{"path":path, "caption":path}}

        elif tool == "box":
            y = args.get("y"); by = args.get("by")
            _, err, path = boxplot(y, by=by)
            if err: raise ValueError(err)
            state.result = {"kind":"image","payload":{"path":path, "caption":path}}

        elif tool == "pivot":
            table, err = pivot_table(args["index"], args["columns"], args["values"], agg=args.get("agg","sum"))
            if err: raise ValueError(err)
            state.result = {"kind":"table","payload":table}

        elif tool == "rank_within":
            table, err = rank_within(args["group"], args["order"])
            if err: raise ValueError(err)
            state.result = {"kind":"table","payload":table}

        elif tool == "cumsum":
            table, err = cumulative_sum(args["group"], args["metric"])
            if err: raise ValueError(err)
            state.result = {"kind":"table","payload":table}

        elif tool == "rolling_mean":
            table, err = rolling_mean(args["time_col"], args["metric"], window=int(args.get("window",3)))
            if err: raise ValueError(err)
            state.result = {"kind":"table","payload":table}

        elif tool == "lag":
            table, err = lag_lead(args["time_col"], args["metric"], shift=int(args.get("shift",1)))
            if err: raise ValueError(err)
            state.result = {"kind":"table","payload":table}

        elif tool == "outliers":
            table, err = outliers_zscore(args["col"], threshold=float(args.get("z",3.0)))
            if err: raise ValueError(err)
            state.result = {"kind":"table","payload":table}

        elif tool == "help":
            state.result = {"kind":"text","payload":"I can help with schema, top-k, aggregates, plots, hist/box, pivot, rolling/lag, outliers. Try: “top 5 Vehicle_Type”, “mean of Range_km”, “plot Range_km by Year, split by Region”."}

        else:
            state.error = f"Unknown tool: {tool}"

    except Exception as e:
        state.error = f"Execution error: {e}"

    return state

def node_narrate(state: S) -> S:
    if not state.result or state.error:
        return state
    # short, creative but faithful summary
    brief = "Return a short 1-2 sentence summary of these results without inventing columns."
    desc = ""
    kind = state.result["kind"]
    if kind.startswith("table"):
        df = state.result["payload"] if kind=="table" else state.result["payload"]["table"]
        try:
            head = df.head(5).to_markdown(index=False)
        except Exception:
            head = str(df.head(5))
        desc = f"Table preview:\n{head}"
    elif kind == "image":
        desc = "A chart image was generated."
    else:
        desc = str(state.result.get("payload",""))

    msgs = [
        SystemMessage("You are a helpful data analyst."),
        HumanMessage(f"{brief}\n\n{desc}")
    ]
    state.narrative = narrator_llm.invoke(msgs).content.strip()
    return state

# --------- Build the graph ---------
graph = StateGraph(S)
graph.add_node("route", node_route)
graph.add_node("execute", node_execute)
graph.add_node("narrate", node_narrate)

graph.set_entry_point("route")
graph.add_edge("route","execute")
graph.add_edge("execute","narrate")
graph.add_edge("narrate", END)

app = graph.compile()

def run_router_graph(user_query: str) -> Tuple[Optional[Dict[str,Any]], Optional[str], Optional[str]]:
    """Returns (result, narrative, error)."""
    state = S(query=user_query)
    out = app.invoke(state)
    return out.result, out.narrative, out.error
