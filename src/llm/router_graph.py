from __future__ import annotations
from typing import Optional, Literal, Dict, Any, Union, List
import os, json
import pandas as pd
import streamlit as st

from pydantic import BaseModel, Field, TypeAdapter, ValidationError
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.chat_models import ChatOllama

# ---- tool imports
from src.tools.analytics import list_columns, top_k_by_group, aggregate_by_group
from src.tools.viz import plot_group_sum
from src.tools.more_funcs import (
    describe_numeric, missing_report, value_counts, correlation_matrix,
    histogram, boxplot, pivot_table, outliers_zscore
)
from src.tools.windows_funcs import rank_within, cumulative_sum, rolling_mean, lag_lead
from src.tools.util import resolve_column

# ---------------- Models / Schemas ----------------
class CallAgg(BaseModel):
    tool: Literal["aggregate_by_group"]
    agg: str
    metric: Optional[str] = None
    group: Optional[str] = None

class CallTopK(BaseModel):
    tool: Literal["top_k_by_group"]
    k: int = 5
    group: str
    metric: str

class CallPlot(BaseModel):
    tool: Literal["plot_group_sum"]
    x: Optional[str] = None
    y: Optional[str] = None
    hue: Optional[str] = None
    agg: Optional[str] = "sum"

class CallValueCounts(BaseModel):
    tool: Literal["value_counts"]
    col: str
    top: int = 20

class CallSimple(BaseModel):
    tool: Literal[
        "list_columns","describe","missing","corr","hist","box","pivot",
        "rank_within","cumsum","rolling_mean","lag","outliers","help"]
    args: Dict[str, Any] = Field(default_factory=dict)

Intent = Union[CallAgg, CallTopK, CallPlot, CallValueCounts, CallSimple]
INTENT_ADAPTER = TypeAdapter(Intent)

class S(BaseModel):
    query: str
    intent: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    narrative: Optional[str] = None

# ---------------- Prompt setup ----------------
def build_tool_system_prompt(cols: list[str]) -> str:
    col_list = ", ".join(map(str, cols)) if len(cols) > 0 else "No dataset loaded yet"
    return f"""
You are Data Explorer's reasoning engine.

You have access to the following tools, with their expected arguments shown in parentheses:

- list_columns()
- top_k_by_group(k, group, metric)
- aggregate_by_group(group, metric, agg)
- plot_group_sum(x, y, hue)
- describe_numeric()
- missing_report()
- value_counts(col, top)
- correlation_matrix()
- histogram(col)
- boxplot(col)
- pivot_table(index, columns, values, aggfunc)
- outliers_zscore(col)
- rank_within(group, metric)
- cumulative_sum(col)
- rolling_mean(col, window)
- lag_lead(col, n)

Your job is to:
1. Analyze the user's query.
2. Select the most appropriate tool.
3. Map the query to the correct arguments using the **current dataset columns** (which may include spelling variations).
4. Return a **valid JSON object**.

---

You MUST return a JSON object with two fields:
- `"tool"`: the name of the function to call (string)
- `"args"`: dictionary of arguments

❗ The output MUST be:
- A single valid JSON object
-do not miss the comma inbetween tools and args
- Not wrapped in triple backticks (no ```json)
- Not followed or preceded by **any** explanations or text
- Free of markdown, natural language, or code formatting

✅ Example valid output:
{{ "tool": "list_columns", "args": {{}} }}

---

Current dataset columns:
{col_list}
""".strip()



# ---------------- LLM setup ----------------
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableMap

MODEL = os.getenv("OLLAMA_MODEL", "llama3")

system_prompt = build_tool_system_prompt(
    st.session_state.get("df", pd.DataFrame()).columns
)

# Build prompt template
router_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("user", "{input}")
])

# Compose final LLM + prompt chain
router_llm = router_prompt | ChatOllama(model=MODEL, temperature=0.2)

# Narrator LLM (can stay as-is)
narrator_llm = ChatOllama(model=MODEL, temperature=0.7)


# ---------------- Helpers ----------------
def repair_intent(data: dict, query: str) -> dict:
    tool = data.get("tool")
    q = query.lower()

    def norm(s):
        return {
            "avg": "mean", "average": "mean", "minimum": "min", "maximum": "max"
        }.get(s.lower(), s.lower())

    def infer_agg():
        if any(x in q for x in ["avg", "average", "mean"]): return "mean"
        if any(x in q for x in ["sum", "total"]): return "sum"
        if any(x in q for x in ["min", "minimum"]): return "min"
        if any(x in q for x in ["max", "maximum"]): return "max"
        if "median" in q: return "median"
        if "count" in q: return "count"
        return "mean"

    if tool == "aggregate_by_group":
        data["agg"] = norm(data.get("agg") or infer_agg())
    elif tool == "plot_group_sum":
        data["agg"] = norm(data.get("agg") or infer_agg())
    elif tool == "top_k_by_group" and not data.get("metric"):
        return {"tool": "value_counts", "col": data.get("group"), "top": data.get("k", 5)}
    elif tool == "value_counts" and not data.get("col"):
        df = st.session_state.get("df")
        if df is not None:
            for c in df.columns:
                if c.lower() in q:
                    data["col"] = c; break
    return data

# ---------------- Nodes ----------------
def node_route(state: S) -> S:
    q = state.query.strip().lower()
    if q in {"hi", "hello", "hey"}:
        state.result = {"kind": "text", "payload": "👋 Hi! Upload a CSV and ask me about your data."}
        return state

    cols = list(st.session_state.get("df", pd.DataFrame()).columns)
    msg = ChatPromptTemplate.from_messages([
        ("system", build_tool_system_prompt(cols)),
        ("user", "{input}")
    ]).format_messages(input=state.query)

    try:
        print("📥 LLM Input Messages:")
        for m in msg:
            print(m)

        response = router_llm.invoke(msg)

        # Parse JSON output from LLM
        out = response.content.strip()
        print("🧠 Raw LLM output (before parse):", repr(out)) 

        if out.startswith("```"):  # Remove Markdown wrappers if present
            out = out.strip("`").replace("json", "").strip()

        data = json.loads(out)
        intent = INTENT_ADAPTER.validate_python(repair_intent(data, state.query))
        state.intent = intent.model_dump()
    except ValidationError as ve:
        state.error = f"❌ JSON validation failed: {ve}"
    except json.JSONDecodeError as je:
        state.error = f"❌ JSON parse error: {je}\nLLM output was:\n{out}"
    except Exception as e:
        state.error = f"❌ Router error: {e}\n\n🔍 LLM output:\n{out}"
    return state


def node_execute(state: S) -> S:
    if state.error or not state.intent: return state
    df = st.session_state.get("df")
    if df is None:
        state.error = "No dataset loaded."
        return state

    tool = state.intent["tool"]
    args = state.intent.get("args", {})
    try:
        if tool == "list_columns":
            state.result = {"kind": "text", "payload": list_columns()}
        elif tool == "top_k_by_group":
            g, m = resolve_column(args["group"], df.columns), resolve_column(args["metric"], df.columns)
            table, err = top_k_by_group(args["k"], g, m)
            if err: raise ValueError(err)
            state.result = {"kind": "table", "payload": table}
        elif tool == "aggregate_by_group":
            g, m = args.get("group"), args.get("metric")
            agg = args.get("agg", "mean")
            if g:
                g = resolve_column(g, df.columns)
                table, err = aggregate_by_group(g, m, agg=agg)
                if err: raise ValueError(err)
                state.result = {"kind": "table", "payload": table}
            else:
                series = pd.to_numeric(df[m], errors="coerce")
                funcs = {
                    "mean": series.mean, "sum": series.sum, "count": series.count,
                    "min": series.min, "max": series.max, "median": series.median
                }
                if agg not in funcs: raise ValueError("Unsupported agg")
                state.result = {"kind": "table", "payload": pd.DataFrame({"metric":[m], agg:[funcs[agg]()]})}
        elif tool == "plot_group_sum":
            path, err = plot_group_sum(args.get("x"), args.get("y"), hue=args.get("hue"))
            if err: raise ValueError(err)
            state.result = {"kind": "image", "payload": {"path": path}}
        elif tool == "value_counts":
            table, err = value_counts(args["col"], top=args.get("top",20))
            if err: raise ValueError(err)
            state.result = {"kind": "table", "payload": table}
        elif tool == "describe":
            table, err = describe_numeric(); state.result = {"kind": "table", "payload": table} if not err else (_:=ValueError(err))
        elif tool == "missing":
            table, err = missing_report(); state.result = {"kind": "table", "payload": table} if not err else (_:=ValueError(err))
        elif tool == "corr":
            table, path, err = correlation_matrix()
            state.result = {"kind": "table+image", "payload": {"table": table, "path": path}} if not err else (_:=ValueError(err))
        elif tool == "help":
            state.result = {"kind": "text", "payload": "Try: 'top 5 X', 'avg of Y by Z', 'plot A vs B'"}
        else:
            state.error = f"Unknown tool: {tool}"
    except Exception as e:
        state.error = f"Execution error: {e}"
    return state

def node_narrate(state: S) -> S:
    if not state.result or state.error: return state
    desc = "" if state.result["kind"] != "table" else str(state.result["payload"].head())
    try:
        out = narrator_llm.invoke([
            SystemMessage("You are a helpful analyst."),
            HumanMessage("Summarize in 1 sentence:\n" + desc)
        ])
        state.narrative = out.content.strip()
    except Exception:
        state.narrative = None
    return state




from typing import Tuple

# ---------------- Graph ----------------
graph = StateGraph(S)
graph.add_node("route", node_route)
graph.add_node("execute", node_execute)
graph.add_node("narrate", node_narrate)
graph.set_entry_point("route")
graph.add_edge("route", "execute")
graph.add_edge("execute", "narrate")
graph.add_edge("narrate", END)

runnable_graph = graph.compile()

# -------- Exported function --------
def run_router_graph(user_input: str) -> Tuple[Optional[dict], Optional[str], Optional[str]]:
    try:
        print("📥 Input to router graph:", user_input)
        state = S(query=user_input)
        output_data = runnable_graph.invoke(state)
        output = S(**output_data) 
        print("✅ Output from graph:", output)
        print("📦 Result:", output.result)
        print("📝 Narrative:", output.narrative)
        print("❗ Error:", output.error)
        return output.result, output.narrative, output.error
    except Exception as e:
        print("❌ Graph exception:", e)
        return None, None, str(e)
