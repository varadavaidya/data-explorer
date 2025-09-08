# src/router.py
import re
import streamlit as st

def simple_router(user_text: str):
    txt = (user_text or "").strip()
    low = txt.lower()

    # schema / basics
    if low in {"what columns do i have", "list columns", "columns", "schema"}:
        return "schema_probe", {}

    # "top 5 <col>" -> value counts
    m = re.search(r"^\s*top\s+(\d+)\s+([A-Za-z0-9_\-\s]+)\s*$", txt, re.I)
    if m:
        return "value_counts", {"col": m.group(2).strip(), "top": int(m.group(1))}

    # "top 5 <group> by <metric>"
    m = re.search(r"top\s+(\d+)\s+([A-Za-z0-9_\-\s]+)\s+by\s+([A-Za-z0-9_\-\s]+)", txt, re.I)
    if m:
        return "qa_numeric", {"k": int(m.group(1)), "group": m.group(2).strip(), "metric": m.group(3).strip()}

    # grouped aggregation: "avg revenue by region", "minimum of marks by subject"
    m = re.search(
        r"(sum|average|avg|mean|count|min|minimum|max|maximum|median)\s+(?:of\s+)?([A-Za-z0-9_\-\s]+?)?\s*by\s+([A-Za-z0-9_\-\s]+)",
        txt, re.I
    )
    if m:
        agg = m.group(1).lower()
        metric = (m.group(2) or "").strip()
        group = m.group(3).strip()
        return "agg", {"agg": agg, "metric": metric, "group": group}

    # overall aggregation: "mean of Range_km", "sum revenue"
    m = re.search(r"(sum|average|avg|mean|count|min|minimum|max|maximum|median)\s+(?:of\s+)?([A-Za-z0-9_\-\s]+)", txt, re.I)
    if m:
        agg = m.group(1).lower()
        metric = (m.group(2) or "").strip()
        return "agg_overall", {"agg": agg, "metric": metric}

    # plot: "plot revenue by month, split by region"
    words = txt.replace(",", " ").split()
    loww = [w.lower() for w in words]
    if "plot" in loww:
        y = x = hue = None
        i = loww.index("plot")
        if i + 1 < len(words): y = words[i + 1]
        if "by" in loww:
            j = loww.index("by")
            if j + 1 < len(words): x = words[j + 1]
        if "split" in loww:
            k = loww.index("split")
            if k + 2 < len(words) and loww[k + 1] == "by": hue = words[k + 2]
        # only plot if user said "plot"
        y = y or st.session_state.defaults.get("metric")
        x = x or st.session_state.defaults.get("time_col")
        return "plot", {"x": x, "y": y, "hue": hue}

    # utilities
    if "describe" in low:
        return "describe", {}
    if "missing" in low or "null" in low or "na " in low:
        return "missing", {}
    if "correlation" in low or low.strip() == "corr":
        return "corr", {}

    # hist / box
    m = re.search(r"hist(?:ogram)?\s+of\s+([A-Za-z0-9_\-\s]+)(?:\s+bins\s+(\d+))?", txt, re.I)
    if m:
        return "hist", {"col": m.group(1).strip(), "bins": int(m.group(2)) if m.group(2) else 30}
    m = re.search(r"box(?:plot)?\s+([A-Za-z0-9_\-\s]+)(?:\s+by\s+([A-Za-z0-9_\-\s]+))?", txt, re.I)
    if m:
        return "box", {"y": m.group(1).strip(), "by": m.group(2).strip() if m.group(2) else None}

    # pivot: "pivot values revenue by region and month agg mean"
    m = re.search(
        r"pivot\s+values\s+([A-Za-z0-9_\-\s]+)\s+by\s+([A-Za-z0-9_\-\s]+)\s+and\s+([A-Za-z0-9_\-\s]+)(?:\s+agg\s+([A-Za-z0-9_]+))?",
        txt, re.I
    )
    if m:
        return "pivot", {"values": m.group(1).strip(), "index": m.group(2).strip(), "columns": m.group(3).strip(), "agg": (m.group(4) or "sum").lower()}

    # windows
    m = re.search(r"rank\s+(.+)\s+by\s+(.+)", txt, re.I)
    if m:
        return "rank_within", {"group": m.group(1).strip(), "order": m.group(2).strip()}
    m = re.search(r"cumulative\s+sum\s+of\s+(.+)\s+by\s+(.+)", txt, re.I)
    if m:
        return "cumsum", {"metric": m.group(1).strip(), "group": m.group(2).strip()}
    m = re.search(r"rolling\s+mean\s+of\s+(.+)\s+by\s+(.+)(?:\s+window\s+(\d+))?", txt, re.I)
    if m:
        return "rolling_mean", {"metric": m.group(1).strip(), "time_col": m.group(2).strip(), "window": int(m.group(3)) if m.group(3) else 3}
    m = re.search(r"lag\s+(.+)\s+by\s+(.+)(?:\s+shift\s+(\d+))?", txt, re.I)
    if m:
        return "lag", {"metric": m.group(1).strip(), "time_col": m.group(2).strip(), "shift": int(m.group(3)) if m.group(3) else 1}

    # outliers
    m = re.search(r"outliers\s+in\s+(.+)(?:\s+z\s*(\d+(\.\d+)?))?", txt, re.I)
    if m:
        return "outliers", {"col": m.group(1).strip(), "z": float(m.group(2)) if m.group(2) else 3.0}

    return "help", {}
