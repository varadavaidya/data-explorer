# src/tools/util.py
from __future__ import annotations
import re
from difflib import get_close_matches
from typing import Iterable, Optional

def _canon(s: str) -> str:
    """Lowercase & strip non-alphanumerics so 'Vehicle_ID', 'vehicle id', 'VEHICLE-ID' look the same."""
    return re.sub(r"[^a-z0-9]", "", s.lower())

def resolve_column(user_col: str, columns: Iterable[str]) -> Optional[str]:
    """
    Map a user-provided column token to a real column.
    - Case-insensitive
    - Ignores spaces/underscores/dashes
    - Fuzzy match for small typos
    Returns the exact column name from `columns`, or None if nothing close.
    """
    cols = list(columns)
    canon_map = {_canon(c): c for c in cols}
    key = _canon(user_col)

    # 1) exact canonical match
    if key in canon_map:
        return canon_map[key]

    # 2) startswith among canonical keys (good for partials like 'veh')
    starters = [canon_map[k] for k in canon_map if k.startswith(key)]
    if starters:
        return starters[0]

    # 3) fuzzy best match (handles small typos)
    best = get_close_matches(key, list(canon_map.keys()), n=1, cutoff=0.72)
    if best:
        return canon_map[best[0]]

    return None
