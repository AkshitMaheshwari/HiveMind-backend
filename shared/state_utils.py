"""
Shared State Reducers for LangGraph Graphs.
Prevents INVALID_CONCURRENT_GRAPH_UPDATE errors when parallel nodes execute.
"""
from typing import Any, Dict, List, Optional


def merge_events(left: Optional[List[Dict[str, Any]]], right: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Merge two event lists without duplicates."""
    if not left:
        return list(right) if right else []
    if not right:
        return list(left) if left else []

    seen = set()
    combined = []
    for ev in left + right:
        if not isinstance(ev, dict):
            continue
        key = (ev.get("event"), ev.get("department"), ev.get("agent"), ev.get("data"), ev.get("timestamp"))
        if key not in seen:
            seen.add(key)
            combined.append(ev)
    return combined


def merge_dicts(left: Optional[Dict[str, Any]], right: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge two dictionaries."""
    res = dict(left or {})
    if right and isinstance(right, dict):
        res.update(right)
    return res


def merge_lists(left: Optional[List[Any]], right: Optional[List[Any]]) -> List[Any]:
    """Merge two lists preserving uniqueness and order."""
    res = list(left or [])
    for item in right or []:
        if item not in res:
            res.append(item)
    return res
