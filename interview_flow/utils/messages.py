from __future__ import annotations

from typing import Dict, Any, List


def last_user_text(state: Dict[str, Any]) -> str:
    msgs = state.get("messages") or []
    for m in reversed(msgs):
        if m.get("role") == "user":
            return (m.get("content") or "").strip()
    return (state.get("input_text") or "").strip()


def merge_list_preserving_order(a: List[str] | None, b: List[str] | None) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in (a or []):
        k = (x or "").strip().lower()
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(x)
    for x in (b or []):
        k = (x or "").strip().lower()
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(x)
    return out

