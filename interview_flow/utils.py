from __future__ import annotations

from typing import Dict, Any, List
from pathlib import Path
from urllib.parse import urlparse, unquote

# --- constants.py ---
DEFAULT_Q = "Walk me through one ML project you’ve done end‑to‑end."


# --- messages.py ---
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


# --- paths.py ---
def _normalize_resume_input(value) -> tuple[str, str]:
    """
    Normalize user-provided resume input into (kind, normalized_string).
    kind ∈ {"http", "file", "local", "none"}.
    """
    if isinstance(value, dict):
        s = (value.get("url") or value.get("path") or value.get("content") or "").strip()
    else:
        s = str(value or "").strip()

    if s.lower() in {"skip", "no", "later", ""}:
        return ("none", "")

    # http(s) URL
    if s.lower().startswith(("http://", "https://")):
        return ("http", s)

    # file:// URL
    if s.lower().startswith("file://"):
        u = urlparse(s)
        p = Path(unquote(u.path)).expanduser()
        return ("file", str(p))

    p = Path(s).expanduser()
    return ("local", str(p))
