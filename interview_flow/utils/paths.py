from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse, unquote


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

