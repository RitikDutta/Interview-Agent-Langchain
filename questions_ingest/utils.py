from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, List, Optional, Union


def snake(s: Optional[str]) -> str:
    if not s:
        return ""
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def sanitize_label(s: Optional[str], fallback: str) -> str:
    out = snake(s)
    return out or fallback


def normalize_text_for_hash(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"^q[\s\-:_]*\d+[a-z]?\s*[).:-]?\s*", "", s)
    return s


def content_hash(text: str) -> str:
    return hashlib.sha256(normalize_text_for_hash(text).encode("utf-8")).hexdigest()


ALLOWED_SCALARS = (str, int, float, bool)


def pinecone_clean_meta(md: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in md.items():
        if v is None:
            continue
        if isinstance(v, list):
            lv: List[str] = []
            for x in v:
                if isinstance(x, ALLOWED_SCALARS):
                    sx = x if isinstance(x, str) else str(x)
                    if sx != "":
                        lv.append(sx)
            if lv:
                out[k] = lv
        elif isinstance(v, ALLOWED_SCALARS):
            out[k] = v
        else:
            out[k] = str(v)
    return out


def _as_list(v: Optional[Union[str, List[str], tuple, set]]) -> List[str]:
    if v is None:
        return []
    if isinstance(v, (list, tuple, set)):
        return [*v]
    return [v]


def _snake_list(vs) -> List[str]:
    out = [sanitize_label(x, "") for x in _as_list(vs)]
    return [x for x in out if x]


def _canon_list(vs) -> List[str]:
    out = [str(x).strip() for x in _as_list(vs)]
    return [x for x in out if x]


def _build_sparse_from_text(text: str) -> Dict[str, List[float]]:
    tokens = list({t for t in re.findall(r"[a-z0-9]+", text.lower()) if t})
    if not tokens:
        return {"indices": [], "values": []}
    return {"indices": list(range(len(tokens))), "values": [1.0] * len(tokens)}


# ----------------- Canonicalization (via vector store) -----------------
def canonicalize_domain(label: Optional[str]) -> Optional[str]:
    """Return a canonical domain label for a user-provided string.

    Tries vector-store lookup first (Pinecone index via interview_flow.normalization.canonical_normalizer),
    then falls back to sanitized snake_case if vector lookup is unavailable or returns nothing.

    Example: "Data Scientist", "data science", "DATA Science" -> "data_science"
    """
    if not label:
        return None
    raw = str(label).strip()
    if not raw:
        return None
    try:
        # Lazy import to avoid hard dependency when env is not configured
        from interview_flow.normalization.canonical_normalizer import search_domains_first
        res = search_domains_first(raw)
        if res and res.get("canonical"):
            return str(res["canonical"]).strip()
    except Exception:
        # fall back silently
        pass
    return "_" + sanitize_label(raw, "misc")


def canonicalize_skill(label: Optional[str]) -> Optional[str]:
    """Return a canonical skill label for a user-provided string.

    Uses the same vector-store canonicalizer as domains, but queries the
    skills namespace. Falls back to sanitized snake_case if unavailable.
    """
    if not label:
        return None
    raw = str(label).strip()
    if not raw:
        return None
    try:
        from interview_flow.normalization.canonical_normalizer import search_skills_first
        res = search_skills_first(raw)
        if res and res.get("canonical"):
            return str(res["canonical"]).strip()
    except Exception:
        pass
    return sanitize_label(raw, "misc")


if __name__ == "__main__":
    cano = canonicalize_domain("Data Scientist")
    print(f"Cano: {cano}")