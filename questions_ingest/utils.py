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

