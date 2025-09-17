from __future__ import annotations

from typing import Dict, Any, Literal
from pathlib import Path

from langgraph.types import interrupt

from ...logging import get_logger
from ...utils.messages import last_user_text
from ...utils.paths import _normalize_resume_input

logger = get_logger("agent.flow")


def _sanitize_label(s: str) -> str:
    try:
        import re
        s = (s or "").strip().lower()
        s = re.sub(r"[^a-z0-9]+", "_", s)
        s = re.sub(r"_+", "_", s).strip("_")
        return s or "misc"
    except Exception:
        return (s or "misc").strip().lower() or "misc"


def node_ask_domain(state: Dict[str, Any]) -> dict:
    logger.debug("node_ask_domain")
    return {"graph_state": "ask_domain"}


def node_get_domain(state: Dict[str, Any]) -> dict:
    user_id = state.get("user_id")
    logger.debug("node get_domain")
    domain_value = interrupt({
        "prompt": "Please provide your domain of interest (e.g., Data Science, AI, Management, Etc). "
    })
    if isinstance(domain_value, dict):
        domain = domain_value.get("domain") or domain_value.get("content") or ""
    else:
        domain = str(domain_value or "")
    domain = domain.strip()
    if domain.lower() in {"skip", "no", "later", ""}:
        logger.info("[get_domain] user skipped providing domain")
        return {"graph_state": "no_domain"}
    logger.info(f"[domain] Received from user user_id={user_id}: '{domain}'")

    # Try to canonicalize via vector store; fall back to sanitized snake_case
    canonical = None
    try:
        from ...normalization.canonical_normalizer import search_domains_first
        hit = search_domains_first(domain)
        canonical = (hit or {}).get("canonical") if hit else None
        if canonical:
            logger.info(f"[domain] Canonicalized name is '{canonical}'")
        else:
            fallback = _sanitize_label(domain)
            logger.info(f"[domain] Using default sanitized form (not found in database): '{fallback}'")
            canonical = fallback
    except Exception as e:
        fallback = _sanitize_label(domain)
        logger.warning(f"[domain] Canonicalization error ({e}); using sanitized fallback '{fallback}'")
        canonical = fallback

    logger.thinking("Captured domain preference (normalized): %s", canonical)
    return {"graph_state": "have_domain", "domain": canonical}


def node_ask_resume(state: Dict[str, Any]) -> dict:
    logger.debug("node_ask_resume")
    return {"graph_state": "ask_resume"}


def node_get_resume_url(state: Dict[str, Any]) -> dict:
    logger.debug("node get_resume_url")
    text = last_user_text(state)
    resume_value = text if text else interrupt({
        "prompt": (
            "Please share your resume — you can paste a URL (http/https), a file:// URL, "
            "or a local path (absolute/relative, ~ ok). Type 'skip' to continue without a resume."
        )
    })
    try:
        kind, normalized = _normalize_resume_input(resume_value)
        logger.thinking("Parsed resume source: kind=%s (len=%d)", kind, len(normalized or ""))
    except Exception as e:
        logger.warning(f"[get_resume_url] normalization error: {e} → continuing without resume")
        return {"graph_state": "no_resume_url"}

    if kind == "none" or not normalized:
        logger.info("[get_resume_url] user skipped providing a resume")
        return {"graph_state": "no_resume_url"}

    if kind in {"file", "local"}:
        p = Path(normalized)
        if not p.exists():
            logger.warning(f"[get_resume_url] local path not found: {p}")
            return {"graph_state": "no_resume_url"}
    logger.info(f"[get_resume_url] source kind={kind} value={normalized}")
    return {"graph_state": "have_resume_url", "resume_url": normalized}


def is_resume_url(state: Dict[str, Any]) -> Literal["resume_is_present", "resume_is_not_present"]:
    has_source = bool((state.get("resume_url") or "").strip())
    return "resume_is_present" if has_source else "resume_is_not_present"
