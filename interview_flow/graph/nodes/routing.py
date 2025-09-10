from __future__ import annotations

from typing import Dict, Any, Literal

from ...logging import get_logger

logger = get_logger("agent.flow")


def gate_is_new_user(state: Dict[str, Any]) -> dict:
    logger.info("System: Checking if user is new or existing")
    return {}


def route_is_new_user(state: Dict[str, Any]) -> Literal["new_user", "existing_user"]:
    if "is_new_user" in state:
        flag = bool(state["is_new_user"])
        if flag:
            logger.thinking("New user flag detected; proceeding with first-time setup")
        else:
            logger.thinking("Found an existing user in database; skipping initialization")
        return "new_user" if flag else "existing_user"
    msgs = state.get("messages") or []
    first = ""
    for m in msgs:
        if m.get("role") == "user":
            first = (m.get("content") or "").lower()
            break
    newbie = any(kw in first for kw in ("start", "new user", "first time", "setup"))
    logger.thinking("Heuristic routing: %s user based on initial message", "new" if newbie else "existing")
    return "new_user" if newbie else "existing_user"


def make_init_user(*, rdb, vs):
    def init_user(state: Dict[str, Any]) -> dict:
        logger.info("System: Initializing new user")
        user_id = state.get("user_id")
        if not user_id:
            raise ValueError("User ID is required for new user initialization")

        rdb.create_tables()
        rdb.upsert_user(
            user_id=user_id,
            name=None,
            domain=None,
            skills=[],
            strengths=[],
            weaknesses=[],
        )
        rdb.ensure_academic_summary(user_id)

        summary = "New user (no resume yet). Domain: Unknown."
        vs.upsert_profile_snapshot(
            user_id=user_id,
            domain="Unknown",
            summary=summary,
            skills=[],
            strengths=[],
            weaknesses=[],
        )

        return {"graph_state": "init_user", "is_new_user": True, "messages": []}

    return init_user


def gate_should_continue(state: Dict[str, Any]) -> dict:
    logger.debug("gate_should_continue")
    return {"graph_state": "should_continue"}


def route_should_continue(state: Dict[str, Any]) -> Literal["continue", "exit"]:
    flag = (state.get("should_continue") or "").strip().lower()
    if flag in {"exit", "quit", "stop"}:
        logger.info("[route_should_continue] flag → EXIT")
        return "exit"
    msg = (state.get("input_text") or "").strip().lower()
    if any(w in msg for w in ("exit", "quit", "stop")):
        logger.info(f"[route_should_continue] message '{msg}' → EXIT")
        return "exit"
    logger.info("[route_should_continue] → CONTINUE")
    return "continue"

