from __future__ import annotations

from typing import Dict, Any

from ...logging import get_logger
from ...utils.constants import DEFAULT_Q
from langgraph.types import interrupt

logger = get_logger("agent.flow")


def make_query_question(*, router, retriever):
    def query_question(state: Dict[str, Any]) -> dict:
        logger.debug("node query_question")
        user_id = state.get("user_id")
        strategy = state.get("strategy")
        logger.thinking("Building question plan using strategy=%s", strategy)
        plan = router.plan(user_id, strategy=strategy)
        q = (plan.query or "").strip()
        logger.thinking("Plan topic → '%s' (diff=%s, domain=%s, skill=%s)", plan.query, plan.difficulty, plan.domain, plan.skill)

        results = retriever.query_search(q, top_k=1)
        if results and len(results) > 0:
            best = results[0]
            meta = getattr(best, "metadata", {}) or {}
            qtext = (meta.get("text") or meta.get("question_text") or "").strip()
            picked_id = getattr(best, "id", None)
            if not qtext:
                qtext = DEFAULT_Q
        else:
            meta = {}
            picked_id = None
            qtext = DEFAULT_Q
        return {
            "graph_state": "have_question",
            "question": qtext,
            "question_meta": meta,
            "picked_question_id": picked_id,
        }

    return query_question


def ask_question(state: Dict[str, Any]) -> dict:
    logger.debug("node ask_question")
    qtext = (state.get("question") or "").strip() or DEFAULT_Q
    ans_value = interrupt({"prompt": qtext})
    if isinstance(ans_value, dict):
        ans = ans_value.get("answer") or ans_value.get("content") or ""
    else:
        ans = str(ans_value or "")
    return {"graph_state": "asked_question", "answer": ans.strip()}


def get_answer(state: Dict[str, Any]) -> dict:
    logger.debug("node get_answer")
    return {"graph_state": "get_answer"}

