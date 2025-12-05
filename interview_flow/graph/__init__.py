from __future__ import annotations

from typing import Any, Dict, Optional

from langgraph.types import Command

from ..logging import get_logger
from ..agents import llm, llm_low_temp, llm_review, llm_overall
from .wiring import build_compiled_graph, get_thread_config


logger = get_logger("agent.flow")

# Lazy singletons to avoid import-time env failures
_graph = None
_rdb = None
_vs = None
_router = None
_retriever = None
_score_updater = None


def _ensure_built():
    global _graph, _rdb, _vs, _router, _retriever, _score_updater
    if _graph is not None:
        return
    # Import heavy deps lazily to avoid import-time network calls via transitive imports
    from ..agents import llm, llm_low_temp, llm_review, llm_overall
    from ..infra.rdb import RelationalDB
    from ..infra.vector_store import VectorStore
    from ..retrieval import StrategicQuestionRouter, QuestionSearch
    from ..scoring import ScoreUpdater
    from ..settings import QUESTIONS_NAMESPACE

    _rdb = RelationalDB()
    _vs = VectorStore()
    _router = StrategicQuestionRouter(rdb=_rdb, vs=_vs, questions_namespace=QUESTIONS_NAMESPACE)
    _retriever = QuestionSearch(namespace=QUESTIONS_NAMESPACE)
    _score_updater = ScoreUpdater(_rdb)
    _graph = build_compiled_graph(
        rdb=_rdb,
        vs=_vs,
        router=_router,
        retriever=_retriever,
        score_updater=_score_updater,
        llm=llm,
        llm_low_temp=llm_low_temp,
        llm_review=llm_review,
        llm_overall=llm_overall,
        logger=logger,
    )


def get_graph():
    _ensure_built()
    return _graph  # type: ignore


def get_current_state(thread_id: str) -> Dict[str, Any]:
    _ensure_built()
    snap = _graph.get_state(get_thread_config(thread_id))  # type: ignore
    try:
        return dict(snap.values or {})
    except Exception:
        return dict(snap or {})


def start_thread(thread_id: str, user_id: str, strategy: Optional[str] = None) -> Dict[str, Any]:
    _ensure_built()
    exists = False
    try:
        exists = _rdb.user_exists_rdb(user_id=user_id)  # type: ignore
    except Exception:
        pass
    init_state: Dict[str, Any] = {
        "user_id": user_id,
        "is_new_user": (not exists),
        "messages": [{"role": "user", "content": "start"}],
    }
    if strategy:
        init_state["strategy"] = strategy
    out = _graph.invoke(init_state, config=get_thread_config(thread_id))  # type: ignore
    # Persist the mapping between user and thread in the relational DB
    try:
        _rdb.set_user_thread_id(user_id=user_id, thread_id=thread_id)  # type: ignore
    except Exception as e:
        logger.warning(f"Could not persist thread_id for user_id={user_id}: {e}")
    return out if isinstance(out, dict) else {}


def resume_thread(thread_id: str, value: Any) -> Dict[str, Any]:
    _ensure_built()
    out = _graph.invoke(Command(resume=value), config=get_thread_config(thread_id))  # type: ignore
    return out if isinstance(out, dict) else {}
