from __future__ import annotations

from typing import Any

import os
import sqlite3
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

from ..schemas.state import State
from ..logging import get_logger
from .nodes import routing as n_routing
from .nodes import intake as n_intake
from .nodes import resume_profile as n_profile
from .nodes import interview as n_interview
from .nodes import metrics as n_metrics
from .nodes import persistence as n_persist


def get_thread_config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def build_compiled_graph(*, rdb, vs, router, retriever, score_updater, llm, llm_low_temp, llm_review, llm_overall, logger=None):
    log = logger or get_logger("agent.flow")
    builder = StateGraph(State)

    # Nodes without DI
    builder.add_node("gate_is_new_user", n_routing.gate_is_new_user)
    builder.add_node("ask_domain", n_intake.node_ask_domain)
    builder.add_node("get_domain", n_intake.node_get_domain)
    builder.add_node("ask_resume", n_intake.node_ask_resume)
    builder.add_node("get_resume_url", n_intake.node_get_resume_url)

    # Nodes with DI (closures)
    builder.add_node("node_resume_ETL", n_profile.make_node_resume_ETL(rdb=rdb, vs=vs))
    builder.add_node("update_profile", n_profile.make_update_profile(rdb=rdb, vs=vs))

    builder.add_node("query_question", n_interview.make_query_question(router=router, retriever=retriever))
    builder.add_node("ask_question", n_interview.ask_question)
    builder.add_node("get_answer", n_interview.get_answer)

    builder.add_node("validate_technical_accuracy", n_metrics.make_validate_metric("technical_accuracy"))
    builder.add_node("validate_reasoning_depth", n_metrics.make_validate_metric("reasoning_depth"))
    builder.add_node("validate_communication_clarity", n_metrics.make_validate_metric("communication_clarity"))
    builder.add_node("metrics_barrier", n_metrics.metrics_barrier)

    builder.add_node("aggregate_feedback", n_metrics.make_aggregate_feedback(llm_overall=llm_overall))

    builder.add_node("update_profile_with_score", n_persist.make_update_profile_with_score(score_updater=score_updater, rdb=rdb, vs=vs))
    builder.add_node("update_strength_weakness", n_persist.make_update_strength_weakness(rdb=rdb, vs=vs))
    builder.add_node("gate_should_continue", n_routing.gate_should_continue)

    # Edges
    builder.add_edge(START, "gate_is_new_user")
    builder.add_conditional_edges(
        "gate_is_new_user",
        n_routing.route_is_new_user,
        {"new_user": "init_user", "existing_user": "query_question"},
    )

    # New user init node (simple helper in routing module)
    builder.add_node("init_user", n_routing.make_init_user(rdb=rdb, vs=vs))
    builder.add_edge("init_user", "ask_domain")

    builder.add_edge("ask_domain", "get_domain")
    builder.add_edge("get_domain", "ask_resume")
    builder.add_edge("ask_resume", "get_resume_url")

    builder.add_conditional_edges(
        "get_resume_url",
        n_intake.is_resume_url,
        {"resume_is_present": "node_resume_ETL", "resume_is_not_present": "update_profile"},
    )

    builder.add_edge("node_resume_ETL", "update_profile")
    builder.add_edge("update_profile", "query_question")
    builder.add_edge("query_question", "ask_question")
    builder.add_edge("ask_question", "get_answer")

    builder.add_edge("get_answer", "validate_technical_accuracy")
    builder.add_edge("get_answer", "validate_reasoning_depth")
    builder.add_edge("get_answer", "validate_communication_clarity")

    builder.add_edge("validate_technical_accuracy", "metrics_barrier")
    builder.add_edge("validate_reasoning_depth", "metrics_barrier")
    builder.add_edge("validate_communication_clarity", "metrics_barrier")

    builder.add_conditional_edges(
        "metrics_barrier",
        n_metrics.metrics_barrier_decider,
        {"go": "aggregate_feedback", "wait": END},
    )

    # Save per-turn history (question, answer, metrics) before updating profile snapshots
    builder.add_node("save_history", n_persist.make_save_history(rdb=rdb))
    builder.add_edge("aggregate_feedback", "save_history")
    builder.add_edge("save_history", "update_strength_weakness")
    builder.add_edge("update_strength_weakness", "update_profile_with_score")

    builder.add_edge("update_profile_with_score", "gate_should_continue")
    builder.add_conditional_edges(
        "gate_should_continue",
        n_routing.route_should_continue,
        {"continue": "query_question", "exit": END},
    )

    # Prefer persistent checkpointing across requests/process restarts.
    # Configure path via LANGRAPH_CHECKPOINT_PATH, default to local sqlite file.
    cp_path = os.getenv("LANGRAPH_CHECKPOINT_PATH") or os.path.join(os.getcwd(), "langgraph_checkpoints.sqlite")
    try:
        # Create a persistent SQLite connection and wrap it in SqliteSaver
        conn = sqlite3.connect(cp_path, check_same_thread=False)
        checkpointer = SqliteSaver(conn)
    except Exception:
        # Fallback to in-memory if sqlite is unavailable
        checkpointer = InMemorySaver()
    return builder.compile(checkpointer=checkpointer)
