from __future__ import annotations

from typing import Any

import os
import sqlite3
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

from ..schemas import State
from ..logging import get_logger
from . import nodes as n


def get_thread_config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def build_compiled_graph(*, rdb, vs, router, retriever, score_updater, llm, llm_low_temp, llm_review, llm_overall, logger=None):
    log = logger or get_logger("agent.flow")
    builder = StateGraph(State)

    # Nodes without DI
    builder.add_node("gate_is_new_user", n.gate_is_new_user)
    builder.add_node("ask_domain", n.node_ask_domain)
    builder.add_node("get_domain", n.node_get_domain)
    builder.add_node("ask_resume", n.node_ask_resume)
    builder.add_node("get_resume_url", n.node_get_resume_url)

    # Nodes with DI (closures)
    builder.add_node("node_resume_ETL", n.make_node_resume_ETL(rdb=rdb, vs=vs))
    builder.add_node("update_profile", n.make_update_profile(rdb=rdb, vs=vs))

    builder.add_node("query_question", n.make_query_question(router=router, retriever=retriever))
    builder.add_node("ask_question", n.ask_question)
    builder.add_node("get_answer", n.get_answer)

    builder.add_node("validate_technical_accuracy", n.make_validate_metric("technical_accuracy"))
    builder.add_node("validate_reasoning_depth", n.make_validate_metric("reasoning_depth"))
    builder.add_node("validate_communication_clarity", n.make_validate_metric("communication_clarity"))
    builder.add_node("metrics_barrier", n.metrics_barrier)

    builder.add_node("aggregate_feedback", n.make_aggregate_feedback(llm_overall=llm_overall))

    builder.add_node("update_profile_with_score", n.make_update_profile_with_score(score_updater=score_updater, rdb=rdb, vs=vs))
    builder.add_node("update_strength_weakness", n.make_update_strength_weakness(rdb=rdb, vs=vs))
    builder.add_node("gate_should_continue", n.gate_should_continue)

    # Edges
    builder.add_edge(START, "gate_is_new_user")
    builder.add_conditional_edges(
        "gate_is_new_user",
        n.route_is_new_user,
        {"new_user": "init_user", "existing_user": "query_question"},
    )

    # New user init node (simple helper in routing module)
    builder.add_node("init_user", n.make_init_user(rdb=rdb, vs=vs))
    builder.add_edge("init_user", "ask_domain")

    builder.add_edge("ask_domain", "get_domain")
    builder.add_edge("get_domain", "ask_resume")
    builder.add_edge("ask_resume", "get_resume_url")

    builder.add_conditional_edges(
        "get_resume_url",
        n.is_resume_url,
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
        n.metrics_barrier_decider,
        {"go": "aggregate_feedback", "wait": END},
    )

    # Save per-turn history (question, answer, metrics) before updating profile snapshots
    builder.add_node("save_history", n.make_save_history(rdb=rdb))
    builder.add_edge("aggregate_feedback", "save_history")
    builder.add_edge("save_history", "update_strength_weakness")
    builder.add_edge("update_strength_weakness", "update_profile_with_score")

    builder.add_edge("update_profile_with_score", "gate_should_continue")
    builder.add_conditional_edges(
        "gate_should_continue",
        n.route_should_continue,
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
