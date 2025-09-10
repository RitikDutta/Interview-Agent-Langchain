from __future__ import annotations

from typing import Dict, Any, Literal

from ...logging import get_logger
from ...scoring.metrics import _eval_metric_with_llm, _overall_with_llm

logger = get_logger("agent.flow")


def make_validate_metric(metric_name: str):
    def _fn(state: Dict[str, Any]) -> Dict[str, Any]:
        q = state.get("question", "")
        a = state.get("answer", "")
        logger.thinking("Evaluating %s for question len=%d answer len=%d", metric_name, len(q), len(a))
        result = _eval_metric_with_llm(metric_name, q, a)
        key = f"metric_{metric_name}"
        return {key: result}
    return _fn


def metrics_barrier(state: Dict[str, Any]) -> Dict[str, Any]:
    logger.debug("[metrics_barrier] keys=%s", list(state.keys()))
    logger.debug("[metrics_barrier] TA=%s", state.get("metric_technical_accuracy"))
    logger.debug("[metrics_barrier] RD=%s", state.get("metric_reasoning_depth"))
    logger.debug("[metrics_barrier] CC=%s", state.get("metric_communication_clarity"))
    return {}


def metrics_barrier_decider(state: Dict[str, Any]) -> Literal["go", "wait"]:
    have_all = all(
        k in state
        for k in (
            "metric_technical_accuracy",
            "metric_reasoning_depth",
            "metric_communication_clarity",
        )
    )
    logger.thinking("Barrier check → have_all=%s", have_all)
    return "go" if have_all else "wait"


def make_aggregate_feedback(*, llm_overall):
    def aggregate_feedback(state: Dict[str, Any]) -> Dict[str, Any]:
        m_ta = state["metric_technical_accuracy"]
        m_rd = state["metric_reasoning_depth"]
        m_cc = state["metric_communication_clarity"]
        s_ta, s_rd, s_cc = int(m_ta["score"]), int(m_rd["score"]), int(m_cc["score"])
        combined = int(round((s_ta + s_rd + s_cc) / 3))
        logger.thinking("Computed scores TA=%d RD=%d CC=%d → combined=%d", s_ta, s_rd, s_cc, combined)
        overall = _overall_with_llm(m_ta, m_rd, m_cc, combined)
        strengths = list(overall.strengths or [])
        weaknesses = list(overall.weaknesses or [])
        return {
            "overall_feedback_summary": (overall.overall_feedback or "").strip(),
            "combined_score": combined,
            "strengths": strengths,
            "weaknesses": weaknesses,
        }

    return aggregate_feedback

