from __future__ import annotations

from typing import Dict, Optional
from langchain_core.messages import SystemMessage, HumanMessage

from .agents import llm_review, llm_overall
from .schemas import MetricEval, OverallEval
from .logging import get_logger

# --- metrics.py ---
_RUBRICS = {
    "technical_accuracy": "factual correctness, valid methods, precise terminology, alignment to the question.",
    "reasoning_depth": "quality of reasoning, decomposition, assumptions, trade-offs, evidence and stepwise logic.",
    "communication_clarity": "structure, brevity, coherence, ordering, audience-appropriate wording.",
}


def _eval_metric_with_llm(metric_name: str, question: str, answer: str) -> dict:
    rubric = _RUBRICS.get(metric_name, "quality relative to the named metric.")
    sys = SystemMessage(content=(
        f"You are a senior interview evaluator for '{metric_name}'. "
        f"Metric focus: {rubric} "
        "Grade conservatively. Penalize errors and unsupported claims. "
        "If off-topic/empty, score 0–2. Return only schema fields."
    ))
    human = HumanMessage(content=(
        f"Question:\n{question}\n\n"
        f"Candidate answer:\n{answer}\n\n"
        "Output:\n"
        "- feedback: ≤50 words, actionable, behavior-focused.\n"
        "- score: integer 0..10."
    ))
    result: MetricEval = llm_review.invoke([sys, human])
    return result.model_dump()


def _overall_with_llm(m_ta: dict, m_rd: dict, m_cc: dict, combined: int) -> OverallEval:
    s_ta, s_rd, s_cc = int(m_ta.get("score", 0)), int(m_rd.get("score", 0)), int(m_cc.get("score", 0))
    sys = SystemMessage(content=(
        "You are a senior interview evaluator. Based strictly on the three metric feedback snippets and scores, "
        "return a structured overall evaluation. Keep overall_feedback ≤100 words, no headings or repetition. "
        "Include strengths/weaknesses ONLY if clearly evidenced; at most 0–3 concise bullet-like phrases each. "
        "Use the provided combined average score below as the combined_score (integer 0–10)."
    ))
    human = HumanMessage(content=(
        f"Technical accuracy: score={s_ta}, feedback={m_ta.get('feedback','')}\n"
        f"Reasoning depth: score={s_rd}, feedback={m_rd.get('feedback','')}\n"
        f"Communication clarity: score={s_cc}, feedback={m_cc.get('feedback','')}\n"
        f"Combined average score (given): {combined}"
    ))
    return llm_overall.with_config({"temperature": 0.2}).invoke([sys, human])


# --- score_updater.py ---
logger = get_logger("scores")


class ScoreUpdater:
    def __init__(self, rdb, default_alpha: float = 0.2, default_hybrid_weight: float = 0.7):
        self.rdb = rdb
        self.default_alpha = float(default_alpha)
        self.default_hybrid_weight = float(default_hybrid_weight)

    def fetch_current(self, user_id: str) -> Dict[str, float]:
        logger.debug(f"fetch_current user_id={user_id}")
        row = self.rdb.get_user_academic_score(user_id) or {}
        return {
            "technical_accuracy": float(row.get("technical_accuracy", 0.0)),
            "reasoning_depth": float(row.get("reasoning_depth", 0.0)),
            "communication_clarity": float(row.get("communication_clarity", 0.0)),
            "score_overall": float(row.get("score_overall", 0.0)),
            "question_attempted": int(row.get("question_attempted", 0)),
        }

    def compute_updated(
        self,
        new_scores: Dict[str, float],
        current_scores: Dict[str, float],
        n_attempts: Optional[int] = None,
        *,
        alpha: Optional[float] = None,
        hybrid_weight: Optional[float] = None,
        round_ndigits: int = 2,
    ) -> Dict[str, float]:
        logger.debug("compute_updated alpha=%s hybrid_weight=%s", alpha, hybrid_weight)
        a = float(alpha if alpha is not None else self.default_alpha)
        w = float(hybrid_weight if hybrid_weight is not None else self.default_hybrid_weight)

        ta_old = float(current_scores.get("technical_accuracy", 0.0))
        rd_old = float(current_scores.get("reasoning_depth", 0.0))
        cc_old = float(current_scores.get("communication_clarity", 0.0))
        ov_old = float(current_scores.get("score_overall", 0.0))
        N = max(0, int(n_attempts if n_attempts is not None else current_scores.get("question_attempted", 0)))

        ta_new_raw = float(new_scores["technical_accuracy"])
        rd_new_raw = float(new_scores["reasoning_depth"])
        cc_new_raw = float(new_scores["communication_clarity"])

        def ema(old: float, new: float) -> float:
            return a * new + (1.0 - a) * old

        ta_new = ema(ta_old, ta_new_raw)
        rd_new = ema(rd_old, rd_new_raw)
        cc_new = ema(cc_old, cc_new_raw)

        overall_from_metrics = (ta_new + rd_new + cc_new) / 3.0
        prev_sum = ov_old * max(1, N)
        lifetime_mean = (prev_sum + overall_from_metrics) / (N + 1)
        overall_hybrid = w * overall_from_metrics + (1.0 - w) * lifetime_mean
        logger.thinking(
            "Computed EMA: TA=%.2f RD=%.2f CC=%.2f | overall_from_metrics=%.2f lifetime_mean=%.2f → hybrid=%.2f",
            ta_new, rd_new, cc_new, overall_from_metrics, lifetime_mean, overall_hybrid,
        )

        return {
            "technical_accuracy": round(ta_new, round_ndigits),
            "reasoning_depth": round(rd_new, round_ndigits),
            "communication_clarity": round(cc_new, round_ndigits),
            "score_overall": round(overall_hybrid, round_ndigits),
        }

    def update_and_save(
        self,
        user_id: str,
        new_scores: Dict[str, float],
        n_attempts_override: Optional[int] = None,
        *,
        alpha: Optional[float] = None,
        hybrid_weight: Optional[float] = None,
        round_ndigits: int = 2,
    ) -> Dict[str, float]:
        logger.debug(f"update_and_save user_id={user_id}")
        self.rdb.ensure_academic_summary(user_id)

        current = self.fetch_current(user_id)
        N = int(n_attempts_override if n_attempts_override is not None else current.get("question_attempted", 0))

        updated = self.compute_updated(
            new_scores=new_scores,
            current_scores=current,
            n_attempts=N,
            alpha=alpha,
            hybrid_weight=hybrid_weight,
            round_ndigits=round_ndigits,
        )

        new_N = N + 1
        self.rdb.update_academic_score(
            user_id=user_id,
            technical_accuracy=updated["technical_accuracy"],
            reasoning_depth=updated["reasoning_depth"],
            communication_clarity=updated["communication_clarity"],
            score_overall=updated["score_overall"],
            question_attempted=new_N,
        )
        logger.info(f"scores updated user_id={user_id} N={new_N} overall={updated['score_overall']}")

        return {**updated, "question_attempted": new_N}
