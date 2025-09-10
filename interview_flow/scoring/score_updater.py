from __future__ import annotations

from typing import Dict, Optional
from ..logging import get_logger

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

