from typing import Dict, Any, Optional
from log_utils import get_logger

logger = get_logger("scores")

class ScoreUpdater:
    """
    Update academic scores:
      - Per-metric (technical_accuracy, reasoning_depth, communication_clarity) via EMA
      - Overall via hybrid of (EMA-of-metrics) and (lifetime mean overall)
      - Uses question_attempted (N) from DB and increments it on save
    """

    def __init__(self, rdb, default_alpha: float = 0.2, default_hybrid_weight: float = 0.7):
        self.rdb = rdb
        self.default_alpha = float(default_alpha)
        self.default_hybrid_weight = float(default_hybrid_weight)

    # ---------- public API ----------
    def fetch_current(self, user_id: str) -> Dict[str, float]:
        """
        Get current scores (incl. question_attempted) from DB.
        Ensures float outputs for metrics and int for N.
        """
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
        """
        Pure function: returns updated floats and does NOT write to DB.
        """
        logger.debug("compute_updated alpha=%s hybrid_weight=%s", alpha, hybrid_weight)
        a = float(alpha if alpha is not None else self.default_alpha)
        w = float(hybrid_weight if hybrid_weight is not None else self.default_hybrid_weight)

        # current values
        ta_old = float(current_scores.get("technical_accuracy", 0.0))
        rd_old = float(current_scores.get("reasoning_depth", 0.0))
        cc_old = float(current_scores.get("communication_clarity", 0.0))
        ov_old = float(current_scores.get("score_overall", 0.0))
        N = max(0, int(n_attempts if n_attempts is not None
                       else current_scores.get("question_attempted", 0)))

        ta_new_raw = float(new_scores["technical_accuracy"])
        rd_new_raw = float(new_scores["reasoning_depth"])
        cc_new_raw = float(new_scores["communication_clarity"])

        def ema(old: float, new: float) -> float:
            return a * new + (1.0 - a) * old

        # metric EMAs
        ta_new = ema(ta_old, ta_new_raw)
        rd_new = ema(rd_old, rd_new_raw)
        cc_new = ema(cc_old, cc_new_raw)

        overall_from_metrics = (ta_new + rd_new + cc_new) / 3.0

        # lifetime mean overall (treat stored overall as mean across N attempts)
        prev_sum = ov_old * max(1, N)
        lifetime_mean = (prev_sum + overall_from_metrics) / (N + 1)

        # hybrid overall
        overall_hybrid = w * overall_from_metrics + (1.0 - w) * lifetime_mean

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
        n_attempts_override: Optional[int] = None,  # usually leave None → use DB N
        *,
        alpha: Optional[float] = None,
        hybrid_weight: Optional[float] = None,
        round_ndigits: int = 2,
    ) -> Dict[str, float]:
        """
        Fetches current scores (incl. N) from DB, computes updated values,
        writes them back, increments N by 1, and returns the updated dict.
        """
        # ensure row exists
        logger.debug(f"update_and_save user_id={user_id}")
        self.rdb.ensure_academic_summary(user_id)

        current = self.fetch_current(user_id)
        N = int(n_attempts_override if n_attempts_override is not None
                else current.get("question_attempted", 0))

        updated = self.compute_updated(
            new_scores=new_scores,
            current_scores=current,
            n_attempts=N,
            alpha=alpha,
            hybrid_weight=hybrid_weight,
            round_ndigits=round_ndigits,
        )

        # persist metrics and bump N
        new_N = N + 1
        self.rdb.update_academic_score(
            user_id=user_id,
            technical_accuracy=updated["technical_accuracy"],
            reasoning_depth=updated["reasoning_depth"],
            communication_clarity=updated["communication_clarity"],
            score_overall=updated["score_overall"],
            question_attempted=new_N,   # <-- bump N
        )
        logger.info(f"scores updated user_id={user_id} N={new_N} overall={updated['score_overall']}")

        return {**updated, "question_attempted": new_N}
