from __future__ import annotations

from typing import Dict, Any

from ...logging import get_logger
from ...utils.messages import merge_list_preserving_order

logger = get_logger("agent.flow")


def make_update_profile_with_score(*, score_updater, rdb, vs):
    def update_profile_with_score(state: Dict[str, Any]) -> dict:
        logger.debug("node update_profile_with_score")
        user_id = state.get("user_id")

        ta = state.get("metric_technical_accuracy", {}).get("score")
        rd = state.get("metric_reasoning_depth", {}).get("score")
        cc = state.get("metric_communication_clarity", {}).get("score")
        scores = {
            'technical_accuracy': ta,
            'reasoning_depth': rd,
            'communication_clarity': cc,
        }
        logger.thinking("Updating scores EMA with TA=%s RD=%s CC=%s", ta, rd, cc)
        score_updater.update_and_save(user_id, scores)

        try:
            strengths = (state.get("strengths") or [])[:]
            weaknesses = (state.get("weaknesses") or [])[:]

            if strengths or weaknesses:
                try:
                    rdb.update_user_profile_partial(
                        user_id=user_id,
                        strengths=strengths or None,
                        weaknesses=weaknesses or None,
                    )
                except Exception as e:
                    logger.error(f"[update_profile_with_score] rdb.update_user_profile_partial failed: {e}")

                try:
                    vs.upsert_profile_snapshot(
                        user_id=user_id,
                        domain=state.get("domain") or "unknown",
                        summary=state.get("summary") or "",
                        skills=state.get("skills") or [],
                        strengths=strengths or [],
                        weaknesses=weaknesses or [],
                    )
                except Exception as e:
                    logger.error(f"[update_profile_with_score] vector upsert failed: {e}")
        except Exception as e:
            logger.error(f"[update_profile_with_score] merge strengths/weaknesses failed: {e}")

        return {"graph_state": "update_profile_with_score"}

    return update_profile_with_score


def make_update_strength_weakness(*, rdb, vs):
    def update_strength_weakness(state: Dict[str, Any]) -> dict:
        logger.debug("node update_strength_weakness")
        user_id = state.get("user_id")
        if not user_id:
            logger.warning("[update_strength_weakness] missing user_id → skip")
            return {"graph_state": "update_strength_weakness"}

        new_strengths = (state.get("strengths") or [])[:]
        new_weaknesses = (state.get("weaknesses") or [])[:]
        logger.thinking("Merging strengths=%d weaknesses=%d into profile", len(new_strengths), len(new_weaknesses))

        if not new_strengths and not new_weaknesses:
            logger.info("[update_strength_weakness] no strengths/weaknesses to add → skip DB update")
            return {"graph_state": "update_strength_weakness"}

        try:
            profile = rdb.get_user_profile(user_id=user_id) or {}
        except Exception as e:
            logger.error(f"[update_strength_weakness] get_user_profile failed: {e}")
            profile = {}

        existing_strengths = (profile.get("strengths") or [])
        existing_weaknesses = (profile.get("weaknesses") or [])

        merged_strengths = merge_list_preserving_order(existing_strengths, new_strengths) if new_strengths else existing_strengths
        merged_weaknesses = merge_list_preserving_order(existing_weaknesses, new_weaknesses) if new_weaknesses else existing_weaknesses

        try:
            rdb.update_user_profile_partial(
                user_id=user_id,
                strengths=merged_strengths if new_strengths else None,
                weaknesses=merged_weaknesses if new_weaknesses else None,
            )
            logger.info("[update_strength_weakness] DB updated")
        except Exception as e:
            logger.error(f"[update_strength_weakness] update_user_profile_partial failed: {e}")

        try:
            vs.upsert_profile_snapshot(
                user_id=user_id,
                domain=state.get("domain") or "unknown",
                summary=state.get("summary") or "",
                skills=state.get("skills") or [],
                strengths=merged_strengths,
                weaknesses=merged_weaknesses,
            )
            logger.info("[update_strength_weakness] Vector snapshot updated")
        except Exception as e:
            logger.error(f"[update_strength_weakness] vector upsert failed: {e}")

        return {"graph_state": "update_strength_weakness"}

    return update_strength_weakness

