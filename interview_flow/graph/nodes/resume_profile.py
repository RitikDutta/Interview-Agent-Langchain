from __future__ import annotations

from typing import Dict, Any

from ...logging import get_logger

logger = get_logger("agent.flow")


def make_node_resume_ETL(*, rdb, vs):
    def node_resume_ETL(state: Dict[str, Any]) -> dict:
        logger.debug("node_resume_ETL")
        logger.info(f"[resume_ETL] source={state.get('resume_url')}")
        logger.thinking("Running ResumeETL for user_id=%s", state.get("user_id"))
        # Import here to avoid import-time network calls from canonical_normalizer
        from ...ingestion.resume_etl import ResumeETL
        resume_etl = ResumeETL(user_id=state.get("user_id"), rdb=rdb, vdb=vs, verbose=True)
        profile = resume_etl.run(resume_url_or_path=state.get("resume_url") or "")
        return {"graph_state": "node_resume_ETL", "profile": profile}

    return node_resume_ETL


def make_update_profile(*, rdb, vs):
    def update_profile(state: Dict[str, Any]) -> dict:
        logger.debug("node update_profile")
        user_id = state.get("user_id")
        etl_out = state.get("profile") or {}
        prof = etl_out.get("profile") or {}

        chosen_domain = (state.get("domain") or prof.get("domain") or None)
        skills = list(prof.get("technical_skills") or [])
        strengths = list(prof.get("strengths") or [])
        weaknesses = list(prof.get("weaknesses") or [])
        summary = (prof.get("user_summary") or "").strip()
        logger.thinking(
            "Updating profile: domain=%s skills=%d strengths=%d weaknesses=%d",
            chosen_domain, len(skills), len(strengths), len(weaknesses),
        )
        try:
            rdb.update_user_profile_partial(
                user_id=user_id,
                domain=chosen_domain,
                skills=skills or None,
                strengths=strengths or None,
                weaknesses=weaknesses or None,
            )
        except Exception as e:
            logger.error(f"[update_profile] RDB update failed: {e}")

        try:
            vs.upsert_profile_snapshot(
                user_id=user_id,
                domain=chosen_domain or "unknown",
                summary=summary,
                skills=skills or [],
                strengths=strengths or [],
                weaknesses=weaknesses or [],
            )
        except Exception as e:
            logger.error(f"[update_profile] Vector upsert failed: {e}")

        return {
            "graph_state": "update_profile",
            "domain": chosen_domain,
            "skills": skills[:],
            "strengths": strengths[:],
            "weaknesses": weaknesses[:],
            "summary": summary,
        }

    return update_profile
