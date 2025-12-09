from __future__ import annotations

from typing import Dict, Any, Literal
from pathlib import Path

from langgraph.types import interrupt

from ..logging import get_logger
from ..utils import last_user_text, _normalize_resume_input, merge_list_preserving_order, DEFAULT_Q
from ..scoring import _eval_metric_with_llm, _overall_with_llm

logger = get_logger("agent.flow")


# --- intake.py ---
def _sanitize_label(s: str) -> str:
    try:
        import re
        s = (s or "").strip().lower()
        s = re.sub(r"[^a-z0-9]+", "_", s)
        s = re.sub(r"_+", "_", s).strip("_")
        return s or "misc"
    except Exception:
        return (s or "misc").strip().lower() or "misc"


def node_ask_domain(state: Dict[str, Any]) -> dict:
    logger.debug("node_ask_domain")
    return {"graph_state": "ask_domain"}


def node_get_domain(state: Dict[str, Any]) -> dict:
    user_id = state.get("user_id")
    logger.debug("node get_domain")
    domain_value = interrupt({
        "prompt": "Please provide your domain of interest (e.g., Data Science, AI, Management, Etc). "
    })
    if isinstance(domain_value, dict):
        domain = domain_value.get("domain") or domain_value.get("content") or ""
    else:
        domain = str(domain_value or "")
    domain = domain.strip()
    if domain.lower() in {"skip", "no", "later", ""}:
        logger.info("[get_domain] user skipped providing domain")
        return {"graph_state": "no_domain"}
    logger.info(f"[domain] Received from user user_id={user_id}: '{domain}'")

    # Try to canonicalize via vector store; fall back to sanitized snake_case
    canonical = None
    try:
        from ..normalization.canonical_normalizer import search_domains_first
        hit = search_domains_first(domain)
        canonical = (hit or {}).get("canonical") if hit else None
        if canonical:
            logger.info(f"[domain] Canonicalized name is '{canonical}'")
        else:
            fallback = _sanitize_label(domain)
            logger.info(f"[domain] Using default sanitized form (not found in database): '{fallback}'")
            canonical = fallback
    except Exception as e:
        fallback = _sanitize_label(domain)
        logger.warning(f"[domain] Canonicalization error ({e}); using sanitized fallback '{fallback}'")
        canonical = fallback

    logger.thinking("Captured domain preference (normalized): %s", canonical)
    return {"graph_state": "have_domain", "domain": canonical}


def node_ask_resume(state: Dict[str, Any]) -> dict:
    logger.debug("node_ask_resume")
    return {"graph_state": "ask_resume"}


def node_get_resume_url(state: Dict[str, Any]) -> dict:
    logger.debug("node get_resume_url")
    text = last_user_text(state)
    resume_value = text if text else interrupt({
        "prompt": (
            "Please share your resume — you can paste a URL (http/https), a file:// URL, "
            "or a local path (absolute/relative, ~ ok). Type 'skip' to continue without a resume."
        )
    })
    try:
        kind, normalized = _normalize_resume_input(resume_value)
        logger.thinking("Parsed resume source: kind=%s (len=%d)", kind, len(normalized or ""))
    except Exception as e:
        logger.warning(f"[get_resume_url] normalization error: {e} → continuing without resume")
        return {"graph_state": "no_resume_url"}

    if kind == "none" or not normalized:
        logger.info("[get_resume_url] user skipped providing a resume")
        return {"graph_state": "no_resume_url"}

    if kind in {"file", "local"}:
        p = Path(normalized)
        if not p.exists():
            logger.warning(f"[get_resume_url] local path not found: {p}")
            return {"graph_state": "no_resume_url"}
    logger.info(f"[get_resume_url] source kind={kind} value={normalized}")
    return {"graph_state": "have_resume_url", "resume_url": normalized}


def is_resume_url(state: Dict[str, Any]) -> Literal["resume_is_present", "resume_is_not_present"]:
    has_source = bool((state.get("resume_url") or "").strip())
    return "resume_is_present" if has_source else "resume_is_not_present"


# --- interview.py ---
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


# --- metrics.py ---
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


# --- persistence.py ---
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


def make_save_history(*, rdb):
    def save_history(state: Dict[str, Any]) -> dict:
        user_id = state.get("user_id") or ""
        qid = state.get("picked_question_id") or "unknown"
        qtext = (state.get("question") or "").strip()
        ans = (state.get("answer") or "").strip()

        m_ta = state.get("metric_technical_accuracy", {}) or {}
        m_rd = state.get("metric_reasoning_depth", {}) or {}
        m_cc = state.get("metric_communication_clarity", {}) or {}
        # combined_score may be computed in aggregate_feedback; default to mean if missing
        overall = state.get("combined_score")
        try:
            ta = float(m_ta.get("score") or 0)
            rd = float(m_rd.get("score") or 0)
            cc = float(m_cc.get("score") or 0)
            if overall is None:
                overall = float(int(round((ta + rd + cc) / 3)))
            else:
                overall = float(overall)
        except Exception:
            ta = rd = cc = 0.0
            overall = 0.0

        try:
            if user_id and qtext and ans:
                rdb.insert_conversation_history(
                    user_id=user_id,
                    question_id=str(qid),
                    question_text=qtext,
                    user_answer=ans,
                    overall=overall,
                    ta=ta,
                    rd=rd,
                    cc=cc,
                )
                logger.info("[save_history] persisted Q/A with metrics")
            else:
                logger.warning("[save_history] missing fields; skip persist")
        except Exception as e:
            logger.error(f"[save_history] persist failed: {e}")

        return {"graph_state": "saved_history"}

    return save_history


# --- resume_profile.py ---
def make_node_resume_ETL(*, rdb, vs):
    def node_resume_ETL(state: Dict[str, Any]) -> dict:
        logger.debug("node_resume_ETL")
        logger.info(f"[resume_ETL] source={state.get('resume_url')}")
        logger.thinking("Running ResumeETL for user_id=%s", state.get("user_id"))
        # Import here to avoid import-time network calls from canonical_normalizer
        from ..ingestion.resume_etl import ResumeETL
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


# --- routing.py ---
def gate_is_new_user(state: Dict[str, Any]) -> dict:
    logger.info("System: Checking if user is new or existing")
    return {}


def route_is_new_user(state: Dict[str, Any]) -> Literal["new_user", "existing_user"]:
    if "is_new_user" in state:
        flag = bool(state["is_new_user"])
        if flag:
            logger.thinking("New user flag detected; proceeding with first-time setup")
        else:
            logger.thinking("Found an existing user in database; skipping initialization")
        return "new_user" if flag else "existing_user"
    msgs = state.get("messages") or []
    first = ""
    for m in msgs:
        if m.get("role") == "user":
            first = (m.get("content") or "").lower()
            break
    newbie = any(kw in first for kw in ("start", "new user", "first time", "setup"))
    logger.thinking("Heuristic routing: %s user based on initial message", "new" if newbie else "existing")
    return "new_user" if newbie else "existing_user"


def make_init_user(*, rdb, vs):
    def init_user(state: Dict[str, Any]) -> dict:
        logger.info("System: Initializing new user")
        user_id = state.get("user_id")
        if not user_id:
            raise ValueError("User ID is required for new user initialization")

        rdb.create_tables()
        rdb.upsert_user(
            user_id=user_id,
            name=None,
            domain=None,
            skills=[],
            strengths=[],
            weaknesses=[],
        )
        rdb.ensure_academic_summary(user_id)

        summary = "New user (no resume yet). Domain: Unknown."
        vs.upsert_profile_snapshot(
            user_id=user_id,
            domain="Unknown",
            summary=summary,
            skills=[],
            strengths=[],
            weaknesses=[],
        )

        return {"graph_state": "init_user", "is_new_user": True, "messages": []}

    return init_user


def gate_should_continue(state: Dict[str, Any]) -> dict:
    logger.debug("gate_should_continue")
    choice = interrupt({
        "prompt": (
            "Would you like another question? Type 'continue' to proceed, or 'stop' to end."
        )
    })
    # Normalize the user's choice after resume
    if isinstance(choice, dict):
        val = choice.get("choice") or choice.get("content") or ""
    else:
        val = str(choice or "")
    v = (val or "").strip().lower()
    if v in {"stop", "exit", "quit", "no", "n", "done"}:
        norm = "exit"
    else:
        # Treat any other value (including blank) as continue to keep the interview flowing
        norm = "continue"
    return {"graph_state": "should_continue", "should_continue": norm}


def route_should_continue(state: Dict[str, Any]) -> Literal["continue", "exit"]:
    flag = (state.get("should_continue") or "").strip().lower()
    if flag in {"exit", "quit", "stop"}:
        logger.info("[route_should_continue] flag → EXIT")
        return "exit"
    msg = (state.get("input_text") or "").strip().lower()
    if any(w in msg for w in ("exit", "quit", "stop")):
        logger.info(f"[route_should_continue] message '{msg}' → EXIT")
        return "exit"
    logger.info("[route_should_continue] → CONTINUE")
    return "continue"
