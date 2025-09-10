# StrategicQuestionRouter (moved from query_plan.py)
from __future__ import annotations
import os
import random
from typing import List, Optional, Dict, Any, Literal

from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from ..logging import get_logger
from ..infra.rdb import RelationalDB
from ..infra.vector_store import VectorStore
from .questions import QuestionSearch


class QueryPlan(BaseModel):
    strategy: Literal["skills", "scores", "weakness", "strength"]
    query: str = Field(..., description="Short semantic query for Pinecone embedding")
    domain: Optional[str] = None
    skill: Optional[str] = None
    difficulty: Optional[Literal["easy", "medium", "hard"]] = None
    lang: str = "en"


import re


def snake(s: Optional[str]) -> str:
    if not s:
        return ""
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


class StrategicQuestionRouter:
    def __init__(
        self,
        rdb: RelationalDB,
        vs: VectorStore,
        questions_namespace: str = "questions_v4",
        model: str = "gemini-2.5-flash",
        temperature: float = 0.2,
        retriever: Optional[QuestionSearch] = None,
    ):
        self.logger = get_logger("planner")
        self.rdb = rdb
        self.vs = vs
        self.ns = questions_namespace
        self.llm = ChatGoogleGenerativeAI(model=model, temperature=temperature)
        self.retriever = retriever or QuestionSearch(namespace=self.ns)

        self.prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a senior interview question planner. Your task is to translate the\n"
                "caller-provided strategy plus the user profile and academic scores into a\n"
                "compact retrieval plan. The plan is for semantic search; the 'query' must\n"
                "be a single topic/subject phrase, not an actual question. Respect the\n"
                "provided strategy and do not switch it.\n\n"
                "Output rules:\n"
                "- 'query' is a short topic (<= 8 words), e.g., 'hypothesis testing', 'hash maps',\n"
                "  'k-means clustering'. Never output a full question; no question marks or directives.\n"
                "- Prefer semantic-first retrieval; keep domain/skill optional unless clearly helpful.\n"
                "- 'skill' (if set) must be a single normalized label from inputs (snake-like).\n"
                "- Use 'lang' as provided; prefer 'en' if unspecified.\n\n"
                "Strategy-specific guidance:\n"
                "- scores: Prioritize 'scores'. If score_overall == 0 (or missing), set\n"
                "  difficulty='easy' and avoid niche skills; include domain if available.\n"
                "  Otherwise, if score_overall < 4, set difficulty='easy'. Choose a foundational\n"
                "  topic to build confidence within the domain.\n"
                "- skills: Focus on exactly one skill (if multiple, pick one). Set 'skill' to that\n"
                "  normalized skill and choose a representative topic for it.\n"
                "- weakness: Target one weakness area; choose a remedial/fundamental topic;\n"
                "  difficulty should be 'easy' or 'medium'.\n"
                "- strength: Choose a deeper/advanced topic in a strength; set difficulty='hard'.\n\n"
                "Return only the structured content for fields: strategy, query, domain, skill, difficulty, lang."
            ),
            (
                "human",
                "STRATEGY: {strategy}\n"
                "DOMAIN: {domain}\n"
                "SKILLS: {skills}\n"
                "STRENGTHS: {strengths}\n"
                "WEAKNESSES: {weaknesses}\n"
                "SCORES: {scores}\n"
                "USER_SUMMARY: {summary}"
            ),
        ])

    def _normalize_query_topic(self, text: str) -> str:
        if not text:
            return ""
        t = str(text).strip()
        t = re.sub(r"(?i)^(topic|subject|area|focus|plan|query|ask( about)?):?\s*", "", t)
        t = t.strip().rstrip("?.! ")
        t = re.sub(r"(?i)^(explain|describe|discuss|how to|what is|why|when|give me|ask about)\s+", "", t)
        t = re.sub(r"\s+", " ", t)
        parts = t.split(" ")
        if len(parts) > 8:
            t = " ".join(parts[:8])
        return t

    def _get_profile(self, user_id: str) -> Dict[str, Any]:
        try:
            r = self.rdb.get_user_profile(user_id) or {}
        except Exception:
            r = {}
        try:
            v = self.vs.get_user_profile(user_id) or {}
            vmd = v.get("metadata") or {}
        except Exception:
            vmd = {}
        domain = (r.get("domain") or vmd.get("domain") or "")
        skills = (r.get("skills") or vmd.get("skills") or [])
        strengths = (r.get("strengths") or vmd.get("strengths") or [])
        weaknesses = (r.get("weaknesses") or vmd.get("weaknesses") or [])
        summary = (vmd.get("user_summary") or "")

        try:
            s = self.rdb.get_user_academic_score(user_id) or {}
            scores = {
                "technical_accuracy": float(s.get("technical_accuracy") or 0.0),
                "reasoning_depth": float(s.get("reasoning_depth") or 0.0),
                "communication_clarity": float(s.get("communication_clarity") or 0.0),
                "score_overall": float(s.get("score_overall") or 0.0),
            }
        except Exception:
            scores = {
                "technical_accuracy": 0.0,
                "reasoning_depth": 0.0,
                "communication_clarity": 0.0,
                "score_overall": 0.0,
            }
        return {
            "domain": snake(domain or ""),
            "skills": [snake(s) for s in skills],
            "strengths": [snake(s) for s in strengths],
            "weaknesses": [snake(s) for s in weaknesses],
            "summary": summary,
            "scores": scores,
        }

    def choose_strategy(self, prof: Dict[str, Any]) -> str:
        scores = prof["scores"]
        self.logger.thinking(
            "Assessing profile to select strategy (overall=%.2f, skills=%d, strengths=%d, weaknesses=%d)",
            (scores.get("score_overall") or 0.0), len(prof.get("skills") or []), len(prof.get("strengths") or []), len(prof.get("weaknesses") or []),
        )
        if prof["weaknesses"]:
            self.logger.thinking("Choosing 'weakness' strategy — weaknesses present")
            return "weakness"
        if (scores.get("score_overall") or 0) < 4:
            self.logger.thinking("Choosing 'scores' strategy — low overall score")
            return "scores"
        if prof["skills"]:
            self.logger.thinking("Choosing 'skills' strategy — skills available")
            return "skills"
        self.logger.thinking("Defaulting to 'strength' strategy")
        return "strength"

    def _sample_skill(self, skills: List[str]) -> Optional[str]:
        return random.choice(skills) if skills else None

    def plan(self, user_id: str, strategy: Optional[str] = None) -> QueryPlan:
        prof = self._get_profile(user_id)
        strategy = (strategy or self.choose_strategy(prof))
        picked_skill = self._sample_skill(prof["skills"]) if strategy == "skills" else None
        if picked_skill:
            self.logger.thinking("Biasing retrieval towards skill: %s", picked_skill)

        chain = self.prompt | self.llm.with_structured_output(QueryPlan)
        self.logger.thinking("Deriving retrieval plan (strategy=%s, domain=%s, skills=%d, strengths=%d, weaknesses=%d)",
                            strategy, prof["domain"] or "", len(prof["skills"]), len(prof["strengths"]), len(prof["weaknesses"]))
        plan = chain.invoke({
            "strategy": strategy,
            "domain": prof["domain"] or "data_science",
            "skills": (prof["skills"][:15] or []),
            "strengths": (prof["strengths"][:10] or []),
            "weaknesses": (prof["weaknesses"][:10] or []),
            "scores": prof["scores"],
            "summary": prof["summary"][:800],
        })

        if strategy == "skills" and not plan.skill:
            plan.skill = picked_skill

        plan.domain = snake(plan.domain or "")
        plan.skill = snake(plan.skill or "")
        plan.difficulty = snake(plan.difficulty or "") or None
        plan.lang = plan.lang or "en"

        score_overall = (prof.get("scores") or {}).get("score_overall") or 0.0
        if strategy == "scores":
            if score_overall == 0:
                plan.difficulty = plan.difficulty or "easy"
                if not plan.domain and prof.get("domain"):
                    plan.domain = prof["domain"]
            elif score_overall < 4:
                plan.difficulty = plan.difficulty or "easy"
        elif strategy == "weakness":
            plan.difficulty = plan.difficulty or "easy"
        elif strategy == "strength":
            plan.difficulty = "hard"

        plan.strategy = strategy  # type: ignore
        plan.query = self._normalize_query_topic(plan.query)
        self.logger.thinking("Planned topic '%s' (difficulty=%s, domain=%s, skill=%s)", plan.query, plan.difficulty, plan.domain, plan.skill)
        if not plan.query:
            base: Optional[str] = None
            if strategy == "skills" and (plan.skill or picked_skill):
                base = (plan.skill or picked_skill or "").replace("_", " ")
            elif strategy == "strength" and prof.get("strengths"):
                base = (prof["strengths"][0] or "").replace("_", " ")
            elif strategy == "weakness" and prof.get("weaknesses"):
                base = (prof["weaknesses"][0] or "").replace("_", " ")
            else:
                base = (prof.get("domain") or "fundamentals").replace("_", " ")

            if strategy in ("scores", "weakness") and base:
                plan.query = self._normalize_query_topic(f"{base} fundamentals")
            else:
                plan.query = self._normalize_query_topic(base)

        self.logger.debug(f"plan: strategy={plan.strategy} domain={plan.domain} skill={plan.skill} diff={plan.difficulty} query='{plan.query}'")
        return plan

    def retrieve(self, user_id: str, plan: QueryPlan) -> Dict[str, Any]:
        first_filter = {"lang": {"$eq": plan.lang}}
        res = self.retriever.get_best_question(
            user_id,
            lang=plan.lang,
            top_k=10,
            custom_filter=first_filter,
        )
        meta = res.get("question_meta", {}) or {}
        got_real = bool(meta) and bool(res.get("picked_question_id"))
        if got_real:
            return res

        second_filter: Dict[str, Any] = {"lang": {"$eq": plan.lang}}
        if plan.skill:
            second_filter["skill"] = {"$eq": plan.skill}
        if plan.difficulty:
            second_filter["difficulty"] = {"$eq": plan.difficulty}
        if plan.domain:
            second_filter["domain"] = {"$eq": plan.domain}

        return self.retriever.get_best_question(
            user_id,
            lang=plan.lang,
            top_k=10,
            custom_filter=second_filter,
        )

