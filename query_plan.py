# strategic_question_router.py
from __future__ import annotations
import random
from typing import List, Optional, Dict, Any, Literal

from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from relational_database import RelationalDB
from vector_store import VectorStore
from questions_retrieval import QuestionSearch


# -------- data models --------
class QueryPlan(BaseModel):
    strategy: Literal["skills", "scores", "weakness", "strength"]
    query: str = Field(..., description="Short semantic query for Pinecone embedding")
    domain: Optional[str] = None         # normalized snake for optional filtering
    skill: Optional[str] = None          # normalized snake_skill for optional filtering
    difficulty: Optional[Literal["easy", "medium", "hard"]] = None     # easy|medium|hard
    lang: str = "en"


# -------- helper: lightweight normalizer mirroring your ingestion snake --------
import re
def snake(s: Optional[str]) -> str:
    if not s: return ""
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


# -------- core router --------
class StrategicQuestionRouter:
    def __init__(
        self,
        rdb: RelationalDB,
        vs: VectorStore,
        questions_namespace: str = "questions_v4",
        model: str = "gemini-2.5-flash",
        temperature: float = 0.2,
    ):
        self.rdb = rdb
        self.vs = vs
        self.ns = questions_namespace
        self.llm = ChatGoogleGenerativeAI(model=model, temperature=temperature)

        

        # one compact prompt to convert profile → query plan
        self.prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a concise interview question planner. "
             "Given profile + chosen strategy, output a compact plan for semantic retrieval. "
             "Prefer semantic-only first (lang filter), because strict domain labels in question bank may differ."),
            ("human",
             "STRATEGY: {strategy}\n"
             "DOMAIN: {domain}\n"
             "SKILLS: {skills}\n"
             "STRENGTHS: {strengths}\n"
             "WEAKNESSES: {weaknesses}\n"
             "SCORES: {scores}\n"
             "USER_SUMMARY: {summary}\n\n"
             "Rules:\n"
             "- Keep 'query' short (<=12 words) but meaningful.\n"
             "- If strategy='scores' and score_overall < 4, set difficulty='easy' and avoid niche skill filters.\n"
             "- If strategy='skills', you may set 'skill' to a single normalized skill to bias retrieval.\n"
             "- If strategy='weakness', focus query on one weakness area; difficulty='easy' or 'medium'.\n"
             "- If strategy='strength', push difficulty='hard' and aim deeper in that strength.\n"
             "- domain is optional; only set if obviously helpful.\n"
             "Return JSON with keys: strategy, query, domain, skill, difficulty, lang.")
        ])

    def _get_profile(self, user_id: str) -> Dict[str, Any]:
        rdb_prof = self.rdb.get_user_profile(user_id) or {}
        vec_prof = (self.vs.get_user_profile(user_id) or {}).get("metadata", {})  # domain/skills/strengths/summary live here too
        # prefer vector canonical fields where present, then rdb
        domain = vec_prof.get("domain") or rdb_prof.get("domain")
        skills = vec_prof.get("skills") or rdb_prof.get("skills") or []
        strengths = vec_prof.get("strengths") or rdb_prof.get("strengths") or []
        weaknesses = vec_prof.get("weaknesses") or rdb_prof.get("weaknesses") or []
        summary = vec_prof.get("user_summary", "")
        scores = self.rdb.get_user_academic_score(user_id) or {
            "technical_accuracy": 0, "reasoning_depth": 0, "communication_clarity": 0, "score_overall": 0.0
        }
        return {
            "domain": snake(domain or ""),
            "skills": [snake(s) for s in skills],
            "strengths": [snake(s) for s in strengths],
            "weaknesses": [snake(s) for s in weaknesses],
            "summary": summary,
            "scores": scores,
        }

    # Deterministic strategy choice (override via arg if needed)
    def choose_strategy(self, prof: Dict[str, Any]) -> str:
        scores = prof["scores"]
        if prof["weaknesses"]:
            return "weakness"
        if (scores.get("score_overall") or 0) < 4:
            return "scores"
        if prof["skills"]:
            return "skills"
        return "strength"

    def _sample_skill(self, skills: List[str]) -> Optional[str]:
        return random.choice(skills) if skills else None

    def plan(self, user_id: str, strategy: Optional[str] = None) -> QueryPlan:
        prof = self._get_profile(user_id)
        strategy = (strategy or self.choose_strategy(prof))
        # seed for skills strategy
        picked_skill = self._sample_skill(prof["skills"]) if strategy == "skills" else None

        # Produce a compact plan (structured output)
        chain = self.prompt | self.llm.with_structured_output(QueryPlan)
        plan = chain.invoke({
            "strategy": strategy,
            "domain": prof["domain"] or "data_science",
            "skills": (prof["skills"][:15] or []),
            "strengths": (prof["strengths"][:10] or []),
            "weaknesses": (prof["weaknesses"][:10] or []),
            "scores": prof["scores"],
            "summary": prof["summary"][:800],
        })

        # Post-processing: use sampled skill if plan omitted it
        if strategy == "skills" and not plan.skill:
            plan.skill = picked_skill

        # Normalize a bit more
        plan.domain = snake(plan.domain or "")
        plan.skill = snake(plan.skill or "")
        plan.difficulty = snake(plan.difficulty or "") or None
        plan.lang = plan.lang or "en"

        return plan

    def retrieve(self, user_id: str, plan: QueryPlan) -> Dict[str, Any]:
        """
        Retrieval policy:
        - Start semantic-first with only lang filter (custom_filter) to avoid domain mismatch issues.
        - If strategy suggests skill/difficulty, try a second pass with selective filters.
        """
        # First pass: semantic-only (+lang) using custom_filter
        first_filter = {"lang": {"$eq": plan.lang}}
        res = self.retriever.get_best_question(
            user_id,
            lang=plan.lang,
            top_k=10,
            custom_filter=first_filter,   # semantic-first
        )
        # If we hit your fallback (p-value) OR the meta shows poor alignment, try a targeted pass
        meta = res.get("question_meta", {}) or {}
        got_real = bool(meta) and bool(res.get("picked_question_id"))

        if got_real:
            return res  # good hit

        # Second pass: add gentle filters if available
        second_filter: Dict[str, Any] = {"lang": {"$eq": plan.lang}}
        if plan.skill:
            second_filter["skill"] = {"$eq": plan.skill}
        if plan.difficulty:
            second_filter["difficulty"] = {"$eq": plan.difficulty}
        # Optional domain filter (use only if corpus labels are trusted)
        if plan.domain:
            second_filter["domain"] = {"$eq": plan.domain}

        return self.retriever.get_best_question(
            user_id,
            lang=plan.lang,
            top_k=10,
            custom_filter=second_filter,
        )


# ------------- convenience runner -------------
if __name__ == "__main__":
    rdb = RelationalDB()
    vs = VectorStore()
    namespace = os.getenv("QUESTIONS_NAMESPACE") or "questions_v4"
    router = StrategicQuestionRouter(rdb=rdb, vs=vs, questions_namespace=namespace)
    USER = os.getenv("TEST_USER_ID") or "test_user_004"
    retriever = QuestionSearch(namespace=namespace)
    
    # Try each strategy explicitly; in production callers may omit the arg
    # strats = ["scores", "skills", "weakness", "strength"]
    strats = ["skills"]

    for strat in strats:
        plan = router.plan(USER, strategy=strat)
        print(f"\n[PLAN/{strat} \n] {plan.model_dump()}\n")
        print("*"*20)
        print(f'asking {plan.query}')
        results = retriever.query_search(plan.query, top_k=1)
        print(results[0].score, results[0].metadata.get("text"))
