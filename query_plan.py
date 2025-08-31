# strategic_question_router.py
from __future__ import annotations
import os
import random
from typing import List, Optional, Dict, Any, Literal

from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from log_utils import get_logger

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
        retriever: Optional[QuestionSearch] = None,
    ):
        self.logger = get_logger("planner")
        self.rdb = rdb
        self.vs = vs
        self.ns = questions_namespace
        self.llm = ChatGoogleGenerativeAI(model=model, temperature=temperature)
        # ensure a default retriever for retrieve()
        self.retriever = retriever or QuestionSearch(namespace=self.ns)

        

        # one compact prompt to convert profile → query plan (topic-only query)
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

    @staticmethod
    def _normalize_query_topic(text: str) -> str:
        """Force the query to be a short topic phrase, never a question."""
        if not text:
            return ""
        t = str(text).strip()
        # remove common prefixes
        t = re.sub(r"(?i)^(topic|subject|area|focus|plan|query|ask( about)?):?\s*", "", t)
        # drop question-like punctuation and trailing dots
        t = t.strip().rstrip("?.! ")
        # remove leading verbs like 'explain', 'describe', 'how to', 'what is'
        t = re.sub(r"(?i)^(explain|describe|discuss|how to|what is|why|when|give me|ask about)\s+", "", t)
        # collapse spaces and cap to 8 words
        t = re.sub(r"\s+", " ", t)
        words = t.split()
        if len(words) > 8:
            t = " ".join(words[:8])
        return t

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
        # Internal reasoning log
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
        # seed for skills strategy
        picked_skill = self._sample_skill(prof["skills"]) if strategy == "skills" else None
        if picked_skill:
            self.logger.thinking("Biasing retrieval towards skill: %s", picked_skill)

        # Produce a compact plan (structured output)
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

        # Post-processing: use sampled skill if plan omitted it
        if strategy == "skills" and not plan.skill:
            plan.skill = picked_skill

        # Normalize a bit more
        plan.domain = snake(plan.domain or "")
        plan.skill = snake(plan.skill or "")
        plan.difficulty = snake(plan.difficulty or "") or None
        plan.lang = plan.lang or "en"

        # Enforce strategy-specific defaults requested
        score_overall = (prof.get("scores") or {}).get("score_overall") or 0.0
        if strategy == "scores":
            # If score is 0, prefer easy and include domain when available
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

        # Force the returned strategy to match caller choice
        plan.strategy = strategy  # type: ignore

        # Ensure the query is a topic (never a full question)
        plan.query = self._normalize_query_topic(plan.query)
        self.logger.thinking("Planned topic '%s' (difficulty=%s, domain=%s, skill=%s)", plan.query, plan.difficulty, plan.domain, plan.skill)
        if not plan.query:
            # Fallback topic based on available signals
            base: Optional[str] = None
            if strategy == "skills" and (plan.skill or picked_skill):
                base = (plan.skill or picked_skill or "").replace("_", " ")
            elif strategy == "strength" and prof.get("strengths"):
                base = (prof["strengths"][0] or "").replace("_", " ")
            elif strategy == "weakness" and prof.get("weaknesses"):
                base = (prof["weaknesses"][0] or "").replace("_", " ")
            else:  # scores or unknown
                base = (prof.get("domain") or "fundamentals").replace("_", " ")

            if strategy in ("scores", "weakness") and base:
                plan.query = self._normalize_query_topic(f"{base} fundamentals")
            else:
                plan.query = self._normalize_query_topic(base)

        self.logger.debug(f"plan: strategy={plan.strategy} domain={plan.domain} skill={plan.skill} diff={plan.difficulty} query='{plan.query}'")
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
        logger = get_logger("planner.demo")
        logger.info(f"PLAN/{strat}: {plan.model_dump()}")
        logger.info(f"asking: {plan.query}")
        results = retriever.query_search(plan.query, top_k=1)
        if results:
            logger.info(f"score={results[0].score} text={results[0].metadata.get('text')}")
