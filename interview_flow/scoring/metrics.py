from __future__ import annotations

from langchain_core.messages import SystemMessage, HumanMessage

from ..agents.llm import llm_review, llm_overall
from ..schemas.eval_models import MetricEval, OverallEval


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

