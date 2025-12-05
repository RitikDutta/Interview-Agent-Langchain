from __future__ import annotations

from typing import TypedDict, Literal, List, Optional, Dict, Any
from pydantic import BaseModel, Field, conint


# --- eval_models.py ---
class MetricEval(BaseModel):
    feedback: str = Field(description="≤50 words, concise and actionable")
    score: conint(ge=0, le=10) = Field(description="integer 0..10")


class OverallEval(BaseModel):
    overall_feedback: str = Field(
        description="≤100 words, balanced, actionable, no headings/repetition"
    )
    combined_score: conint(ge=0, le=10) = Field(
        description="average score of metrics, integer 0–10"
    )
    strengths: Optional[List[str]] = Field(
        default=None,
        description=(
            "OPTIONAL. in 1-2 words strengths"
            "Only include if clearly evidenced across the provided metrics; otherwise omit."
        ),
    )
    weaknesses: Optional[List[str]] = Field(
        default=None,
        description=(
            "OPTIONAL. in 1-2 words weaknesses."
            "Only include if clearly evidenced across the provided metrics; otherwise omit."
        ),
    )


# --- state.py ---
class Message(TypedDict):
    role: Literal["user", "assistant", "system"]
    content: str


class State(TypedDict, total=False):
    graph_state: str
    input_text: str
    is_new_user: bool
    should_continue: str
    messages: List[Message]

    user_id: str
    domain: Optional[str]
    resume_url: Optional[str]
    strategy: Optional[str]
    profile: Dict[str, Any]
    skills: List[str]
    summary: str

    # interview loop
    question: str
    question_meta: Dict[str, Any]
    picked_question_id: Optional[str]
    answer: str

    # metrics
    metric_technical_accuracy: Dict[str, Any]
    metric_reasoning_depth: Dict[str, Any]
    metric_communication_clarity: Dict[str, Any]

    # aggregation
    overall_feedback_summary: str
    combined_score: int
    strengths: List[str]
    weaknesses: List[str]
