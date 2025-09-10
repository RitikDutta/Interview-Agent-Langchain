from __future__ import annotations

from typing import TypedDict, Literal, List, Optional, Dict, Any


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

