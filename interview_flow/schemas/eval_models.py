from __future__ import annotations

from typing import Optional, List
from pydantic import BaseModel, Field, conint


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

