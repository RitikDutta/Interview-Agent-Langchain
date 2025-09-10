from __future__ import annotations

from typing import List, Literal
from pydantic import BaseModel, Field


class QuestionItem(BaseModel):
    question_text: str
    skill: str = "misc"
    subskill: str = "general"
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    domain: str = "misc"
    tags: List[str] = Field(default_factory=list)
    lang: str = "en"


class QuestionBatch(BaseModel):
    questions: List[QuestionItem] = Field(default_factory=list)

