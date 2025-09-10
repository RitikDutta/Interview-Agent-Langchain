from __future__ import annotations

from langchain_google_genai import ChatGoogleGenerativeAI

from ..settings import GENAI_MODEL, GENAI_MODEL_LOW_TEMP
from ..schemas.eval_models import MetricEval, OverallEval

# Primary LLM clients
llm = ChatGoogleGenerativeAI(model=GENAI_MODEL)
llm_low_temp = ChatGoogleGenerativeAI(model=GENAI_MODEL_LOW_TEMP, temperature=0.2)

# Structured binders for scoring
llm_review = llm.with_structured_output(MetricEval)
llm_overall = llm.with_structured_output(OverallEval)

