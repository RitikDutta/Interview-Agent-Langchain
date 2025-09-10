from __future__ import annotations

from typing import Any, Dict, List

import fitz  # PyMuPDF
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from interview_flow.logging import get_logger
from .models import QuestionItem, QuestionBatch

logger = get_logger("ingest.segment")


def extract_pdf_blocks(pdf_path: str) -> List[Dict[str, Any]]:
    doc = fitz.open(pdf_path)
    blocks = []
    for i in range(len(doc)):
        txt = doc[i].get_text("text")
        if txt and txt.strip():
            blocks.append({"page": i + 1, "text": txt})
    doc.close()
    return blocks


def get_segment_chain():
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an expert dataset builder. Extract interview questions from the user-provided text.\n"
            "For each question, return a JSON with fields: question_text, skill, subskill, difficulty, domain, tags, lang.\n"
            "Use short labels for skill/subskill/domain and keep difficulty in {easy|medium|hard}. Use 'en' for lang."
        ),
        ("human", "{page_text}")
    ])
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    chain = prompt | llm.with_structured_output(QuestionBatch)
    return chain


def segment_page_into_questions(chain, text: str) -> List[QuestionItem]:
    try:
        res: QuestionBatch = chain.invoke({"page_text": text})
        return list(res.questions or [])
    except Exception as e:
        logger.warning(f"segment_page_into_questions error: {e}")
        return []

