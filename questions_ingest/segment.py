from __future__ import annotations

from typing import Any, Dict, List, Optional

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


def get_segment_chain(base_domain: Optional[str] = None):
    base_note = ""
    if base_domain and str(base_domain).strip():
        bd = str(base_domain).strip()
        base_note = (
            "\nContext: The provided PDF is primarily about the domain '"
            + bd +
            "'. Unless the page text clearly indicates otherwise, set the domain field to '" + bd + "'."
        )
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an expert curator of interview question datasets. "
            "Your job is to split raw text into atomic interview questions and label them with normalized metadata.\n\n"

            "For each question, return a JSON object with:\n"
            "- question_text: full question string (plain text)\n"
            "- skill: the most relevant specific skill (short snake_case)\n"
            "- subskill: narrower variant of skill if applicable (else 'general')\n"
            "- domain: the higher-level discipline the question belongs to "
            "(e.g., data_science, devops, software_engineering, management, finance). "
            "⚠️ Always prefer generalized domains, not niche areas. "
            "Example: domain=devops (not cloud_computing), domain=data_science (not deep_learning).\n"
            "- difficulty: one of {{easy|medium|hard}} (relative to a typical interview context)\n"
            "- tags: 0–5 short, helpful keywords\n"
            "- lang: 'en'\n\n"

            "Rules:\n"
            "- Use concise, normalized snake_case labels for skill/subskill/domain.\n"
            "- If no clear match, default to skill='misc', subskill='general', domain='misc'.\n"
            "- Multi-part questions should be split into separate atomic entries.\n"
            "- Do not invent overly specific domains. Keep domain broad and high-level.\n"
            "- Keep outputs valid JSON objects only (no commentary).\n" + base_note
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
