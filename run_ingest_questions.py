#!/usr/bin/env python3
"""
Env-driven runner for question ingestion.

Reads configuration from environment (.env is loaded if present) and ingests
a PDF of interview questions into Pinecone.

Environment variables (set in your .env):
  - INGEST_PDF_PATH         (required) Path to the questions PDF
  - QUESTIONS_VERSION       (required) Version label to stamp in metadata (e.g., v4)
  - QUESTIONS_NAMESPACE     (required) Pinecone namespace for the question bank
  - PINECONE_INDEX_NAME     (optional) Pinecone index name (default: interview_questions)
  - SKILLS_TAXONOMY         (ignored) kept for legacy; now canonicalized via vector search
  - DOMAINS_TAXONOMY        (ignored) kept for legacy; now canonicalized via vector search
  - NEAR_DUP_SIM            (optional) Near-duplicate threshold (e.g., 0.9)
  - LOG_LEVEL               (optional) Set to THINKING or DEBUG for rich logs

Usage:
  python run_ingest_questions.py
"""

from __future__ import annotations

import os
from typing import Optional
from dotenv import load_dotenv

from interview_flow.logging import get_logger
from questions_ingest import ingest_pdf


def _env_first(*names: str, default: Optional[str] = None) -> Optional[str]:
    for n in names:
        v = os.getenv(n)
        if v is not None and str(v).strip():
            return str(v).strip()
    return default


def main() -> int:
    load_dotenv()
    os.environ.setdefault("LOG_LEVEL", "INFO")
    logger = get_logger("ingest.runner")

    pdf_path = _env_first("INGEST_PDF_PATH")
    version = _env_first("QUESTIONS_VERSION", "INGEST_VERSION")
    namespace = _env_first("QUESTIONS_NAMESPACE")
    index_name = _env_first("PINECONE_INDEX_NAME", default="interview_questions")
    skills_tax = _env_first("SKILLS_TAXONOMY")  # ignored
    domains_tax = _env_first("DOMAINS_TAXONOMY")  # ignored
    near_dup_raw = _env_first("NEAR_DUP_SIM")
    near_dup: Optional[float] = None
    try:
        if near_dup_raw is not None:
            near_dup = float(near_dup_raw)
    except Exception:
        logger.warning("Invalid NEAR_DUP_SIM value; ignoring")

    # Basic validation
    missing = []
    if not pdf_path:
        missing.append("INGEST_PDF_PATH")
    if not version:
        missing.append("QUESTIONS_VERSION")
    if not namespace:
        missing.append("QUESTIONS_NAMESPACE")
    if missing:
        logger.error("Missing required env: %s", ", ".join(missing))
        return 2

    logger.info("Starting ingestion")
    logger.info("pdf=%s version=%s namespace=%s index=%s", pdf_path, version, namespace, index_name)
    if skills_tax:
        logger.info("(ignored) skills_taxonomy=%s", skills_tax)
    if domains_tax:
        logger.info("(ignored) domains_taxonomy=%s", domains_tax)
    if near_dup is not None:
        logger.info("near_dup_sim=%.3f", near_dup)

    ingest_pdf(
        pdf_path=pdf_path,
        version=version,
        namespace=namespace,
        index_name=index_name,
        near_dup_sim=near_dup,
        skills_taxonomy_path=None,
        domains_taxonomy_path=None,
    )

    logger.info("Ingestion finished successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
