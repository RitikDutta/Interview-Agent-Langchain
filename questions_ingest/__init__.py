"""
Standalone question ingestion/search module (separate from interview_flow).

Public APIs:
  - ingest_pdf(...)
  - search_questions(...)
"""

from __future__ import annotations

from .ingest import ingest_pdf
from .search import search_questions

__all__ = ["ingest_pdf", "search_questions"]
