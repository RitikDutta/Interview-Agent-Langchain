from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

from interview_flow.logging import get_logger

from .models import QuestionItem
from .utils import (
    sanitize_label,
    content_hash,
    pinecone_clean_meta,
)
from .segment import extract_pdf_blocks, get_segment_chain, segment_page_into_questions
from .clients import get_openai_client, embed_texts, ensure_pinecone_index, upsert_to_pinecone
from interview_flow.normalization.canonical_normalizer import (
    search_domains_first,
    search_skills_first,
    search_domains_batch_first,
    search_skills_batch_first,
)

logger = get_logger("ingest")


def _clean_batch(items: List[QuestionItem]) -> List[QuestionItem]:
    cleaned: List[QuestionItem] = []
    for q in items:
        qt = (q.question_text or "").strip()
        if len(qt.split()) >= 4:
            q.subskill = sanitize_label(q.subskill or "general", "general")
            q.skill = sanitize_label(q.skill, "misc")
            q.difficulty = sanitize_label(q.difficulty, "medium")
            q.domain = sanitize_label(q.domain, "misc")
            q.lang = sanitize_label(q.lang, "en")
            q.tags = [sanitize_label(t, "") for t in (q.tags or []) if t]
            cleaned.append(q)
    return cleaned


def ingest_pdf(
    pdf_path: str,
    version: str,
    namespace: str,
    index_name: Optional[str] = None,
    near_dup_sim: Optional[float] = None,
    skills_taxonomy_path: Optional[str] = None,   # ignored (legacy)
    domains_taxonomy_path: Optional[str] = None,  # ignored (legacy)
):
    index_name = index_name or os.getenv("PINECONE_INDEX_NAME", "interview_questions")
    index = ensure_pinecone_index(index_name=index_name, dimension=1536, metric="cosine")
    oa = get_openai_client()

    # Canonical mapping will be done via vector search (canonical_normalizer)

    source_pdf = os.path.basename(pdf_path)
    pages = extract_pdf_blocks(pdf_path)
    logger.info(f"[INGEST] File: {source_pdf}  Pages with text: {len(pages)}")
    logger.thinking("Segmenting PDF pages into interview questions with LLM")

    seg_chain = get_segment_chain()
    candidates: List[Dict[str, Any]] = []
    seen_hashes = set()

    for p in pages:
        items = segment_page_into_questions(seg_chain, p["text"])
        if not items:
            continue

        items = _clean_batch(items)

        # Batch canonicalization per page to reduce API calls
        def _key(s: str) -> str:
            return (s or "").strip().lower()

        uniq_skills_order: List[str] = []
        uniq_domains_order: List[str] = []
        seen_s, seen_d = set(), set()
        for it in items:
            ks, kd = _key(it.skill), _key(it.domain)
            if ks and ks not in seen_s:
                seen_s.add(ks); uniq_skills_order.append(it.skill)
            if kd and kd not in seen_d:
                seen_d.add(kd); uniq_domains_order.append(it.domain)

        skill_hits: List[Optional[Dict[str, Any]]] = []
        domain_hits: List[Optional[Dict[str, Any]]] = []
        try:
            skill_hits = search_skills_batch_first(uniq_skills_order) if uniq_skills_order else []
        except Exception:
            skill_hits = [None] * len(uniq_skills_order)
        try:
            domain_hits = search_domains_batch_first(uniq_domains_order) if uniq_domains_order else []
        except Exception:
            domain_hits = [None] * len(uniq_domains_order)

        smap = {_key(lbl): (hit or {}) for lbl, hit in zip(uniq_skills_order, skill_hits)}
        dmap = {_key(lbl): (hit or {}) for lbl, hit in zip(uniq_domains_order, domain_hits)}

        for it in items:
            qtext = it.question_text.strip()
            h = content_hash(qtext)
            if h in seen_hashes:
                continue
            seen_hashes.add(h)

            s_hit = smap.get(_key(it.skill)) or {}
            d_hit = dmap.get(_key(it.domain)) or {}

            skill_std = s_hit.get("canonical") or sanitize_label(it.skill, "misc")
            domain_std = d_hit.get("canonical") or sanitize_label(it.domain, "misc")
            skill_cat = s_hit.get("category") or ""
            domain_cat = d_hit.get("category") or ""

            categories = sorted({
                skill_cat if skill_cat else "uncategorized",
                domain_cat if domain_cat else "uncategorized",
            })

            meta = {
                "type": "question",
                "version": version,
                "text": qtext,
                "skill": skill_std,
                "subskill": it.subskill,
                "difficulty": it.difficulty,
                "domain": domain_std,
                "categories": categories,
                "tags": it.tags,
                "lang": it.lang,
                "source_pdf": source_pdf,
                "page": p["page"],
                "content_hash": h,
                "ingested_at": int(time.time()),
            }
            meta = pinecone_clean_meta(meta)

            skill_prefix = skill_std if skill_std != "misc" else "q"
            qid = f"{skill_prefix}_{h[:10]}"

            candidates.append({
                "id": qid,
                "text": qtext,
                "metadata": meta,
            })

    logger.info(f"[SEGMENT] Found unique questions: {len(candidates)}")

    if near_dup_sim:
        logger.info(f"[DEDUP] Near-dup threshold: {near_dup_sim}")
        pruned, seen_texts = [], []
        for c in candidates:
            a = set(c["text"].lower().split())
            if any(len(a & set(t.lower().split())) / max(1, len(a | set(t.lower().split()))) >= near_dup_sim
                   for t in seen_texts):
                continue
            pruned.append(c); seen_texts.append(c["text"])
        logger.thinking("Near-duplicate pruning retained %d/%d questions", len(pruned), len(candidates))
        logger.info(f"[DEDUP] Kept after near-dup pruning: {len(pruned)} (from {len(candidates)})")
        candidates = pruned

    if not candidates:
        logger.warning("[EXIT] No candidates to embed/upsert.")
        return

    texts = [c["text"] for c in candidates]
    logger.thinking("Embedding %d question texts for vector search", len(texts))
    embeddings = embed_texts(oa, texts)
    batch, BATCH, total = [], 200, 0
    for c, vec in zip(candidates, embeddings):
        batch.append({"id": c["id"], "embedding": vec, "metadata": c["metadata"]})
        if len(batch) >= BATCH:
            upsert_to_pinecone(index, batch, namespace=namespace)
            total += len(batch); logger.info(f"[UPSERT] {total} vectors upserted...")
            batch = []
    if batch:
        upsert_to_pinecone(index, batch, namespace=namespace)
        total += len(batch); logger.info(f"[UPSERT] {total} vectors upserted...")

    logger.info("[DONE] Ingestion complete.")
