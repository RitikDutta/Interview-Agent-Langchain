from __future__ import annotations

import os
from typing import Any, Dict, List, Literal, Optional, Union

from interview_flow.logging import get_logger
from .clients import ensure_pinecone_index, get_openai_client, embed_texts
from .utils import _build_sparse_from_text, _snake_list, _canon_list

logger = get_logger("ingest.search")


def _build_filter(
    *,
    skill=None, difficulty=None, lang=None, domain=None,
    categories=None, subskill=None, tags=None,
    type=None, version=None,
) -> Dict[str, Any]:
    filt: Dict[str, Any] = {}

    def put(key, vals, snakefy=True):
        if vals is None:
            return
        vv = _snake_list(vals) if snakefy else _canon_list(vals)
        if vv:
            filt[key] = {"$in": vv}

    put("skill", skill)
    put("difficulty", difficulty)
    put("lang", lang)
    put("domain", domain)
    put("subskill", subskill)
    put("type", type)
    put("version", version, snakefy=False)  # keep raw version

    cv = _snake_list(categories)
    if cv:
        filt["categories"] = {"$in": cv}
    tv = _snake_list(tags)
    if tv:
        filt["tags"] = {"$in": tv}
    return filt


def search_questions(
    *,
    query: Optional[str],
    namespace: str,
    index_name: Optional[str] = None,
    top_k: int = 5,
    mode: Literal["semantic", "filters", "hybrid"] = "semantic",
    # metadata filters (string or list[str], ANY-of semantics)
    skill: Optional[Literal["easy", "medium", "hard"]] = None,
    difficulty: Optional[Union[str, List[str]]] = None,
    lang: Optional[Union[str, List[str]]] = None,
    domain: Optional[Union[str, List[str]]] = None,
    categories: Optional[Union[str, List[str]]] = None,
    subskill: Optional[Union[str, List[str]]] = None,
    tags: Optional[Union[str, List[str]]] = None,
    type: Optional[Union[str, List[str]]] = None,
    version: Optional[Union[str, List[str]]] = None,
):
    index_name = index_name or os.getenv("PINECONE_INDEX_NAME", "interview_questions")
    index = ensure_pinecone_index(index_name=index_name, dimension=1536, metric="cosine")

    filt = _build_filter(
        skill=skill, difficulty=difficulty, lang=lang, domain=domain,
        categories=categories, subskill=subskill, tags=tags,
        type=type, version=version,
    )

    oa = get_openai_client()
    dense_vec: Optional[List[float]] = None
    sparse_vec: Optional[Dict[str, List[float]]] = None

    if mode == "semantic":
        if not query:
            raise ValueError("semantic mode requires a non-empty 'query'")
        dense_vec = embed_texts(oa, [query])[0]
    elif mode == "hybrid":
        if not query:
            raise ValueError("hybrid mode requires a non-empty 'query'")
        dense_vec = embed_texts(oa, [query])[0]
        sparse_vec = _build_sparse_from_text(query)
    elif mode == "filters":
        dense_vec = [0.0] * 1536
    else:
        raise ValueError("mode must be one of: 'semantic', 'filters', 'hybrid'")

    kwargs = dict(
        top_k=top_k,
        include_metadata=True,
        namespace=namespace,
        filter=filt or None,
    )
    if dense_vec is not None:
        kwargs["vector"] = dense_vec
    if sparse_vec is not None:
        kwargs["sparse_vector"] = sparse_vec

    logger.thinking("Search intent: mode=%s, top_k=%d, filter_keys=%s, has_query=%s", mode, top_k, list((filt or {}).keys()), bool(query))
    res = index.query(**kwargs)
    matches = res.get("matches", []) if isinstance(res, dict) else res.matches

    logger.info("=== SEARCH RESULTS ===")
    for m in matches:
        md = m.get("metadata", {})
        logger.info(f"- score={m.get('score'):.4f}  id={m.get('id')}")
        logger.info(f"  skill={md.get('skill')} | domain={md.get('domain')} | categories={md.get('categories')}")
        logger.info(f"  subskill={md.get('subskill')}, difficulty={md.get('difficulty')}, lang={md.get('lang')}")
        logger.info(f"  tags={md.get('tags')} | type={md.get('type')} | version={md.get('version')}")
        text = (md.get('text') or '')
        trunc = (text[:220].strip() + ('...' if len(text) > 220 else ''))
        logger.info(f"  text: {trunc}")
        logger.info(f"  src: {md.get('source_pdf')} (p.{md.get('page')})  content_hash={md.get('content_hash')}")
        logger.info("")

