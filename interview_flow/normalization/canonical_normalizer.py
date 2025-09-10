from __future__ import annotations
import os
from typing import List, Dict, Any, Optional
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
from pinecone import Pinecone

# Namespaces
NAMESPACE_SKILLS  = "vector_skills"
NAMESPACE_DOMAINS = "vector_domains"

# Model
OPENAI_EMBED_MODEL = "text-embedding-3-small"  # fast & cheap

# Index
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX")
if not PINECONE_INDEX_NAME:
    raise ValueError("PINECONE_INDEX env var missing (e.g., 'interview-ai-db').")

# Clients (initialized at import time; assumes env keys are present)
_oai = OpenAI()      # needs OPENAI_API_KEY
_pc  = Pinecone()    # needs PINECONE_API_KEY
_index = _pc.Index(PINECONE_INDEX_NAME)


def _prep_text(s: str) -> str:
    return " ".join(s.strip().lower().split())


def _parse_canonical_from_id(_id: str) -> str:
    return _id.split(":")[-1]


def embed_batch(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    normed = [_prep_text(t) for t in texts]
    out = _oai.embeddings.create(model=OPENAI_EMBED_MODEL, input=normed)
    return [list(d.embedding) for d in out.data]


@lru_cache(maxsize=4096)
def _embed_cached_single(norm_text: str) -> List[float]:
    out = _oai.embeddings.create(model=OPENAI_EMBED_MODEL, input=norm_text)
    return list(out.data[0].embedding)


def embed_single(text: str) -> List[float]:
    return _embed_cached_single(_prep_text(text))


def pinecone_query_first(
    vector: List[float],
    namespace: str,
    category: Optional[str] = None,
    include_metadata: bool = False,
) -> Optional[Dict[str, Any]]:
    flt: Dict[str, Any] = {}
    if category:
        flt["category"] = {"$eq": category}

    res = _index.query(
        vector=vector,
        namespace=namespace,
        top_k=1,
        include_values=False,
        include_metadata=include_metadata,
        filter=flt or {},
    )
    if not res.matches:
        return None

    m = res.matches[0]
    canonical = _parse_canonical_from_id(m.id)
    return {
        "id": m.id,
        "canonical": canonical,
        "score": float(m.score),
        "metadata": getattr(m, "metadata", None) if include_metadata else None,
    }


def search_domains_first(query_text: str, category: Optional[str] = None) -> Optional[Dict[str, Any]]:
    vec = embed_single(query_text)
    return pinecone_query_first(vec, namespace=NAMESPACE_DOMAINS, category=category)


def search_skills_first(query_text: str, category: Optional[str] = None) -> Optional[Dict[str, Any]]:
    vec = embed_single(query_text)
    return pinecone_query_first(vec, namespace=NAMESPACE_SKILLS, category=category)


def search_batch_first(
    labels: List[str],
    namespace: str,
    category: Optional[str] = None,
    max_workers: int = 8,
) -> List[Optional[Dict[str, Any]]]:
    if not labels:
        return []

    idx_map: Dict[str, List[int]] = {}
    clean_labels = []
    for i, lab in enumerate(labels):
        key = _prep_text(lab)
        if key not in idx_map:
            idx_map[key] = []
            clean_labels.append(lab)
        idx_map[key].append(i)

    vectors = embed_batch(clean_labels)

    results_temp: List[Optional[Dict[str, Any]]] = [None] * len(clean_labels)

    def _worker(j: int):
        return j, pinecone_query_first(vectors[j], namespace=namespace, category=category)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_worker, j) for j in range(len(clean_labels))]
        for fut in as_completed(futures):
            j, res = fut.result()
            results_temp[j] = res

    final: List[Optional[Dict[str, Any]]] = [None] * len(labels)
    for key, positions in idx_map.items():
        j = next(k for k, lab in enumerate(clean_labels) if _prep_text(lab) == key)
        for pos in positions:
            final[pos] = results_temp[j]

    return final


def search_domains_batch_first(labels: List[str], category: Optional[str] = None, max_workers: int = 8):
    return search_batch_first(labels, namespace=NAMESPACE_DOMAINS, category=category, max_workers=max_workers)


def search_skills_batch_first(labels: List[str], category: Optional[str] = None, max_workers: int = 8):
    return search_batch_first(labels, namespace=NAMESPACE_SKILLS, category=category, max_workers=max_workers)

