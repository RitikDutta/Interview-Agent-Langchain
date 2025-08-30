from __future__ import annotations
import os
from typing import List, Dict, Any, Optional, Tuple
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

# Clients
_oai = OpenAI()      # needs OPENAI_API_KEY
_pc  = Pinecone()    # needs PINECONE_API_KEY
_index = _pc.Index(PINECONE_INDEX_NAME)

# ---------- Utilities ----------
def _prep_text(s: str) -> str:
    return " ".join(s.strip().lower().split())

def _parse_canonical_from_id(_id: str) -> str:
    # expects "namespace:canonical"
    # fallback: last segment after colon(s)
    return _id.split(":")[-1]

# ---------- Batch embedding ----------
def embed_batch(texts: List[str]) -> List[List[float]]:
    """Embed many texts in a single OpenAI call."""
    if not texts:
        return []
    normed = [_prep_text(t) for t in texts]
    out = _oai.embeddings.create(model=OPENAI_EMBED_MODEL, input=normed)
    return [list(d.embedding) for d in out.data]

# Optional tiny cache for single embeds (used by single-query helpers)
@lru_cache(maxsize=4096)
def _embed_cached_single(norm_text: str) -> List[float]:
    out = _oai.embeddings.create(model=OPENAI_EMBED_MODEL, input=norm_text)
    return list(out.data[0].embedding)

def embed_single(text: str) -> List[float]:
    return _embed_cached_single(_prep_text(text))

# ---------- Pinecone single query (top_k=1) ----------
def pinecone_query_first(
    vector: List[float],
    namespace: str,
    category: Optional[str] = None,
    include_metadata: bool = False,  # speed: False (canonical comes from id)
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

# ---------- Public: single search ----------
def search_domains_first(query_text: str, category: Optional[str] = None) -> Optional[Dict[str, Any]]:
    vec = embed_single(query_text)
    return pinecone_query_first(vec, namespace=NAMESPACE_DOMAINS, category=category)

def search_skills_first(query_text: str, category: Optional[str] = None) -> Optional[Dict[str, Any]]:
    vec = embed_single(query_text)
    return pinecone_query_first(vec, namespace=NAMESPACE_SKILLS, category=category)

# ---------- Public: batch search (fast path) ----------
def search_batch_first(
    labels: List[str],
    namespace: str,
    category: Optional[str] = None,
    max_workers: int = 8,           # tune (<= 8 is usually fine)
) -> List[Optional[Dict[str, Any]]]:
    """
    Returns a list aligned with 'labels', each entry is best match dict or None.
    Embeds all labels in ONE call, then queries Pinecone in parallel (top_k=1).
    """
    if not labels:
        return []

    # 1) Deduplicate to avoid duplicate Pinecone queries
    #    (still preserve order via results map)
    idx_map: Dict[str, List[int]] = {}
    clean_labels = []
    for i, lab in enumerate(labels):
        key = _prep_text(lab)
        if key not in idx_map:
            idx_map[key] = []
            clean_labels.append(lab)   # keep original text for embedding (case doesn't matter)
        idx_map[key].append(i)

    # 2) Single batch embed
    vectors = embed_batch(clean_labels)

    # 3) Query Pinecone in parallel
    results_temp: List[Optional[Dict[str, Any]]] = [None] * len(clean_labels)

    def _worker(j: int):
        return j, pinecone_query_first(vectors[j], namespace=namespace, category=category)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_worker, j) for j in range(len(clean_labels))]
        for fut in as_completed(futures):
            j, res = fut.result()
            results_temp[j] = res

    # 4) Fan-out results back to original order
    final: List[Optional[Dict[str, Any]]] = [None] * len(labels)
    for key, positions in idx_map.items():
        # find index j for this key in clean_labels
        # since we appended in order, first occurrence index is:
        j = next(k for k, lab in enumerate(clean_labels) if _prep_text(lab) == key)
        for pos in positions:
            final[pos] = results_temp[j]

    return final

# Convenience wrappers
def search_domains_batch_first(labels: List[str], category: Optional[str] = None, max_workers: int = 8):
    return search_batch_first(labels, namespace=NAMESPACE_DOMAINS, category=category, max_workers=max_workers)

def search_skills_batch_first(labels: List[str], category: Optional[str] = None, max_workers: int = 8):
    return search_batch_first(labels, namespace=NAMESPACE_SKILLS, category=category, max_workers=max_workers)

# ---------- Quick CLI test ----------
if __name__ == "__main__":
    print("Index:", PINECONE_INDEX_NAME)
    dom = ["data scientist", "product owner", "cyber security", "finance"]
    skl = ["python3", "ml engineer", "search engine optimization", "netsec", "python3"]  # dup to test dedup

    # print("\n-- DOMAIN batch --")
    # for label, res in zip(dom, search_domains_batch_first(dom)):
    #     print(f"{label!r:28s} -> {res}")

    # print("\n-- SKILL batch --")
    # for label, res in zip(skl, search_skills_batch_first(skl)):
    #     print(f"{label!r:28s} -> {res}")

    # print(search_skills_first('descriptive_statistics'))
    print(search_skills_batch_first(skl))