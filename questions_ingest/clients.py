from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

load_dotenv()


def get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set")
    return OpenAI(api_key=api_key)


def embed_texts(client: OpenAI, texts: List[str], model: str = "text-embedding-3-small") -> List[List[float]]:
    if not texts:
        return []
    vectors: List[List[float]] = []
    BATCH = 128
    for i in range(0, len(texts), BATCH):
        sl = texts[i:i + BATCH]
        resp = client.embeddings.create(model=model, input=sl)
        vectors.extend([d.embedding for d in resp.data])
    return vectors


def get_pinecone_client() -> Pinecone:
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY not set")
    return Pinecone(api_key=api_key)


def ensure_pinecone_index(index_name: str, dimension: int = 1536, metric: str = "cosine"):
    pc = get_pinecone_client()
    try:
        names = pc.list_indexes().names()
    except Exception:
        names = [ix["name"] if isinstance(ix, dict) else ix.name for ix in pc.list_indexes()]
    if index_name not in names:
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(
                cloud=os.getenv("PINECONE_CLOUD", "aws"),
                region=os.getenv("PINECONE_REGION", "us-east-1"),
            ),
        )
    return pc.Index(index_name)


def upsert_to_pinecone(index, items: List[Dict[str, Any]], namespace: Optional[str] = None):
    if not items:
        return
    vectors = [(it["id"], it["embedding"], it.get("metadata", {})) for it in items]
    index.upsert(vectors=vectors, namespace=namespace)

