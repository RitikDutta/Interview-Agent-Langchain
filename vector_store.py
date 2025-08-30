# vector_store.py

import os
import re
import time
from typing import List, Optional, Dict, Any

from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec


def _ts() -> int:
    return int(time.time())

_SNAKE_RE_1 = re.compile(r"[^A-Za-z0-9]+")
_SNAKE_RE_2 = re.compile(r"_+")

def _to_snake_lower(s: str) -> str:
    s = _SNAKE_RE_1.sub("_", (s or "").strip())
    s = _SNAKE_RE_2.sub("_", s).strip("_")
    return s.lower()

def _snake_list(xs: Optional[List[str]]) -> List[str]:
    return [_to_snake_lower(x) for x in (xs or []) if isinstance(x, str) and x.strip()]


class VectorStore:
    def __init__(self):
        self.index_name = os.getenv("PINECONE_INDEX_NAME", "interview_questions")
        self.namespace = os.getenv("PROFILES_NAMESPACE", "profiles_v1")

        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("PINECONE_API_KEY missing")
        self.pc = Pinecone(api_key=api_key)

        # create serverless index if missing (free/starter: aws/us-east-1)
        names = self.pc.list_indexes().names()
        if self.index_name not in names:
            self.pc.create_index(
                name=self.index_name,
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=os.getenv("PINECONE_CLOUD", "aws"),
                    region=os.getenv("PINECONE_REGION", "us-east-1"),
                ),
            )
        # prod best-practice is to target by host; name is fine for dev
        self.index = self.pc.Index(self.index_name)

        # embeddings
        oai_key = os.getenv("OPENAI_API_KEY")
        if not oai_key:
            raise ValueError("OPENAI_API_KEY missing")
        self.emb_client = OpenAI(api_key=oai_key)
        self.embedding_model = "text-embedding-3-small"

    def _embed_one(self, text: str) -> List[float]:
        # keep it tiny; it's a short summary
        resp = self.emb_client.embeddings.create(
            model=self.embedding_model,
            input=[text],
        )
        return resp.data[0].embedding

    def upsert_profile_snapshot(
        self,
        user_id: str,
        domain: Optional[str],
        summary: str,
        skills: Optional[List[str]] = None,
        strengths: Optional[List[str]] = None,
        weaknesses: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,  # retained for compatibility; ignored
    ) -> str:
        """
        Create/update the single profile vector for a user.

        Notes:
        - Stores ONLY canonical names for `domain` and `skills`.
        - Drops *_key fields and categories entirely.
        - `categories` arg is ignored for compatibility (safe no-op).
        """
        # Soft warning if callers still pass categories
        try:
            if categories:
                import logging
                logging.getLogger(__name__).debug(
                    "VectorStore.upsert_profile_snapshot: 'categories' is ignored now."
                )
        except Exception:
            pass

        vid = user_id
        vec = self._embed_one(summary)

        md: Dict[str, Any] = {
            "type": "profile_snapshot",
            "version": "v3",         # bumped: removed *_key + categories
            "user_id": user_id,

            # Canonical values only
            "domain": (domain or "unknown"),
            "skills": (skills or []),

            # Other attrs
            "strengths": strengths or [],
            "weaknesses": weaknesses or [],
            "user_summary": summary,
            "updated_at": _ts(),
        }

        self.index.upsert(
            vectors=[(vid, vec, md)],
            namespace=self.namespace,
        )
        print(f"[vector] upsert OK  ns={self.namespace} id={vid}")
        return vid

    def get_user_profile(self, user_id: str) -> dict | None:
        """
        Fetch the user's profile vector metadata from Pinecone.
        Returns a dict with {id, namespace, metadata} or None if absent.
        """
        vid = user_id
        out = self.index.fetch(ids=[vid], namespace=self.namespace)
        vec = out.vectors.get(vid) if hasattr(out, "vectors") else None
        if not vec:
            print(f"[vector] get_user_profile: not found  ns={self.namespace} id={vid}")
            return None

        profile = {
            "id": vid,
            "namespace": self.namespace,
            "metadata": dict(vec.metadata) if hasattr(vec, "metadata") else {},
            # To include values:
            # "values": list(vec.values) if hasattr(vec, "values") else None,
        }
        print(f"[vector] get_user_profile OK  ns={self.namespace} id={vid}")
        return profile
