# search_questions.py
import os
import re
from typing import Any, Dict, List, Optional, Union

from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv

# normalization helpers you provided
from canonical_normalizer import (
    search_domains_first,
    search_skills_first,
    search_domains_batch_first,
    search_skills_batch_first,
)

load_dotenv()

EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM   = 1536  # must match your ingestion

StrOrList = Union[str, List[str], tuple, set]

class QuestionSearch:
    def __init__(
        self,
        *,
        index_name: Optional[str] = None,
        namespace: Optional[str] = None,
        embedding_model: str = EMBED_MODEL,
        embedding_dim: int = EMBED_DIM,
    ):
        self.index_name = index_name or os.getenv("PINECONE_INDEX_NAME", "interview_questions")
        self.namespace  = namespace  or os.getenv("PINECONE_NAMESPACE", "questions_v3")
        self.embedding_model = embedding_model
        self.embedding_dim   = embedding_dim

        self._oa = self._get_openai_client()
        self._pc = self._get_pinecone_client()
        self._index = self._pc.Index(self.index_name)

        # known metadata fields (snake-case in DB)
        self._SCALAR_SNAKE = {"skill", "subskill", "difficulty", "domain", "lang", "type"}
        self._LIST_SNAKE   = {"categories", "tags"}
        self._RAW_FIELDS   = {"version"}  # keep raw, e.g. "v3"

    # ---------- public APIs ----------
    def query_search(self, query: str, *, top_k: int = 5) -> List[Dict[str, Any]]:
        """Semantic-only search."""
        qvec = self._embed_query(query)
        res = self._index.query(
            namespace=self.namespace,
            vector=qvec,
            top_k=top_k,
            include_metadata=True,
        )
        return self._extract_matches(res)

    def hybrid_search(
        self,
        query: str,
        filters: Optional[Dict[str, StrOrList]] = None,
        *,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Semantic + metadata filters (skill/domain are normalized via canonical_normalizer)."""
        qvec = self._embed_query(query)
        filt = self._build_filter_with_normalization(filters or {})
        res = self._index.query(
            namespace=self.namespace,
            vector=qvec,
            top_k=top_k,
            include_metadata=True,
            filter=filt or None,
        )
        return self._extract_matches(res)

    # ---------- internals ----------
    def _get_openai_client(self) -> OpenAI:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")
        return OpenAI(api_key=api_key)

    def _get_pinecone_client(self) -> Pinecone:
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("PINECONE_API_KEY not set")
        return Pinecone(api_key=api_key)

    def _embed_query(self, query: str) -> List[float]:
        resp = self._oa.embeddings.create(model=self.embedding_model, input=[query])
        return resp.data[0].embedding

    @staticmethod
    def _as_list(v: Optional[StrOrList]) -> List[str]:
        if v is None: return []
        if isinstance(v, (list, tuple, set)): return [*v]
        return [v]

    @staticmethod
    def _snake(s: Optional[str]) -> str:
        if not s: return ""
        s = s.strip().lower()
        s = re.sub(r"[^a-z0-9]+", "_", s)
        s = re.sub(r"_+", "_", s).strip("_")
        return s

    def _snake_list(self, vs: Optional[StrOrList]) -> List[str]:
        vals = [self._snake(x) for x in self._as_list(vs)]
        return [v for v in vals if v]

    def _raw_list(self, vs: Optional[StrOrList]) -> List[str]:
        vals = [str(x).strip() for x in self._as_list(vs)]
        return [v for v in vals if v]

    # --- NEW: normalize skill/domain in filters using your canonical_normalizer ---
    def _normalize_skill_filter(self, value: StrOrList) -> List[str]:
        items = self._as_list(value)
        if not items:
            return []
        if len(items) == 1:
            res = search_skills_first(items[0])
            canon = (res or {}).get("canonical")
            return [self._snake(canon or items[0])]
        # batch
        res_list = search_skills_batch_first(items)
        out = []
        for fallback, res in zip(items, res_list or []):
            canon = (res or {}).get("canonical") if isinstance(res, dict) else None
            out.append(self._snake(canon or fallback))
        return out

    def _normalize_domain_filter(self, value: StrOrList) -> List[str]:
        items = self._as_list(value)
        if not items:
            return []
        if len(items) == 1:
            res = search_domains_first(items[0])
            canon = (res or {}).get("canonical")
            return [self._snake(canon or items[0])]
        # batch
        res_list = search_domains_batch_first(items)
        out = []
        for fallback, res in zip(items, res_list or []):
            canon = (res or {}).get("canonical") if isinstance(res, dict) else None
            out.append(self._snake(canon or fallback))
        return out

    def _build_filter_with_normalization(self, filters: Dict[str, StrOrList]) -> Optional[Dict[str, Any]]:
        if not filters:
            return None

        out: Dict[str, Any] = {}

        for key, val in filters.items():
            k = str(key).strip()

            # allow native pinecone ops (power-user path)
            if isinstance(val, dict) and any(str(op).startswith("$") for op in val.keys()):
                out[k] = val
                continue

            if k == "skill":
                vals = self._normalize_skill_filter(val)
                if vals: out[k] = {"$in": vals}
                continue

            if k == "domain":
                vals = self._normalize_domain_filter(val)
                if vals: out[k] = {"$in": vals}
                continue

            if k in self._SCALAR_SNAKE or k in self._LIST_SNAKE:
                vals = self._snake_list(val)
                if vals: out[k] = {"$in": vals}
                continue

            if k in self._RAW_FIELDS:
                vals = self._raw_list(val)
                if vals: out[k] = {"$in": vals}
                continue

            # unknown key → raw strings
            vals = self._raw_list(val)
            if vals: out[k] = {"$in": vals}

        return out or None

    @staticmethod
    def _extract_matches(res) -> List[Dict[str, Any]]:
        return res.get("matches", []) if isinstance(res, dict) else res.matches


# ---------------- Example ----------------
if __name__ == "__main__":
    qs = QuestionSearch(namespace="questions_v4")

    print("=== Query search ===")
    results = qs.query_search("hypothesis testing", top_k=3)
    for r in results:
        print(r.score, r.metadata.get("text"))

    print("\n=== Hybrid search (normalized filters) ===")
    # Note: even if you pass variants, they’ll be normalized via canonical_normalizer.
    filt = {"difficulty": ["easy", "Medium"], "domain": ["Data Science", "analytics_science"]}
    results = qs.hybrid_search("anova assumptions", filt, top_k=5)
    for r in results:
        print(r.score, r.metadata.get("text"))
