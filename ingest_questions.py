import os
import re
import json
import time
import hashlib
from typing import List, Dict, Any, Optional, Tuple, Literal, Union
from difflib import get_close_matches

# ------------------------------- Dependencies --------------------------------
import fitz  # PyMuPDF
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
load_dotenv()
from log_utils import get_logger
logger = get_logger("ingest")

# =============================== MODELS ======================================

class QuestionItem(BaseModel):
    question_text: str
    skill: str = "misc"                       # default
    subskill: str = "general"                 # default
    difficulty: Literal["easy","medium","hard"] = "medium"
    domain: str = "misc"                      # default
    tags: List[str] = Field(default_factory=list)
    lang: str = "en"

class QuestionBatch(BaseModel):
    questions: List[QuestionItem] = Field(default_factory=list)

# ========================= COMMON HELPERS ====================================

def snake(s: Optional[str]) -> str:
    if not s:
        return ""
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s

def sanitize_label(s: Optional[str], fallback: str) -> str:
    out = snake(s)
    return out or fallback

def normalize_text_for_hash(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"^q[\s\-:_]*\d+[a-z]?\s*[).:-]?\s*", "", s)
    return s

def content_hash(text: str) -> str:
    return hashlib.sha256(normalize_text_for_hash(text).encode("utf-8")).hexdigest()

# Pinecone-safe metadata cleaner
ALLOWED_SCALARS = (str, int, float, bool)
def pinecone_clean_meta(md: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in md.items():
        if v is None:
            continue
        if isinstance(v, list):
            lv: List[str] = []
            for x in v:
                if isinstance(x, ALLOWED_SCALARS):
                    sx = x if isinstance(x, str) else str(x)
                    if sx != "":
                        lv.append(sx)
            if lv:
                out[k] = lv
        elif isinstance(v, ALLOWED_SCALARS):
            out[k] = v
        else:
            out[k] = str(v)
    return out

# ============================== TAXONOMY =====================================

class TaxIndex:
    """Token -> {canonical_snake, category} where token is any alias in snake_case."""
    def __init__(self, records: List[Dict[str, Any]]):
        self.by_token: Dict[str, Dict[str, str]] = {}
        for rec in records or []:
            canon_snake = snake(rec.get("canonical", ""))
            cat_snake = snake(rec.get("category", ""))  # may be empty
            toks = {canon_snake} | {snake(s) for s in rec.get("synonyms", [])}
            for t in toks:
                if t:
                    self.by_token[t] = {"canonical_snake": canon_snake, "category": cat_snake}

    def lookup(self, raw: str) -> Tuple[str, str, bool]:
        """
        Return (canonical_snake, category_or_empty, matched_bool).
        If no match: returns (snake(raw) or 'misc', '', False).
        """
        tok = snake(raw)
        if tok in self.by_token:
            info = self.by_token[tok]
            return info["canonical_snake"], info["category"], True
        if tok:
            match = get_close_matches(tok, self.by_token.keys(), n=1, cutoff=0.9)
            if match:
                info = self.by_token[match[0]]
                return info["canonical_snake"], info["category"], True
        return (tok or "misc"), "", False

def load_taxonomy(path: Optional[str]) -> TaxIndex:
    if not path:
        return TaxIndex([])
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Taxonomy at {path} must be a JSON array of records")
    return TaxIndex(data)

# ============================ PDF + LLM ======================================

def extract_pdf_blocks(pdf_path: str) -> List[Dict[str, Any]]:
    doc = fitz.open(pdf_path)
    blocks = []
    for i in range(len(doc)):
        txt = doc[i].get_text("text")
        if txt and txt.strip():
            blocks.append({"page": i + 1, "text": txt})
    doc.close()
    return blocks

def get_segment_chain():
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=1.0)
    structured_llm = llm.with_structured_output(QuestionBatch)
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an expert curator of interview question banks. "
         "Split the given page text into a LIST of ATOMIC interview questions. "
         "Each item must be a single ask-able unit (split multi-part only if parts stand alone). "
         "Ignore non-questions. Fill all schema fields precisely and concisely."),
        ("human",
         "Page text:\n{page_text}\n\n"
         "IMPORTANT:\n"
         "- Return many questions if present.\n"
         "- Keep fields concise. If a label is unclear, use fallbacks: skill=misc, subskill=general, domain=misc.\n"),
    ])
    return prompt | structured_llm

def segment_page_into_questions(chain, page_text: str) -> List[QuestionItem]:
    txt = (page_text or "").strip()
    if not txt:
        return []
    if len(txt) > 20000:
        txt = txt[:20000]
    try:
        batch: QuestionBatch = chain.invoke({"page_text": txt})
        items = batch.questions or []
    except Exception:
        items = []
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

# ===================== OPENAI + PINECONE =====================================

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

# ============================ INGEST =========================================

def ingest_pdf(
    pdf_path: str,
    version: str,
    namespace: str,
    index_name: Optional[str] = None,
    near_dup_sim: Optional[float] = None,
    skills_taxonomy_path: Optional[str] = None,
    domains_taxonomy_path: Optional[str] = None,
):
    index_name = index_name or os.getenv("PINECONE_INDEX_NAME", "interview_questions")
    index = ensure_pinecone_index(index_name=index_name, dimension=1536, metric="cosine")
    oa = get_openai_client()

    # load taxonomies once
    skill_tax = load_taxonomy(skills_taxonomy_path)
    domain_tax = load_taxonomy(domains_taxonomy_path)

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

        for it in items:
            qtext = it.question_text.strip()
            h = content_hash(qtext)
            if h in seen_hashes:
                continue
            seen_hashes.add(h)

            # Standardize via taxonomies
            skill_std, skill_cat, _ = skill_tax.lookup(it.skill)
            domain_std, domain_cat, _ = domain_tax.lookup(it.domain)

            # Categories: taxonomy category if present; else 'uncategorized'
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
                "categories": categories,     # list[str]
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

    # Optional near-duplicate pruning (token Jaccard)
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

    # Embed + upsert
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

# ============================== SEARCH =======================================

def _as_list(v: Optional[Union[str, List[str], tuple, set]]) -> List[str]:
    if v is None: return []
    if isinstance(v, (list, tuple, set)): return [*v]
    return [v]

def _snake_list(vs) -> List[str]:
    out = [sanitize_label(x, "") for x in _as_list(vs)]
    return [x for x in out if x]

def _canon_list(vs) -> List[str]:
    out = [str(x).strip() for x in _as_list(vs)]
    return [x for x in out if x]

def _build_sparse_from_text(text: str) -> Dict[str, List[float]]:
    """
    Simple keyword sparse vector for hybrid search.
    Weights are 1.0 per unique token.
    """
    tokens = list({t for t in re.findall(r"[a-z0-9]+", text.lower()) if t})
    if not tokens:
        return {"indices": [], "values": []}
    # Local compact indices per query. Pinecone accepts indices and values.
    return {"indices": list(range(len(tokens))), "values": [1.0] * len(tokens)}

def _build_filter(
    *,
    skill=None, difficulty=None, lang=None, domain=None,
    categories=None, subskill=None, tags=None,
    type=None, version=None
) -> Dict[str, Any]:
    filt: Dict[str, Any] = {}
    def put(key, vals, snakefy=True):
        if vals is None: return
        if snakefy:
            vv = _snake_list(vals)
        else:
            vv = _canon_list(vals)
        if vv:
            filt[key] = {"$in": vv}
    put("skill", skill)
    put("difficulty", difficulty)
    put("lang", lang)
    put("domain", domain)
    put("subskill", subskill)
    put("type", type)           # e.g., "question"
    put("version", version, snakefy=False)  # keep version raw ("v3", "v2.1")
    # list fields
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
    """
    mode:
      - 'semantic': dense vector from embeddings (needs query)
      - 'filters' : metadata filter only; uses zero vector
      - 'hybrid'  : dense + sparse keywords (needs query)
    """
    index_name = index_name or os.getenv("PINECONE_INDEX_NAME", "interview_questions")
    index = ensure_pinecone_index(index_name=index_name, dimension=1536, metric="cosine")

    filt = _build_filter(
        skill=skill, difficulty=difficulty, lang=lang, domain=domain,
        categories=categories, subskill=subskill, tags=tags,
        type=type, version=version
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
        # Pinecone requires a vector or an ID; use a zero-vector for neutral similarity
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
        logger.info(f"  text: {md.get('text')[:220].strip()}{'...' if len(md.get('text',''))>220 else ''}")
        logger.info(f"  src: {md.get('source_pdf')} (p.{md.get('page')})  content_hash={md.get('content_hash')}")
        logger.info("")

# ============================== EXAMPLE RUN ==================================
if __name__ == "__main__":
    pdf_path = "/home/codered/mystuff/progs/python/interview-mentor-Langraph/questionbank/data_science_2.pdf"
    version = "v4"
    namespace = "questions_v4"
    index_name = os.getenv("PINECONE_INDEX_NAME", "interview-ai-db")
    skills_taxonomy_path = os.getenv("SKILLS_TAXONOMY", "case_skills.json")
    domains_taxonomy_path = os.getenv("DOMAINS_TAXONOMY", "case_domain.json")

    # --- Ingest ---
    # ingest_pdf(
    #     pdf_path=pdf_path,
    #     version=version,
    #     namespace=namespace,
    #     index_name=index_name,
    #     near_dup_sim=None,
    #     skills_taxonomy_path=skills_taxonomy_path,
    #     domains_taxonomy_path=domains_taxonomy_path,
    # )

    # --- Search examples ---
    # 1) semantic search
    search_questions(query="type 1 and type 2 error", namespace=namespace, mode="semantic")

    # 2) filters-only search (no query vector, uses zero dense vector)
    # search_questions(query=None, namespace=namespace, mode="filters",
    #                  categories=["technology"], difficulty=["easy","medium"], domain="data_science")

    # 3) hybrid search (dense + simple keyword sparse)
    # search_questions(query="regularization l1 l2", namespace=namespace, mode="hybrid",
    #                  categories=["ai_ml"], tags=["l1_regularization", "l2_regularization"], top_k=5)
