import os
import re
import json
import time
import hashlib
import argparse
from typing import List, Dict, Any, Optional, Tuple, Literal

# PDF extraction
import fitz  # PyMuPDF

# LangChain (structured output with Gemini)
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# Embeddings
from openai import OpenAI

# Pinecone (classic client)
from pinecone import Pinecone, ServerlessSpec

from dotenv import load_dotenv
load_dotenv()


# --------------------------------------------------------------------
# Pydantic models for structured output (one atomic question + batch)
# --------------------------------------------------------------------
class QuestionItem(BaseModel):
    """Single, ask-able question + labels produced by the LLM."""
    question_text: str = Field(..., description="Single atomic question text; no numbering")
    skill: str = Field(..., description="Primary skill label, e.g., sql, python, stats, ml, rag, etc.")
    subskill: str = Field(..., description="Narrower area, e.g., joins, window_functions; 'general' if unclear")
    difficulty: Literal["easy", "medium", "hard"] = Field(..., description="Difficulty level")
    domain: str = Field(..., description="Domain: data_science, genai, backend, analytics, misc")
    tags: List[str] = Field(default_factory=list, description="1-5 lowercase keywords")
    lang: str = Field(default="en", description="2-letter language code (e.g., en)")

class QuestionBatch(BaseModel):
    """Top-level object to avoid provider issues with raw arrays."""
    questions: List[QuestionItem] = Field(default_factory=list)


# --------------------------------------------------------------------
# Helpers: normalization, hashing, labels
# --------------------------------------------------------------------
def normalize_text_for_hash(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"^q[\s\-:_]*\d+[a-z]?\s*[).:-]?\s*", "", s)  # strip "Q12)", "12." etc
    return s

def content_hash(text: str) -> str:
    return hashlib.sha256(normalize_text_for_hash(text).encode("utf-8")).hexdigest()

def sanitize_label(s: Optional[str], fallback: str) -> str:
    if not s:
        return fallback
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or fallback


# --------------------------------------------------------------------
# PDF extraction
# --------------------------------------------------------------------
def extract_pdf_blocks(pdf_path: str) -> List[Dict[str, Any]]:
    """Return [{page: int, text: str}, ...] for pages with text."""
    doc = fitz.open(pdf_path)
    blocks = []
    for page_index in range(len(doc)):
        page = doc[page_index]
        text = page.get_text("text")
        if text and text.strip():
            blocks.append({"page": page_index + 1, "text": text})
    doc.close()
    return blocks


# --------------------------------------------------------------------
# LangChain structured segmentation (Gemini 2.5 Flash)
# --------------------------------------------------------------------
def get_segment_chain():
    """Build a LangChain pipeline that returns a QuestionBatch (structured)."""
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.0,
    )

    structured_llm = llm.with_structured_output(QuestionBatch)  # enforce schema

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an expert curator of interview question banks. "
            "Split the given page text into a LIST of ATOMIC interview questions. "
            "Each item must be a single ask-able unit (split multi-part only if parts stand alone). "
            "Ignore non-questions. Fill all schema fields precisely and concisely."
        ),
        (
            "human",
            "Page text:\n{page_text}\n\n"
            "IMPORTANT:\n"
            "- Return many questions if present.\n"
            "- Keep fields concise. If a label is unclear, use fallbacks: skill=misc, subskill=general, domain=misc.\n"
        )
    ])

    # LCEL chain
    chain = prompt | structured_llm
    return chain


def segment_page_into_questions(chain, page_text: str) -> List[QuestionItem]:
    """Run the structured chain on a single page of text."""
    txt = (page_text or "").strip()
    if not txt:
        return []

    # Keep prompt sizes sane; Gemini Flash handles long inputs, but no need to overdo
    if len(txt) > 20000:
        txt = txt[:20000]

    try:
        batch: QuestionBatch = chain.invoke({"page_text": txt})
        items = batch.questions or []
    except Exception:
        # If structured parse fails, return empty (keep pipeline robust)
        items = []

    # Minimal cleanup + guardrails
    cleaned: List[QuestionItem] = []
    for q in items:
        qt = (q.question_text or "").strip()
        if len(qt.split()) >= 4:
            if not q.subskill:
                q.subskill = "general"
            # Normalize labels
            q.skill = sanitize_label(q.skill, "misc")
            q.subskill = sanitize_label(q.subskill, "general")
            q.difficulty = sanitize_label(q.difficulty, "medium")  # pydantic already restricts to literals
            q.domain = sanitize_label(q.domain, "misc")
            q.lang = sanitize_label(q.lang, "en")
            q.tags = [sanitize_label(t, "") for t in (q.tags or []) if t]
            cleaned.append(q)
    return cleaned


# --------------------------------------------------------------------
# OpenAI embeddings
# --------------------------------------------------------------------
def get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set")
    return OpenAI(api_key=api_key)

def embed_texts(client: OpenAI, texts: List[str], model: str = "text-embedding-3-small") -> List[List[float]]:
    if not texts:
        return []
    vectors: List[List[float]] = []
    BATCH = 128  # faster; safe in most environments
    for i in range(0, len(texts), BATCH):
        sl = texts[i:i+BATCH]
        resp = client.embeddings.create(model=model, input=sl)
        vectors.extend([d.embedding for d in resp.data])
    return vectors


# --------------------------------------------------------------------
# Pinecone utils
# --------------------------------------------------------------------
def get_pinecone_client() -> Pinecone:
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY not set")
    return Pinecone(api_key=api_key)

def ensure_pinecone_index(index_name: str, dimension: int = 1536, metric: str = "cosine"):
    """
    Creates/returns a serverless index.
    Starter/free plan: AWS us-east-1 only.
    """
    pc = get_pinecone_client()
    # list_indexes() may return objects; normalize to names safely:
    try:
        names = pc.list_indexes().names()
    except Exception:
        names = [ix["name"] if isinstance(ix, dict) else ix.name for ix in pc.list_indexes()]

    if index_name not in names:
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(cloud=os.getenv("PINECONE_CLOUD", "aws"),
                                region=os.getenv("PINECONE_REGION", "us-east-1"))
        )
        # optional: wait for ready

    # You can target by name (dev) or by host (prod best-practice).
    # Name is simpler and fine here:
    index = pc.Index(index_name)

    # If you want prod-style targeting by host (one fewer network hop):
    # host = pc.describe_index(index_name).host
    # index = pc.Index(host=host)

    return index

def upsert_to_pinecone(index, items: List[Dict[str, Any]], namespace: Optional[str] = None):
    if not items:
        return
    vectors = [(it["id"], it["embedding"], it.get("metadata", {})) for it in items]
    index.upsert(vectors=vectors, namespace=namespace)


# --------------------------------------------------------------------
# ingestion pipeline (structured)
# --------------------------------------------------------------------
def ingest_pdf(
    pdf_path: str,
    version: str,
    namespace: str,
    index_name: Optional[str] = None,
    near_dup_sim: Optional[float] = None
):
    index_name = index_name or os.getenv("PINECONE_INDEX_NAME", "interview_questions")
    index = ensure_pinecone_index(index_name=index_name, dimension=1536, metric="cosine")
    oa = get_openai_client()

    source_pdf = os.path.basename(pdf_path)
    pages = extract_pdf_blocks(pdf_path)
    print(f"[INGEST] File: {source_pdf}  Pages with text: {len(pages)}")

    # Build the structured segmentation chain once
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
                continue  # exact dup in this run
            seen_hashes.add(h)

            meta = {
                "type": "question",
                "text": qtext,
                "skill": it.skill,
                "subskill": it.subskill,
                "difficulty": it.difficulty,
                "domain": it.domain,
                "tags": it.tags,
                "lang": it.lang,
                "source_pdf": source_pdf,
                "page": p["page"],
                "version": version,
                "content_hash": h,
                "ingested_at": int(time.time())
            }
            skill_prefix = it.skill if it.skill != "misc" else "q"
            qid = f"{skill_prefix}_{h[:10]}"

            candidates.append({
                "id": qid,
                "text": qtext,
                "metadata": meta
            })

    print(f"[SEGMENT] Found unique questions: {len(candidates)}")

    # Optional cheap near-dup pruning (token Jaccard); keep pipeline light
    if near_dup_sim:
        print(f"[DEDUP] Near-dup threshold: {near_dup_sim}")
        pruned = []
        seen_texts = []
        for c in candidates:
            keep = True
            a = set(c["text"].lower().split())
            for t in seen_texts:
                b = set(t.lower().split())
                sim = len(a & b) / max(1, len(a | b))
                if sim >= near_dup_sim:
                    keep = False
                    break
            if keep:
                pruned.append(c)
                seen_texts.append(c["text"])
        print(f"[DEDUP] Kept after near-dup pruning: {len(pruned)} (from {len(candidates)})")
        candidates = pruned

    if not candidates:
        print("[EXIT] No candidates to embed/upsert.")
        return

    # Embed + upsert
    texts = [c["text"] for c in candidates]
    embeddings = embed_texts(oa, texts)
    batch = []
    BATCH = 200
    total = 0
    for c, vec in zip(candidates, embeddings):
        batch.append({"id": c["id"], "embedding": vec, "metadata": c["metadata"]})
        if len(batch) >= BATCH:
            upsert_to_pinecone(index, batch, namespace=namespace)
            total += len(batch)
            print(f"[UPSERT] {total} vectors upserted...")
            batch = []
    if batch:
        upsert_to_pinecone(index, batch, namespace=namespace)
        total += len(batch)
        print(f"[UPSERT] {total} vectors upserted...")

    print("[DONE] Ingestion complete.")


# --------------------------------------------------------------------
# Search
# --------------------------------------------------------------------
def search_questions(
    query: str,
    namespace: str,
    index_name: Optional[str] = None,
    top_k: int = 5,
    skill: Optional[str] = None,
    difficulty: Optional[str] = None,
    lang: Optional[str] = None,
    domain: Optional[str] = None
):
    index_name = index_name or os.getenv("PINECONE_INDEX_NAME", "interview_questions")
    index = ensure_pinecone_index(index_name=index_name, dimension=1536, metric="cosine")
    oa = get_openai_client()

    filt: Dict[str, Any] = {}
    if skill:      filt["skill"] = {"$eq": sanitize_label(skill, "misc")}
    if difficulty: filt["difficulty"] = {"$eq": sanitize_label(difficulty, "medium")}
    if lang:       filt["lang"] = {"$eq": sanitize_label(lang, "en")}
    if domain:     filt["domain"] = {"$eq": sanitize_label(domain, "misc")}

    qvec = embed_texts(oa, [query])[0]

    res = index.query(
        vector=qvec,
        top_k=top_k,
        include_metadata=True,
        namespace=namespace,
        filter=filt if filt else None
    )

    matches = res.get("matches", []) if isinstance(res, dict) else res.matches
    print("=== SEARCH RESULTS ===")
    for m in matches:
        md = m.get("metadata", {})
        print(f"- score={m.get('score'):.4f}  id={m.get('id')}")
        print(f"  skill={md.get('skill')}, subskill={md.get('subskill')}, difficulty={md.get('difficulty')}, domain={md.get('domain')}, lang={md.get('lang')}")
        print(f"  text: {md.get('text')[:220].strip()}{'...' if len(md.get('text',''))>220 else ''}")
        print(f"  src: {md.get('source_pdf')} (p.{md.get('page')})  version={md.get('version')}")
        print()


# --------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Ingest and search interview questions in Pinecone.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ingest = sub.add_parser("ingest", help="Ingest a PDF into Pinecone.")
    p_ingest.add_argument("--pdf", required=True, help="Path to the PDF containing questions.")
    p_ingest.add_argument("--version", required=True, help="Version label for this ingestion (e.g., v1).")
    p_ingest.add_argument("--namespace", required=True, help="Pinecone namespace to store vectors (e.g., questions_v1).")
    p_ingest.add_argument("--index-name", default=None, help="Override Pinecone index name (default from env).")
    p_ingest.add_argument("--near-dup-sim", type=float, default=None, help="Optional 0..1 similarity for near-dup pruning within batch (rough Jaccard).")

    p_search = sub.add_parser("search", help="Semantic search with optional filters.")
    p_search.add_argument("--query", required=True, help="What you’re looking for.")
    p_search.add_argument("--namespace", required=True, help="Namespace that holds your questions.")
    p_search.add_argument("--index-name", default=None, help="Override Pinecone index name (default from env).")
    p_search.add_argument("--top-k", type=int, default=5)
    p_search.add_argument("--skill", default=None)
    p_search.add_argument("--difficulty", default=None)
    p_search.add_argument("--lang", default=None)
    p_search.add_argument("--domain", default=None)

    args = parser.parse_args()

    if args.cmd == "ingest":
        ingest_pdf(
            pdf_path=args.pdf,
            version=args.version,
            namespace=args.namespace,
            index_name=args.index_name,
            near_dup_sim=args.near_dup_sim
        )
    elif args.cmd == "search":
        search_questions(
            query=args.query,
            namespace=args.namespace,
            index_name=args.index_name,
            top_k=args.top_k,
            skill=args.skill,
            difficulty=args.difficulty,
            lang=args.lang,
            domain=args.domain
        )


if __name__ == "__main__":
    main()
