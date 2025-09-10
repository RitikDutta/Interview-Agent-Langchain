# resume_etl.py (moved from root)
from __future__ import annotations
import os, tempfile, shutil, requests, json, logging, re
from pathlib import Path
from urllib.parse import urlparse, unquote

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
try:
    from langchain_community.document_loaders import PyMuPDFLoader
    HAS_PYMUPDF = True
except Exception:
    HAS_PYMUPDF = False

from ..infra.rdb import RelationalDB
from ..infra.vector_store import VectorStore

from ..normalization.canonical_normalizer import (
    search_domains_first,
    search_skills_first,
    search_domains_batch_first,
    search_skills_batch_first,
)

from ..logging import get_logger
log = get_logger("resume_etl")
_lvl = os.getenv("RESUME_ETL_LOG_LEVEL")
if _lvl:
    try:
        log.setLevel(getattr(logging, _lvl.upper()))
    except Exception:
        pass

load_dotenv()


class InterviewProfile(BaseModel):
    user_name: str = Field(description="Name of the user")
    domain: str = Field(description="Primary domain in snake_case")
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    technical_skills: List[str] = Field(default_factory=list)
    project_experience: List[str] = Field(default_factory=list)
    personality_traits: List[str] = Field(default_factory=list)
    user_summary: str = Field(default="")


_SNAKE_RE_1 = re.compile(r"[^A-Za-z0-9]+")
_SNAKE_RE_2 = re.compile(r"_+")


def to_snake_lower(s: str) -> str:
    s = _SNAKE_RE_1.sub("_", (s or "").strip())
    s = _SNAKE_RE_2.sub("_", s).strip("_")
    return s.lower()


class ResumeETL:
    def __init__(
        self,
        user_id: str,
        user_name_hint: Optional[str] = None,
        gemini_model: str = "gemini-2.5-flash",
        max_pages: int = 3,
        page_joiner: str = "\n\n--- PAGE BREAK ---\n\n",
        rdb: Optional[RelationalDB] = None,
        vdb: Optional[VectorStore] = None,
        verbose: bool = False,
    ):
        self.user_id = user_id
        self.user_name_hint = user_name_hint or ""
        self.max_pages = max_pages
        self.page_joiner = page_joiner
        self.verbose = verbose

        self.llm = ChatGoogleGenerativeAI(model=gemini_model)
        self.structured_llm = self.llm.with_structured_output(InterviewProfile)

        self.rdb = rdb or RelationalDB()
        self.vdb = vdb or VectorStore()

    def _resolve_to_local_path(self, resume_url_or_path: str) -> tuple[str, str | None]:
        src = (resume_url_or_path or "").strip()
        if not src:
            raise ValueError("Empty resume path/URL")

        if src.lower().startswith("file://"):
            url = urlparse(src)
            p = Path(unquote(url.path)).expanduser()
            if not p.exists():
                raise FileNotFoundError(f"file:// path not found: {p}")
            return str(p), None

        if src.lower().startswith(("http://", "https://")):
            tmp_dir = tempfile.mkdtemp(prefix="resume_etl_")
            local_path = os.path.join(tmp_dir, "resume.pdf")
            log.info(f"[extract] downloading resume from URL -> {src}")
            with requests.get(src, stream=True, timeout=60) as r:
                r.raise_for_status()
                ctype = (r.headers.get("Content-Type") or "").lower()
                if "pdf" not in ctype and not src.lower().endswith(".pdf"):
                    log.warning(f"[extract] Content-Type looks non-PDF: {ctype!r}. Proceeding anyway.")
                with open(local_path, "wb") as f:
                    for chunk in r.iter_content(8192):
                        if chunk:
                            f.write(chunk)
            log.info(f"[extract] saved to {local_path}")
            return local_path, tmp_dir

        p = Path(src).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Local resume path not found: {p}")
        return str(p), None

    def extract(self, resume_url_or_path: str) -> Dict[str, Any]:
        tmp_dir = None
        try:
            local_path, tmp_dir = self._resolve_to_local_path(resume_url_or_path)

            pages = []
            try:
                for i, d in enumerate(PyPDFLoader(local_path).load()):
                    if i >= self.max_pages:
                        break
                    pages.append((d.page_content or "").strip())
                log.info(f"[extract] PyPDFLoader OK, pages captured={len(pages)}")
            except Exception as e_pdf:
                if HAS_PYMUPDF:
                    docs = PyMuPDFLoader(local_path).load()
                    for i, d in enumerate(docs[: self.max_pages]):
                        pages.append((d.page_content or "").strip())
                    log.warning(
                        f"[extract] fallback PyMuPDFLoader OK, pages captured={len(pages)}; err={e_pdf}"
                    )
                else:
                    raise RuntimeError(f"PDF extraction failed and PyMuPDF not available: {e_pdf}") from e_pdf

            raw_text = self.page_joiner.join([p for p in pages if p])
            if not raw_text.strip():
                raise RuntimeError("No text extracted from resume")

            return {"raw_text": raw_text, "pages_used": len(pages)}
        finally:
            if tmp_dir and os.path.isdir(tmp_dir):
                shutil.rmtree(tmp_dir, ignore_errors=True)

    def transform(self, raw_text: str, pages_used: int) -> Dict[str, Any]:
        system_nudge = (
            "Extract a structured interview profile from the resume text. "
            "Be concise; if a field is missing, return an empty list/string. "
            "Use short bullet nouns for lists. "
            "Also write a short summary of the user in user_summary."
        )
        prompt = (
            f"{system_nudge}\n\n"
            f"Optional name hint: {self.user_name_hint}\n"
            f"=== RESUME EXCERPT (first {pages_used} pages) ===\n{raw_text}\n"
        )
        log.info("[transform] invoking structured LLM…")
        prof: InterviewProfile = self.structured_llm.invoke(prompt)

        log.debug("==== PROFILE (raw LLM) ====\n%s\n==========================",
                  json.dumps(prof.model_dump(), indent=2, ensure_ascii=False))
        log.thinking("LLM extracted domain=%s skills=%d strengths=%d weaknesses=%d",
                     prof.domain, len(prof.technical_skills or []), len(prof.strengths or []), len(prof.weaknesses or []))

        dom_key = to_snake_lower(prof.domain)
        dom_hit = search_domains_first(dom_key)
        dom_canon = dom_hit.get("canonical") if dom_hit else prof.domain
        dom_cat = dom_hit.get("category") if dom_hit else None
        log.thinking("Domain normalization: raw=%s → canonical=%s category=%s", prof.domain, dom_canon, dom_cat)

        skill_keys = [to_snake_lower(s) for s in (prof.technical_skills or [])]
        hits = search_skills_batch_first(skill_keys) if skill_keys else []
        skills_canon: List[str] = []
        misses: List[str] = []
        skill_cats: set[str] = set()

        for i, raw_lab in enumerate(skill_keys):
            h = hits[i] if i < len(hits) else None
            if h:
                skills_canon.append(h.get("canonical", raw_lab))
                if h.get("category"):
                    skill_cats.add(h["category"])
            else:
                original = (prof.technical_skills or [])[i] if i < len(prof.technical_skills or []) else raw_lab
                skills_canon.append(original)
                misses.append(original)
        if skills_canon:
            log.thinking("Normalized skills → %s", skills_canon)
        if misses:
            log.thinking("Unmatched skills → %s", misses)

        categories: List[str] = []
        seen = set()
        if dom_cat:
            seen.add(dom_cat); categories.append(dom_cat)
        for c in sorted(skill_cats):
            if c not in seen:
                seen.add(c); categories.append(c)

        normalized = {
            "user_name": prof.user_name,
            "domain": dom_canon,
            "technical_skills": sorted(set(skills_canon), key=str.lower),
            "strengths": prof.strengths,
            "weaknesses": prof.weaknesses,
            "project_experience": prof.project_experience,
            "personality_traits": prof.personality_traits,
            "user_summary": prof.user_summary,
            "categories": categories,
            "_normalize_debug": {
                "domain_raw": prof.domain,
                "skills_raw": prof.technical_skills,
                "unmatched_skills": misses,
            }
        }

        log.debug("==== PROFILE (standardized) ====\n%s\n================================",
                  json.dumps(normalized, indent=2, ensure_ascii=False))
        return normalized

    def load(self, normalized: Dict[str, Any], pages_used: int = 1) -> Dict[str, Any]:
        log.info("[load] ensuring tables…")
        self.rdb.create_tables()

        log.info("[load] upsert users row in MySQL…")
        self.rdb.upsert_user(
            user_id=self.user_id,
            name=normalized["user_name"] or None,
            domain=normalized["domain"] or None,
            skills=normalized["technical_skills"],
            strengths=normalized["strengths"],
            weaknesses=normalized["weaknesses"],
            categories=normalized.get("categories", []),
        )

        log.info("[load] ensure academic_summary row…")
        self.rdb.ensure_academic_summary(self.user_id)

        log.info("[load] upsert profile snapshot vector…")
        vec_id = self.vdb.upsert_profile_snapshot(
            user_id=self.user_id,
            domain=normalized["domain"],
            summary=normalized["user_summary"],
            skills=normalized["technical_skills"],
            strengths=normalized["strengths"],
            weaknesses=normalized["weaknesses"],
            categories=normalized.get("categories", []),
        )

        result = {
            "user_id": self.user_id,
            "vector_id": vec_id,
            "namespace": self.vdb.namespace,
            "pages_used": pages_used,
        }
        log.debug("==== LOAD RESULT ====\n%s\n=====================",
                  json.dumps(result, indent=2))
        log.info("[load] done")
        return result

    def run(self, resume_url_or_path: str) -> Dict[str, Any]:
        log.info(f"[run] start ETL for user_id={self.user_id}")
        ex = self.extract(resume_url_or_path)
        log.info(f"[run] extracted pages={ex['pages_used']}")
        normalized = self.transform(ex["raw_text"], ex["pages_used"])
        load_res = self.load(normalized, ex["pages_used"])
        log.info("[run] ETL complete")
        return {"profile": normalized, "load_result": load_res}
