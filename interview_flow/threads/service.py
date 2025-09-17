from __future__ import annotations

from typing import Any, Dict, Optional

from ..logging import get_logger
from ..infra.rdb import RelationalDB
from .. import start_thread as _start_thread, resume_thread as _resume_thread, get_current_state as _get_state


class ThreadService:
    """Backend thread/session helper to centralize thread_id handling and conversation logging.

    Responsibilities:
    - Resolve thread_id using provided value → stored in RDB → fallback to user_id
    - Start/resume interview threads via LangGraph
    - Persist thread_id mapping to the users table
    - Persist conversation messages to RDB (conversation_logs)
    - Fetch conversation history by thread_id
    """

    def __init__(self, rdb: Optional[RelationalDB] = None):
        self.logger = get_logger("threads")
        self.rdb = rdb or RelationalDB()
        # Ensure tables exist for conversation logging
        try:
            self.rdb.create_tables()
        except Exception:
            pass

    # ----------------- helpers -----------------
    def resolve_thread_id(self, user_id: str, provided: Optional[str] = None) -> str:
        tid = (provided or "").strip() if isinstance(provided, str) else (provided or "")
        if tid:
            return tid
        try:
            existing = self.rdb.get_user_thread_id(user_id) or ""
            if str(existing).strip():
                return str(existing).strip()
        except Exception:
            pass
        return user_id

    def save_thread_id(self, user_id: str, thread_id: str) -> None:
        try:
            self.rdb.set_user_thread_id(user_id=user_id, thread_id=thread_id)
        except Exception as e:
            self.logger.warning(f"save_thread_id: could not persist thread_id for user_id={user_id}: {e}")

    # ----------------- conversation logging -----------------
    def log(self, thread_id: str, user_id: str, role: str, content: str) -> None:
        try:
            if not content:
                return
            self.rdb.log_conversation_message(thread_id=thread_id, user_id=user_id, role=role, content=content)
        except Exception as e:
            self.logger.warning(f"log: failed to persist message ({role}): {e}")

    # ----------------- core actions -----------------
    def start(self, *, user_id: str, thread_id: Optional[str] = None, strategy: Optional[str] = None, username: Optional[str] = None) -> Dict[str, Any]:
        tid = self.resolve_thread_id(user_id, thread_id)
        out = _start_thread(thread_id=tid, user_id=user_id, strategy=strategy)
        self.save_thread_id(user_id, tid)
        state = _get_state(tid)

        # Try to extract assistant prompt/question to log
        try:
            intr = out.get("__interrupt__") if isinstance(out, dict) else None
            if intr and isinstance(intr, dict):
                prompt = intr.get("prompt") or intr.get("content") or "Continue…"
                self.log(tid, user_id, "assistant", str(prompt))
            else:
                gstate = str(state.get("graph_state") or "").strip().lower()
                if gstate in {"ask_domain", "get_domain"}:
                    self.log(tid, user_id, "assistant", "Please provide your domain of interest (e.g., Data Science, AI, Management, Etc). ")
                elif gstate in {"ask_resume", "get_resume_url"}:
                    self.log(tid, user_id, "assistant", "Please share your resume — you can paste a URL (http/https), a file:// URL, or a local path (absolute/relative, ~ ok). Type 'skip' to continue without a resume.")
                else:
                    qtext = (state.get("question") or "").strip() or "Proceed when ready."
                    self.log(tid, user_id, "assistant", qtext)
        except Exception:
            pass

        return {"thread_id": tid, "state": state, "out": out}

    def resume(self, *, user_id: str, content: Any, thread_id: Optional[str] = None) -> Dict[str, Any]:
        tid = self.resolve_thread_id(user_id, thread_id)
        # Log user message before invoking
        try:
            text = content if isinstance(content, (str, int, float)) else (content.get("content") if isinstance(content, dict) else str(content))
            if text:
                self.log(tid, user_id, "user", str(text))
        except Exception:
            pass

        out = _resume_thread(thread_id=tid, value=content)
        state = _get_state(tid)
        self.save_thread_id(user_id, tid)

        # Log assistant prompt/question after turn
        try:
            intr = out.get("__interrupt__") if isinstance(out, dict) else None
            if intr and isinstance(intr, dict):
                prompt = intr.get("prompt") or intr.get("content") or "Continue…"
                self.log(tid, user_id, "assistant", str(prompt))
            else:
                gstate = str(state.get("graph_state") or "").strip().lower()
                if gstate in {"ask_domain", "get_domain"}:
                    self.log(tid, user_id, "assistant", "Please provide your domain of interest (e.g., Data Science, AI, Management, Etc). ")
                elif gstate in {"ask_resume", "get_resume_url"}:
                    self.log(tid, user_id, "assistant", "Please share your resume — you can paste a URL (http/https), a file:// URL, or a local path (absolute/relative, ~ ok). Type 'skip' to continue without a resume.")
                else:
                    qtext = (state.get("question") or "").strip() or "Proceed when ready."
                    self.log(tid, user_id, "assistant", qtext)
        except Exception:
            pass

        return {"thread_id": tid, "state": state, "out": out}

    def get_conversation(self, *, thread_id: str, limit: Optional[int] = None) -> Dict[str, Any]:
        rows = self.rdb.get_conversation(thread_id, limit=limit)
        return {"thread_id": thread_id, "messages": rows}

