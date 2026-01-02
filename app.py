# monitor_app.py
from __future__ import annotations
import json
from datetime import datetime
from typing import Any, Dict, Optional

from flask import Flask, jsonify, render_template, render_template_string, request, send_from_directory, url_for
import logging
import os
from threading import Thread
from queue import Queue, Empty
from interview_flow.logging import get_logger
import tempfile
import secrets
import time
from pathlib import Path
from werkzeug.utils import secure_filename

# Imports
from interview_flow.infra.rdb import RelationalDB
from interview_flow.infra.vector_store import VectorStore
from interview_flow import (
    get_current_state,
    start_thread,
    resume_thread,
)
from interview_flow.threads import ThreadService


app = Flask(__name__)
logger = get_logger("monitor")

# Create clients. Requires env vars.
rdb = RelationalDB()
vdb = VectorStore()
threads = ThreadService(rdb)

# Uploaded resume storage (served back as a temp link)
UPLOAD_ROOT = Path(tempfile.gettempdir()) / "resume_uploads"
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
UPLOAD_TTL_SECONDS = int(os.getenv("RESUME_UPLOAD_TTL_SECONDS") or 3600)

# -----------------------------------------------------------------------------
# Runtime state cache (in-process).
# Updated by the LangGraph loop.
# -----------------------------------------------------------------------------
_STATE_CACHE: dict[str, dict] = {}


def set_state_snapshot(user_id: str, state: Dict[str, Any]) -> None:
    """Register the latest runtime state for a user."""
    _STATE_CACHE[user_id] = dict(state or {})

def get_state_snapshot(user_id: str) -> Dict[str, Any]:
    """Return latest state, or {}."""
    return _STATE_CACHE.get(user_id, {})


# -----------------------------------------------------------------------------
# Index
# -----------------------------------------------------------------------------
@app.get("/")
def index_page():
    """Simple index with links to Agent UI and Monitor UI."""
    default_user = os.getenv("Test_USER_ID") or os.getenv("TEST_USER_ID") or "test_user_022"
    username = os.getenv("TEST_USERNAME") or default_user
    strategy = os.getenv("TEST_STRATEGY") or "scores"
    return render_template("index.html", user_id=default_user, username=username, strategy=strategy)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _safe(obj: Any) -> Any:
    """Make values JSON-friendly for API and template."""
    if isinstance(obj, (datetime,)):
        return obj.isoformat()
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="ignore")
    return obj

def _deep_safe(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): _deep_safe(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_deep_safe(x) for x in o]
    return _safe(o)

def fetch_all(user_id: str) -> Dict[str, Any]:
    # RDBMS
    rdb_profile = rdb.get_user_profile(user_id) or {}
    rdb_scores  = rdb.get_user_academic_score(user_id) or {}
    try:
        rdb_history = rdb.get_conversation_history(user_id) or []
    except Exception:
        rdb_history = []

    # Vector DB
    vec = vdb.get_user_profile(user_id) or {}
    vec_meta = (vec.get("metadata") or {})

    # State
    state = get_state_snapshot(user_id) or {}

    data = {
        "user_id": user_id,
        "thread_id": rdb_profile.get("thread_id"),
        "rdb": {
            "profile": rdb_profile,
            "scores": rdb_scores,
            "history": rdb_history,
        },
        "vector": {
            "id": vec.get("id"),
            "namespace": vec.get("namespace"),
            "metadata": vec_meta,
        },
        "state": state,
    }
    return _deep_safe(data)

def _cleanup_old_uploads(max_age_seconds: int = UPLOAD_TTL_SECONDS) -> None:
    """Best-effort cleanup for stale temp uploads."""
    try:
        now = time.time()
        for p in UPLOAD_ROOT.glob("*"):
            if not p.is_file():
                continue
            age = now - p.stat().st_mtime
            if age > max_age_seconds:
                try:
                    p.unlink(missing_ok=True)
                except Exception:
                    pass
    except Exception as e:
        logger.warning(f"upload cleanup failed: {e}")

# -----------------------------------------------------------------------------
# User Uploads
# -----------------------------------------------------------------------------
@app.post("/upload-resume-to-graph")
def upload_resume_to_graph():
    """
    1. Receives PDF from browser.
    2. Saves it to a temp file exposed as an HTTP link.
    3. Returns the link so the client can send it through the normal chat flow.
    """
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file part"}), 400
    
    file = request.files['file']
    user_id = (request.form.get('user_id') or "").strip()
    
    # Resolve the thread_id just like the chat endpoint does
    thread_id = threads.resolve_thread_id(user_id, request.form.get('thread_id'))

    if not user_id:
        return jsonify({"status": "error", "message": "Missing user_id"}), 400

    if file.filename == '':
        return jsonify({"status": "error", "message": "No selected file"}), 400

    try:
        _cleanup_old_uploads()
        raw_name = secure_filename(file.filename) or "resume.pdf"
        token = secrets.token_hex(8)
        stored_name = f"{token}_{raw_name}"
        dest = UPLOAD_ROOT / stored_name
        dest.parent.mkdir(parents=True, exist_ok=True)

        file.save(dest)

        resume_url = url_for("serve_uploaded_resume", filename=stored_name, _external=True)
        logger.info(f"Saved upload to {dest}, generated temp link for user {user_id}")

        return jsonify({
            "status": "success",
            "resume_url": resume_url,
            "resume_path": str(dest),
            "thread_id": thread_id,
            "message": "Resume uploaded. Paste the link into chat to process."
        })

    except Exception as e:
        logger.error(f"Upload failed: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.get("/uploads/resumes/<path:filename>")
def serve_uploaded_resume(filename: str):
    safe_name = secure_filename(filename)
    if not safe_name:
        return jsonify({"status": "error", "message": "Invalid filename"}), 404
    try:
        return send_from_directory(str(UPLOAD_ROOT), safe_name, mimetype="application/pdf")
    except Exception:
        return jsonify({"status": "error", "message": "File not found"}), 404

# -----------------------------------------------------------------------------
# API (JSON)
# -----------------------------------------------------------------------------
@app.get("/monitor/api/<user_id>.json")
def api_monitor(user_id: str):
    logger.info(f"GET /monitor/api/{user_id}.json")
    return jsonify(fetch_all(user_id))

# -----------------------------------------------------------------------------
# Agent (1:1 Interview) UI
# -----------------------------------------------------------------------------
@app.get("/agent")
def agent_page():
    """Serve the 1:1 interview UI.
    Accepts optional query params: user_id, username, strategy
    """
    user_id = (request.args.get("user_id") or "").strip()
    username = (request.args.get("username") or request.args.get("user") or request.args.get("name") or "").strip()
    # Default user_id to username when user_id not provided
    if not user_id and username:
        user_id = username
    strategy = (request.args.get("strategy") or "").strip()
    logger.info(f"GET /agent user_id={user_id or '-'} username={username or '-'} strategy={strategy or '-'}")
    return render_template("agent_lang.html", user_id=user_id, username=username, strategy=strategy)


def _sse(data: dict) -> str:
    return "data: " + json.dumps(data) + "\n\n"


def _resolve_thread_id(user_id: str, provided: Optional[str]) -> str:
    return threads.resolve_thread_id(user_id, provided)


@app.post("/chat")
def chat_sse():
    """
    Stream server-sent events for agent conversation.
    Expects JSON body: {user_id, thread_id, message}
    """
    body = request.get_json(silent=True) or {}
    user_id = str(body.get("user_id") or "").strip() or "guest_user"
    thread_id = _resolve_thread_id(user_id, body.get("thread_id"))
    message = str(body.get("message") or "").strip()
    username = (body.get("username") or "").strip()
    strategy = (body.get("strategy") or "scores").strip() or "scores"

    logger.info(f"POST /chat user_id={user_id} thread_id={thread_id} msg_len={len(message)}")

    def generate():
        # Stream events as they happen using a background worker + queue
        q: Queue = Queue()

        # THINKING log handler → enqueue status events immediately
        base_logger = get_logger()

        class _ThinkingHandler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                try:
                    if getattr(logging, "THINKING", 15) != record.levelno:
                        return
                    if not str(record.name).startswith("app."):
                        return
                    q.put({"type": "status", "content": f"Thinking: {record.getMessage()}"})
                except Exception:
                    pass

        h = _ThinkingHandler(level=getattr(logging, "THINKING", 15))
        base_logger.addHandler(h)

        # Worker function that runs the turn and enqueues events
        def _worker():
            try:
                if message == "<BEGIN_INTERVIEW>":
                    q.put({"type": "status", "content": "Starting interview session…"})
                    res = threads.start(user_id=user_id, thread_id=thread_id, strategy=strategy, username=username)
                    out = res.get("out", {})
                    try:
                        if username:
                            rdb.upsert_user(user_id=user_id, name=username)
                    except Exception:
                        pass
                else:
                    q.put({"type": "status", "content": "Processing your input…"})
                    res = threads.resume(user_id=user_id, content=message, thread_id=thread_id)
                    out = res.get("out", {})

                state = get_current_state(thread_id)
                set_state_snapshot(user_id, state)

                intr = out.get("__interrupt__") if isinstance(out, dict) else None
                if intr and isinstance(intr, dict):
                    prompt = intr.get("prompt") or intr.get("content") or "Continue…"
                    q.put({"type": "status", "content": "Agent ready. Awaiting your response."})
                    q.put({"type": "final_response", "content": prompt})
                    return

                # Fallback: Some interrupt prompts might not surface in out; infer from graph_state
                gstate = str(state.get("graph_state") or "").strip().lower()
                if gstate in {"ask_domain", "get_domain"}:
                    domain_prompt = "Please provide your domain of interest (e.g., Data Science, AI, Management, Etc). "
                    q.put({"type": "status", "content": "Awaiting your domain."})
                    q.put({"type": "final_response", "content": domain_prompt})
                    return
                if gstate in {"ask_resume", "get_resume_url"}:
                    resume_prompt = (
                        "Please share your resume — you can paste a URL (http/https), a file:// URL, "
                        "or a local path (absolute/relative, ~ ok). Type 'skip' to continue without a resume."
                    )
                    q.put({"type": "status", "content": "Awaiting your resume (URL or path)."})
                    q.put({"type": "final_response", "content": resume_prompt})
                    return

                # Post-turn summary
                try:
                    sc = state.get("combined_score")
                    if sc is not None:
                        q.put({"type": "status", "content": f"Turn scored. Combined score: {sc}"})
                    fb = (state.get("overall_feedback_summary") or "").strip()
                    if fb:
                        q.put({"type": "status", "content": f"Feedback: {fb}"})
                except Exception:
                    pass

                qtext = (state.get("question") or "").strip() or "Proceed when ready."
                q.put({"type": "final_response", "content": qtext})
            except Exception as e:
                logger.error(f"/chat error: {e}")
                q.put({"type": "error", "content": str(e)})
            finally:
                q.put({"type": "done"})

        t = Thread(target=_worker, daemon=True)
        t.start()

        try:
            while True:
                try:
                    event = q.get(timeout=0.2)
                except Empty:
                    # keep the stream alive while worker runs
                    if t.is_alive():
                        continue
                    else:
                        break
                if not isinstance(event, dict):
                    continue
                if event.get("type") == "done":
                    break
                yield _sse(event)
        finally:
            try:
                base_logger.removeHandler(h)
            except Exception:
                pass

    from flask import Response, stream_with_context
    resp = Response(stream_with_context(generate()), mimetype="text/event-stream")
    resp.headers["Cache-Control"] = "no-cache"
    resp.headers["X-Accel-Buffering"] = "no"
    return resp


@app.get("/monitor/api/conversation/<user_id>.json")
def api_conversation(user_id: str):
    """Return conversation history for a user's thread as JSON.
    Query params: thread_id (optional). Defaults to stored thread_id or user_id.
    """
    tid = _resolve_thread_id(user_id, request.args.get("thread_id"))
    data = threads.get_conversation(thread_id=tid)
    return jsonify({"user_id": user_id, **data})


@app.get("/monitor/conversation/<user_id>")
def conversation_view(user_id: str):
    """Simple HTML view of a conversation for a user's thread.
    Query params: thread_id (optional)
    """
    tid = _resolve_thread_id(user_id, request.args.get("thread_id"))
    msgs = threads.get_conversation(thread_id=tid).get("messages") or []
    return render_template_string(
        r"""
<!doctype html>
<html lang="en" data-bs-theme="dark">
  <head>
    <meta charset="utf-8">
    <title>Conversation • {{ user_id }} • {{ thread_id }}</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet" crossorigin="anonymous">
    <style>
      body{ padding:24px; }
      .bubble{ border-radius: 12px; padding: 10px 14px; margin: 6px 0; max-width: 80%; }
      .user{ background: #0d6efd22; border: 1px solid #0d6efd55; margin-left:auto; }
      .assistant{ background: #19875422; border: 1px solid #19875455; }
      .ts{ font-size:.75rem; color:#aaa; }
      code{ background: rgba(255,255,255,.08); padding:.1rem .35rem; border-radius: .25rem; }
    </style>
  </head>
  <body>
    <div class="container-xxl">
      <h1 class="h5">Conversation — <span class="text-info">{{ user_id }}</span> <small class="text-secondary">(<code>{{ thread_id }}</code>)</small></h1>
      <div class="mt-3">
        {% for m in messages %}
          <div class="bubble {{ 'user' if m.role=='user' else 'assistant' }}">
            <div class="small text-secondary">{{ m.role|capitalize }} <span class="ts">{{ m.ts }}</span></div>
            <div>{{ m.content }}</div>
          </div>
        {% else %}
          <div class="text-secondary">No messages yet for this thread.</div>
        {% endfor %}
      </div>
      <hr class="my-4">
      <div class="text-secondary small">Source: in‑memory log. Use the JSON API at <code>/monitor/api/conversation/{{ user_id }}.json?thread_id={{ thread_id }}</code></div>
    </div>
  </body>
</html>
        """,
        user_id=user_id,
        thread_id=tid,
        messages=msgs,
    )


@app.get("/get_profile")
def get_profile_api():
    """Return a compact live profile for the agent UI."""
    user_id = request.args.get("user_id", "").strip()
    logger.info(f"GET /get_profile user_id={user_id}")

    rdb_profile = rdb.get_user_profile(user_id) or {}
    vec = vdb.get_user_profile(user_id) or {}
    md = vec.get("metadata") or {}

    def _merge_unique(a, b):
        seen = set()
        out = []
        for x in (a or []) + (b or []):
            x = str(x).strip()
            k = x.lower()
            if not x or k in seen:
                continue
            seen.add(k)
            out.append(x)
        return out

    profile = {
        "user_name": rdb_profile.get("name") or user_id,
        "domain": rdb_profile.get("domain") or md.get("domain"),
        "technical_skills": _merge_unique(rdb_profile.get("skills") or [], md.get("skills") or []),
        "strengths": _merge_unique(rdb_profile.get("strengths") or [], md.get("strengths") or []),
        "weaknesses": _merge_unique(rdb_profile.get("weaknesses") or [], md.get("weaknesses") or []),
        "project_experience": [],
        "user_summary": md.get("user_summary") or "",
    }
    return jsonify(profile)

# -----------------------------------------------------------------------------
# UI (HTML)
# -----------------------------------------------------------------------------
_PAGE = r"""
<!doctype html>
<html lang="en" data-bs-theme="dark">
  <head>
    <meta charset="utf-8">
    <title>Interview Mentor • Monitor • {{ data.user_id }}</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <!-- Bootstrap -->
    <link
      href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css"
      rel="stylesheet"
      crossorigin="anonymous"
    >
    <style>
      body { padding: 24px; }
      .chip {
        display:inline-block; padding:.25rem .5rem; border-radius:999px;
        font-size:.85rem; background: rgba(255,255,255,.08); margin:.125rem .25rem .25rem 0;
      }
      pre.json { max-height: 420px; overflow:auto; background: #111; border:1px solid #333; padding:12px; border-radius:8px; }
      .metric-label{ width: 180px; }
      .card{ border-radius: 14px; }
      .section-title{ font-size: 1.05rem; letter-spacing: .5px; text-transform: uppercase; color:#aaa; }
    </style>
  </head>
  <body>
    <div class="container-xxl">
      <div class="d-flex align-items-center justify-content-between mb-3">
        <h1 class="h4 m-0">Monitor: <span class="text-info">{{ data.user_id }}</span></h1>
        <div class="d-flex gap-2">
          <a class="btn btn-outline-secondary btn-sm" href="/monitor/api/{{ data.user_id }}.json" target="_blank">Open JSON</a>
          <button class="btn btn-primary btn-sm" id="copyBtn">Copy JSON</button>
          <button class="btn btn-outline-light btn-sm" onclick="location.reload()">Refresh</button>
        </div>
      </div>

      <!-- Row: RDB (Profile + Scores) -->
      <div class="row g-3">
        <div class="col-lg-6">
          <div class="card h-100">
            <div class="card-body">
              <div class="section-title mb-2">Relational DB – Profile</div>
              {% set prof = data.rdb.profile or {} %}
              <div class="mb-2"><b>Name:</b> {{ prof.name or "—" }}</div>
              <div class="mb-2"><b>Domain:</b> <span class="badge bg-info-subtle text-info-emphasis">{{ prof.domain or "—" }}</span></div>
              <div class="mb-2"><b>Updated:</b> {{ prof.updated_at or "—" }}</div>

              <div class="mb-2"><b>Skills:</b><br>
                {% for s in (prof.skills or []) %}
                  <span class="chip">{{ s }}</span>
                {% else %} <span class="text-muted">—</span>
                {% endfor %}
              </div>

              <div class="mb-2"><b>Strengths:</b><br>
                {% for s in (prof.strengths or []) %}
                  <span class="chip">{{ s }}</span>
                {% else %} <span class="text-muted">—</span>
                {% endfor %}
              </div>

              <div class="mb-2"><b>Weaknesses:</b><br>
                {% for s in (prof.weaknesses or []) %}
                  <span class="chip">{{ s }}</span>
                {% else %} <span class="text-muted">—</span>
                {% endfor %}
              </div>

              <div class="mb-2"><b>Categories:</b><br>
                {% for s in (prof.categories or []) %}
                  <span class="chip">{{ s }}</span>
                {% else %} <span class="text-muted">—</span>
                {% endfor %}
              </div>

              <details class="mt-2">
                <summary class="text-secondary">Raw (users row)</summary>
                <pre class="json">{{ data.rdb.profile|tojson(indent=2) }}</pre>
              </details>
            </div>
          </div>
        </div>

        <div class="col-lg-6">
          <div class="card h-100">
            <div class="card-body">
              <div class="section-title mb-2">Relational DB – Academic Summary</div>
              {% set sc = data.rdb.scores or {} %}
              <div class="d-flex align-items-center mb-2">
                <span class="metric-label text-secondary">Attempts</span>
                <span class="badge text-bg-dark">{{ sc.question_attempted or 0 }}</span>
              </div>

              {% macro meter(label, val) -%}
                <div class="mb-2">
                  <div class="d-flex justify-content-between">
                    <span class="metric-label text-secondary">{{ label }}</span>
                    <span class="text-muted">{{ (val or 0) | round(2) }}/10</span>
                  </div>
                  <div class="progress" role="progressbar" aria-label="{{ label }}">
                    {% set p = ((val or 0)/10*100) | round(0) %}
                    <div class="progress-bar" style="width: {{ p }}%"></div>
                  </div>
                </div>
              {%- endmacro %}

              {{ meter("Technical Accuracy", sc.technical_accuracy) }}
              {{ meter("Reasoning Depth", sc.reasoning_depth) }}
              {{ meter("Communication Clarity", sc.communication_clarity) }}
              {{ meter("Overall (hybrid)", sc.score_overall) }}

              <details class="mt-2">
                <summary class="text-secondary">Raw (academic_summary row)</summary>
                <pre class="json">{{ data.rdb.scores|tojson(indent=2) }}</pre>
              </details>
            </div>
          </div>
        </div>
      </div>

      <!-- Row: Vector + State -->
      <div class="row g-3 mt-1">
        <div class="col-lg-6">
          <div class="card h-100">
            <div class="card-body">
              <div class="section-title mb-2">Vector DB – Profile Snapshot</div>
              {% set md = data.vector.metadata or {} %}
              <div class="mb-2"><b>Vector:</b> {{ data.vector.id or "—" }} <small class="text-secondary">({{ data.vector.namespace or "—" }})</small></div>
              <div class="mb-2"><b>Domain:</b> <span class="badge bg-info-subtle text-info-emphasis">{{ md.domain or "—" }}</span></div>

              <div class="mb-2"><b>Skills:</b><br>
                {% for s in (md.skills or []) %}
                  <span class="chip">{{ s }}</span>
                {% else %} <span class="text-muted">—</span>
                {% endfor %}
              </div>

              <div class="mb-2"><b>Strengths:</b><br>
                {% for s in (md.strengths or []) %}
                  <span class="chip">{{ s }}</span>
                {% else %} <span class="text-muted">—</span>
                {% endfor %}
              </div>

              <div class="mb-2"><b>Weaknesses:</b><br>
                {% for s in (md.weaknesses or []) %}
                  <span class="chip">{{ s }}</span>
                {% else %} <span class="text-muted">—</span>
                {% endfor %}
              </div>

              <div class="mb-2"><b>User Summary:</b>
                <div class="text-body-tertiary">{{ md.user_summary or "—" }}</div>
              </div>

              <details class="mt-2">
                <summary class="text-secondary">Raw (vector metadata)</summary>
                <pre class="json">{{ md|tojson(indent=2) }}</pre>
              </details>
            </div>
          </div>
        </div>

        <div class="col-lg-6">
          <div class="card h-100">
            <div class="card-body">
              <div class="section-title mb-2">Runtime State (LangGraph)</div>
              {% set st = data.state or {} %}

              <div class="mb-2"><b>Graph State:</b> {{ st.graph_state or "—" }}</div>
              <div class="mb-2"><b>Strategy:</b> <span class="badge text-bg-secondary">{{ st.strategy or "—" }}</span></div>
              <div class="mb-2"><b>Question:</b> {{ st.question or "—" }}</div>
              <div class="mb-2"><b>Answer:</b> <div class="text-body-tertiary">{{ st.answer or "—" }}</div></div>

              {% set mta = st.metric_technical_accuracy or {} %}
              {% set mrd = st.metric_reasoning_depth or {} %}
              {% set mcc = st.metric_communication_clarity or {} %}

              <div class="mb-2"><b>Metrics:</b>
                <ul class="mb-2">
                  <li>TA: {{ mta.score or "—" }} — <span class="text-body-tertiary">{{ mta.feedback or "" }}</span></li>
                  <li>RD: {{ mrd.score or "—" }} — <span class="text-body-tertiary">{{ mrd.feedback or "" }}</span></li>
                  <li>CC: {{ mcc.score or "—" }} — <span class="text-body-tertiary">{{ mcc.feedback or "" }}</span></li>
                </ul>
              </div>

              <div class="mb-2"><b>Combined Score:</b> {{ st.combined_score if st.combined_score is defined else "—" }}</div>
              <div class="mb-2"><b>Overall Feedback:</b> <div class="text-body-tertiary">{{ st.overall_feedback_summary or "—" }}</div></div>

              <details class="mt-2">
                <summary class="text-secondary">Raw (full state)</summary>
                <pre class="json">{{ st|tojson(indent=2) }}</pre>
              </details>
            </div>
          </div>
        </div>
      </div>

      <hr class="my-4">

      <div class="text-center text-secondary small">
        Built for Interview Mentor • Flask + Bootstrap
      </div>
    </div>

    <script>
      const raw = {{ data|tojson(indent=2) }};
      document.getElementById('copyBtn').onclick = async () => {
        try {
          await navigator.clipboard.writeText(JSON.stringify(raw, null, 2));
          const btn = document.getElementById('copyBtn');
          btn.innerText = "Copied!";
          setTimeout(()=>btn.innerText="Copy JSON", 900);
        } catch(e) { alert("Copy failed"); }
      };
      // Respect user dark preference
      const prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
      if (!prefersDark) document.documentElement.setAttribute("data-bs-theme","light");
    </script>
  </body>
</html>
"""

@app.get("/monitor/<user_id>")
def monitor(user_id: str):
    logger.info(f"GET /monitor/{user_id}")
    data = fetch_all(user_id)
    return render_template("monitor.html", data=data)

# -----------------------------------------------------------------------------
# (Optional) quick manual setter for testing: curl -XPOST -H 'Content-Type: application/json' \
#   localhost:5001/monitor/state/test_user_004 -d '{"graph_state":"asked_question"}'
# -----------------------------------------------------------------------------
@app.post("/monitor/state/<user_id>")
def post_state(user_id: str):
    payload = request.get_json(silent=True) or {}
    logger.debug(f"POST /monitor/state/{user_id} keys={list(payload.keys())}")
    set_state_snapshot(user_id, payload)
    return jsonify({"ok": True, "user_id": user_id, "keys": list(payload.keys())})


# -----------------------------------------------------------------------------
# Graph control (start/resume) — runs LangGraph inside this Flask process
# -----------------------------------------------------------------------------
@app.post("/monitor/graph/<user_id>/start")
def graph_start(user_id: str):
    body = request.get_json(silent=True) or {}
    thread_id = _resolve_thread_id(user_id, request.args.get("thread_id") or body.get("thread_id"))
    strategy = body.get("strategy") or "scores"
    logger.info(f"POST /monitor/graph/{user_id}/start thread_id={thread_id} strategy={strategy}")
    res = threads.start(user_id=user_id, thread_id=thread_id, strategy=strategy)
    out = res.get("out", {})
    state = res.get("state", {})
    set_state_snapshot(user_id, state)
    return jsonify({
        "ok": True,
        "thread_id": thread_id,
        "interrupt": out.get("__interrupt__"),
        "state": state,
    })


@app.post("/monitor/graph/<user_id>/resume")
def graph_resume(user_id: str):
    body = request.get_json(silent=True) or {}
    thread_id = _resolve_thread_id(user_id, request.args.get("thread_id") or body.get("thread_id"))
    value = body.get("value") if "value" in body else (body.get("content") or body)
    logger.info(f"POST /monitor/graph/{user_id}/resume thread_id={thread_id}")
    res = threads.resume(user_id=user_id, content=value, thread_id=thread_id)
    out = res.get("out", {})
    state = res.get("state", {})
    set_state_snapshot(user_id, state)
    return jsonify({
        "ok": True,
        "thread_id": thread_id,
        "interrupt": out.get("__interrupt__"),
        "state": state,
    })

if __name__ == "__main__":
    # Run: FLASK_ENV=development python monitor_app.py
    app.run(host="0.0.0.0", port=5001, debug=True)
