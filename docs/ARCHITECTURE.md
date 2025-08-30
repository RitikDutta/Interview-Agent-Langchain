# Interview Mentor — Architecture & Flow

This document provides a detailed walkthrough of how the system works end‑to‑end: the LangGraph runtime and nodes, the resume ETL pipeline, data persistence, question retrieval, metrics aggregation, and the live monitor UI.

## Overview

- The agent collects a candidate’s domain and resume, extracts a structured profile, persists it to a relational DB (MySQL) and vector DB (Pinecone), retrieves interview questions, evaluates answers on three metrics, aggregates feedback, and streams live status to a Flask monitor UI.
- Core runtime is a LangGraph in `concept_agent_architecture_test.py` that orchestrates nodes and manages partial state updates across steps using an in‑memory checkpointer.
- Resume processing is in `resume_ETL.py` (extract → transform via LLM → canonical normalization → load to DBs).
- Persistence: `relational_database.py` (MySQL) and `vector_store.py` (Pinecone + OpenAI embeddings).
- Retrieval: `query_plan.py` (builds compact query text) and `questions_retrieval.py` (vector search) to pick a question.
- Scoring logic uses LLM structured outputs and saves metrics; `set_scores.py` helps store aggregate scores.
- Monitor UI: `monitor_app.py` with `templates/monitor.html` streams LangGraph node status live via `monitor_bus.py` using Server‑Sent Events (SSE).

## Key Modules

- `concept_agent_architecture_test.py`: LangGraph runtime, state schema, nodes, and helpers (`start_thread`, `resume_thread`, `get_current_state`). Emits live status per node.
- `resume_ETL.py`: ETL for resumes (download/parse, LLM structured profile, canonical normalization, load to DBs).
- `relational_database.py`: MySQL pool, schema creation, profile upsert/merge, academic score reads/writes.
- `vector_store.py`: Pinecone serverless index + OpenAI embeddings; upserts single profile vector snapshot per user.
- `query_plan.py`: builds query text using profile and strategy to guide retrieval.
- `questions_retrieval.py`: searches vector index to select a best matching interview question.
- `set_scores.py`: utility to update academic scores.
- `monitor_app.py`: Flask monitor app with JSON API, graph control endpoints, and SSE stream.
- `monitor_bus.py`: small in‑process pub/sub for streaming agent events.
- `templates/monitor.html`: 4‑panel UI (RDB Profile, RDB Scores, Vector Profile, Runtime State + live activity + input box).

## State & Graph

The LangGraph maintains a single, merged `State` (a TypedDict). Nodes return partial updates; only keys declared on `State` persist across steps.

Important fields:

- Core: `graph_state`, `user_id`, `is_new_user`, `messages`, `should_continue`, `strategy`
- Resume: `resume_url` (URL or local path), `profile` (ETL output: `{ "profile": normalized, "load_result": {...} }`), `skills`, `summary`, `domain`
- Interview loop: `question`, `question_meta`, `picked_question_id`, `answer`
- Metrics: `metric_technical_accuracy`, `metric_reasoning_depth`, `metric_communication_clarity`
- Aggregation: `overall_feedback_summary`, `combined_score`, `strengths`, `weaknesses`

Graph helpers:

- `get_graph()` compiles a singleton `StateGraph` with `InMemorySaver` for pause/resume.
- `start_thread(thread_id, user_id, strategy)` starts a run; `resume_thread(thread_id, value)` resumes at interrupts.
- `get_current_state(thread_id)` returns the latest snapshot for the monitor.
- `emit_status(message, node=..., **extra)` publishes live events to `monitor_bus` (used by nearly all nodes).

## Node Catalog (What each node does)

Nodes are functions `(state) -> dict` returning partial state updates. Key nodes:

- New/existing user
  - `gate_is_new_user`: no‑op node; signals start of routing; emits status.
  - `route_is_new_user` (conditional, not a node): returns `"new_user"` or `"existing_user"` using explicit `is_new_user` if present, otherwise a small heuristic.
  - `init_user`: ensures DB rows (users + academic_summary), seeds a minimal vector snapshot, returns initial state; emits status.

- Domain collection
  - `node_ask_domain`: signals UI to request a domain; emits status.
  - `get_domain`: interrupts and captures the domain string, handling skip/empty; writes `domain` to state; emits status.

- Resume capture & ETL
  - `node_ask_resume`: asks for a resume source (URL or local path); emits status.
  - `get_resume_url`: accepts `http(s)`, `file://`, or local path, performs normalization and basic validation; writes `resume_url` to state (name kept for compatibility); emits status.
  - `is_resume_url` (conditional): routes to ETL when a usable source is present.
  - `node_resume_ETL`: orchestrates `ResumeETL.run(resume_url_or_path=...)`; writes the ETL `profile` output to state; emits status on start/finish.
  - `update_profile`: merges ETL‑derived `domain`, `skills`, `strengths`, `weaknesses`, `summary` into RDB (`update_user_profile_partial`, union semantics) and updates the vector snapshot (`upsert_profile_snapshot`); writes those fields back into state; emits status.

- Interview loop
  - `query_question`: builds plan via `StrategicQuestionRouter`, queries via `QuestionSearch`, and sets the current `question`, `question_meta`, `picked_question_id`; emits status.
  - `ask_question`: interrupts with the chosen question to collect `answer`; emits status.
  - `get_answer`: no‑op marker (answer already stored); emits status.

- Metrics & aggregation
  - `validate_technical_accuracy` / `validate_reasoning_depth` / `validate_communication_clarity`: each calls LLM structured evaluation (`MetricEval`) and returns `{ metric_*: {feedback, score} }`; emit status.
  - `metrics_barrier`: synchronization node; emits status.
  - `metrics_barrier_decider` (conditional): returns "go" when all three metrics are present.
  - `aggregate_feedback`: computes integer average `combined_score`, calls LLM structured `OverallEval` to produce an `overall_feedback` and optional `strengths`/`weaknesses` only if clearly evidenced. Merges those into state with de‑dup; emits status.

- Profile amendment & continuation
  - `update_profile_with_score`: writes scores via `set_score.update_and_save`, appends `strengths`/`weaknesses` to RDB (union) and refreshes vector snapshot; emits status.
  - `update_strength_weakness`: explicit node to append strength/weakness into both RDB and vector, using order‑preserving de‑dup merge; emits status.
  - `gate_should_continue` + `route_should_continue`: routes to next question or `END`; emits status.

## Process Flow

Simplified sequence (new user):

1) `START` → `gate_is_new_user` → `route_is_new_user` → `init_user`  
2) `ask_domain` → `get_domain` (interrupt)  
3) `ask_resume` → `get_resume_url` (interrupt if not provided) → `is_resume_url`  
4) `node_resume_ETL` → `update_profile`  
5) `query_question` → `ask_question` (interrupt) → `get_answer`  
6) `validate_*` (3 parallel nodes) → `metrics_barrier` → `aggregate_feedback`  
7) `update_strength_weakness` → `update_profile_with_score`  
8) `gate_should_continue` → `route_should_continue` → continue loop or `END`

Interrupts: `get_domain`, `get_resume_url`, `ask_question` — resume via `Command(resume=...)`.

## Resume ETL (`resume_ETL.py`)

1) Extract
   - If URL, download to a temp file; else use local path.
   - Read up to `max_pages` using `PyPDFLoader` (fallback to PyMuPDF if available).

2) Transform
   - Prompt an LLM for a structured `InterviewProfile` (name, domain, skills, strengths, weaknesses, projects, traits, user_summary).
   - Canonical normalization with `canonical_normalizer.py` helpers:
     - Domain → canonical name + category
     - Skills (batch) → canonical names + categories
     - `categories` aggregated from domain and skills
   - Outputs a normalized profile (includes `user_summary` and `_normalize_debug`).

3) Load
   - RDB: ensure tables → upsert/update `users` (including `categories`) → ensure `academic_summary` row.
   - Vector: upsert a single `profile_<user_id>` record with an embedding of the `summary` and rich metadata (canonical domain/skills, strengths/weaknesses, user_summary, updated_at).

4) Return
   - `run()` returns `{ "profile": <normalized>, "load_result": { user_id, vector_id, namespace, pages_used } }`.

## Relational Database (`relational_database.py`)

- MySQL connection pool; env vars required: `MYSQL_HOST`, `MYSQL_USER`, `MYSQL_PASSWORD`, `MYSQL_DB`.
- Schema:
  - `users`: `user_id` (PK), `name`, `domain`, JSON‑string list columns: `skills`, `strengths`, `weaknesses`, `categories`, `updated_at`.
  - `academic_summary`: aggregate scores (`question_attempted`, `technical_accuracy`, `reasoning_depth`, `communication_clarity`, `score_overall`).
- Writes:
  - `upsert_user(...)`, `update_user_profile_partial(...)` (union‑dedup of list fields, sets domain if provided), `ensure_academic_summary(user_id)`.
- Reads: `get_user_profile(user_id)`, `get_user_academic_score(user_id)`.
- Metrics update: `update_academic_score(...)` (per‑field updates).

## Vector Database (`vector_store.py`)

- Pinecone serverless index + OpenAI embeddings (`text-embedding-3-small`).
- Env vars: `PINECONE_API_KEY`, `PINECONE_INDEX_NAME`, `PROFILES_NAMESPACE`, `OPENAI_API_KEY`, plus Pinecone cloud/region.
- `upsert_profile_snapshot(user_id, domain, summary, skills, strengths, weaknesses, ...)`:
  - Embeds `summary` and stores metadata with canonical domain/skills, strengths/weaknesses, user_summary, timestamps.
  - One vector per user (`id = "profile_<user_id>"`).
- `get_user_profile(user_id)` fetches the vector metadata for UI display.

## Retrieval and Planning

- `query_plan.py`: builds a compact query from available profile signals and strategy.
- `questions_retrieval.py`: performs vector search in the questions namespace and returns hits; the pipeline picks the top result and extracts `question_text`/`metadata`.

## Scoring and Feedback

- Metric review: `llm_review` produces `MetricEval` (`{ feedback, score }`) per metric.
- Aggregation: `llm_overall` produces `OverallEval` (≤100 words overall feedback and optional concise strengths/weaknesses only if clearly evidenced). The average of the three metric scores is used as the single source of truth for `combined_score` (0–10 integer).
- Persistence:
  - `update_profile_with_score` saves academic scores (via `set_score`) and appends strengths/weaknesses to both RDB (union) and vector snapshot.
  - `update_strength_weakness` also appends strengths/weaknesses to both stores with order‑preserving de‑dup.

## Monitor & UI

- `monitor_app.py` (Flask):
  - `GET /monitor/<user_id>` → renders `templates/monitor.html`.
  - `GET /monitor/api/<user_id>.json` → combined snapshot (RDB profile/scores, vector metadata, latest LangGraph state).
  - `POST /monitor/graph/<user_id>/start` → starts a graph run; returns any interrupt payload and caches state.
  - `POST /monitor/graph/<user_id>/resume` → resumes with value; returns new state and interrupt payload.
  - `GET /monitor/stream/<user_id>?thread_id=...` → SSE stream of live agent activity.
- `monitor_bus.py`: in‑process pub/sub used by `emit_status(...)` to stream node updates to SSE.
- `templates/monitor.html`:
  - Four panels: RDB Profile, RDB Scores, Vector Profile, Runtime State.
  - Live “Agent Activity” window streams node updates (with timestamps and `[node]` tag).
  - Text input + Start/Send buttons to drive interrupts.

## End‑to‑End Runbook

1) Start monitor server:

```bash
python monitor_app.py
```

2) Open in browser:

```
http://127.0.0.1:5001/monitor/<your_user_id>
```

3) Click Start. Follow the prompts in the Runtime State card (domain → resume source → answer). Watch “Agent Activity” for live stream and the other panels update as data is persisted.

## Extending the Agent

- Add a node: write `def my_node(state: State) -> dict:` and `builder.add_node("my_node", my_node)`; wire with `add_edge`/`add_conditional_edges`. Return only keys declared in `State` if they must persist.
- Add a new metric: implement `validate_<metric>` returning `{ "metric_<metric>": {feedback, score} }`; include in barrier and aggregation.
- Add new state fields: update `class State` (TypedDict) so LangGraph retains them across nodes.
- Emit UI status: call `emit_status("message", node="my_node")` inside your node.

## Configuration

- MySQL: `MYSQL_HOST`, `MYSQL_USER`, `MYSQL_PASSWORD`, `MYSQL_DB`
- Pinecone: `PINECONE_API_KEY`, `PINECONE_INDEX_NAME`, `PROFILES_NAMESPACE`, `PINECONE_CLOUD`, `PINECONE_REGION`
- OpenAI: `OPENAI_API_KEY`
- LLM: uses Google Generative AI (`gemini-2.5-flash`) for structured outputs in ETL and metrics.

## Gotchas & Tips

- State merging: keys not present in `State` will be dropped between nodes. Declare new keys before returning them.
- Resume source: `get_resume_url` accepts http(s), file://, and local paths; local paths must exist.
- List merges: RDB updates union lists server‑side (skills/strengths/weaknesses/categories). Vector snapshot is a full refresh with merged values.
- Interrupts: ensure you resume the correct thread with `resume_thread(thread_id, value)`. The monitor UI uses `user_id` as `thread_id` by default.

