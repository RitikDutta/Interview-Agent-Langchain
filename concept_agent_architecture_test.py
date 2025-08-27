from dotenv import load_dotenv
import time
load_dotenv()

from typing import TypedDict, Literal, List, Optional, Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import InMemorySaver
from IPython.display import Image, display
from pydantic import BaseModel, Field, conint
from relational_database import RelationalDB
from vector_store import VectorStore
from resume_ETL import ResumeETL
from query_plan import StrategicQuestionRouter
from questions_retrieval import QuestionSearch
from set_scores import ScoreUpdater


rdb = RelationalDB()
vs = VectorStore()
question_router = StrategicQuestionRouter(rdb=rdb, vs=vs)
retriever = QuestionSearch(namespace="questions_v4")
set_score = ScoreUpdater(rdb)


# ------------------------------------------------------------------------------
# logs
# ------------------------------------------------------------------------------
def now_iso() -> str:
    return time.strftime("[%Y-%m-%d] [%H:%M:%S]", time.localtime())

def log(msg: str):
    print(f"{now_iso()} {msg}", flush=True)

# ------------------------------------------------------------------------------
# LLM
# ------------------------------------------------------------------------------
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
llm_low_temp = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)

# ------------------------------------------------------------------------------
# State
# ------------------------------------------------------------------------------
class Message(TypedDict):
    role: Literal["user", "assistant", "system"]
    content: str

class State(TypedDict, total=False):
    graph_state: str
    input_text: str
    is_new_user: bool
    should_continue: str
    messages: List[Message]

    user_id: str
    domain: Optional[str]
    resume_url: Optional[str]
    strategy: Optional[str]

    # interview loop
    question: str
    question_meta: Dict[str, Any]
    picked_question_id: Optional[str]
    answer: str

    # metrics
    metric_technical_accuracy: Dict[str, Any]
    metric_reasoning_depth: Dict[str, Any]
    metric_communication_clarity: Dict[str, Any]

    # aggregation
    overall_feedback_summary: str
    combined_score: int
    strengths: List[str]
    weaknesses: List[str]


class MetricEval(BaseModel):
    feedback: str = Field(description="≤50 words, concise and actionable")
    score: conint(ge=0, le=10) = Field(description="integer 0..10")

class OverallEval(BaseModel):
    overall_feedback: str = Field(
        description="≤100 words, balanced, actionable, no headings/repetition"
    )
    combined_score: conint(ge=0, le=10) = Field(
        description="average score of metrics, integer 0–10"
    )
    strengths: Optional[List[str]] = Field(
        default=None,
        description=(
            "OPTIONAL. 0–3 bullet-like strengths, each a short phrase (≤10 words). "
            "Only include if clearly evidenced across the provided metrics; otherwise omit."
        ),
    )
    weaknesses: Optional[List[str]] = Field(
        default=None,
        description=(
            "OPTIONAL. 0–3 bullet-like weaknesses/opportunities, short phrases (≤10 words). "
            "Only include if clearly evidenced across the metrics; otherwise omit."
        ),
    )


# ------------------------------------------------------------------------------
# Binders
# ------------------------------------------------------------------------------
llm_review = llm.with_structured_output(MetricEval)
llm_overall = llm.with_structured_output(OverallEval)
# ------------------------------------------------------------------------------
# short models
# ------------------------------------------------------------------------------

def _eval_metric_with_llm(metric_name: str, question: str, answer: str) -> dict:
    rubric = {
        "technical_accuracy": "factual correctness, valid methods, precise terminology, alignment to the question.",
        "reasoning_depth": "quality of reasoning, decomposition, assumptions, trade-offs, evidence and stepwise logic.",
        "communication_clarity": "structure, brevity, coherence, ordering, audience-appropriate wording."
    }.get(metric_name, "quality relative to the named metric.")

    sys = SystemMessage(content=(
        f"You are a senior interview evaluator for '{metric_name}'. "
        f"Metric focus: {rubric} "
        "Grade conservatively. Penalize errors and unsupported claims. "
        "If off-topic/empty, score 0–2. Return only schema fields."
    ))

    human = HumanMessage(content=(
        f"Question:\n{question}\n\n"
        f"Candidate answer:\n{answer}\n\n"
        "Output:\n"
        "- feedback: ≤50 words, actionable, behavior-focused.\n"
        "- score: integer 0..10."
    ))

    result: MetricEval = llm_review.invoke([sys, human])
    return result.model_dump()

# ------------------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------------------
def last_user_text(state: State) -> str:
    msgs = state.get("messages") or []
    for m in reversed(msgs):
        if m.get("role") == "user":
            return (m.get("content") or "").strip()
    return (state.get("input_text") or "").strip()

# ------------------------------------------------------------------------------
# new/existing user routing
# ------------------------------------------------------------------------------
def gate_is_new_user(state: State) -> dict:
    log("System: Checking if user is new or existing")
    return {}

def route_is_new_user(state: State) -> Literal["new_user", "existing_user"]:
    if "is_new_user" in state:
        flag = bool(state["is_new_user"])
        log(f"[route_is_new_user] flag={flag}")
        return "new_user" if flag else "existing_user"
    # heuristic fallback (optional)
    first = ""
    msgs = state.get("messages") or []
    for m in msgs:
        if m.get("role") == "user":
            first = (m.get("content") or "").lower()
            break
    newbie = any(kw in first for kw in ("start", "new user", "first time", "setup"))
    log(f"[route_is_new_user] inferred_new={newbie}")
    return "new_user" if newbie else "existing_user"

def init_user(state: State) -> dict:
    """
    Initialize the user state for a new user.
    """
    log("System: Initializing new user")
    user_id = state.get("user_id")
    if not user_id:
        raise ValueError("User ID is required for new user initialization")
    
    # RelationalDB initialization
    rdb.create_tables()  # ensure tables exist
    rdb.upsert_user(
        user_id=user_id,
        name=None,
        domain=None,
        skills=[],
        strengths=[],
        weaknesses=[],
    )
    rdb.ensure_academic_summary(user_id) # ensure academic summary exists

    # VectorStore initialization mostly stub
    summary = "New user (no resume yet). Domain: Unknown."
    vs.upsert_profile_snapshot(
        user_id=user_id,
        domain="Unknown",
        summary=summary,
        skills=[],
        strengths=[],
        weaknesses=[],
    )

    return {"graph_state": "init_user", "is_new_user": True, "messages": []}

# ------------------------------------------------------------------------------
# nodes
# ------------------------------------------------------------------------------
def node_ask_domain(state: State) -> dict:
    """
    ask ther user for domain
    """
    log("node_ask_domain")
    return {"graph_state": "ask_domain"}

def get_domain(state: State) -> dict:
    """
    If the latest user message already contains a domain, store it.
    Otherwise, interrupt to ask for it and wait for a reply.
    """
    user_id = state.get("user_id")
    log("node get_domain")
    # 1) already give a domain?
    text = last_user_text(state)
    if text:
        log(f"[get_domain] found domain in message: {text}")
        return {"graph_state": "have_domain", "domain": text}

    # 2) no domain
    domain_value = interrupt({
        "prompt": "Please provide your domain of interest (e.g., Data Science, AI, Management, Etc). "
    })

    # 3) normalize the domain_value into a string
    if isinstance(domain_value, dict):
        # accept common shapes: {"domain": ...} or {"content": ...}
        domain = domain_value.get("domain") or domain_value.get("content") or ""
    else:
        domain = str(domain_value or "")

    domain = domain.strip()

    # 4) handle skip or invalid
    if domain.lower() in {"skip", "no", "later", ""}:
        log("[user_id] user skipped providing domain")
        return {"graph_state": "no_domain"}  # no domain key set

    log(f"got domain from user: {user_id} → {domain}")
    return {"graph_state": "have_domain", "domain": domain}

def node_ask_resume(state: State) -> dict:
    log("node2_ask_resume")
    return {"graph_state": "ask_resume"}

def get_resume_url(state: State) -> dict:
    """
    If the latest user message already contains a URL, store it.
    Otherwise, interrupt to ask for it and wait for a reply.
    """
    log("node get_resume_url")
    # 1) already give a URL?
    text = last_user_text(state)
    if text.startswith(("http://", "https://")):
        log(f"[get_resume_url] found URL in message: {text}")
        return {"graph_state": "have_resume_url", "resume_url": text}

    # 2) no URL
    resume_value = interrupt({
        "prompt": "Please paste your resume URL (http/https). "
                  "Type 'skip' to continue without a URL."
    })

    # 3) normalize the resume_value into a string
    if isinstance(resume_value, dict):
        # accept common shapes: {"url": ...} or {"content": ...}
        url = resume_value.get("url") or resume_value.get("content") or ""
    else:
        url = str(resume_value or "")

    url = url.strip()

    # 4) handle skip or invalid
    if url.lower() in {"skip", "no", "later", ""}:
        log("[get_resume_url] user skipped providing URL")
        return {"graph_state": "no_resume_url"}  # no resume_url key set

    if not url.startswith(("http://", "https://")):
        log(f"[get_resume_url] invalid URL provided: {url!r} (continuing without)")
        return {"graph_state": "no_resume_url"}

    log(f"got URL from user: {state.get('user_id')} → {url}")

    return {"graph_state": "have_resume_url", "resume_url": url}

# ----- Resume routing + pipeline -----
def is_resume_url(state: State) -> Literal["resume_is_present", "resume_is_not_present"]:
    # prefer the explicit field set by get_resume_url; fall back to last message
    text = (state.get("resume_url") or last_user_text(state)).strip()
    log(f"[is_resume_url] checking: {text!r}")
    return "resume_is_present" if text.startswith(("http://", "https://")) else "resume_is_not_present"

def node_resume_ETL(state: State) -> dict:
    log("node node_resume_ETL")
    log(f"Extracting resume from URL: {state.get('resume_url')}")
    resume_etl = ResumeETL(
        user_id=state.get("user_id"),
        verbose=True,
    )
    return {"graph_state": "transform_resume"}

def update_profile(state: State) -> dict:
    log("node update_profile")
    return {"graph_state": "update_profile"}

# ----- interview loop -----

def query_question(state: State) -> dict:
    log("node query_question")
    user_id = state.get("user_id") or "test_user_004"

    # build a query using your router
    plan = question_router.plan(user_id, strategy=state.get('strategy'))
    q = (plan.query or "").strip()

    # run your vector-store search
    results = retriever.query_search(q, top_k=1)

    # unpack the best hit with safe fallbacks
    if results and len(results) > 0:
        best = results[0]
        meta = getattr(best, "metadata", {}) or {}
        # many pipelines store the text in 'text' or 'question_text' – support both
        qtext = (meta.get("text") or meta.get("question_text") or "").strip()
        picked_id = getattr(best, "id", None)
        if not qtext:
            qtext = "Walk me through one ML project you’ve done end‑to‑end."
    else:
        meta = {}
        picked_id = None
        qtext = "Walk me through one ML project you’ve done end‑to‑end."

    # set state and move on
    return {
        "graph_state": "have_question",
        "question": qtext,
        "question_meta": meta,
        "picked_question_id": picked_id,
    }


def ask_question(state: State) -> dict:
    log("node ask_question")

    qtext = (state.get("question") or "").strip()
    if not qtext:
        qtext = "Tell me about a project you’re proud of."

    # user input
    ans_value = interrupt({
        "prompt": qtext
    })

    # normalize the interrupt return into a plain string
    if isinstance(ans_value, dict):
        ans = ans_value.get("answer") or ans_value.get("content") or ""
    else:
        ans = str(ans_value or "")

    return {
        "graph_state": "asked_question",
        "answer": ans.strip()
    }

def validate_technical_accuracy(state: State) -> Dict[str, Any]:
    q = state.get("question", "")
    a = state.get("answer", "")
    result = _eval_metric_with_llm("technical_accuracy", q, a)
    return {"metric_technical_accuracy": result}

def validate_reasoning_depth(state: State) -> Dict[str, Any]:
    q = state.get("question", "")
    a = state.get("answer", "")
    result = _eval_metric_with_llm("reasoning_depth", q, a)
    return {"metric_reasoning_depth": result}

def validate_communication_clarity(state: State) -> Dict[str, Any]:
    q = state.get("question", "")
    a = state.get("answer", "")
    result = _eval_metric_with_llm("communication_clarity", q, a)
    return {"metric_communication_clarity": result}

def metrics_barrier(state: State) -> Dict[str, Any]:
    # Debug what keys we have each time the barrier runs
    print("barrier keys:", list(state.keys()))
    print("TA:", state.get("metric_technical_accuracy"))
    print("RD:", state.get("metric_reasoning_depth"))
    print("CC:", state.get("metric_communication_clarity"))
    return {}  # no state change

def metrics_barrier_decider(state: State) -> Literal["go", "wait"]:
    have_all = all(
        k in state
        for k in (
            "metric_technical_accuracy",
            "metric_reasoning_depth",
            "metric_communication_clarity",
        )
    )
    return "go" if have_all else "wait"

def aggregate_feedback(state: State) -> Dict[str, Any]:
    m_ta, m_rd, m_cc = state["metric_technical_accuracy"], state["metric_reasoning_depth"], state["metric_communication_clarity"]
    s_ta, s_rd, s_cc = int(m_ta["score"]), int(m_rd["score"]), int(m_cc["score"])
    combined = int(round((s_ta + s_rd + s_cc) / 3))  # single source of truth

    sys = SystemMessage(content=(
        "You are a senior interview evaluator. Write ONE overall feedback (≤100 words) "
        "based strictly on the three metric feedback snippets. No headings, no repetition."
    ))
    human = HumanMessage(content=(
        f"Technical accuracy: {m_ta['feedback']}\n"
        f"Reasoning depth: {m_rd['feedback']}\n"
        f"Communication clarity: {m_cc['feedback']}\n"
    ))
    text_only = llm.with_config({"temperature": 0.2}).invoke([sys, human]).content.strip()

    return {
        "overall_feedback_summary": text_only,
        "combined_score": combined,
        "strengths": state.get("strengths") or [],
        "weaknesses": state.get("weaknesses") or [],
    }

def get_answer(state: State) -> dict:
    log("node get_answer")
    # answer is already captured in ask_question
    return {"graph_state": "get_answer"}

def update_profile_with_score(state: State) -> dict:
    log("node update_profile_with_score")
    user_id = state.get("user_id")

    # scores collected in previous step (calculate_score)
    ta = state.get("metric_technical_accuracy", {}).get("score")
    rd = state.get("metric_reasoning_depth", {}).get("score")
    cc = state.get("metric_communication_clarity", {}).get("score")
    overall = state.get("combined_score")  # or compute avg if you store raw
    scores = {
        'technical_accuracy': ta,
        'reasoning_depth': rd,
        'communication_clarity': cc,
    }
    set_score.update_and_save(user_id, scores)

    return {"graph_state": "update_profile_with_score"}

def update_strength_weakness(state: State) -> dict:
    log("node update_strength_weakness")
    user_id = state.get("user_id")
    if not user_id:
        log("[update_strength_weakness] missing user_id → skip")
        return {"graph_state": "update_strength_weakness"}

    # strengths/weaknesses from LLM optional can be empty
    new_strengths = (state.get("strengths") or [])[:]
    new_weaknesses = (state.get("weaknesses") or [])[:]

    # nothing to add → no-op
    if not new_strengths and not new_weaknesses:
        log("[update_strength_weakness] no strengths/weaknesses to add → skip DB update")
        return {"graph_state": "update_strength_weakness"}

    # fetch existing profile to *append* (not overwrite)
    try:
        profile = rdb.get_user_profile(user_id=user_id) or {}
    except Exception as e:
        log(f"[update_strength_weakness] get_user_profile failed: {e}")
        profile = {}

    existing_strengths = (profile.get("strengths") or [])
    existing_weaknesses = (profile.get("weaknesses") or [])

    # de-dup while preserving order
    def _merge(old_list, add_list):
        seen = set()
        merged = []
        for x in (old_list + add_list):
            x = (x or "").strip()
            if not x:
                continue
            key = x.lower()
            if key in seen:
                continue
            seen.add(key)
            merged.append(x)
        return merged

    merged_strengths = _merge(existing_strengths, new_strengths) if new_strengths else existing_strengths
    merged_weaknesses = _merge(existing_weaknesses, new_weaknesses) if new_weaknesses else existing_weaknesses

    # Only pass fields we actually want to modify (avoid touching domain/skills/categories here)
    try:
        rdb.update_user_profile_partial(
            user_id=user_id,
            strengths=merged_strengths if new_strengths else None,
            weaknesses=merged_weaknesses if new_weaknesses else None,
        )
        log("[update_strength_weakness] DB updated")
    except Exception as e:
        log(f"[update_strength_weakness] update_user_profile_partial failed: {e}")

    return {"graph_state": "update_strength_weakness"}

# ----- continue / exit routing -----
def gate_should_continue(state: State) -> dict:
    log("gate_should_continue")
    return {"graph_state": "should_continue"}

def route_should_continue(state: State) -> Literal["continue", "exit"]:
    # 1) explicit flag wins
    flag = (state.get("should_continue") or "").strip().lower()
    if flag in {"exit", "quit", "stop"}:
        log("[route_should_continue] flag → EXIT")
        return "exit"
    # 2) infer from last user message
    msg = last_user_text(state).lower()
    if any(w in msg for w in ("exit", "quit", "stop")):
        log(f"[route_should_continue] message '{msg}' → EXIT")
        return "exit"
    log("[route_should_continue] → CONTINUE")
    return "continue"

# ------------------------------------------------------------------------------
# Build Graph
# ------------------------------------------------------------------------------
builder = StateGraph(State)

builder.add_node("gate_is_new_user", gate_is_new_user)
builder.add_node("init_user", init_user)
builder.add_node("ask_domain", node_ask_domain)
builder.add_node("get_domain", get_domain)
builder.add_node("ask_resume", node_ask_resume)
builder.add_node("get_resume_url", get_resume_url)

builder.add_node("node_resume_ETL", node_resume_ETL)
builder.add_node("update_profile", update_profile)

builder.add_node("query_question", query_question)
builder.add_node("ask_question", ask_question)
builder.add_node("get_answer", get_answer)

builder.add_node("validate_technical_accuracy", validate_technical_accuracy)
builder.add_node("validate_reasoning_depth", validate_reasoning_depth)
builder.add_node("validate_communication_clarity", validate_communication_clarity)
builder.add_node("metrics_barrier", metrics_barrier)

builder.add_node("aggregate_feedback", aggregate_feedback)

builder.add_node("update_profile_with_score", update_profile_with_score)
builder.add_node("update_strength_weakness", update_strength_weakness)
builder.add_node("gate_should_continue", gate_should_continue)

# Start → new/existing
builder.add_edge(START, "gate_is_new_user")
builder.add_conditional_edges(
    "gate_is_new_user",
    route_is_new_user,
    {"new_user": "init_user", "existing_user": "query_question"},
)
# New user initialization
builder.add_edge("init_user", "ask_domain")

# Domain → Resume
builder.add_edge("ask_domain", "get_domain")
builder.add_edge("get_domain", "ask_resume")
builder.add_edge("ask_resume", "get_resume_url")

# Route based on URL presence (after get_resume_url)
builder.add_conditional_edges(
    "get_resume_url",
    is_resume_url,
    {"resume_is_present": "node_resume_ETL", "resume_is_not_present": "update_profile"},
)

# Resume processing
builder.add_edge("node_resume_ETL", "update_profile")

# Interview loop
builder.add_edge("update_profile", "query_question")
builder.add_edge("query_question", "ask_question")
builder.add_edge("ask_question", "get_answer")

builder.add_edge("get_answer", "validate_technical_accuracy")
builder.add_edge("get_answer", "validate_reasoning_depth")
builder.add_edge("get_answer", "validate_communication_clarity")

builder.add_edge("validate_technical_accuracy", "metrics_barrier")
builder.add_edge("validate_reasoning_depth", "metrics_barrier")
builder.add_edge("validate_communication_clarity", "metrics_barrier")

builder.add_conditional_edges(
    "metrics_barrier",
    metrics_barrier_decider,
    {"go": "aggregate_feedback", "wait": END},
)

# After aggregation, continue as before
builder.add_edge("aggregate_feedback", "update_strength_weakness")
builder.add_edge("update_strength_weakness", "update_profile_with_score")

# builder.add_edge("update_profile_with_score", "give_feedback")

# continue / Exit
builder.add_edge("update_profile_with_score", "gate_should_continue")
builder.add_conditional_edges(
    "gate_should_continue",
    route_should_continue,
    {"continue": "query_question", "exit": END},
)

# compile with a checkpointer so interrupts can pause/resume
checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

thread = {"configurable": {"thread_id": "demo-1"}}
out = graph.invoke(
        { "user_id": "test_user_004", "is_new_user": (rdb.user_exists_rdb(user_id="123")), "messages": [{"role": "user"}], "should_continue": "exit", "strategy": "scores"},
    config=thread
)



log(out.get("__interrupt__"))
out = graph.invoke(Command(resume="Data Science"), config=thread)

log(out.get("__interrupt__"))
out = graph.invoke(Command(resume="/home/codered/mystuff/progs/python/interview-mentor-Langraph/Ritik Dutta Resume-June.pdf"), config=thread)

answer = "For model testing, common techniques include train/test split, cross-validation (like k-fold or stratified), and bootstrapping to evaluate performance reliability. Sometimes a hold-out validation set or A/B testing in production is used. Models are also subjected to robustness or stress testing with noisy or adversarial data, followed by error analysis to understand misclassifications and improve performance."
log(out.get("__interrupt__"))
out = graph.invoke(Command(resume=answer), config=thread)