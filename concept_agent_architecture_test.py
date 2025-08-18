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

from relational_database import RelationalDB
from vector_store import VectorStore

rdb = RelationalDB()
vs = VectorStore()

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

# ------------------------------------------------------------------------------
# Tool: pretty print the domain in a decorative box
# ------------------------------------------------------------------------------
@tool("pretty_print_domain")
def pretty_print_domain(domain: str) -> str:
    """Pretty-print a domain string in a box for the console."""
    label = f" DOMAIN: {domain.strip()} "
    top = "╔" + "═" * len(label) + "╗"
    mid = "║" + label + "║"
    bot = "╚" + "═" * len(label) + "╝"
    return f"\n{top}\n{mid}\n{bot}\n"

llm_with_tools = llm.bind_tools([pretty_print_domain])

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
    question: str
    domain: Optional[str]
    resume_url: Optional[str]

    user_id: str
    profile: Dict[str, Any]   # holds name/domain/skills/strengths/weaknesses/summary (optional)
    step: Optional[str]

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
    # 1) quick pass: did the user already give a domain?
    text = last_user_text(state)
    if text:
        log(f"[get_domain] found domain in message: {text}")
        return {"graph_state": "have_domain", "domain": text}

    # 2) no domain → ask the user via interrupt (human-in-the-loop)
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
    # 1) quick pass: did the user already give a URL?
    text = last_user_text(state)
    if text.startswith(("http://", "https://")):
        log(f"[get_resume_url] found URL in message: {text}")
        return {"graph_state": "have_resume_url", "resume_url": text}

    # 2) no URL → ask the user via interrupt (human-in-the-loop)
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
    
    # 5) store the URL in the state
    state["resume_url"] = url

    return {"graph_state": "have_resume_url", "resume_url": url}

# ----- Resume routing + pipeline -----
def is_resume_url(state: State) -> Literal["resume_is_present", "resume_is_not_present"]:
    # prefer the explicit field set by get_resume_url; fall back to last message
    text = (state.get("resume_url") or last_user_text(state)).strip()
    log(f"[is_resume_url] checking: {text!r}")
    return "resume_is_present" if text.startswith(("http://", "https://")) else "resume_is_not_present"

def extract_resume(state: State) -> dict:
    log("node extract_resume")
    log(f"Extracting resume from URL: {state.get('resume_url')}")
    # actual extraction would use state["resume_url"]
    return {"graph_state": "transform_resume"}

def transform_resume(state: State) -> dict:
    log("node transform_resume")
    return {"graph_state": "load_resume"}

def load_resume(state: State) -> dict:
    log("node load_resume")
    return {"graph_state": "update_profile"}

def update_profile(state: State) -> dict:
    log("node update_profile")
    return {"graph_state": "update_profile"}

# ----- interview loop -----
def query_question(state: State) -> dict:
    log("node query_question")
    return {"graph_state": "query_question"}

def ask_question(state: State) -> dict:
    log("node ask_question")
    return {"graph_state": "ask_question"}

def get_answer(state: State) -> dict:
    log("node get_answer")
    return {"graph_state": "get_answer"}

def review_answer(state: State) -> dict:
    log("node review_answer")
    return {"graph_state": "review_answer"}

def calculate_score(state: State) -> dict:
    log("node calculate_score")
    return {"graph_state": "calculate_score"}

def update_profile_with_score(state: State) -> dict:
    log("node update_profile_with_score")
    return {"graph_state": "update_profile_with_score"}

def give_feedback(state: State) -> dict:
    log("node give_feedback")
    return {"graph_state": "give_feedback"}

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

builder.add_node("extract_resume", extract_resume)
builder.add_node("transform_resume", transform_resume)
builder.add_node("load_resume", load_resume)
builder.add_node("update_profile", update_profile)

builder.add_node("query_question", query_question)
builder.add_node("ask_question", ask_question)
builder.add_node("get_answer", get_answer)
builder.add_node("review_answer", review_answer)
builder.add_node("calculate_score", calculate_score)
builder.add_node("update_profile_with_score", update_profile_with_score)
builder.add_node("give_feedback", give_feedback)
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
    {"resume_is_present": "extract_resume", "resume_is_not_present": "update_profile"},
)

# Resume processing
builder.add_edge("extract_resume", "transform_resume")
builder.add_edge("transform_resume", "load_resume")
builder.add_edge("load_resume", "update_profile")

# Interview loop
builder.add_edge("update_profile", "query_question")
builder.add_edge("query_question", "ask_question")
builder.add_edge("ask_question", "get_answer")
builder.add_edge("get_answer", "review_answer")
builder.add_edge("review_answer", "calculate_score")
builder.add_edge("calculate_score", "update_profile_with_score")
builder.add_edge("update_profile_with_score", "give_feedback")

# continue / Exit
builder.add_edge("give_feedback", "gate_should_continue")
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
        { "user_id": "123", "is_new_user": not(rdb.user_exists_rdb(user_id="123")), "messages": [{"role": "user"}], "should_continue": "exit"},
    config=thread
)



log(out.get("__interrupt__"))
out = graph.invoke(Command(resume="Data Science"), config=thread)

log(out.get("__interrupt__"))
out = graph.invoke(Command(resume="https://abc.com/resume.pdf"), config=thread)