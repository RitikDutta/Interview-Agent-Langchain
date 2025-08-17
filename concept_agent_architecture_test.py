from dotenv import load_dotenv
load_dotenv()

from typing import TypedDict, Literal, List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import InMemorySaver
from IPython.display import Image, display

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
    domain: str
    resume_url: str  # set by get_resume_url when available

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
    print("gate_is_new_user")
    return {}

def route_is_new_user(state: State) -> Literal["new_user", "existing_user"]:
    if "is_new_user" in state:
        flag = bool(state["is_new_user"])
        print(f"[route_is_new_user] flag={flag}")
        return "new_user" if flag else "existing_user"
    # heuristic fallback (optional)
    first = ""
    msgs = state.get("messages") or []
    for m in msgs:
        if m.get("role") == "user":
            first = (m.get("content") or "").lower()
            break
    newbie = any(kw in first for kw in ("start", "new user", "first time", "setup"))
    print(f"[route_is_new_user] inferred_new={newbie}")
    return "new_user" if newbie else "existing_user"

# ------------------------------------------------------------------------------
# nodes
# ------------------------------------------------------------------------------
def node_ask_domain(state: State) -> dict:
    """
    Ask the user for their domain — demo a tool call that prints it in style.
    We read the latest user message as the domain for this simple demo.
    """
    print("node1_ask_domain")
    domain_text = last_user_text(state) or "unknown"

    prompt = (
        "You have a tool named 'pretty_print_domain'. "
        "Call it exactly once using the user's domain below.\n\n"
        f"User domain: {domain_text}"
    )
    ai_msg = llm_with_tools.invoke([
        SystemMessage(content="You MUST call the tool."),
        HumanMessage(content=prompt)
    ])

    called = False
    for tc in getattr(ai_msg, "tool_calls", []) or []:
        if tc["name"] == "pretty_print_domain":
            res = pretty_print_domain.invoke(tc["args"])
            print(res)  # show tool output in console
            called = True

    if not called:
        # fallback if the model didn't call the tool
        res = pretty_print_domain.invoke({"domain": domain_text})
        print(res)

    return {"graph_state": "ask_domain", "domain": domain_text}

def node_ask_resume(state: State) -> dict:
    print("node2_ask_resume")
    return {"graph_state": "ask_resume"}

def get_resume_url(state: State) -> dict:
    """
    If the latest user message already contains a URL, store it.
    Otherwise, interrupt to ask for it and wait for a reply.
    """
    print("node get_resume_url")
    # 1) quick pass: did the user already give a URL?
    text = last_user_text(state)
    if text.startswith(("http://", "https://")):
        print(f"[get_resume_url] found URL in message: {text}")
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
        print("[get_resume_url] user skipped providing URL")
        return {"graph_state": "no_resume_url"}  # no resume_url key set

    if not url.startswith(("http://", "https://")):
        print(f"[get_resume_url] invalid URL provided: {url!r} (continuing without)")
        return {"graph_state": "no_resume_url"}

    print(f"[get_resume_url] got URL from user: {url}")
    return {"graph_state": "have_resume_url", "resume_url": url}

# ----- Resume routing + pipeline -----
def is_resume_url(state: State) -> Literal["resume_is_present", "resume_is_not_present"]:
    # prefer the explicit field set by get_resume_url; fall back to last message
    text = (state.get("resume_url") or last_user_text(state)).strip()
    print(f"[is_resume_url] checking: {text!r}")
    return "resume_is_present" if text.startswith(("http://", "https://")) else "resume_is_not_present"

def extract_resume(state: State) -> dict:
    print("node extract_resume")
    # actual extraction would use state["resume_url"]
    return {"graph_state": "transform_resume"}

def transform_resume(state: State) -> dict:
    print("node transform_resume")
    return {"graph_state": "load_resume"}

def load_resume(state: State) -> dict:
    print("node load_resume")
    return {"graph_state": "update_profile"}

def update_profile(state: State) -> dict:
    print("node update_profile")
    return {"graph_state": "update_profile"}

# ----- interview loop -----
def query_question(state: State) -> dict:
    print("node query_question")
    return {"graph_state": "query_question"}

def ask_question(state: State) -> dict:
    print("node ask_question")
    return {"graph_state": "ask_question"}

def get_answer(state: State) -> dict:
    print("node get_answer")
    return {"graph_state": "get_answer"}

def review_answer(state: State) -> dict:
    print("node review_answer")
    return {"graph_state": "review_answer"}

def calculate_score(state: State) -> dict:
    print("node calculate_score")
    return {"graph_state": "calculate_score"}

def update_profile_with_score(state: State) -> dict:
    print("node update_profile_with_score")
    return {"graph_state": "update_profile_with_score"}

def give_feedback(state: State) -> dict:
    print("node give_feedback")
    return {"graph_state": "give_feedback"}

# ----- continue / exit routing -----
def gate_should_continue(state: State) -> dict:
    print("gate_should_continue")
    return {"graph_state": "should_continue"}

def route_should_continue(state: State) -> Literal["continue", "exit"]:
    # 1) explicit flag wins
    flag = (state.get("should_continue") or "").strip().lower()
    if flag in {"exit", "quit", "stop"}:
        print("[route_should_continue] flag → EXIT")
        return "exit"
    # 2) infer from last user message
    msg = last_user_text(state).lower()
    if any(w in msg for w in ("exit", "quit", "stop")):
        print(f"[route_should_continue] message '{msg}' → EXIT")
        return "exit"
    print("[route_should_continue] → CONTINUE")
    return "continue"

# ------------------------------------------------------------------------------
# Build Graph
# ------------------------------------------------------------------------------
builder = StateGraph(State)

builder.add_node("gate_is_new_user", gate_is_new_user)
builder.add_node("ask_domain", node_ask_domain)
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
    {"new_user": "ask_domain", "existing_user": "query_question"},
)

# Domain → Resume
builder.add_edge("ask_domain", "ask_resume")
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
    {"is_new_user": True, "messages": [{"role": "user", "content": "Data Science"}], "should_continue": "exit"},
    config=thread
)
print(out.get("__interrupt__"))   # shows: {"prompt": "...paste your resume URL..."}
out = graph.invoke(Command(resume="https://abc.com/resume.pdf"), config=thread)