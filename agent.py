import os
import re
import time
from typing import Optional, Dict, Any, List, Literal, TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from relational_database import RelationalDB
from vector_store import VectorStore
from dotenv import load_dotenv
load_dotenv()


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def log(msg: str):
    print(f"{now_iso()} {msg}", flush=True)


def looks_like_url(s: str) -> bool:
    return bool(re.match(r"^https?://", s.strip(), flags=re.I))


# ---- LangGraph state ----
class AgentState(TypedDict):
    messages: List[Any]
    user_id: str
    domain: Optional[str]
    resume_url: Optional[str]
    profile: Dict[str, Any]   # holds name/domain/skills/strengths/weaknesses/summary (optional)
    step: Optional[str]


# ---- Agent wrapper ----
class InterviewAgent:
    def __init__(self, user_id: str):
        """
        user_id: you create it first and pass it here.
        """
        self.user_id = user_id
        self.thread_id = f"thread_{user_id}"
        self.db = RelationalDB()
        self.vs = VectorStore()
        self.graph = self._build_graph()

    # -------- public API --------
    def init_user(self):
        """Initialize user rows in RDBMS and a mostly-empty profile snapshot in Pinecone."""
        log(f"system: initialized user  user_id={self.user_id}")
        self.db.create_tables()
        # mostly empty row
        self.db.upsert_user(
            user_id=self.user_id,
            name=None,
            domain=None,
            skills=[],
            strengths=[],
            weaknesses=[],
        )
        self.db.ensure_academic_summary(self.user_id)

        # vector snapshot (mostly stub)
        summary = "New user (no resume yet). Domain: unknown."
        self.vs.upsert_profile_snapshot(
            user_id=self.user_id,
            domain="unknown",
            summary=summary,
            skills=[],
            strengths=[],
            weaknesses=[],
        )

    def run_turn(self, messages: List[Any], domain: Optional[str] = None, profile: Optional[Dict[str, Any]] = None):
        config = {"configurable": {"user_id": self.user_id, "thread_id": self.thread_id}}  # <-- add thread_id
        state: AgentState = {
            "messages": messages,
            "user_id": self.user_id,
            "domain": domain,
            "resume_url": None,
            "profile": profile or {},
            "step": None,
        }
        return self.graph.invoke(state, config)

    # -------- internal graph --------
    def _build_graph(self):
        graph = StateGraph(AgentState)

        def route_node(state: AgentState) -> Literal[
            "ask_domain", "handle_domain", "ask_resume", "route_resume", "persist_profile", "respond", "__end__"
        ]:
            msgs = state["messages"]
            last = msgs[-1] if msgs else None
            last_text = last.content.strip() if isinstance(last, HumanMessage) else ""

            # ask for domain if not set
            if not state.get("domain"):
                return "ask_domain"

            # ask for resume to update profile
            if looks_like_url(last_text):
                return "route_resume"
            if last_text.lower() == "skip":
                # proceed to persist as-is
                return "persist_profile"

            # If we got domain in this turn, handle it
            if isinstance(last, HumanMessage) and state.get("domain") and state["profile"].get("domain") is None:
                return "handle_domain"

            # Default path: ask resume if not asked
            if not state.get("resume_url"):
                return "ask_resume"

            # Persist then respond
            if state.get("profile"):
                return "respond"

            return "ask_resume"

        def ask_domain_node(state: AgentState) -> AgentState:
            # If user already typed something, treat it as domain; else ask
            msgs = state["messages"]
            last = msgs[-1] if msgs else None
            if isinstance(last, HumanMessage) and last.content.strip():
                # treat their message as domain
                domain = last.content.strip().lower().replace(" ", "_")
                state["domain"] = domain
                ai = AIMessage(content=f"Great—preparing for **{domain}**.\nYou can paste a public **PDF resume link** (optional), or say **skip**.")
                log(f"agent: ask_domain -> captured domain='{domain}'")
                return {"messages": msgs + [ai], "domain": domain, "profile": state.get("profile", {})}
            else:
                ai = AIMessage(content="Hi! What job domain are you preparing for? (e.g., Data Science, Backend Engineering)")
                log("agent: ask_domain")
                return {"messages": msgs + [ai], "profile": state.get("profile", {})}

        def handle_domain_node(state: AgentState) -> AgentState:
            # Update DB + vector snapshot with the domain only
            domain = state.get("domain") or "unknown"
            log(f"agent tool call: update_profile(domain='{domain}')")
            self.db.update_user_profile_partial(user_id=self.user_id, domain=domain)
            # keep profile mirror in state
            prof = state.get("profile", {})
            prof["domain"] = domain

            # update the vector snapshot summary in a light way
            summary = f"New user (no resume yet). Domain: {domain}."
            self.vs.upsert_profile_snapshot(
                user_id=self.user_id,
                domain=domain,
                summary=summary,
                skills=prof.get("skills", []),
                strengths=prof.get("strengths", []),
                weaknesses=prof.get("weaknesses", []),
            )
            msgs = state["messages"]
            ai = AIMessage(content=f"Saved your domain as **{domain}**.\nYou can paste a public **PDF resume link** now (optional), or say **skip**.")
            return {"messages": msgs + [ai], "profile": prof}

        def ask_resume_node(state: AgentState) -> AgentState:
            msgs = state["messages"]
            ai = AIMessage(content="Optionally paste a public **PDF resume link** now, or reply **skip** to continue.")
            log("agent: ask_resume")
            return {"messages": msgs + [ai]}

        def route_resume_node(state: AgentState) -> AgentState:
            msgs = state["messages"]
            last = msgs[-1]
            url = last.content.strip()
            if looks_like_url(url):
                # Placeholder tool call (no logic)
                log(f"agent tool call: read_resume(url='{url}')")
                log("system: read_resume placeholder executed (no parsing).")
                # record it in state to affect summary
                return {"messages": msgs + [AIMessage(content="Got your resume link. (Parsing will be added later.)")],
                        "resume_url": url}
            else:
                return {"messages": msgs + [AIMessage(content="Got it. Proceeding without a resume.")],
                        "resume_url": None}

        def persist_profile_node(state: AgentState) -> AgentState:
            # ensure academic_summary row exists (zeros); we already created on init, but idempotent is fine
            self.db.ensure_academic_summary(self.user_id)

            domain = state.get("domain") or "unknown"
            resume_note = "resume link provided (pending parse)" if state.get("resume_url") else "no resume"
            summary = f"User profile snapshot: {resume_note}. Domain: {domain}."

            prof = state.get("profile", {})
            # Merge minimal known fields into RDBMS (lists remain empty at this stage)
            self.db.update_user_profile_partial(
                user_id=self.user_id,
                domain=domain,
                skills=prof.get("skills", []),
                strengths=prof.get("strengths", []),
                weaknesses=prof.get("weaknesses", []),
            )
            # Upsert vector snapshot
            self.vs.upsert_profile_snapshot(
                user_id=self.user_id,
                domain=domain,
                summary=summary,
                skills=prof.get("skills", []),
                strengths=prof.get("strengths", []),
                weaknesses=prof.get("weaknesses", []),
            )
            msgs = state["messages"]
            log("agent: persist_profile")
            ai = AIMessage(content="Your profile is saved. Let’s begin.")
            return {"messages": msgs + [ai], "profile": prof}

        def respond_node(state: AgentState) -> AgentState:
            msgs = state["messages"]
            # No question bank retrieval yet—ask a generic starter
            ai = AIMessage(content="**Q1:** To start, could you tell me about yourself and one project you’re proud of in this domain?")
            log("agent: respond (generic starter asked)")
            return {"messages": msgs + [ai]}

        # register nodes
        graph.add_node("router", lambda s: s)
        graph.add_node("ask_domain", ask_domain_node)
        graph.add_node("handle_domain", handle_domain_node)
        graph.add_node("ask_resume", ask_resume_node)
        graph.add_node("route_resume", route_resume_node)
        graph.add_node("persist_profile", persist_profile_node)
        graph.add_node("respond", respond_node)

        # edges
        graph.add_edge(START, "router")

        def _route(s: AgentState) -> str:
            return route_node(s)

        graph.add_conditional_edges(
            "router",
            _route,
            {
                "ask_domain": "ask_domain",
                "handle_domain": "handle_domain",
                "ask_resume": "ask_resume",
                "route_resume": "route_resume",
                "persist_profile": "persist_profile",
                "respond": "respond",
                "__end__": END,
            },
        )

        # Most nodes end the turn (wait for user), except the persist→respond chain
            graph.add_edge("ask_domain", END)
            graph.add_edge("handle_domain", END)
            graph.add_edge("ask_resume", END)
            graph.add_edge("route_resume", "persist_profile")
            graph.add_edge("persist_profile", "respond")
            graph.add_edge("respond", END)

        checkpointer = MemorySaver()
        return graph.compile(checkpointer=checkpointer)


# ---- quick REPL to demo logs ----
if __name__ == "__main__":
    user_id = os.getenv("DEMO_USER_ID", "u_demo_001")
    agent = InterviewAgent(user_id=user_id)
    agent.init_user()

    # first pass (no user input) -> ask domain
    res = agent.run_turn(messages=[])
    for m in res["messages"]:
        if isinstance(m, AIMessage):
            print("AI:", m.content)

    # loop
    domain = res.get("domain")
    profile = res.get("profile", {})
    while True:
        try:
            text = input("You: ").strip()
        except EOFError:
            break
        out = agent.run_turn(messages=[HumanMessage(content=text)], domain=domain, profile=profile)
        domain = out.get("domain", domain)
        profile = out.get("profile", profile)
        for m in out["messages"]:
            if isinstance(m, AIMessage):
                print("AI:", m.content)
