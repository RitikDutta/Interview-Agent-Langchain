#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
import uuid

from dotenv import load_dotenv

# Load .env first and ensure THINKING logs are enabled before any other imports
load_dotenv()
os.environ.setdefault("LOG_LEVEL", "THINKING")
from interview_flow.logging import get_logger as _base_get_logger  # configures root using env
_base_get_logger(level_name="THINKING")

from interview_flow import start_thread, resume_thread, get_current_state


def _env(name: str, default: str) -> str:
    v = os.getenv(name)
    return v if v is not None and str(v).strip() else default


def main() -> int:
    # Root logger already configured above via LOG_LEVEL and _base_get_logger
    # Read required values from environment (.env loaded earlier)
    def _env_first(*names: str, default: str = "") -> str:
        for n in names:
            v = os.getenv(n)
            if v is not None and str(v).strip():
                return str(v).strip()
        return default

    user_id = _env_first("Test_USER_ID", "TEST_USER_ID", default="test_user_022")
    strategy = _env_first("TEST_STRATEGY", default="scores")
    thread_id = _env_first("TEST_THREAD_ID", default=f"test-{uuid.uuid4().hex[:8]}")
    ns = os.getenv("QUESTIONS_NAMESPACE")

    test_domain = _env("TEST_DOMAIN", "Data Science")
    test_resume = _env("TEST_RESUME", "skip")
    test_answer = _env("TEST_ANSWER", f"Here's a brief answer about my experience in {test_domain}.")

    header = f"Starting interview: user={user_id} thread_id={thread_id} strategy={strategy}"
    if ns:
        header += f" namespace={ns}"
    print(header)

    # Kick off
    def _pick_prompt(intr) -> str:
        if intr is None:
            return ""
        if isinstance(intr, dict):
            return (intr.get("prompt") or intr.get("content") or "").strip()
        if isinstance(intr, (list, tuple)):
            for x in intr:
                if isinstance(x, dict):
                    v = (x.get("prompt") or x.get("content") or "").strip()
                    if v:
                        return v
            return ""
        return str(intr).strip()

    out = start_thread(thread_id=thread_id, user_id=user_id, strategy=strategy)

    # Respond to up to 3 interrupts in order: domain → resume → question
    values = [test_domain, test_resume, test_answer]
    step = 0
    while step < len(values):
        intr = out.get("__interrupt__") if isinstance(out, dict) else None
        if not intr:
            break
        prompt = _pick_prompt(intr)
        if prompt:
            print(f"Interrupt → {prompt}")
        out = resume_thread(thread_id=thread_id, value=values[step])
        step += 1

    # If we haven’t yet answered the question, try once based on state
    if step < 3:
        st = get_current_state(thread_id)
        qtext = (st.get("question") or "").strip()
        if qtext and step < 2:  # we must have skipped directly to question
            print(f"Question → {qtext}")
            out = resume_thread(thread_id=thread_id, value=test_answer)
            step = 3
        elif qtext and step == 2:
            # Already answered resume; now answer question
            print(f"Question → {qtext}")
            out = resume_thread(thread_id=thread_id, value=test_answer)
            step = 3

    # Final state
    state = get_current_state(thread_id)
    print("\n=== Final State (summary) ===")
    print(f"thread_id: {thread_id}")
    print(f"question:  {state.get('question')}")
    print(f"answer:    {state.get('answer')}")
    try:
        mta = state.get("metric_technical_accuracy") or {}
        mrd = state.get("metric_reasoning_depth") or {}
        mcc = state.get("metric_communication_clarity") or {}
        print(f"TA: {mta.get('score')} — {mta.get('feedback')}")
        print(f"RD: {mrd.get('score')} — {mrd.get('feedback')}")
        print(f"CC: {mcc.get('score')} — {mcc.get('feedback')}")
    except Exception:
        pass
    print(f"combined_score: {state.get('combined_score')}")
    print(f"overall_feedback: {state.get('overall_feedback_summary')}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        raise
