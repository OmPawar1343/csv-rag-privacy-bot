"""
Simple "agent" CLI for CSV RAG Privacy Bot

What this does:
  - Reuses your existing safety_orchestrator and rag.rag_query.
  - Implements a Data Privacy Agent in pure Python (no LangChain agent),
    which controls the workflow:

      0) RBAC/self-only pre-check (same logic as rag.rag_query).
      1) Call guard_input_agent(question)  -> input safety checks.
      2) If blocked: return a polite refusal (with reason).
      3) If allowed: call rag.rag_query(question) and return its answer.

  - The local LLM (TinyLlama via rag.py) is only used INSIDE rag.rag_query to
    answer from retrieved context, not to control tools.

  - For explainability, the agent prints a small "Thought / Action /
    Observation" trace to show the decision steps (similar to ReAct)
    but fully controlled by Python, not by the LLM.

  - If OPENROUTER_API_KEY is set, we also call a remote LLM (via OpenRouter)
    to generate a short, nice text summary of the table answer.
"""

from typing import Tuple
import os
import json
import requests

from dotenv import load_dotenv
load_dotenv()

# Reuse your existing core logic
import rag
from rag import (
    ANY_TRUE_BLOCK,
    ADMIN_BYPASS_ANYTRUE,
    parse_question,
    maybe_add_self_target,
    enforce_self_scope_for_user,
    is_self_overall,
)

from safety_orchestrator import (
    guard_input_anytrue_guards_only,
    guard_input_regex_only,
    FRIENDLY_MSG,
)

# ----- OpenRouter config (for nicer text summaries) -----

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "mistralai/mistral-7b-instruct:free")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def summarize_with_openrouter(question: str, md_answer: str) -> str | None:
    """
    Optional helper: use a stronger remote LLM (via OpenRouter) to
    generate a concise text summary from the markdown answer.

    If no API key is set or any error occurs, returns None.
    """
    if not OPENROUTER_API_KEY:
        return None

    # Use only the "Results" section (ignore Privacy Meter in summary)
    parts = md_answer.split("## Privacy Meter")
    main_part = parts[0].strip()

    system_prompt = (
        "You are a helpful HR data assistant. You will receive a QUESTION and a "
        "markdown RESULT table produced by a secure CSV RAG system.\n\n"
        "Rules:\n"
        "- Answer the QUESTION using ONLY the information in the RESULT.\n"
        "- If the answer is not present, say exactly: \"Not available in context.\"\n"
        "- Keep the answer short (1–3 sentences).\n"
        "- Do not show extra columns beyond what is clearly in the RESULT.\n"
    )

    user_message = f"QUESTION:\n{question}\n\nRESULT:\n{main_part}"

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-Title": "CSV-RAG-Agent-CLI",
    }

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "max_tokens": 200,
        "temperature": 0.3,
    }

    try:
        resp = requests.post(OPENROUTER_URL, headers=headers, data=json.dumps(payload), timeout=30)
        if resp.status_code != 200:
            return None
        data = resp.json()
        if "choices" in data and data["choices"]:
            text = data["choices"][0]["message"]["content"]
            return (text or "").strip()
        return None
    except Exception:
        return None


def guard_input_agent(question: str) -> Tuple[bool, str]:
    """
    Privacy Agent safety tool.

    Uses your safety_orchestrator input checks, but aligns with rag.rag_query:

      - Respect ANY_TRUE_BLOCK toggle.
      - Admin bypass if ADMIN_BYPASS_ANYTRUE=1.
      - Skip Stage 2 regex/denylist for self-only queries (self_overall=True).

    Returns:
      allowed (bool): True if stages pass (given the above rules), False otherwise.
      reason  (str):  "ALLOWED" if safe, otherwise a human-friendly reason.
    """
    # Global off switch: if ANY_TRUE_BLOCK=0, agent does not block on these gates.
    if not ANY_TRUE_BLOCK:
        return True, "ALLOWED"

    user = rag.CURRENT_USER or {}
    is_admin = user.get("role") == "admin"

    # Admin bypass for ANY-TRUE gates
    if is_admin and ADMIN_BYPASS_ANYTRUE:
        return True, "ALLOWED"

    # Determine if this is a self-only request (same logic as rag.rag_query)
    parsed = parse_question(question)
    empids, names = parsed["EmpIDs"], parsed["Names"]
    names = maybe_add_self_target(question, names)
    empids, names, _ = enforce_self_scope_for_user(user, empids, names)
    self_overall = is_self_overall(user, empids, names)

    # Stage 1C: third-party guards-only (llm-guard, presidio, etc.)
    ok1, msg1, who1, kinds1 = guard_input_anytrue_guards_only(question)

    # Stage 2: regex + denylist (skip for self-only, as in rag.rag_query)
    if self_overall:
        ok2, msg2, who2, kinds2 = True, "", [], []
    else:
        ok2, msg2, who2, kinds2 = guard_input_regex_only(question)

    if ok1 and ok2:
        return True, "ALLOWED"

    # Blocked: build a combined reason from both stages
    blocked_parts = []
    if not ok1:
        blocked_parts.append(msg1 or FRIENDLY_MSG)
    if not ok2:
        blocked_parts.append(msg2 or FRIENDLY_MSG)
    reason = " ".join(blocked_parts) if blocked_parts else FRIENDLY_MSG
    return False, reason


def agent_answer(question: str, with_trace: bool = True) -> str:
    """
    Main agent controller.

    Workflow:
      0) RBAC/self-only pre-check (same logic as rag.rag_query).
      1) Run input guards via guard_input_agent(question).
      2) If blocked:
           - Print ReAct-style trace (if with_trace=True).
           - Return a polite refusal as a markdown table.
      3) If allowed:
           - Call rag.rag_query(question) to run the full secure RAG engine
             (RBAC, self-only, retrieval/output gates, redaction, Privacy Meter).
           - Print trace and return the RAG answer (markdown).
    """
    trace = []

    user = rag.CURRENT_USER or {}

    # ---- 0) RBAC / self-only pre-check (mirror rag.rag_query) ----
    parsed = parse_question(question)
    empids, names = parsed["EmpIDs"], parsed["Names"]
    names = maybe_add_self_target(question, names)
    empids, names, self_blocked = enforce_self_scope_for_user(user, empids, names)

    if self_blocked:
        trace.append(
            "Thought: Policy is self-only; this question targets another employee, "
            "so I will refuse before calling RAG."
        )
        polite = "\n".join([
            "## Results",
            "| Column | Value |",
            "|---|---|",
            "| Notice | For privacy reasons, you can only view your own record. |",
        ])
        if with_trace:
            print("\n--- Agent Trace ---")
            print("\n".join(trace))
            print("--- End Trace ---")
        return polite

    # ---- 1) Safety gate (prompt injection / secrets / denylist) ----
    trace.append("Thought: I should first check if the question is safe to answer.")
    allowed, reason = guard_input_agent(question)
    trace.append("Action: guard_input_agent (ANY-TRUE + regex/denylist)")
    trace.append(f"Observation: {reason if not allowed else 'ALLOWED'}")

    if not allowed:
        trace.append("Thought: The question is not safe, so I will refuse and not call RAG.")
        polite = "\n".join([
            "## Results",
            "| Column | Value |",
            "|---|---|",
            f"| Notice | {reason} |",
        ])
        if with_trace:
            print("\n--- Agent Trace ---")
            print("\n".join(trace))
            print("--- End Trace ---")
        return polite

    # ---- 2) Delegate to the full secure RAG engine ----
    trace.append("Thought: The question is allowed, so I will call rag.rag_query to answer it.")
    trace.append("Action: rag.rag_query(question)")

    answer_md = rag.rag_query(question)

    trace.append("Observation: RAG answer (with RBAC, redaction, Privacy Meter) produced.")

    if with_trace:
        print("\n--- Agent Trace ---")
        print("\n".join(trace))
        print("--- End Trace ---")

    return answer_md


if __name__ == "__main__":
    # Reuse your existing authentication and CURRENT_USER from rag.py
    rag.CURRENT_USER = rag.authenticate()
    print("👋 Welcome to the CSV RAG Agent Chatbot! 🗂️🤖")
    print("Commands: login | switch | whoami | unlock pii | lock pii | logout | exit | quit")
    print("Note: This CLI uses a simple Python-based Data Privacy Agent on top of your RAG engine.")
    if OPENROUTER_API_KEY:
        print("Note: OpenRouter summarization is ENABLED.\n")
    else:
        print("Note: OpenRouter summarization is DISABLED (no OPENROUTER_API_KEY).\n")

    while True:
        question = input("\nYour Question: ").strip()
        if not question:
            continue

        ql = question.lower()

        # End session
        if ql in ("exit", "quit", "logout"):
            print("Thank You Goodbye! 👋")
            break

        # Re-login / switch user (no restart needed)
        if ql in ("login", "switch", "change user", "relogin"):
            try:
                rag.CURRENT_USER = rag.authenticate()
                print(
                    f"✅ Switched to {rag.CURRENT_USER.get('FullName','-')} "
                    f"({rag.CURRENT_USER.get('role','-')})"
                )
            except SystemExit:
                print("Switch cancelled.")
            continue

        # Admin: unlock PII (step-up auth) by delegating to rag's logic
        if ql in ("unlock pii", "unlock", "unlock pii data"):
            if not rag.CURRENT_USER or rag.CURRENT_USER.get("role") != "admin":
                print("Only admin users can unlock sensitive fields.")
                continue
            try:
                from pwinput import pwinput
                pw = pwinput("Enter PII unlock password: ", mask="*")
            except Exception:
                from getpass import getpass
                pw = getpass("Enter PII unlock password: ")
            expected = rag.ADMIN_PII_UNLOCK_PASSWORD or rag.CURRENT_USER.get("password")
            if pw == expected:
                rag.CURRENT_USER["pii_unlocked"] = True
                print("🔓 Sensitive fields unlocked for this session.")
            else:
                print("❌ Incorrect unlock password; sensitive fields remain masked.")
            continue

        # Lock PII again
        if ql in ("lock pii", "lock"):
            if rag.CURRENT_USER:
                rag.CURRENT_USER["pii_unlocked"] = False
                print("🔒 Sensitive fields locked.")
            else:
                print("No active user.")
            continue

        # Show current user
        if ql in ("whoami",):
            u = rag.CURRENT_USER or {}
            unlocked = "yes" if u.get("pii_unlocked") else "no"
            print(
                f"👤 Current user: {u.get('FullName','-')} "
                f"(role: {u.get('role','-')}, EmpID: {u.get('EmpID','-')}, PII unlocked: {unlocked})"
            )
            continue

        print("🤔 Agent controller deciding... (RBAC → safety → RAG), please wait...")
        try:
            md_response = agent_answer(question, with_trace=True)

            # Pretty-print the table answer
            try:
                print("\nTable Answer:\n", rag.markdown_to_table(md_response))
            except Exception:
                print("\nTable Answer:\n", md_response)

            # Optional: nicer text summary via OpenRouter
            summary = summarize_with_openrouter(question, md_response)
            if summary:
                print("\nLLM Summary (OpenRouter):\n", summary)

        except Exception as e:
            print(f"\n[!] Agent error: {e}")