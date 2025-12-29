# CSV RAG Privacy Bot

A retrieval‑augmented generation (RAG) chatbot for querying CSV data (e.g. HR/employee records) with **strong privacy and security guardrails**.

It lets users ask natural‑language questions over CSVs while enforcing:

- Role‑based access control (admin vs normal user),
- Self‑only access for sensitive data,
- Multi‑layer PII / secrets detection,
- Detailed privacy telemetry for audits (Privacy Meter).

The LLM is **never** allowed to bypass policy: all RBAC and PII rules are enforced in Python (`rag.py` + `safety_orchestrator.py`). The LLM is only used to **answer from already-filtered context** or to **summarize** answers.

---

## Features

- **CSV → vector index** with ChromaDB (`populate.py`)
  - Each row converted into a simple `"Column: Value"` key–value document.
  - Embeddings created via `embeddings/embedding_model.py`.

- **Dynamic schema & sensitivity detection**
  - Infers columns directly from stored documents.
  - Flags sensitive-looking columns (email, phone, IDs, DOB, etc.) via regex and heuristics.

- **RBAC / self‑only policy**
  - `USERS` defined in `rag.py` with `role`, `EmpID`, `FullName`.
  - **Admin**:
    - Can see all rows and all columns (subject to PII unlock and optional gating).
  - **Normal user**:
    - Can see their **own full record**.
    - Can only see **non‑sensitive** fields for other employees (and may be fully blocked depending on toggles).
  - Self-only policy is controlled by `ENFORCE_SELF_ONLY`.

- **Centralized safety orchestrator** (`safety_orchestrator.py`)
  - LLM Guard detectors (prompt injection, jailbreak, secrets, PII, toxicity).
  - Presidio / regex PII detection.
  - Denylist + regex for value-level secrets / injection phrases.
  - Guardrails AI for optional output validation.
  - **ANY‑TRUE** policy: if any detector flags a high‑risk issue, the request or answer can be blocked.
  - Separate stages:
    - Stage 1A: `llm_guard_scan_prompt` (sanitize + telemetry, no blocking).
    - Stage 1C: `guard_input_anytrue_guards_only` (third‑party detectors).
    - Stage 2: `guard_input_regex_only` (value-only regex + denylist).
    - Retrieval/output: `guard_retrieval_anytrue`, `guard_output_anytrue`.

- **RAG over CSVs** (`rag.py`)
  - Targeted queries by EmpID or name.
  - Generic questions over the dataset.
  - Uses an embedding model (e.g. BGE) + Chroma to find relevant rows.
  - LLM (local Ollama) only sees **filtered context** and is constrained to answer from context:
    - `USE_LLM_FOR_TARGETED`: optional LLM answer for targeted queries.
    - `USE_LLM_FOR_GENERIC`: optional LLM answer for generic queries.
  - Column-level privacy via `allowed_columns_for_request` and `SENSITIVE_COLS`.
  - Optional Presidio-based redaction as defense‑in‑depth.

- **Privacy Meter & logging** (`privacy_utils.py`, `privacy_report.py`)
  - Per‑answer telemetry:
    - Status (Allowed / Blocked),
    - Stage (Input / Retrieval / Output),
    - Risk scores,
    - Flags (prompt injection, secrets, toxicity, etc.),
    - Requested sensitive fields,
    - `Output PII shown`.
  - Logs to CSV/JSONL/pretty JSON.
  - `privacy_report.py` can aggregate logs into summary privacy reports.

- **Python-based agent controller** (`agent_cli.py`)
  - No LangGraph agent; control is **pure Python**:
    - Step 0: RBAC/self‑only pre‑check (same logic as `rag.rag_query`).
    - Step 1: `guard_input_agent(question)` → input safety gate (ANY‑TRUE + regex/denylist).
    - Step 2: If blocked → ReAct-style trace + small markdown “Notice” table.
    - Step 3: If allowed → call `rag.rag_query(question)` and print:
      - ReAct-style trace,
      - Table Answer,
      - Privacy Meter.
  - Optional: If `OPENROUTER_API_KEY` is set, uses OpenRouter LLM once per answer to generate a **short, nice summary** of the table (no DB access, only sees the safe markdown).

---

## Project structure

```text
.
├─ embeddings/
│  └─ embedding_model.py        # Encapsulates the embedding model used by RAG
├─ audits/
│  ├─ giskard_rag_security.py   # Optional security tests (Giskard)
│  └─ run_giskard.py
├─ populate.py                  # Index CSVs into ChromaDB
├─ rag.py                       # Core RAG engine + CLI (RBAC + safety + Privacy Meter)
├─ agent_cli.py                 # Python "agent" on top of rag.rag_query (ReAct trace + input gate)
├─ safety_orchestrator.py       # Central safety / guardrails logic
├─ privacy_utils.py             # Privacy Meter & logging utilities
├─ privacy_report.py            # Aggregate privacy risk reports from logs
├─ .env                         # Local config (not committed)
└─ .gitignore






##Configuration (.env)
# Local Ollama model used inside rag.py
# e.g. tinyllama:latest, llama3.1, phi3:mini, etc.
OLLAMA_MODEL=llama3.1

# Optional: OpenRouter for nicer summaries in agent_cli.py
OPENROUTER_API_KEY=         # (leave empty to disable)
OPENROUTER_MODEL=mistralai/mistral-7b-instruct:free

# Privacy / policy toggles (defaults shown here)
ANY_TRUE_BLOCK=1            # Enable ANY-TRUE gating (input/retrieval/output)
ENFORCE_SELF_ONLY=1         # Non-admin users restricted to self-only
ADMIN_BYPASS_ANYTRUE=1      # Admin can bypass ANY-TRUE gates
ADMIN_BYPASS_GUARDRAILS=1   # Admin can bypass Guardrails output validator
SENSITIVE_BLOCK_ALL=0       # If 1, hard-blocks sensitive “all details” for everyone

# Optional LLM usage in rag.py
USE_LLM_FOR_TARGETED=0      # If 1, local LLM adds an "Answer" row for targeted queries
USE_LLM_FOR_GENERIC=0       # If 1, local LLM summarizes generic query context




###Install dependencies
pip install \
  chromadb \
  pandas \
  tabulate \
  llm-guard \
  guardrails-ai \
  presidio-analyzer \
  presidio-anonymizer \
  langchain-core \
  langchain-ollama \
  langchain-openai \
  python-dotenv \
  requests
