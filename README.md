# CSV RAG Privacy Bot

A retrieval‑augmented generation (RAG) chatbot for querying CSV data (e.g. HR/employee records) with **strong privacy and security guardrails**.

It lets users ask natural‑language questions over CSVs while enforcing:

- role‑based access control (admin vs normal user),
- self‑only access for sensitive data,
- multi‑layer PII / secrets detection,
- detailed privacy telemetry for audits.

---

## Features

- **CSV → vector index** with ChromaDB (`populate.py`)
  - Each row becomes a simple `"Column: Value"` key‑value document.
- **Dynamic schema & sensitivity detection**
  - Automatically infers columns and flags sensitive ones (email, phone, IDs, etc).
- **RBAC / self‑only policy**
  - Admin: see everything.
  - Normal user: can see their own full record; only non‑sensitive fields for others.
- **Centralized safety orchestrator** (`safety_orchestrator.py`)
  - LLM Guard (prompt, secrets, toxicity, PII),
  - Presidio / regex PII detection,
  - Guardrails validation,
  - ANY‑TRUE policy: if any detector flags, the request/answer can be blocked.
- **RAG over CSVs** (`rag.py`)
  - Targeted lookups by EmpID or name.
  - Generic questions over the dataset.
  - LLM constrained to context‑only answers.
- **Privacy Meter & logging** (`privacy_utils.py`, `privacy_report.py`)
  - Per‑answer risk score, flags, and “Output PII shown”.
  - CSV/JSONL/pretty JSON logs.
  - Aggregate privacy reports via `privacy_report.py`.

---

## Project structure

```text
.
├─ embeddings/
│  └─ embedding_model.py        # Encapsulates the embedding model
├─ audits/
│  ├─ giskard_rag_security.py   # Optional security tests (Giskard)
│  └─ run_giskard.py
├─ populate.py                  # Index CSVs into ChromaDB
├─ rag.py                       # Main CLI RAG chatbot (RBAC + safety)
├─ safety_orchestrator.py       # Central safety/guardrails logic
├─ privacy_utils.py             # Privacy Meter & logging utilities
├─ privacy_report.py            # Generate privacy risk reports from logs
└─ .gitignore

Install dependencies
chromadb
langchain-community
pandas
tabulate
ollama
llm-guard
guardrails-ai
presidio-analyzer
presidio-anonymizer
