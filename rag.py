# rag.py
# Updated pipeline:
# - Stage 1A: LLM Guard sanitize
# - Stage 1B: RBAC/self-only early (same level as LLM Guard)
# - Stage 1C: Input third-party guards-only (injection/jailbreak/secrets/toxicity)
# - Stage 2: Input fallback (value-only regex) + denylist (skipped for self)
# Retrieval/Output: for self = secrets-only block; for others = full PII block.
# Raga/Giskard = offline/shadow only.




# --- Silence third‑party logs and progress bars (place at very top) ---
import os, sys, logging, contextlib
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["DISABLE_TQDM"] = "1"
os.environ["TQDM_DISABLE"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["LOGURU_LEVEL"] = "ERROR"

logging.basicConfig(level=logging.ERROR)
for name in [
    "transformers", "huggingface_hub", "urllib3",
    "presidio", "presidio-analyzer", "presidio-anonymizer",
    "sentence_transformers", "llm_guard", "protectai", "inference",
]:
    logging.getLogger(name).setLevel(logging.ERROR)

try:
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
    try:
        hf_logging.disable_progress_bar()
    except Exception:
        pass
except Exception:
    pass

try:
    from loguru import logger as loguru_logger
    loguru_logger.remove()
    loguru_logger.add(sys.stderr, level="ERROR")
except Exception:
    pass
# --- end silence block ---

import re
import warnings
warnings.filterwarnings("ignore")

from collections import defaultdict
import pandas as pd
from tabulate import tabulate
from chromadb import PersistentClient
from langchain_community.llms.ollama import Ollama

from embeddings.embedding_model import get_embedding_model
from privacy_utils import PRIVACY_METER, privacy_meter_report

# Updated orchestrator imports
from safety_orchestrator import (
    guard_input_anytrue_guards_only,  # third-party guards only at input (no PII intent)
    guard_input_regex_only,           # regex-only input check (value-only, after RBAC)
    guard_retrieval_anytrue,
    guard_output_anytrue,
    FRIENDLY_MSG,
)

# Config toggles
USE_LLM_FOR_TARGETED = False
USE_LLM_FOR_GENERIC = os.getenv("USE_LLM_FOR_GENERIC", "0").lower() in ("1","true","yes","on")
RAG_DISTANCE_THRESHOLD = float(os.getenv("RAG_DISTANCE_THRESHOLD", "0.0"))
SELF_TARGET_ENABLED = os.getenv("SELF_TARGET_ENABLED", "0").lower() in ("1","true","yes","on")
MAX_TARGETS = int(os.getenv("MAX_TARGETS", "1"))
ANY_TRUE_BLOCK = os.getenv("ANY_TRUE_BLOCK", "1").lower() in ("1","true","yes","on")

# Self-only policy
ENFORCE_SELF_ONLY = os.getenv("ENFORCE_SELF_ONLY", "1").lower() in ("1","true","yes","on")
SELF_TARGET_PREFER = os.getenv("SELF_TARGET_PREFER", "empid").lower()

# Admin bypass toggles
ADMIN_BYPASS_ANYTRUE     = os.getenv("ADMIN_BYPASS_ANYTRUE", "1").lower() in ("1","true","yes","on")
ADMIN_BYPASS_LLM_GUARD   = os.getenv("ADMIN_BYPASS_LLM_GUARD", "1").lower() in ("1","true","yes","on")
ADMIN_BYPASS_DENYLIST    = os.getenv("ADMIN_BYPASS_DENYLIST", "1").lower() in ("1","true","yes","on")
ADMIN_BYPASS_GUARDRAILS  = os.getenv("ADMIN_BYPASS_GUARDRAILS", "1").lower() in ("1","true","yes","on")

# ========= LLM Guard (third-party) =========
# Why: Keep local wrappers to sanitize and log input/output when orchestrator isn't the one doing Stage 1A.
try:
    with open(os.devnull, "w") as _devnull, \
         contextlib.redirect_stdout(_devnull), \
         contextlib.redirect_stderr(_devnull):
        from llm_guard import scan_prompt, scan_output
        try:
            from llm_guard.input_scanners import PromptInjection, Secrets, Toxicity, Jailbreak
            INPUT_SCANNERS = [
                PromptInjection(threshold=0.1),
                Jailbreak(threshold=0.1),
                Secrets(),
                Toxicity(threshold=0.5),
            ]
        except Exception:
            from llm_guard.input_scanners import PromptInjection, Secrets, Toxicity
            INPUT_SCANNERS = [
                PromptInjection(threshold=0.1),
                Secrets(),
                Toxicity(threshold=0.5),
            ]
        from llm_guard.output_scanners import PII as PIIOut, Toxicity as ToxicityOut
        OUTPUT_SCANNERS = [PIIOut(), ToxicityOut()]
        LLM_GUARD_AVAILABLE = True

    try:
        from loguru import logger as _loguru_logger
        _loguru_logger.disable("llm_guard")
        _loguru_logger.disable("protectai")
        _loguru_logger.disable("inference")
        _loguru_logger.remove()
        _loguru_logger.add(sys.stderr, level="ERROR")
    except Exception:
        pass
except Exception:
    LLM_GUARD_AVAILABLE = False
    INPUT_SCANNERS, OUTPUT_SCANNERS = [], []

# Denylist (kept for injection/jailbreak phrases)
# Why: Extra phrasing guard (legacy). Stage 2 regex/denylist in orchestrator generally covers this.
DENY_PATTERNS = [
    r"ignore all previous instructions",
    r"hidden dataset",
    r"reveal secrets",
    r"system prompt",
    r"\bjailbreak\b",
    r"bypass (?:your|the) (?:rules|guardrails)",
    r"training data( prompt)?",
    r"show me (?:the )?hidden",
]
def is_policy_denied(q: str) -> bool:
    ql = q.lower()
    return any(re.search(p, ql) for p in DENY_PATTERNS)

# Heuristic scans for meter visibility (used only when LLM Guard unavailable)
# Why: Minimal fallback so Privacy table has some signal if llm_guard import is unavailable.
HEURISTIC_INJECTION_PATTERNS = [
    r"(?:remove|disable).*(?:safety|guardrails|constraints)",
    r"(?:print|disclose|reveal).*(?:internal|configuration|policy|guideline)",
    r"for this (?:message|response) only.*(?:priority|rule)",
]
HEURISTIC_JAILBREAK_PATTERNS = [r"\bact as admin\b", r"\bdeveloper mode\b", r"\bdo anything now\b"]
HEURISTIC_SECRETS_PATTERNS = [r"\bAKIA[0-9A-Z]{16}\b", r"-----BEGIN (?:RSA )?PRIVATE KEY-----"]
HEURISTIC_TOXICITY_PATTERNS = [r"\bidiot\b", r"\bstupid\b", r"\bmoron\b", r"\bshut up\b"]

class _SimpleScanResult:
    def __init__(self, name: str, valid: bool):
        self.name = name
        self.valid = valid
def _heuristic_scan(text: str):
    t = (text or "")
    results = []
    if any(re.search(p, t, flags=re.IGNORECASE) for p in HEURISTIC_INJECTION_PATTERNS):
        results.append(_SimpleScanResult("PromptInjection", False))
    if any(re.search(p, t, flags=re.IGNORECASE) for p in HEURISTIC_JAILBREAK_PATTERNS):
        results.append(_SimpleScanResult("Jailbreak", False))
    if any(re.search(p, t, flags=re.IGNORECASE) for p in HEURISTIC_SECRETS_PATTERNS):
        results.append(_SimpleScanResult("Secrets", False))
    if any(re.search(p, t, flags=re.IGNORECASE) for p in HEURISTIC_TOXICITY_PATTERNS):
        results.append(_SimpleScanResult("Toxicity", False))
    return results

# ========= Guardrails AI (optional) =========
# Why: Extra validation option for generations. We still enforce output gates.
try:
    import guardrails as gd
    GUARDRAILS_AVAILABLE = True
    RAIL_SPEC = """
<rail version="0.1">
  <output type="string" />
  <prompt>
You are a helpful assistant. Answer ONLY using the Context.
If not present, reply exactly: "Not available in context."
Question:
${question}
Context:
${context}
  </prompt>
  <validate>
    <regex_match name="no_injection"
                 pattern="^(?:(?!(ignore all previous instructions|hidden dataset|reveal secrets|system prompt|jailbreak|bypass (?:your|the) (?:rules|guardrails)|training data|show me (?:the )?hidden)).)*$"
                 on-fail="refrain" />
    <validator name="guardrails/toxic_language" on-fail="refrain" />
  </validate>
</rail>
"""
    guard = gd.Guard.from_rail_string(RAIL_SPEC)

    SELF_RAIL_SPEC = """
<rail version="0.1">
  <output type="string" />
  <validate>
    <regex_match name="no_injection"
                 pattern="^(?:(?!(ignore all previous instructions|system prompt|jailbreak|bypass (?:your|the) (?:rules|guardrails)|reveal secrets)).)*$"
                 on-fail="refrain" />
  </validate>
</rail>
"""
    guard_self = gd.Guard.from_rail_string(SELF_RAIL_SPEC)
except Exception:
    GUARDRAILS_AVAILABLE = False
    guard = None
    guard_self = None

# ========= Presidio (optional) =========
# Why: Optional redaction layer; gates should already control PII. This is defense-in-depth.
try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine, OperatorConfig
    PRESIDIO_AVAILABLE = True
except Exception:
    PRESIDIO_AVAILABLE = False

# Initialize LLM
llm = Ollama(model=os.getenv("OLLAMA_MODEL", "tinyllama"), temperature=0.0, num_ctx=512, num_predict=250)
def llm_call(prompt: str, **kwargs) -> str:
    return llm.invoke(prompt)

# ========= DB paths (configurable) =========
DB_PATH = os.getenv("CHROMA_DB_PATH", "db/chroma_db")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "csv_collection")

# Initialize ChromaDB
client = PersistentClient(path=DB_PATH)
collection = client.get_collection(name=COLLECTION_NAME)

# Embedding model
embedding_model = get_embedding_model()

# ===== Dynamic schema inference (no hardcoding) =====
# Why: Infer columns directly from stored KV docs; mark sensitive columns via regex heuristics.
EMAIL_RE  = re.compile(r'[\w\.-]+@[\w\.-]+', re.I)
PHONE_RE  = re.compile(r'\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b')
SSN_RE    = re.compile(r'\b\d{3}-\d{2}-\d{4}\b')
DOB_RE    = re.compile(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b')
IP_RE     = re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')
V6_RE     = re.compile(r'\b([0-9a-f]{0,4}:){2,7}[0-9a-f]{0,4}\b', re.I)
CC_RE     = re.compile(r'\b(?:\d[ -]*?){13,19}\b')

SCHEMA_SAMPLE_LIMIT = int(os.getenv("SCHEMA_SAMPLE_LIMIT", "200"))
SENSITIVE_MIN_RATIO = float(os.getenv("SENSITIVE_MIN_RATIO", "0.15"))

def _parse_kv_lines(doc_text: str) -> dict:
    out = {}
    for line in (doc_text or "").splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        k, v = k.strip(), v.strip()
        if k and k not in out:
            out[k] = v
    return out

def _normalize_value(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())

def _match_target_in_doc(doc_text: str, target_label: str) -> bool:
    # Why: Ensure the document belongs to the requested target (EmpID/Name) before using it.
    tl = _normalize_value(target_label)
    kv = _parse_kv_lines(doc_text)
    vals = [_normalize_value(v) for v in kv.values()]
    if tl in vals:
        return True
    parts = [p for p in tl.split() if p]
    if len(parts) == 2:
        first, last = parts
        if first in vals and last in vals:
            return True
        low = (doc_text or "").lower()
        if re.search(rf"\b{re.escape(first)}\b", low) and re.search(rf"\b{re.escape(last)}\b", low):
            return True
    return False

def _get_sample_docs(collection, limit=SCHEMA_SAMPLE_LIMIT) -> list[str]:
    try:
        got = collection.get(limit=limit, include=["documents"])
        docs = got.get("documents", []) or []
        if docs and isinstance(docs[0], list):
            docs = [d for sub in docs for d in sub]
        return [d for d in docs if isinstance(d, str)]
    except Exception:
        return []

def _is_id_like(values: list[str]) -> bool:
    # Why: Columns that look mostly numeric/unique are treated as ID-like, likely sensitive.
    vals = [v for v in values if isinstance(v, str)]
    digits = [v for v in vals if re.fullmatch(r"\d{3,}", v or "")]
    if not vals or len(digits) / max(1, len(vals)) < 0.6:
        return False
    unique_ratio = len(set(digits)) / max(1, len(digits))
    return unique_ratio >= 0.7

def _auto_detect_sensitive(col_values: list[str]) -> bool:
    # Why: Tag a column as sensitive if enough values match PII-like patterns.
    n = max(1, len(col_values))
    hits = sum(
        1 for v in col_values
        if EMAIL_RE.search(str(v or "")) or PHONE_RE.search(str(v or "")) or
           SSN_RE.search(str(v or "")) or DOB_RE.search(str(v or "")) or
           IP_RE.search(str(v or "")) or CC_RE.search(str(v or ""))
    )
    return (hits / n) >= SENSITIVE_MIN_RATIO

def infer_schema_from_collection(collection) -> tuple[list[str], set[str]]:
    docs = _get_sample_docs(collection)
    by_col = defaultdict(list)
    for t in docs:
        kv = _parse_kv_lines(t)
        for k, v in kv.items():
            by_col[k].append(v)
    all_cols = list(by_col.keys())
    sensitive = set()
    for c, vals in by_col.items():
        if _auto_detect_sensitive(vals) or _is_id_like(vals):
            sensitive.add(c)
    return all_cols, sensitive

ALL_COLUMNS, SENSITIVE_COLS = infer_schema_from_collection(collection)
NON_SENSITIVE_COLS = [c for c in ALL_COLUMNS if c not in SENSITIVE_COLS]
SUGGESTED_SAFE_FIELDS = NON_SENSITIVE_COLS[:6]

# -------- Sensitive category detection (schema-free) --------
# Why: Lightweight, schema-free detection for user intent and context categories (not hard regex PII).
SENSITIVE_BLOCK_ALL = os.getenv("SENSITIVE_BLOCK_ALL", "0").lower() in ("1","true","yes","on")
SENSITIVE_KEYWORDS = {
    "salary": ["salary","compensation","pay grade","ctc","wage","pay band","package"],
    "performance": ["performance score","rating","appraisal","kpi","okrs","disciplinary","termination"],
    "health": ["mrn","medical record","insurance id","icd","cpt","rx"],
    "demographics": ["gender","race","ethnicity","marital","religion","sexual orientation","political"],
    "address": ["address","street","city","state","zip","postal"],
    "secrets": ["password","api key","token","jwt","secret","private key"],
}
UUID_RE = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b", re.I)
MAC_RE  = re.compile(r"\b(?:[0-9a-f]{2}[:-]){5}[0-9a-f]{2}\b", re.I)
IMEI_RE = re.compile(r"\b\d{15}\b")
IBAN_RE = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b")

def _luhn_ok(s: str) -> bool:
    s = "".join(ch for ch in s if ch is not None and ch.isdigit())
    if not (13 <= len(s) <= 19): return False
    total = 0
    parity = len(s) % 2
    for i, ch in enumerate(s):
        d = int(ch)
        if i % 2 == parity:
            d *= 2
            if d > 9: d -= 9
        total += d
    return total % 10 == 0

def detect_sensitive_intent(question: str) -> set[str]:
    # Why: Identify sensitive intent in the question (helps Privacy Meter reasoning).
    q = (question or "").lower()
    cats = set()
    for cat, keys in SENSITIVE_KEYWORDS.items():
        if any(k in q for k in keys):
            cats.add(cat)
    if any(x in q for x in ["email","e-mail"]): cats.add("email")
    if "phone" in q or "mobile" in q: cats.add("phone")
    if "dob" in q or "date of birth" in q: cats.add("dob")
    if "ip" in q: cats.add("ip")
    if "uuid" in q: cats.add("uuid")
    if "mac" in q: cats.add("mac")
    if "gps" in q or "latitude" in q or "longitude" in q: cats.add("gps")
    if "credit card" in q or "card number" in q: cats.add("card")
    if "ssn" in q or "aadhaar" in q or "passport" in q or "driver" in q: cats.add("national_id")
    return cats

def detect_sensitive_categories_in_text(text: str) -> set[str]:
    # Why: Lightweight scan of retrieved context for sensitive categories (for reasons/refusals).
    t = text or ""
    cats = set()
    if EMAIL_RE.search(t): cats.add("email")
    if PHONE_RE.search(t): cats.add("phone")
    if DOB_RE.search(t): cats.add("dob")
    if SSN_RE.search(t): cats.add("national_id")
    if IP_RE.search(t) or V6_RE.search(t): cats.add("ip")
    if UUID_RE.search(t): cats.add("uuid")
    if MAC_RE.search(t): cats.add("mac")
    if IBAN_RE.search(t): cats.add("bank")
    if IMEI_RE.search(t): cats.add("device_id")
    for cand in re.findall(r"\b(?:\d[ -]?){13,19}\b", t):
        if _luhn_ok(cand): cats.add("card"); break
    if re.search(r"\b-?\d{1,2}\.\d{3,}\s*,\s*-?\d{1,3}\.\d{3,}\b", t): cats.add("gps")
    if re.search(r"-----BEGIN [^-]+ PRIVATE KEY-----", t) or re.search(r"\bAKIA[0-9A-Z]{16}\b", t) or re.search(r"\b(bearer|eyJ)[A-Za-z0-9\._\-]+\b", t, re.I):
        cats.add("secrets")
    low = t.lower()
    if any(k in low for k in SENSITIVE_KEYWORDS["salary"]): cats.add("salary")
    if any(k in low for k in SENSITIVE_KEYWORDS["performance"]): cats.add("performance")
    if any(k in low for k in SENSITIVE_KEYWORDS["health"]): cats.add("health")
    if any(k in low for k in SENSITIVE_KEYWORDS["address"]): cats.add("address")
    if any(k in low for k in SENSITIVE_KEYWORDS["demographics"]): cats.add("demographics")
    if any(k in low for k in SENSITIVE_KEYWORDS["secrets"]): cats.add("secrets")
    return cats

# ========== AUTH / RBAC ==========
USERS = {
    "admin": {"password": "admin123", "role": "admin", "EmpID": "0", "FullName": "Admin User"},
    "nehmat": {"password": "user123", "role": "user", "EmpID": "1000", "FullName": "Nehmat Anne"},
}
CURRENT_USER = None

def authenticate(max_attempts=3):
    """
    What: Simple CLI login for demo/testing.
    Why: We need a user context (role, EmpID, FullName) to enforce RBAC/self-only policy.
    """
    print("🔐 Login required")
    for attempt in range(1, max_attempts + 1):
        username = input("Username: ").strip()
        try:
            from pwinput import pwinput
            password = pwinput(f"Password (attempt {attempt}/{max_attempts}): ", mask='*')
        except Exception:
            from getpass import getpass
            password = getpass(f"Password (attempt {attempt}/{max_attempts}, typing hidden): ").strip()
        user = USERS.get(username)
        if user and user["password"] == password:
            print(f"✅ Logged in as {username} ({user['role']})")
            return user
        remaining = max_attempts - attempt
        print("❌ Invalid credentials." + (f" Attempts left: {remaining}" if remaining>0 else ""))
    raise SystemExit(1)

def is_self_request(user, empid: str, fullname: str) -> bool:
    """
    What: Check if the current request is for the logged-in user's own record.
    Why: Self requests get broader access (PII allowed; secrets still blocked).
    """
    if not user: return False
    if empid and str(user.get("EmpID")) == str(empid): return True
    if fullname and user.get("FullName") and fullname.lower().strip() == user["FullName"].lower().strip(): return True
    return False

def maybe_add_self_target(question: str, names_list: list) -> list:
    """
    What: If user says "my", auto-scope target to self (no explicit name/ID provided).
    Why: Usability — helps users get their own info without typing their name/ID.
    """
    if not (SELF_TARGET_ENABLED and CURRENT_USER and CURRENT_USER.get("FullName")):
        return names_list
    if names_list:
        return names_list
    q = f" {question.lower()} "
    if " my " in q and CURRENT_USER["FullName"].lower() not in [n.lower() for n in names_list]:
        names_list.append(CURRENT_USER["FullName"])
    return names_list

def get_self_target(user):
    """
    What: Extract the user's (EmpID, FullName) tuple if available.
    Why: Shared helper for scoping and self checks.
    """
    if not user:
        return None, None
    empid = str(user.get("EmpID", "")).strip() or None
    fullname = (user.get("FullName") or "").strip() or None
    return empid, fullname

def enforce_self_scope_for_user(user, empids: list[str], names: list[str]):
    """
    What: Enforce self-only access for non-admin users.
    Why: Prevent lateral access as an early policy gate, before retrieval/output.

    Behavior:
      - If non-admin asks for others, scope back to self and flag self_blocked=True for clear notice.
      - If no target is provided, auto-scope to self when possible.
    """
    if not user or user.get("role") == "admin" or not ENFORCE_SELF_ONLY:
        return empids, names, False

    self_empid, self_name = get_self_target(user)

    asked_other = False
    if empids and self_empid and any(str(e) != str(self_empid) for e in empids):
        asked_other = True
    if names and self_name and any(n.strip().lower() != (self_name or "").strip().lower() for n in names):
        asked_other = True

    if asked_other:
        if self_empid:
            return [self_empid], [], True
        if self_name:
            return [], [self_name], True
        return [], [], True

    if not empids and not names:
        if SELF_TARGET_PREFER == "empid" and self_empid:
            return [self_empid], [], False
        if self_name:
            return [], [self_name], False

    if self_empid:
        empids = [e for e in empids if str(e) == str(self_empid)]
    if self_name:
        names = [n for n in names if n.strip().lower() == (self_name or "").strip().lower()]

    if not empids and not names:
        if self_empid:
            return [self_empid], [], False
        if self_name:
            return [], [self_name], False

    return empids, names, False

def is_self_overall(user, empids: list[str], names: list[str]) -> bool:
    """
    What: Determine if the entire request is for the logged-in user's own record.
    Why: Stage 2 regex is skipped for self to reduce friction.
    """
    if not user:
        return False
    self_empid, self_name = get_self_target(user)
    if empids and self_empid and str(empids[0]) == str(self_empid):
        return True
    if names and self_name and names[0].strip().lower() == (self_name or "").strip().lower():
        return True
    return False

# ---------- Name parsing helpers (no spaCy) ----------
# Why: Lightweight heuristic to pull "First Last"; STOP_TOKENS avoid injection tokens being treated as names.
STOP_TOKENS = {
    "a","an","the","this","that","these","those",
    "i","i'm","im","me","my","myself","mine",
    "you","your","yours","we","our","ours","they","them","their","theirs",
    "am","is","are","was","were","be","been","being",
    "please","kindly","just","only","also",
    "and","or","but","if","of","in","on","for","to","with","by","from","as","at","about","regarding",
    "assume","assuming","assumed","suppose","supposing","supposed",
    "admin","user","team","department","dept","hr",
    "details","info","information",
    "tell","show","give","provide","find","fetch","list","what","whats","what's",
    "gender","dob","date","birth","got","developer","mode","unfiltered","restrictions","safeguards","constraints",
    "rule","priority","top","response","only","enable","bypass","now","ok","safe","guards","guardrails",
    # extra stop tokens to avoid misclassifying injection phrases as names
    "system","prompt","previous","instructions","ignore","hidden","dataset","enabled","reveal"
}
FILLER_PAIRS = {
    ("tell","me"), ("give","me"), ("show","me"), ("provide","me"), ("list","me"),
    ("am","admin"), ("i","am"), ("as","admin"), ("as","user"),
    ("the","gender"), ("the","dob"), ("all","details")
}

def extract_names_from_text(text: str) -> list[str]:
    """
    What: Heuristically extract a likely 'First Last' name from free text.
    Why: We avoid spaCy for speed/size; use STOP_TOKENS to prevent phrases like
         "Developer Mode" or "System Prompt" from being misread as names.
    """
    text = text or ""
    names = set()
    for chunk in re.findall(r"<([^<>]+)>", text):
        for m in re.finditer(r"\b([A-Za-z][a-zA-Z'`-]+)\s+([A-Za-z][a-zA-Z'`-]+)\b", chunk):
            names.add(f"{m.group(1).strip()} {m.group(2).strip()}")
    for m in re.finditer(r"\b(?:of|for)\s+([A-Za-z][a-zA-Z'`-]+)\s+([A-Za-z][a-zA-Z'`-]+)\b", text, flags=re.I):
        first, last = m.group(1).strip(), m.group(2).strip()
        names.add(f"{first.title()} {last.title()}")
    # Capitalized pairs, filter out stop tokens to avoid misclassifying injection phrases.
    for m in re.finditer(r"\b([A-Z][a-zA-Z'`-]+)\s+([A-Z][a-zA-Z'`-]+)\b", text):
        a, b = m.group(1).strip(), m.group(2).strip()
        if a.lower() in STOP_TOKENS or b.lower() in STOP_TOKENS:
            continue
        names.add(f"{a} {b}")
    filtered, seen = [], set()
    for n in names:
        parts = n.split()
        if len(parts) == 2 and len(parts[0]) >= 3 and len(parts[1]) >= 3:
            k = n.lower()
            if k not in seen:
                seen.add(k)
                filtered.append(n)
    if not filtered:
        low = (text or "").lower().replace("<"," ").replace(">"," ")
        tokens = re.findall(r"[a-zA-Z][a-zA-Z'`-]*", low)
        for i in range(len(tokens) - 1):
            a, b = tokens[i], tokens[i + 1]
            if a in STOP_TOKENS or b in STOP_TOKENS:
                continue
            if (a, b) in FILLER_PAIRS:
                continue
            if len(a) < 3 or len(b) < 3:
                continue
            filtered.append(f"{a.title()} {b.title()}")
            break
    return filtered[:MAX_TARGETS]

# ========== PII REDACTION ==========
# Why: Optional last-mile redaction for printed output (defense-in-depth).
if PRESIDIO_AVAILABLE:
    analyzer = AnalyzerEngine()
    anonymizer = AnonymizerEngine()
    PRESIDIO_ENTITIES = ["EMAIL_ADDRESS","PHONE_NUMBER","LOCATION","DATE_TIME","NRP","IP_ADDRESS","CREDIT_CARD","US_SSN"]
    def redact_pii_text(text: str) -> str:
        try:
            results = analyzer.analyze(text=text, entities=PRESIDIO_ENTITIES, language="en")
            operators = {
                "DEFAULT": OperatorConfig("redact", {}),
                "EMAIL_ADDRESS": OperatorConfig("mask", {"masking_char": "*", "chars_to_mask": 100, "from_end": False})
            }
            return anonymizer.anonymize(text=text, analyzer_results=results, operators=operators).text
        except Exception:
            return text
else:
    EMAIL_HIDE_RE = re.compile(r'[\w\.-]+@[\w\.-]+', re.IGNORECASE)
    DOB_HIDE_RE   = re.compile(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b')
    PHONE_HIDE_RE = re.compile(r'\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b')
    SSN_HIDE_RE   = re.compile(r'\b\d{3}-\d{2}-\d{4}\b')
    IP_HIDE_RE    = re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')
    def redact_pii_text(text: str) -> str:
        text = EMAIL_HIDE_RE.sub('***@***', text)
        text = DOB_HIDE_RE.sub('REDACTED_DATE', text)
        text = PHONE_HIDE_RE.sub('REDACTED_PHONE', text)
        text = SSN_HIDE_RE.sub('REDACTED_SSN', text)
        text = IP_HIDE_RE.sub('REDACTED_IP', text)
        return text

# ========== Guard helpers ==========
def guard_input(user_text: str):
    """
    What: Stage 1A input sanitize via LLM Guard (+ heuristic fallback when unavailable).
    Why: Run early so we can sanitize prompt and log scanner hits in the Privacy Meter.
    Returns:
      - ok, sanitized_text, results (llm_guard objects or heuristics for logging)
    """
    if not LLM_GUARD_AVAILABLE:
        results = _heuristic_scan(user_text)
        return True, user_text, results
    try:
        sanitized, results = scan_prompt(user_text, INPUT_SCANNERS)
        ok = all(getattr(r, "valid", getattr(r, "is_valid", False)) for r in results)
        results.extend(_heuristic_scan(user_text))
        return ok, sanitized, results
    except Exception as e:
        print(f"[LLM Guard input error] {e}")
        results = _heuristic_scan(user_text)
        return True, user_text, results

def guard_output(text: str):
    """
    What: Optional output scan (PII/Toxicity) via LLM Guard when available.
    Why: Defense-in-depth and additional evidence for the Privacy Meter.
    """
    if not LLM_GUARD_AVAILABLE:
        return True, text, []
    try:
        sanitized, results = scan_output(text, OUTPUT_SCANNERS)
        ok = all(getattr(r, "valid", getattr(r, "is_valid", False)) for r in results)
        return ok, sanitized, results
    except Exception as e:
        print(f"[LLM Guard output error] {e}")
        return True, text, []

# ========== Formatting ==========
RE_SPACE = re.compile(r"\s+")
MAX_VALUE_COL_CHARS = int(os.getenv("TABLE_VALUE_MAX_CHARS", "80"))

def sanitize_cell_value(v) -> str:
    s = "" if v is None else str(v)
    s = s.replace("\r\n", " ").replace("\n", " ").replace("\r", " ").replace("\t", " ")
    s = s.replace("|", "/")
    s = RE_SPACE.sub(" ", s).strip()
    return s

def clamp_cell_value(s: str, limit: int = MAX_VALUE_COL_CHARS) -> str:
    s = "" if s is None else str(s)
    if limit and limit > 0 and len(s) > limit:
        return s[: limit - 1] + "…"
    return s

def tabulate_grid(df: pd.DataFrame) -> str:
    df = df.applymap(sanitize_cell_value)
    try:
        if "Value" in df.columns and MAX_VALUE_COL_CHARS > 0:
            maxwidths = [None] * len(df.columns)
            maxwidths[list(df.columns).index("Value")] = MAX_VALUE_COL_CHARS
            return tabulate(df, headers="keys", tablefmt="grid", maxcolwidths=maxwidths)
        return tabulate(df, headers="keys", tablefmt="grid")
    except TypeError:
        if "Value" in df.columns and MAX_VALUE_COL_CHARS > 0:
            df["Value"] = df["Value"].apply(lambda x: clamp_cell_value(x, MAX_VALUE_COL_CHARS))
        return tabulate(df, headers="keys", tablefmt="grid")

# ========== Column parsing (no synonyms) ==========
def parse_columns(question: str) -> list[str]:
    """
    What: Extract requested columns by exact/compact match against inferred schema.
    Why: Avoid magic synonyms; keep mapping deterministic and auditable.
    """
    q = (question or "").lower()
    found = []
    for col in ALL_COLUMNS:
        col_norm = re.sub(r"\s+", " ", (col or "").strip().lower())
        if not col_norm:
            continue
        if re.search(rf"\b{re.escape(col_norm)}\b", q):
            found.append(col); continue
        col_nospace = col_norm.replace(" ", "")
        q_nospace = re.sub(r"\s+", "", q)
        if col_nospace and col_nospace in q_nospace:
            found.append(col)
    seen, out = set(), []
    for c in found:
        if c not in seen:
            seen.add(c); out.append(c)
    return out

def parse_question(question: str):
    """
    What: Parse EmpIDs/Names/Columns/AllDetails from user question.
    Why: Downstream RBAC, targeting, and column filtering depend on these signals.
    """
    qlow = question.lower()
    all_details = any(kw in qlow for kw in ["all details", "everything", "complete info", "full details"])
    columns = parse_columns(question)
    empids = re.findall(r"\b\d{3,6}\b", question)
    names = extract_names_from_text(question)
    return {"EmpIDs": empids, "Names": names, "Columns": columns, "AllDetails": all_details}

def extract_values_from_context(context, columns):
    """
    What: Pull values for a given set of columns from a KV text context (one doc).
    Why: Keep the LLM prompt tight with only the requested/allowed columns.
    """
    mapping = {}
    for line in (context or "").splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        nk = re.sub(r"[^a-z0-9]", "", (k or "").lower())
        if nk and nk not in mapping:
            mapping[nk] = (v or "").strip()
    result = {}
    for col in (columns or []):
        candidates = set()
        canonical = re.sub(r"[^a-z0-9]", "", (col or "").lower())
        candidates.add(canonical)
        spaced = re.sub(r"(?<!^)(?=[A-Z])", " ", col or "").strip()
        candidates.add(re.sub(r"[^a-z0-9]", "", spaced.lower()))
        val = "Not available"
        for nk in candidates:
            if nk in mapping and mapping[nk]:
                val = mapping[nk]; break
        result[col] = val
    return result

def build_table_from_dict(extracted: dict):
    """
    What: Build a simple 2-column markdown table from a {column: value} map.
    Why: Consistent rendering path for targeted/generic answers.
    """
    lines = ["| Column | Value |", "|---|---|"]
    for col in ALL_COLUMNS:
        if col in extracted:
            lines.append(f"| {col} | {sanitize_cell_value(extracted[col])} |")
    return "\n".join(lines)

# ========== RBAC ==========
def allowed_columns_for_request(user, target_empid: str, target_name: str, requested_cols: list, all_details: bool):
    """
    What: Compute which columns this user may see for the given target.
    Why: Enforce column-level privacy:
         - Admin: can view everything
         - Self: can view all columns for their own record
         - Others: only NON_SENSITIVE_COLS
    """
    cols_all = ALL_COLUMNS
    cols_safe = NON_SENSITIVE_COLS
    if user["role"] == "admin":
        return requested_cols if (requested_cols and not all_details) else cols_all
    self_req = is_self_request(user, target_empid, target_name)
    base = set(cols_all) if self_req else set(cols_safe)
    if all_details or not requested_cols:
        return [c for c in cols_all if c in base]
    return [c for c in (requested_cols or []) if c in base]

def mask_value(col: str, val, for_other_employee: bool) -> str:
    """
    What: Mask value for other-employee views if the column is sensitive.
    Why: Defense-in-depth on top of column filtering.
    """
    s = "" if val is None else str(val)
    if not for_other_employee:
        return s
    if col in SENSITIVE_COLS:
        return "REDACTED"
    return s

# ========== RAG helpers ==========
def build_allowed_context(context_text: str, allowed_cols: list, for_other: bool) -> str:
    """
    What: Construct a minimal context with only allowed columns and masked values.
    Why: Principle of least privilege — never send disallowed columns to the LLM.
    """
    extracted = extract_values_from_context(context_text, allowed_cols)
    masked = {col: mask_value(col, val, for_other) for col, val in extracted.items()}
    return "\n".join(f"{col}: {masked[col]}" for col in allowed_cols)

def compose_prompt(user_question: str, allowed_context: str) -> str:
    """
    What: Constrain the LLM to context-only answers.
    Why: Reduce hallucination and prevent leakage — LLM must answer using only provided context.
    """
    return f"""
You are a data assistant. Use ONLY the information in the Context to answer the Question.
If information is missing, say "Not available in context".
Keep answers concise.
Question:
{user_question}
Context:
{allowed_context}
"""

def generate_answer(user_question: str, allowed_context: str, is_admin: bool, for_other: bool, is_self: bool = False) -> str:
    """
    What: Generate an answer strictly from allowed context, with output gating.
    Why:
      - Keep generations context-bound to reduce leakage/hallucinations.
      - Apply output ANY-TRUE gate:
        * self: allow PII (secrets still blocked)
        * others: block PII + secrets
      - Do not overexpose content to the LLM (allowed_context is already filtered/masked).
    """
    # Prefer a guardrails-validated path if available for extra safety.
    use_guard = None
    if GUARDRAILS_AVAILABLE:
        if is_self and guard_self is not None:
            use_guard = guard_self
        elif guard is not None:
            use_guard = guard

    if use_guard is not None and not (is_admin and ADMIN_BYPASS_GUARDRAILS):
        try:
            raw, validated = use_guard(llm_call, prompt_params={"question": user_question, "context": allowed_context})
            if validated:
                ans = str(validated).strip()
                # For self, skip LLM Guard output scan unless admin requires; for others keep it.
                if (not is_self) or (is_admin and not ADMIN_BYPASS_LLM_GUARD):
                    if not (is_admin and ADMIN_BYPASS_LLM_GUARD):
                        ok_out, safe_ans, _ = guard_output(ans)
                        if not ok_out and not is_admin:
                            return "Sorry, I can’t share that content. Please ask a different question."
                        final_text = safe_ans if ok_out else ans
                    else:
                        final_text = ans
                else:
                    final_text = ans

                # Output ANY-TRUE — secrets hard-block for all; PII blocked for others.
                if ANY_TRUE_BLOCK and not (is_admin and ADMIN_BYPASS_ANYTRUE):
                    ok_any, msg_any, _, _ = guard_output_anytrue(
                        final_text, question=user_question, contexts=[allowed_context], self_ok_pii=bool(is_self)
                    )
                    if not ok_any:
                        return FRIENDLY_MSG

                # Redaction layer for others (defense-in-depth; may be redundant with gates).
                if not is_admin and for_other:
                    final_text = redact_pii_text(final_text)
                return final_text
            else:
                return "Not available in context."
        except Exception:
            pass

    # Fallback: plain LLM call (still gated by output ANY-TRUE)
    answer = llm.invoke(compose_prompt(user_question, allowed_context))
    if is_admin and ADMIN_BYPASS_LLM_GUARD:
        final_text = str(answer)
    else:
        if not is_self:
            ok_out, safe_answer, _ = guard_output(str(answer))
            if not ok_out and not is_admin:
                return "Sorry, I can’t share that content. Please ask a different question."
            final_text = safe_answer if ok_out else str(answer)
        else:
            final_text = str(answer)

    if ANY_TRUE_BLOCK and not (is_admin and ADMIN_BYPASS_ANYTRUE):
        ok_any, msg_any, _, _ = guard_output_anytrue(
            final_text, question=user_question, contexts=[allowed_context], self_ok_pii=bool(is_self)
        )
        if not ok_any:
            return FRIENDLY_MSG

    if not is_admin and for_other:
        final_text = redact_pii_text(final_text)
    return final_text

def make_polite_refusal(target_label: str, requested_sensitive_cols: list, is_all_details: bool) -> str:
    """
    What: Explain clearly why a request is refused and suggest safer alternatives.
    Why: Good UX — users understand policy and can reformulate safely.
    """
    reason = "Your request asked for complete details, which include restricted personal/sensitive information." if is_all_details else ("Restricted fields requested: " + ", ".join(requested_sensitive_cols))
    msg = "We’re sorry, but we can’t share personal or sensitive information. We take privacy seriously."
    lines = [
        "| Column | Value |","|---|---|",
        f"| Target | {target_label} |",
        f"| Message | {msg} |",
        f"| Reason | {reason} |"
    ]
    return "\n".join(lines)

def build_targeted_answer_table(question: str, context: str, allowed_cols: list, for_other: bool, is_admin: bool, is_self: bool = False) -> str:
    """
    What: Build the final table for targeted answers using only allowed columns (masked when needed).
    Why: Control exposure strictly; never pass disallowed data to the LLM or the user.
    """
    allowed_cols = [c for c in allowed_cols if isinstance(c, str) and c.strip()]
    lines = ["| Column | Value |", "|---|---|"]
    if USE_LLM_FOR_TARGETED:
        allowed_context = build_allowed_context(context, allowed_cols, for_other)
        summary = generate_answer(question, allowed_context, is_admin=is_admin, for_other=for_other, is_self=is_self)
        if summary:
            s = sanitize_cell_value(str(summary).splitlines()[0])
            if s:
                lines.append(f"| Answer | {s} |")
    extracted = extract_values_from_context(context, allowed_cols)
    masked = {col: mask_value(col, val, for_other) for col, val in extracted.items()}
    for col in allowed_cols:
        val = masked.get(col, "Not available")
        val = "Not available" if val is None or str(val).strip() == "" else str(val)
        lines.append(f"| {sanitize_cell_value(col)} | {sanitize_cell_value(val)} |")
    answer_table = "\n".join(lines)
    if not is_admin and for_other:
        answer_table = redact_pii_text(answer_table)
    return answer_table

# ========== RAG query ==========
def rag_query(question: str, top_k=1):
    """
    What: Full RAG orchestration — input guards -> RBAC -> retrieval/output gates -> answer + Privacy Meter.
    Why:
      - 1A LLM Guard first: sanitize input + log scanners for Privacy.
      - 1B RBAC early: self-only policy blocks lateral access before retrieval.
      - 1C/2 input gates: block injection/jailbreak/secrets; regex/denylist fallback after RBAC (skip for self).
      - Retrieval/Output: self_ok_pii controls PII policy; secrets always blocked.
    """
    is_admin = (CURRENT_USER or {}).get("role") == "admin"

    # Stage 1A) LLM Guard sanitize (no denylist here)
    # Why: Run llm_guard early to sanitize + capture scanner hits for the Privacy Meter.
    if LLM_GUARD_AVAILABLE:
        try:
            ok_in, safe_question, input_results = guard_input(question)
        except Exception:
            ok_in, safe_question, input_results = True, question, []
    else:
        ok_in, safe_question, input_results = True, question, []

    # Why: If llm_guard flags and admin is not bypassing, block immediately (prevents prompt-hijack).
    if not ok_in and not (is_admin and ADMIN_BYPASS_LLM_GUARD):
        polite = "\n".join([
            "| Column | Value |","|---|---|",
            f"| Notice | {FRIENDLY_MSG} |"
        ])
        out = f"## Results\n{polite}"
        if PRIVACY_METER:
            meter = privacy_meter_report(
                section_name="Results",
                question=question,
                answer_text=polite,
                user_role=(CURRENT_USER or {}).get("role"),
                requested_sensitive_cols=[],
                input_scan_results=input_results,
                denylist_hit=False,
                guardrails_blocked=True,
                trigger_reason="ANY-TRUE input gate (llm_guard_input)",
                role_verified=is_admin,
                is_self=False,
            )
            out += f"\n\n## Privacy Meter\n{meter}"
        return out

    # Use sanitized input
    question = safe_question

    # Stage 1B) RBAC early (same level as LLM Guard)
    # Why: Enforce self-only before retrieval/output to minimize exposure.
    parsed = parse_question(question)
    empids, names, requested_columns, all_details = parsed["EmpIDs"], parsed["Names"], parsed["Columns"], parsed["AllDetails"]
    names = maybe_add_self_target(question, names)
    empids, names, self_blocked = enforce_self_scope_for_user(CURRENT_USER, empids, names)
    if self_blocked:
        # Why: RBAC refuses lateral access; we still audit-run input guards so Privacy shows all hits.
        audit_who, audit_kinds = [], set()
        if ANY_TRUE_BLOCK and not (is_admin and ADMIN_BYPASS_ANYTRUE):
            try:
                ok_guard, _, who_guard, kinds_guard = guard_input_anytrue_guards_only(question)
                if who_guard: audit_who.extend(who_guard)
                if kinds_guard: audit_kinds |= set(kinds_guard)
            except Exception:
                pass
            try:
                ok_rx, _, who_rx, kinds_rx = guard_input_regex_only(question)
                if who_rx: audit_who.extend(who_rx)
                if kinds_rx: audit_kinds |= set(kinds_rx)
            except Exception:
                pass

        gates_list = ["rbac"] + sorted(list(set(audit_who)))
        gates_str = " | ".join(gates_list)
        requested_sensitive = sorted(list(set(detect_sensitive_intent(question)) | audit_kinds))
        deny_hit = ("denylist" in audit_who) or ("denylist" in audit_kinds)

        polite = "\n".join([
            "| Column | Value |", "|---|---|",
            "| Notice | For privacy reasons, you can only view your own record. |"
        ])
        out = f"## Results\n{polite}"
        if PRIVACY_METER:
            meter = privacy_meter_report(
                section_name="Results",
                question=question,
                answer_text=polite,
                user_role=(CURRENT_USER or {}).get("role"),
                requested_sensitive_cols=requested_sensitive,
                input_scan_results=input_results,
                denylist_hit=deny_hit,
                guardrails_blocked=True,
                trigger_reason=f"Input gate ({gates_str})",
                role_verified=is_admin,
                is_self=True,
            )
            out += f"\n\n## Privacy Meter\n{meter}"
        return out

    # Determine if this is overall a self request (skip Stage 2 regex for self)
    self_overall = is_self_overall(CURRENT_USER, empids, names)

    # Stage 1C) Input third-party guards-only (block injection/jailbreak/secrets/toxicity)
    # Why: Block on llm_guard; still record who/kinds for Privacy Meter.
    if ANY_TRUE_BLOCK and not (is_admin and ADMIN_BYPASS_ANYTRUE):
        ok_guard, msg_guard, who_guard, kinds_guard = guard_input_anytrue_guards_only(question)
        if not ok_guard:
            polite = "\n".join([
                "| Column | Value |","|---|---|",
                f"| Notice | {FRIENDLY_MSG} |"
            ])
            out = f"## Results\n{polite}"
            if PRIVACY_METER:
                meter = privacy_meter_report(
                    section_name="Results",
                    question=question,
                    answer_text=polite,
                    user_role=(CURRENT_USER or {}).get("role"),
                    requested_sensitive_cols=sorted(list(kinds_guard)),
                    input_scan_results=input_results,
                    denylist_hit=False,
                    guardrails_blocked=True,
                    trigger_reason=f"ANY-TRUE input gate ({', '.join(who_guard) or 'llm_guard_input'})",
                    role_verified=is_admin,
                    is_self=False,
                )
                out += f"\n\n## Privacy Meter\n{meter}"
            return out

    # Stage 2) Input fallback: value-only regex + denylist (after RBAC) — skip for self
    # Why: Catch literal PII/secrets or common injection phrases (conservative fallback).
    if ANY_TRUE_BLOCK and not (is_admin and ADMIN_BYPASS_ANYTRUE) and not self_overall:
        ok_rx, msg_rx, who_rx, kinds_rx = guard_input_regex_only(question)
        if not ok_rx:
            polite = "\n".join([
                "| Column | Value |","|---|---|",
                f"| Notice | {FRIENDLY_MSG} |"
            ])
            out = f"## Results\n{polite}"
            if PRIVACY_METER:
                meter = privacy_meter_report(
                    section_name="Results",
                    question=question,
                    answer_text=polite,
                    user_role=(CURRENT_USER or {}).get("role"),
                    requested_sensitive_cols=sorted(list(kinds_rx)),
                    input_scan_results=input_results,
                    denylist_hit=("denylist" in (kinds_rx or set())),
                    guardrails_blocked=True,
                    trigger_reason=f"ANY-TRUE input gate ({', '.join(who_rx) or 'regex'})",
                    role_verified=is_admin,
                    is_self=False,
                )
                out += f"\n\n## Privacy Meter\n{meter}"
            return out

    # Optional denylist (legacy; consider centralizing in orchestrator)
    denied = is_policy_denied(question)
    if denied and not (is_admin and ADMIN_BYPASS_DENYLIST):
        polite = "\n".join([
            "| Column | Value |","|---|---|",
            "| Notice | Sorry, I can’t process that request. Please rephrase your question. |"
        ])
        out = f"## Results\n{polite}"
        if PRIVACY_METER:
            meter = privacy_meter_report(
                section_name="Results",
                question=question,
                answer_text=polite,
                user_role=(CURRENT_USER or {}).get("role"),
                requested_sensitive_cols=[],
                input_scan_results=input_results,
                denylist_hit=True,
                guardrails_blocked=True,
                trigger_reason="Input denylist hit",
                role_verified=is_admin,
                is_self=False,
            )
            out += f"\n\n## Privacy Meter\n{meter}"
        return out

    # ===== Targeted lookups by EmpID/Name =====
    all_answers = []

    def _apply_distance_threshold(results):
        # Why: Filter out weak matches for better precision if a distance threshold is set.
        docs = results.get('documents', [[]])[0]
        if RAG_DISTANCE_THRESHOLD > 0:
            dists = results.get('distances', [[]])[0] or [None]*len(docs)
            docs = [d for d, dist in zip(docs, dists) if (dist is None or dist <= RAG_DISTANCE_THRESHOLD)]
        return docs

    def handle_target(target_label: str, target_empid: str = None, target_name: str = None) -> bool:
        # Why: Embed the question with target bias, then exact-match target in retrieved docs.
        q_emb = embedding_model.encode([f"{question} [target: {target_label}]"])[0]
        results = collection.query(
            query_embeddings=[q_emb.tolist()],
            n_results=max(5, top_k),
            include=["documents", "distances"]
        )
        docs = _apply_distance_threshold(results)
        if not docs:
            return False

        # Retrieval gate (self: secrets-only; others: full)
        # Why: self_ok_pii=True for self (PII allowed; secrets still blocked).
        is_self = is_self_request(CURRENT_USER, target_empid, target_name)
        if ANY_TRUE_BLOCK and not (is_admin and ADMIN_BYPASS_ANYTRUE):
            ok_ret, safe_docs, who_ret, kinds_ret = guard_retrieval_anytrue(docs, self_ok_pii=is_self)
            if not ok_ret:
                polite_table = "\n".join([
                    "| Column | Value |","|---|---|",
                    f"| Notice | {FRIENDLY_MSG} |"
                ])
                section_md = f"## Employee {target_label}\n{polite_table}"
                if PRIVACY_METER:
                    meter = privacy_meter_report(
                        section_name=f"Employee {target_label}",
                        question=question,
                        answer_text=polite_table,
                        user_role=(CURRENT_USER or {}).get("role"),
                        requested_sensitive_cols=sorted(list(kinds_ret)),
                        input_scan_results=[],
                        denylist_hit=False,
                        guardrails_blocked=True,
                        trigger_reason=f"ANY-TRUE retrieval gate ({', '.join(who_ret) or 'detector'})",
                        role_verified=is_admin,
                        is_self=False,
                    )
                    section_md += f"\n\n## Privacy Meter\n{meter}"
                all_answers.append(section_md)
                return True
            else:
                docs = safe_docs

        chosen = None
        for doc_text in docs:
            if _match_target_in_doc(doc_text, target_label):
                chosen = doc_text
                break
        if not chosen:
            return False

        context = chosen
        is_self = is_self_request(CURRENT_USER, target_empid, target_name)
        for_other = not (is_admin or is_self)

        restricted_set = set(SENSITIVE_COLS)
        requested_sensitive = []
        if parsed["Columns"]:
            requested_sensitive = [c for c in parsed["Columns"] if c in restricted_set]
        elif parsed["AllDetails"]:
            requested_sensitive = sorted(list(restricted_set))

        # Why: Compute categories for a clear refusal reason if needed.
        q_sensitive_cats = detect_sensitive_intent(question)
        doc_cats = detect_sensitive_categories_in_text(context)
        cats_hit = (q_sensitive_cats & doc_cats) if q_sensitive_cats else set()
        blocked_reasons = list(requested_sensitive) + sorted(list(cats_hit))

        if (SENSITIVE_BLOCK_ALL or for_other) and (parsed["AllDetails"] or blocked_reasons):
            polite_table = make_polite_refusal(
                target_label=target_label,
                requested_sensitive_cols=blocked_reasons or ["(restricted fields)"],
                is_all_details=parsed["AllDetails"]
            )
            polite_table = redact_pii_text(polite_table)
            section_md = f"## Employee {target_label}\n{polite_table}"
            if PRIVACY_METER:
                meter = privacy_meter_report(
                    section_name=f"Employee {target_label}",
                    question=question,
                    answer_text=polite_table,
                    user_role=(CURRENT_USER or {}).get("role"),
                    requested_sensitive_cols=blocked_reasons,
                    input_scan_results=[],
                    denylist_hit=False,
                    guardrails_blocked=True,
                    trigger_reason=("Requested all details" if parsed["AllDetails"] else f"Sensitive: {', '.join(blocked_reasons)}"),
                    role_verified=is_admin,
                    is_self=is_self,
                )
                section_md += f"\n\n## Privacy Meter\n{meter}"
            all_answers.append(section_md)
            return True

        allowed_cols = allowed_columns_for_request(
            CURRENT_USER,
            target_empid=target_empid,
            target_name=target_name,
            requested_cols=parsed["Columns"],
            all_details=parsed["AllDetails"],
        )
        if not allowed_cols:
            polite_table = make_polite_refusal(
                target_label=target_label,
                requested_sensitive_cols=blocked_reasons or ["(restricted fields)"],
                is_all_details=parsed["AllDetails"]
            )
            polite_table = redact_pii_text(polite_table)
            section_md = f"## Employee {target_label}\n{polite_table}"
            if PRIVACY_METER:
                meter = privacy_meter_report(
                    section_name=f"Employee {target_label}",
                    question=question,
                    answer_text=polite_table,
                    user_role=(CURRENT_USER or {}).get("role"),
                    requested_sensitive_cols=blocked_reasons or ["(restricted fields)"],
                    input_scan_results=[],
                    denylist_hit=False,
                    guardrails_blocked=True,
                    trigger_reason="No permitted columns for this role/target",
                    role_verified=is_admin,
                    is_self=is_self,
                )
                section_md += f"\n\n## Privacy Meter\n{meter}"
            all_answers.append(section_md)
            return True

        answer_table = build_targeted_answer_table(
            question=question,
            context=context,
            allowed_cols=allowed_cols,
            for_other=for_other,
            is_admin=is_admin,
            is_self=is_self
        )
        section_md = f"## Employee {target_label}\n{answer_table}"
        if PRIVACY_METER:
            meter = privacy_meter_report(
                section_name=f"Employee {target_label}",
                question=question,
                answer_text=answer_table,
                user_role=(CURRENT_USER or {}).get("role"),
                requested_sensitive_cols=requested_sensitive,
                input_scan_results=[],
                denylist_hit=False,
                guardrails_blocked=False,
                role_verified=is_admin,
                is_self=is_self,
            )
            section_md += f"\n\n## Privacy Meter\n{meter}"
        all_answers.append(section_md)
        return True

    # Targets by EmpID
    successes = 0
    for empid in parsed["EmpIDs"]:
        if handle_target(target_label=str(empid), target_empid=empid, target_name=None):
            successes += 1
            if successes >= MAX_TARGETS:
                break

    # Targets by name
    if successes < MAX_TARGETS:
        for name in parsed["Names"]:
            if handle_target(target_label=name, target_empid=None, target_name=name):
                successes += 1
                if successes >= MAX_TARGETS:
                    break

    if (parsed["EmpIDs"] or parsed["Names"]) and not all_answers:
        polite = "\n".join([
            "| Column | Value |",
            "|---|---|",
            "| Notice | No matching records were found for the target(s). Try specifying First Last or an ID, or ask a general question. |"
        ])
        out = f"## Results\n{polite}"
        if PRIVACY_METER:
            meter = privacy_meter_report(
                section_name="Results",
                question=question,
                answer_text=polite,
                user_role=(CURRENT_USER or {}).get("role"),
                requested_sensitive_cols=[],
                input_scan_results=[],
                denylist_hit=False,
                guardrails_blocked=False,
                trigger_reason="No matching records",
                role_verified=is_admin,
                is_self=False,
            )
            out += f"\n\n## Privacy Meter\n{meter}"
        return out

    # ===== Generic RAG =====
    if not parsed["EmpIDs"] and not parsed["Names"]:

        # Why: For non-admins, generic PII requests are refused — ask for non-sensitive fields.
        if (CURRENT_USER or {}).get("role") != "admin":
            restricted_set = set(SENSITIVE_COLS)
            if parsed["Columns"]:
                requested_sensitive = [c for c in parsed["Columns"] if c in restricted_set]
            elif parsed["AllDetails"]:
                requested_sensitive = sorted(list(restricted_set))
            else:
                requested_sensitive = []
            if parsed["AllDetails"] or (parsed["Columns"] and requested_sensitive):
                polite_table = make_polite_refusal(
                    target_label="(multiple rows)",
                    requested_sensitive_cols=requested_sensitive or ["(restricted fields)"],
                    is_all_details=parsed["AllDetails"]
                )
                polite_table = redact_pii_text(polite_table)
                section_md = f"## Results\n{polite_table}"
                if PRIVACY_METER:
                    meter = privacy_meter_report(
                        section_name="Results",
                        question=question,
                        answer_text=polite_table,
                        user_role=(CURRENT_USER or {}).get("role"),
                        requested_sensitive_cols=requested_sensitive or ["(restricted fields)"],
                        input_scan_results=[],
                        denylist_hit=False,
                        guardrails_blocked=True,
                        trigger_reason=("Requested all details" if parsed["AllDetails"] else f"Sensitive field(s): {', '.join(requested_sensitive or [])}"),
                        role_verified=is_admin,
                        is_self=False,
                    )
                    section_md += f"\n\n## Privacy Meter\n{meter}"
                return section_md

        q_emb = embedding_model.encode([question])[0]
        results = collection.query(
            query_embeddings=[q_emb.tolist()],
            n_results=max(3, top_k),
            include=["documents", "distances"]
        )
        docs = _apply_distance_threshold(results)
        if not docs:
            return "## Results\n(No data found)"

        # Retrieval gate for generic queries (not self)
        if ANY_TRUE_BLOCK and not (is_admin and ADMIN_BYPASS_ANYTRUE):
            ok_ret, safe_docs, who_ret, kinds_ret = guard_retrieval_anytrue(docs, self_ok_pii=False)
            if not ok_ret:
                polite_table = "\n".join([
                    "| Column | Value |","|---|---|",
                    f"| Notice | {FRIENDLY_MSG} |"
                ])
                section_md = f"## Results\n{polite_table}"
                if PRIVACY_METER:
                    meter = privacy_meter_report(
                        section_name="Results",
                        question=question,
                        answer_text=polite_table,
                        user_role=(CURRENT_USER or {}).get("role"),
                        requested_sensitive_cols=sorted(list(kinds_ret)),
                        input_scan_results=[],
                        denylist_hit=False,
                        guardrails_blocked=True,
                        trigger_reason=f"ANY-TRUE retrieval gate ({', '.join(who_ret) or 'detector'})",
                        role_verified=is_admin,
                        is_self=False,
                    )
                    section_md += f"\n\n## Privacy Meter\n{meter}"
                return section_md
            else:
                docs = safe_docs

        allowed_cols_generic = (ALL_COLUMNS if is_admin else NON_SENSITIVE_COLS)
        if parsed["Columns"]:
            allowed_cols_generic = [c for c in parsed["Columns"] if c in allowed_cols_generic]
        if not allowed_cols_generic:
            polite_table = make_polite_refusal(
                target_label="(multiple rows)",
                requested_sensitive_cols=["(restricted fields)"],
                is_all_details=parsed["AllDetails"]
            )
            polite_table = redact_pii_text(polite_table)
            section_md = f"## Results\n{polite_table}"
            if PRIVACY_METER:
                meter = privacy_meter_report(
                    section_name="Results",
                    question=question,
                    answer_text=polite_table,
                    user_role=(CURRENT_USER or {}).get("role"),
                    requested_sensitive_cols=["(restricted fields)"],
                    input_scan_results=[],
                    denylist_hit=False,
                    guardrails_blocked=True,
                    trigger_reason="No permitted columns for this role",
                    role_verified=is_admin,
                    is_self=False,
                )
                section_md += f"\n\n## Privacy Meter\n{meter}"
            return section_md

        if not USE_LLM_FOR_GENERIC:
            best_doc = docs[0]
            table = build_targeted_answer_table(
                question=question,
                context=best_doc,
                allowed_cols=allowed_cols_generic,
                for_other=(not is_admin),
                is_admin=is_admin,
                is_self=False
            )
            section_md = f"## Results\n{table}"
            if PRIVACY_METER:
                restricted_set = set(SENSITIVE_COLS)
                if parsed["Columns"]:
                    requested_sensitive = [c for c in parsed["Columns"] if c in restricted_set]
                elif parsed["AllDetails"]:
                    requested_sensitive = sorted(list(restricted_set))
                else:
                    requested_sensitive = []
                meter = privacy_meter_report(
                    section_name="Results",
                    question=question,
                    answer_text=section_md,
                    user_role=(CURRENT_USER or {}).get("role"),
                    requested_sensitive_cols=requested_sensitive,
                    input_scan_results=[],
                    denylist_hit=False,
                    guardrails_blocked=False,
                    role_verified=is_admin,
                    is_self=False,
                )
                section_md += f"\n\n## Privacy Meter\n{meter}"
            return section_md

        allowed_snippets = []
        for i, doc in enumerate(docs, start=1):
            snippet = build_allowed_context(doc, allowed_cols_generic, for_other=(not is_admin))
            allowed_snippets.append(f"Record {i}:\n{snippet}")
        allowed_context = "\n\n---\n\n".join(allowed_snippets)

        summary = generate_answer(question, allowed_context, is_admin=is_admin, for_other=(not is_admin), is_self=False)
        summary_text = sanitize_cell_value(str(summary).splitlines()[0]) if summary and str(summary).strip() else "Not available in context."
        answer_table = "\n".join(["| Column | Value |", "|---|---|", f"| Answer | {summary_text} |"])
        if not is_admin:
            answer_table = redact_pii_text(answer_table)
        section_md = f"## Results\n{answer_table}"
        if PRIVACY_METER:
            restricted_set = set(SENSITIVE_COLS)
            if parsed["Columns"]:
                requested_sensitive = [c for c in parsed["Columns"] if c in restricted_set]
            elif parsed["AllDetails"]:
                requested_sensitive = sorted(list(restricted_set))
            else:
                requested_sensitive = []
            meter = privacy_meter_report(
                section_name="Results",
                question=question,
                answer_text=section_md,
                user_role=(CURRENT_USER or {}).get("role"),
                requested_sensitive_cols=requested_sensitive,
                input_scan_results=[],
                denylist_hit=False,
                guardrails_blocked=False,
                role_verified=is_admin,
                is_self=False,
            )
            section_md += f"\n\n## Privacy Meter\n{meter}"
        return section_md

    return "\n\n".join(all_answers) if all_answers else "## Results\n(No data found)"

def markdown_to_table(md_text):
    """
    What: Render markdown sections with tables as pretty grid text for CLI.
    Why: Improves readability without changing the markdown structure.
    """
    def _split_md_row(l: str):
        s = (l or "").strip()
        if not s:
            return []
        if not s.startswith("|"):
            return []
        if s.startswith("|"):
            s = s[1:]
        if s.endswith("|"):
            s = s[:-1]
        parts = s.split("|")
        if len(parts) >= 2:
            left = sanitize_cell_value(parts[0])
            right = sanitize_cell_value("|".join(parts[1:]))
            return [left, right]
        return [sanitize_cell_value(p) for p in parts]

    sections = re.split(r"^##\s+", md_text or "", flags=re.MULTILINE)
    output = []
    for sec in sections:
        if not sec.strip():
            continue
        lines = sec.splitlines()
        header = lines[0].strip()
        table_lines = [l for l in lines[1:] if l.strip().startswith("|")]
        if not table_lines:
            output.append(f"## {header}\n(No data found)")
            continue
        rows = []
        for l in table_lines:
            if re.match(r"^\s*\|\s*-{3,}", l):
                continue
            parts = _split_md_row(l)
            if parts:
                rows.append(parts)
        if not rows:
            output.append(f"## {header}\n(No data found)")
            continue

        header_cells = [sanitize_cell_value(c) for c in rows[0]]
        data_rows = rows[1:]
        width = len(header_cells)
        fixed = []
        for r in data_rows:
            if len(r) < width:
                r = r + [""] * (width - len(r))
            elif len(r) > width:
                r = r[:width]
            fixed.append([sanitize_cell_value(x) for x in r])

        df = pd.DataFrame(fixed, columns=header_cells)
        if not df.empty:
            df = df.applymap(lambda x: str(x).strip())
            df = df[~(df == "").all(axis=1)]
        if df.empty:
            output.append(f"## {header}\n(No data found)")
            continue

        output.append(f"## {header}\n" + tabulate_grid(df))
    return "\n\n".join(output)

if __name__ == "__main__":
    CURRENT_USER = authenticate()
    print("👋 Welcome to the CSV RAG Chatbot! 🗂️💬")
    print("Commands: login | switch | whoami | logout | exit | quit")

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
                CURRENT_USER = authenticate()
                print(f"✅ Switched to {CURRENT_USER.get('FullName','-')} ({CURRENT_USER.get('role','-')})")
            except SystemExit:
                print("Switch cancelled.")
            continue

        # Show current user
        if ql in ("whoami",):
            u = CURRENT_USER or {}
            print(f"👤 Current user: {u.get('FullName','-')} (role: {u.get('role','-')}, EmpID: {u.get('EmpID','-')})")
            continue

        print("🤔 Thinking... finding the best answer, please wait...")
        response = rag_query(question)
        try:
            print("\nAnswer:\n", markdown_to_table(response))
        except Exception:
            print("\nAnswer:\n", response)