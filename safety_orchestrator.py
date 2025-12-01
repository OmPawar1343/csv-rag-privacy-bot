# safety_orchestrator.py


"""
Short reason (Why this file exists):
- Centralizes all safety checks (third-party + regex/denylist) in one place.
- Implements "ANY-TRUE" policy: if any detector flags unsafe content, we can block.
- Standardizes outputs (who matched, which kinds) for the Privacy Meter.
- Keeps policy logic consistent across input, retrieval, and output stages.

What:
- Provides small, focused guard functions (input, retrieval, output) used by rag.py.
- Reports detector hits (who/kinds) even when not blocking, so audits are complete.
"""

import os
import re
import typing as T

FRIENDLY_MSG = "Sorry, I can’t share that information. It may contain sensitive data. Please verify access or provide an approved ticket."

# ---------------- Toggles ----------------
# Why: Feature flags to enable/disable detectors without code changes.
USE_PRESIDIO    = os.getenv("PRIV_USE_PRESIDIO", "1").lower() in ("1","true","yes","on")
USE_REGEX       = os.getenv("PRIV_USE_REGEX", "1").lower() in ("1","true","yes","on")
USE_LLM_GUARD   = os.getenv("PRIV_USE_LLM_GUARD", "1").lower() in ("1","true","yes","on")
USE_GUARDRAILS  = os.getenv("PRIV_USE_GUARDRAILS", "1").lower() in ("1","true","yes","on")

# Regex behavior controls (input only)
# Why: Value-only input regex detects literal PII values; skip if off to reduce false positives.
REGEX_VALUE_ONLY_INPUT = os.getenv("PRIV_REGEX_VALUE_ONLY_INPUT", "1").lower() in ("1","true","yes","on")
USE_REGEX_INPUT        = os.getenv("PRIV_USE_REGEX_INPUT", "1").lower() in ("1","true","yes","on")

# Presidio add-on: optional "intent" recognizers (e.g., "email", "phone" intent)
USE_PRESIDIO_INTENT    = os.getenv("PRIV_PRESIDIO_INTENT", "1").lower() in ("1","true","yes","on")

# Fail-safe: block when no detectors are available (hardening against misconfig)
FAIL_SAFE_BLOCK_IF_NO_DETECTOR = os.getenv("PRIV_FAIL_SAFE", "1").lower() in ("1","true","yes","on")

# ---------------- Optional Presidio ----------------
# Why: ML/regex-based entity detection; used for visibility (kinds), not always block.
PRESIDIO = None
if USE_PRESIDIO:
    try:
        from presidio_analyzer import AnalyzerEngine
        PRESIDIO = AnalyzerEngine()
        if USE_PRESIDIO_INTENT:
            try:
                from presidio_analyzer import Pattern, PatternRecognizer
                # Why: Add light “intent” recognizers so Privacy Meter can reflect user intent (email/phone/etc.).
                intent_defs = [
                    ("EMAIL_INTENT",   r"\b(e[-\s]?mail|emails)\b", 0.6),
                    ("PHONE_INTENT",   r"\b(phone|mobile|contact(?:\s*number)?)\b", 0.6),
                    ("PAN_INTENT",     r"\bpan(\s*no\.?)?\b", 0.6),
                    ("AADHAAR_INTENT", r"\b(aadhaar|aadhar)\b", 0.6),
                    ("SALARY_INTENT",  r"\b(salary|ctc|wage|compensation)\b", 0.6),
                    ("ADDRESS_INTENT", r"\b(address|street|road|avenue|city|zip|postal|pin)\b", 0.6),
                ]
                for ent, pat, score in intent_defs:
                    pr = PatternRecognizer(
                        supported_entity=ent,
                        patterns=[Pattern(name=f"{ent}_pat", regex=pat, score=score)],
                    )
                    PRESIDIO.registry.add_recognizer(pr)
            except Exception:
                pass
    except Exception:
        PRESIDIO = None

# ---------------- Optional LLM Guard ----------------
# Why: Third-party safety scanners for injection/jailbreak/secrets/toxicity (input) and PII/toxicity (output).
LLM_GUARD_INPUT = None
LLM_GUARD_OUTPUT = None
if USE_LLM_GUARD:
    try:
        from llm_guard import scan_prompt, scan_output  # noqa
        from llm_guard.input_scanners import PromptInjection, Secrets, Toxicity, Jailbreak
        from llm_guard.output_scanners import PII as PIIOut, Toxicity as ToxicityOut
        LLM_GUARD_INPUT = [PromptInjection(threshold=0.1), Jailbreak(threshold=0.1), Secrets(), Toxicity(threshold=0.5)]
        LLM_GUARD_OUTPUT = [PIIOut(), ToxicityOut()]
    except Exception:
        LLM_GUARD_INPUT = None
        LLM_GUARD_OUTPUT = None

def llm_guard_scan_prompt(text: str) -> tuple[bool, str, list]:
    """
    What: Run LLM Guard input scanners and return (ok, sanitized_text, results).
    Why: Stage 1A sanitation + rich scanner objects so the Privacy Meter can display input flags.
    """
    if not (USE_LLM_GUARD and LLM_GUARD_INPUT):
        return True, text, []
    try:
        from llm_guard import scan_prompt
        sanitized, results = scan_prompt(text, LLM_GUARD_INPUT)
        ok = all(getattr(r, "valid", getattr(r, "is_valid", True)) for r in (results or []))
        return ok, sanitized or text, (results or [])
    except Exception:
        return True, text, []

def detect_llm_guard_input(text: str) -> bool:
    """
    What: Boolean helper - did any input scanner fail?
    Why: Used by any_true_block to mark input-stage hits and decide block policy.
    """
    if not (USE_LLM_GUARD and LLM_GUARD_INPUT):
        return False
    try:
        from llm_guard import scan_prompt
        _, results = scan_prompt(text, LLM_GUARD_INPUT)
        return not all(getattr(r, "valid", getattr(r, "is_valid", True)) for r in results)
    except Exception:
        return False

def detect_llm_guard_output(text: str) -> bool:
    """
    What: Boolean helper - did any output scanner fail?
    Why: Used by retrieval/output stages for ANY-TRUE evidence and optional block.
    """
    if not (USE_LLM_GUARD and LLM_GUARD_OUTPUT):
        return False
    try:
        from llm_guard import scan_output
        _, results = scan_output(text, LLM_GUARD_OUTPUT)
        return not all(getattr(r, "valid", getattr(r, "is_valid", True)) for r in results)
    except Exception:
        return False

# ---------------- Optional Guardrails ----------------
# Why: Additional validation layer for generated text (regex validators).
GUARDRAILS_GUARD = None
if USE_GUARDRAILS:
    try:
        from guardrails import Guard
        RAIL_SPEC = """
<rail version="0.1">
  <output type="string" />
  <validate>
    <regex_match name="no_injection"
                 pattern="^(?:(?!(ignore all previous instructions|system prompt|jailbreak|bypass (?:your|the) (?:rules|guardrails)|reveal secrets)).)*$"
                 on-fail="refrain" />
    <regex_match name="no_pii"
                 pattern="^(?:(?!(\\b\\d{3}-\\d{2}-\\d{4}\\b|[A-Z]{5}\\d{4}[A-Z]|\\b[2-9]\\d{3}\\s?\\d{4}\\s?\\d{4}\\b|\\b(?:\\+?\\d{1,3}[-.\\s]?)?\\d{3}[-.\\s]?\\d{3}[-.\\s]?\\d{4}\\b|[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,})).)*$"
                 on-fail="refrain" />
  </validate>
</rail>
"""
        GUARDRAILS_GUARD = Guard.from_rail_string(RAIL_SPEC)
    except Exception:
        GUARDRAILS_GUARD = None

def detect_guardrails_output(text: str) -> bool:
    """
    What: Boolean helper - did Guardrails validators fail on output?
    Why: Extra guardrail signal for ANY-TRUE and audit (Privacy Meter).
    """
    if not (USE_GUARDRAILS and GUARDRAILS_GUARD):
        return False
    try:
        result = GUARDRAILS_GUARD.validate(output=text)
        return bool(result and getattr(result, "validation_passed", False) is False)
    except Exception:
        return False

# ---------------- Regex PII + Secrets + Denylist -------------------
# Why: Lightweight, fast detectors for PII tokens and common jailbreak/injection phrases.
RE_EMAIL   = re.compile(r"\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[A-Za-z]{2,}\b")
RE_PHONE   = re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b")
RE_AADHAAR = re.compile(r"\b[2-9]\d{3}\s?\d{4}\s?\d{4}\b")
RE_PAN     = re.compile(r"(?<![A-Z])[A-Z]{5}[0-9]{4}[A-Z](?![A-Z])")
RE_IFSC    = re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b")
RE_IBAN    = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b")
RE_SSN     = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
RE_IP4     = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
RE_DATE    = re.compile(
    r"\b(19|20)\d{2}[-/](0?[1-9]|1[0-2])[-/](0?[1-9]|[12]\d|3[01])\b"
    r"|"
    r"\b(0?[1-9]|[12]\d|3[01])[-/](0?[1-9]|1[0-2])[-/](19|20)\d{2}\b",
    re.I
)

INJECTION_DENY_PATTERNS = [
    r"ignore all previous instructions",
    r"\bsystem prompt\b",
    r"\bjailbreak\b",
    r"bypass (?:your|the) (?:rules|guardrails)",
    r"reveal secrets",
    r"hidden dataset",
    r"training data( prompt)?",
    r"show me (?:the )?hidden",
]

# Secrets (keys/tokens/private keys, JWTs)
SECRET_RE = re.compile(
    r"-----BEGIN [^-]+ PRIVATE KEY-----|AKIA[0-9A-Z]{16}|\b(bearer|eyJ)[A-Za-z0-9\._\-]{20,}\b",
    re.I
)
def _has_secrets(text: str) -> bool:
    """
    What: Detect high-risk secrets (keys/JWTs/private keys).
    Why: Secrets are always hard-blocked for everyone.
    """
    return bool(SECRET_RE.search(text or ""))

def _digits(s: str) -> str:
    return "".join(ch for ch in str(s) if ch.isdigit())

def _luhn_ok(num: str) -> bool:
    """
    What: Luhn check for credit card-like values.
    Why: Confirm likely card numbers among digit sequences.
    """
    d = _digits(num)
    if not (13 <= len(d) <= 19):
        return False
    total, alt = 0, False
    for ch in d[::-1]:
        n = ord(ch) - 48
        if alt:
            n *= 2
            if n > 9: n -= 9
        total += n
        alt = not alt
    return total % 10 == 0

def detect_presidio(text: str) -> T.Set[str]:
    """
    What: Run Presidio to detect PII categories (kinds), including optional intents.
    Why: Provide rich categories for the Privacy Meter and retrieval/output decisions.
    """
    kinds: T.Set[str] = set()
    if not PRESIDIO or not text:
        return kinds
    try:
        res = PRESIDIO.analyze(text=text, language="en")
        for r in res:
            et = r.entity_type
            if et == "EMAIL_ADDRESS": kinds.add("Email")
            elif et == "PHONE_NUMBER": kinds.add("Phone")
            elif et == "IBAN_CODE": kinds.add("IBAN")
            elif et == "CREDIT_CARD": kinds.add("CardNumber")
            elif et == "US_SSN": kinds.add("SSN")
            elif et == "IP_ADDRESS": kinds.add("IP")
            elif et == "DATE_TIME": kinds.add("DOB")
            elif et == "LOCATION": kinds.add("Address")
            # intents (for visibility)
            elif et == "EMAIL_INTENT": kinds.add("Email")
            elif et == "PHONE_INTENT": kinds.add("Phone")
            elif et == "PAN_INTENT": kinds.add("PAN")
            elif et == "AADHAAR_INTENT": kinds.add("Aadhaar")
            elif et == "SALARY_INTENT": kinds.add("Salary")
            elif et == "ADDRESS_INTENT": kinds.add("Address")
    except Exception:
        pass
    return kinds

def detect_regex(text: str, stage: str = "", value_only: bool = False) -> T.Set[str]:
    """
    What: Regex-based detector for PII-like tokens and related terms.
    Why: Fast, deterministic backstop that complements third-party scanners.
    """
    s = text or ""
    kinds: T.Set[str] = set()
    if RE_EMAIL.search(s): kinds.add("Email")
    if RE_PHONE.search(s): kinds.add("Phone")
    if RE_AADHAAR.search(s): kinds.add("Aadhaar")
    if RE_PAN.search(s): kinds.add("PAN")
    if RE_IFSC.search(s): kinds.add("IFSC"); kinds.add("BankAccount")
    if RE_IBAN.search(s): kinds.add("IBAN"); kinds.add("BankAccount")
    if RE_SSN.search(s): kinds.add("SSN")
    if RE_IP4.search(s): kinds.add("IP")
    if RE_DATE.search(s): kinds.add("DOB")
    for cand in re.findall(r"\b(?:\d[ -]?){13,19}\b", s):
        # Luhn
        d = _digits(cand)
        if _luhn_ok(d):
            kinds.update({"CardNumber", "BankAccount"}); break
    if not value_only:
        if re.search(r"\b(bank|account|acct|acc\s*no|iban|ifsc|routing)\b", s, re.I): kinds.add("BankAccount")
        if re.search(r"\b(address|street|road|avenue|city|zip|postal|pin)\b", s, re.I): kinds.add("Address")
        if re.search(r"\b(salary|ctc|wage|compensation)\b", s, re.I): kinds.add("Salary")
    return kinds

def detect_denylist(text: str) -> bool:
    """
    What: Check for common jailbreak/injection phrases.
    Why: Conservative fallback block for known bad patterns.
    """
    t = (text or "").lower()
    return any(re.search(p, t) for p in INJECTION_DENY_PATTERNS)

# ---------------- ANY-TRUE engine -------------------
def any_true_block(text: str, stage: str, *, question: str | None = None, contexts: T.List[str] | None = None, include_regex: bool = True) -> tuple[bool, list[str], set[str]]:
    """
    What: Run detectors for a given stage and aggregate evidence.
    Why: ANY-TRUE policy — if any enabled detector flags (or PII categories are found),
         we report hits (who/kinds) and let the caller decide whether to block.

    Returns:
      - flagged (bool): True if any detector matched or kinds found.
      - who (list[str]): detector sources (e.g., ['llm_guard_input','presidio','regex'])
      - kinds (set[str]): normalized categories (e.g., {'Email','Phone','Aadhaar'})
    """
    flagged_by: list[str] = []
    kinds: set[str] = set()

    if stage == "input":
        if detect_llm_guard_input(text):
            flagged_by.append("llm_guard_input")
        k1 = detect_presidio(text)
        if k1: kinds |= k1; flagged_by.append("presidio")
        if include_regex and USE_REGEX and USE_REGEX_INPUT:
            k2 = detect_regex(text, stage="input", value_only=REGEX_VALUE_ONLY_INPUT)
            if k2: kinds |= k2; flagged_by.append("regex")

    elif stage == "retrieval":
        if detect_llm_guard_output(text): flagged_by.append("llm_guard_output")
        if detect_guardrails_output(text): flagged_by.append("guardrails")
        k1 = detect_presidio(text)
        if k1: kinds |= k1; flagged_by.append("presidio")
        if include_regex and USE_REGEX:
            k2 = detect_regex(text, stage="retrieval", value_only=False)
            if k2: kinds |= k2; flagged_by.append("regex")

    elif stage == "output":
        if detect_llm_guard_output(text): flagged_by.append("llm_guard_output")
        if detect_guardrails_output(text): flagged_by.append("guardrails")
        k1 = detect_presidio(text)
        if k1: kinds |= k1; flagged_by.append("presidio")
        if include_regex and USE_REGEX:
            k2 = detect_regex(text, stage="output", value_only=False)
            if k2: kinds |= k2; flagged_by.append("regex")

    # Why: If all detectors are disabled or unavailable, fail-safe can block to avoid unsafe leakage.
    no_detectors = not any([
        USE_LLM_GUARD and (LLM_GUARD_INPUT if stage=="input" else LLM_GUARD_OUTPUT),
        USE_GUARDRAILS and GUARDRAILS_GUARD,
        USE_PRESIDIO and PRESIDIO,
        USE_REGEX and (USE_REGEX_INPUT if stage=="input" else True),
    ])
    if no_detectors and FAIL_SAFE_BLOCK_IF_NO_DETECTOR:
        return True, ["fail_safe"], set()

    flagged = bool(flagged_by) or bool(kinds)
    return flagged, list(dict.fromkeys(flagged_by)), kinds

# ---------------- Public API ------------------------
def guard_input_anytrue_guards_only(query: str) -> tuple[bool, str, list[str], set[str]]:
    """
    What: Stage 1C input guards (third-party only). REPORT presidio + llm_guard; BLOCK only on llm_guard.
    Why: Minimize false positives (Presidio = visibility; llm_guard = decision).
    """
    who: list[str] = []
    kinds: set[str] = set()

    k_pres = detect_presidio(query)
    if k_pres:
        kinds |= k_pres
        who.append("presidio")

    if detect_llm_guard_input(query):
        who.append("llm_guard_input")
        return False, FRIENDLY_MSG, list(dict.fromkeys(who)), kinds

    return True, "", list(dict.fromkeys(who)), kinds

def guard_input_regex_only(query: str) -> tuple[bool, str, list[str], set[str]]:
    """
    What: Stage 2 input fallback (after RBAC): value-only PII regex intent + injection denylist.
    Why: Conservative net for literal values and known jailbreak phrases. Caller can
         skip for self to reduce friction (policy decision in rag.py).
    """
    if not (USE_REGEX and USE_REGEX_INPUT):
        return True, "", [], set()
    kinds = detect_regex(query, stage="input", value_only=REGEX_VALUE_ONLY_INPUT)
    deny = detect_denylist(query)
    if kinds or deny:
        who = []
        if kinds: who.append("regex")
        if deny: who.append("denylist")
        kinds2 = set(kinds or set())
        if deny: kinds2.add("denylist")
        return False, FRIENDLY_MSG, who, kinds2
    return True, "", [], set()

def guard_retrieval_anytrue(chunks: T.List[str], *, self_ok_pii: bool = False) -> tuple[bool, T.List[str], list[str], set[str]]:
    """
    What: Scan retrieved docs (Stage: retrieval). Filter/allow docs based on PII policy.
    Why:
      - Always drop secrets (for everyone).
      - If self_ok_pii=False (others), drop docs containing hard PII categories.
      - If self_ok_pii=True (self), allow PII but still drop secrets.
    Returns: (ok, safe_docs, who, kinds)
    """
    who_all: list[str] = []
    kinds_all: set[str] = set()
    safe_docs: list[str] = []
    hard_pii = {"PAN", "Aadhaar", "CardNumber", "IBAN", "IFSC", "SSN", "BankAccount", "DOB", "Email", "Phone", "Address", "IP"}
    bad = False

    for doc in (chunks or []):
        flagged, who, kinds = any_true_block(doc, stage="retrieval", include_regex=True)
        secrets = _has_secrets(doc)
        who_all.extend(who); kinds_all |= kinds

        if secrets:
            bad = True
            continue  # drop this doc entirely

        if (not self_ok_pii) and (kinds & hard_pii):
            bad = True
            continue  # drop this doc for non-self

        safe_docs.append(doc)

    if bad and not safe_docs:
        return False, [], list(dict.fromkeys(who_all)), kinds_all
    return True, safe_docs if safe_docs else (chunks or []), list(dict.fromkeys(who_all)), kinds_all

def guard_output_anytrue(answer: str, *, question: str | None = None, contexts: T.List[str] | None = None, self_ok_pii: bool = False) -> tuple[bool, str, list[str], set[str]]:
    """
    What: Scan final answer (Stage: output) and enforce PII/secrets policy.
    Why:
      - Secrets are always blocked (for everyone).
      - If self_ok_pii=False (others), block if hard PII detected.
      - If self_ok_pii=True (self), allow PII; still block secrets.
    Returns: (ok, safe_text, who, kinds)
    """
    flagged, who, kinds = any_true_block(answer, stage="output", question=question, contexts=contexts, include_regex=True)
    hard_pii = {"PAN", "Aadhaar", "CardNumber", "IBAN", "IFSC", "SSN", "BankAccount", "DOB", "Email", "Phone", "Address", "IP"}

    if _has_secrets(answer):
        return False, FRIENDLY_MSG, list(dict.fromkeys(who + ["regex"])), kinds | {"Secrets"}

    if (not self_ok_pii) and (kinds & hard_pii):
        return False, FRIENDLY_MSG, list(dict.fromkeys(who)), kinds

    return True, answer, [], set()