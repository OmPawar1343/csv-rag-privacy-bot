"""
privacy_utils.py


  Build and log a Privacy Meter for each response: stage, gates fired, detectors,
  input flags, requested sensitive kinds, and any PII shown in output. Supports
  CSV, JSONL, and pretty JSON logs and renders a compact or full markdown table
  for the UI/CLI.


"""

import os
import re
import time
import random
import string
import csv
import json

# Top-level toggle: render the Privacy Meter section
PRIVACY_METER = True

# Rendering mode
METER_COMPACT = os.getenv("PRIV_METER_COMPACT", "1").lower() in ("1", "true", "yes", "on")

# CSV logging toggles
METER_LOG = os.getenv("PRIV_METER_LOG", "1").lower() in ("1", "true", "yes", "on")
METER_LOG_PATH = os.getenv("PRIV_METER_LOG_PATH", "logs/privacy_meter.csv")

# JSONL logging toggles
METER_LOG_JSONL = os.getenv("PRIV_METER_LOG_JSONL", "1").lower() in ("1", "true", "yes", "on")
METER_JSONL_PATH = os.getenv("PRIV_METER_JSONL_PATH", "logs/privacy_meter.jsonl")

# Pretty JSON toggles (vertical, human-readable) — ENABLED BY DEFAULT
METER_LOG_JSON_PRETTY = os.getenv("PRIV_METER_LOG_JSON_PRETTY", "1").lower() in ("1","true","yes","on")
METER_JSON_PRETTY_PATH = os.getenv("PRIV_METER_JSON_PRETTY_PATH", "logs/privacy_meter.pretty.json")

# Content controls for logs (preview sizes and full dumps)
METER_LOG_PREVIEW_CHARS = int(os.getenv("PRIV_METER_LOG_PREVIEW_CHARS", "160"))
METER_LOG_FULL = os.getenv("PRIV_METER_LOG_FULL", "0").lower() in ("1", "true", "yes", "on")

# Optional: show exposure risk (attempt vs actual exposure)
SHOW_EXPOSURE_RISK = os.getenv("PRIV_SHOW_EXPOSURE_RISK", "1").lower() in ("1", "true", "yes", "on")


# ---------- PII detectors for final answer ----------
# What: Lightweight regex to check final answer text for visible PII.
# Why: Improves the Privacy Meter signal by listing “Output PII shown”.
RE_EMAIL   = re.compile(r"\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[A-Za-z]{2,}\b")

# Phones: detect sequences that look like real phone numbers (10–15 digits after stripping separators)
RE_PHONE_CAND = re.compile(r"(?:\+?\d[\d\s().-]{8,}\d)")

# Date candidates like 2020-04-03 or 4/3/2020 (generic date, not necessarily DOB)
RE_DATE_CAND = re.compile(
    r"\b(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b"
)

# Label hints to disambiguate DOB vs employment/joining dates
DOB_LABEL_HINT = re.compile(r"\b(dob|date\s*of\s*birth|birth\s*date|born)\b", re.I)
JOINING_LABEL_HINT = re.compile(
    r"\b(doj|joining\s*date|date\s*of\s*joining|hire\s*date|hiring\s*date|start\s*date)\b",
    re.I,
)

RE_AADHAAR = re.compile(r"\b[2-9]\d{3}\s?\d{4}\s?\d{4}\b")
RE_PAN     = re.compile(r"(?<![A-Z])[A-Z]{5}[0-9]{4}[A-Z](?![A-Z])")
RE_ADDR_L  = re.compile(r"\b(address|street|road|avenue|city|zip|postal|pin|location)\b", re.I)
RE_EMP_ID  = re.compile(r"\bemp[\s_-]?id\b", re.I)
RE_GENDER  = re.compile(r"\bgender\b", re.I)
RE_SALARY  = re.compile(r"\b(salary|ctc|compensation|wage|pay)\b", re.I)

def _has_phone_like(s: str) -> bool:
    """
    Return True if text contains a phone-like sequence with 10–15 digits
    (after removing separators), e.g., +91 98765 43210 or (202) 555-0199.
    Avoids 7-digit numbers like 1626886.
    """
    for m in RE_PHONE_CAND.finditer(s or ""):
        digits = re.sub(r"\D", "", m.group(0))
        if 10 <= len(digits) <= 15:
            return True
    return False

def _has_date_near_label(s: str, label_re: re.Pattern, window: int = 32) -> bool:
    """
    True if a generic date appears near a specific label hint (DOB / joining).
    Helps disambiguate DOB vs employment dates.
    """
    s = s or ""
    for m in RE_DATE_CAND.finditer(s):
        start, end = m.span()
        left = max(0, start - window)
        right = min(len(s), end + window)
        if label_re.search(s[left:right]):
            return True
    return False


# Canonicalization of requested sensitive kinds
SENSITIVE_CANON = {
    "phone": "Phone", "mobile": "Phone", "contact": "Phone", "contact no": "Phone",
    "contact number": "Phone", "telephone": "Phone", "tel": "Phone",
    "salary": "Salary", "ctc": "Salary", "compensation": "Salary", "wage": "Salary", "pay": "Salary",
    "dob": "DOB", "date of birth": "DOB", "birth date": "DOB", "born": "DOB",
    "joiningdate": "JoiningDate", "date of joining": "JoiningDate", "doj": "JoiningDate",
    "hire date": "JoiningDate", "start date": "JoiningDate",
    "address": "Address", "location": "Address",
    "email": "Email", "mail": "Email",
    "empid": "EmpID", "employee id": "EmpID", "emp id": "EmpID",
    "ssn": "SSN", "passport": "Passport", "pan": "PAN", "aadhaar": "Aadhaar", "aadhar": "Aadhaar",
}

def canonicalize_sensitive(items: list[str]) -> list[str]:
    """
    Map various wordings to canonical sensitive kinds and deduplicate.
    """
    out = []
    seen = set()
    for x in items or []:
        t = str(x).strip()
        if not t:
            continue
        k = re.sub(r"\s+", " ", t).lower()
        canon = SENSITIVE_CANON.get(k)
        if not canon:
            # heuristics
            if any(w in k for w in ("phone", "mobile", "contact", "tel")):
                canon = "Phone"
            elif any(w in k for w in ("salary", "ctc", "compensation", "wage", "pay")):
                canon = "Salary"
            elif "dob" in k or "date of birth" in k or "birth" in k or "born" in k:
                canon = "DOB"
            elif any(w in k for w in ("joining", "doj", "hire", "start date")):
                canon = "JoiningDate"
            elif any(w in k for w in ("address", "location")):
                canon = "Address"
            elif "email" in k or "mail" in k:
                canon = "Email"
            elif "emp" in k and "id" in k:
                canon = "EmpID"
            elif "passport" in k:
                canon = "Passport"
            elif "pan" in k:
                canon = "PAN"
            elif "aadhaar" in k or "aadhar" in k:
                canon = "Aadhaar"
        if canon and canon not in seen:
            seen.add(canon)
            out.append(canon)
    return out


def _event_id() -> str:
    """
    What: Generate a unique event id for each meter entry.
    Why: Makes correlating UI logs and CSV/JSON easy during audits.
    """
    ts = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    suf = "".join(random.choices(string.hexdigits.upper(), k=6))
    return f"PRIV-{ts}-{suf}"

def _input_flags(scan_results) -> dict:
    """
    What: Summarize LLM-Guard input scanner results into booleans.
    Why: Compact “input flags” line in the meter (injection/jailbreak/secrets/toxicity).
    """
    flags = {"prompt_injection": False, "jailbreak": False, "secrets": False, "toxicity": False}
    for r in (scan_results or []):
        name = (getattr(r, "name", "") or "").lower()
        valid = bool(getattr(r, "valid", getattr(r, "is_valid", True)))
        if not valid:
            if "prompt" in name and "injection" in name:
                flags["prompt_injection"] = True
            elif "jailbreak" in name:
                flags["jailbreak"] = True
            elif "secret" in name:
                flags["secrets"] = True
            elif "toxic" in name:
                flags["toxicity"] = True
    return flags

def _detect_output_pii(answer_text: str) -> set[str]:
    """
    What: Heuristically detect PII categories present in the final answer text.
    Why: If content leaked, display which kinds (Email/Phone/etc.) were shown.
         Values that are clearly masked (REDACTED, *** , Not available) do not count as PII shown.
    """
    s = answer_text or ""
    shown = set()

    # Direct value patterns (real-looking values)
    # These ignore redacted placeholders because they don't match these regexes.
    if RE_EMAIL.search(s):        shown.add("Email")
    if _has_phone_like(s):        shown.add("Phone")
    if RE_AADHAAR.search(s):      shown.add("Aadhaar")
    if RE_PAN.search(s):          shown.add("PAN")

    def _is_redacted_value(v: str) -> bool:
        """Return True if the displayed value is obviously masked / not real PII."""
        v = (v or "").strip().lower()
        if not v:
            return True
        if "not available" in v:
            return True
        if "redacted" in v:
            return True
        if "***" in v:
            return True
        return False

    # Table/Markdown label presence, but only count if value is not redacted/masked
    def _label_has_value(label: str) -> bool:
        # Grid/table row: | Label | Value |
        pat_grid = rf"\|\s*{re.escape(label)}\s*\|\s*([^|]+)\|"
        m_grid = re.search(pat_grid, s, re.I)
        if m_grid:
            val = m_grid.group(1)
            if not _is_redacted_value(val):
                return True

        # Inline: Label: Value
        pat_md = rf"\b{re.escape(label)}\b\s*:\s*([^\n]+)"
        m_md = re.search(pat_md, s, re.I)
        if m_md:
            val = m_md.group(1)
            if not _is_redacted_value(val):
                return True

        return False

    label_map = [
        ("Aadhaar", "Aadhaar"), ("Aadhar", "Aadhaar"),
        ("PAN", "PAN"),
        ("DOB", "DOB"), ("Date of Birth", "DOB"),
        ("Email", "Email"), ("Phone", "Phone"),
        ("Address", "Address"), ("Salary", "Salary"),
        ("EmpID", "EmpID"), ("Gender", "Gender"),
        # Joining / employment dates
        ("JoiningDate", "JoiningDate"), ("Date of Joining", "JoiningDate"),
        ("DOJ", "JoiningDate"), ("Hire Date", "JoiningDate"), ("Start Date", "JoiningDate"),
    ]
    for label, kind in label_map:
        if _label_has_value(label):
            shown.add(kind)

    # Disambiguate DOB vs other dates (only when dates appear near label hints)
    if _has_date_near_label(s, DOB_LABEL_HINT):
        shown.add("DOB")
    if _has_date_near_label(s, JOINING_LABEL_HINT):
        shown.add("JoiningDate")

    # Loose label matches in narrative text are DISABLED to avoid false positives
    # when only words like "EmpID" or "Address" appear without real values.
    # if RE_EMP_ID.search(s):  shown.add("EmpID")
    # if RE_GENDER.search(s):  shown.add("Gender")
    # if RE_ADDR_L.search(s):  shown.add("Address")
    # if RE_SALARY.search(s):  shown.add("Salary")

    return shown

def _parse_trigger_info(trigger_reason: str | None) -> tuple[str | None, list[str]]:
    """
    What: Parse 'trigger_reason' into (stage, detectors) for the meter.
    Why: Normalize free-text reasons into structured fields for CSV/JSON and display.
    """
    if not trigger_reason:
        return None, []
    s = trigger_reason.lower()
    stage = None
    if "input gate" in s: stage = "input"
    elif "retrieval gate" in s: stage = "retrieval"
    elif "output gate" in s: stage = "output"
    elif "policy" in s or "self-only" in s: stage = "policy"
    dets = []
    m = re.search(r"\(([^)]+)\)", s)
    if m:
        dets = [d.strip().replace(" ", "_") for d in m.group(1).split(",") if d.strip()]
    if stage == "policy" and "rbac" not in dets:
        dets.append("rbac")
    # Normalize common sources
    for key in ["presidio","regex","llm_guard","guardrails","giskard","fail_safe","rbac"]:
        if key in s and key not in dets:
            dets.append(key)
    return stage, dets

def _risk_score_guard_aware(requested_sensitive: list[str], role_verified: bool,
                            input_flags: dict, stage: str | None, detectors: list[str]) -> int:
    """
    What: Compute a heuristic risk score based on what was asked for, where it was caught,
         and which detectors fired (plus LLM-Guard input flags).
    Why: Single “Risk” number makes triage and dashboards easier.
    """
    base = {
        "BankAccount":70,"CardNumber":85,"IBAN":70,"IFSC":55,
        "Aadhaar":80,"PAN":65,"Passport":65,"SSN":80,
        "Salary":55,"DOB":45,"JoiningDate":20,
        "Address":35,"Location":35,"Phone":25,"Email":20,"EmpID":10,"Gender":10
    }
    score = sum(base.get(str(k), 10) for k in set(requested_sensitive or []))
    stage_w = {"input":15, "retrieval":25, "output":35, "policy":10}
    if stage in stage_w:
        score += stage_w[stage]
    det_w = {"presidio":12, "regex":8, "llm_guard":10, "guardrails":8, "giskard":6, "fail_safe":15, "rbac":10}
    for d in set(detectors or []):
        score += det_w.get(d, 5)
    if input_flags.get("prompt_injection"): score += 15
    if input_flags.get("jailbreak"):        score += 15
    if input_flags.get("secrets"):          score += 20
    if input_flags.get("toxicity"):         score += 5
    # Slight uplift if unverified role with sensitive ask/detectors
    if (requested_sensitive or detectors) and not role_verified:
        score += 10
    return max(0, min(100, score))

def _level(score: int) -> str:
    """
    What: Map numeric risk score to a level.
    Why: Quick, manager-friendly severity indicator.
    """
    return "High" if score >= 70 else "Medium" if score >= 35 else "Low"

# ---- CSV/JSON helpers ----
def _ensure_dir(path: str):
    """
    What: Ensure directory exists for any log path.
    Why: Avoid IO errors when writing CSV/JSON/pretty logs.
    """
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    except Exception:
        pass

def _bool_str(v) -> str:
    """
    What: Normalize booleans to capitalized strings (True/False).
    Why: Uniform CSV text for easy filtering.
    """
    return "True" if bool(v) else "False"

def _flags_str(flags: dict) -> str:
    """
    What: Format input flags in a single line.
    Why: Compact view in the meter and CSV.
    """
    return f"prompt_injection:{_bool_str(flags.get('prompt_injection'))}, " \
           f"jailbreak:{_bool_str(flags.get('jailbreak'))}, " \
           f"secrets:{_bool_str(flags.get('secrets'))}, " \
           f"toxicity:{_bool_str(flags.get('toxicity'))}"

def _preview(text: str, n: int) -> str:
    """
    What: Safe, single-line preview of question/answer.
    Why: Logs should show enough context without dumping full content.
    """
    s = (text or "").replace("\r", " ").replace("\n", " ").strip()
    return s if len(s) <= n else s[: n - 1] + "…"

# CSV/JSONL schema (kept stable for dashboards)
CSV_HEADERS = [
    "time_utc","event_id","section","status","stage","risk","risk_level",
    "role","role_verified","is_self","admin_bypass","guardrails_blocked","denylist_hit",
    "requested_sensitive","gates_fired","detectors","trigger_reason","output_pii_shown",
    "input_flags","question_preview","answer_preview"
]

def _log_to_csv(row: dict):
    """
    What: Append a single meter row to a CSV.
    Why: Easy ad-hoc analysis and Excel-friendly.
    """
    if not METER_LOG or not METER_LOG_PATH:
        return
    try:
        _ensure_dir(METER_LOG_PATH)
        file_exists = os.path.exists(METER_LOG_PATH) and os.path.getsize(METER_LOG_PATH) > 0
        with open(METER_LOG_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADERS, extrasaction="ignore")
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
    except Exception:
        pass

def _log_to_jsonl(obj: dict):
    """
    What: Append a structured JSON object (one per line).
    Why: Machine-friendly logs for pipelines and BI tools.
    """
    if not METER_LOG_JSONL or not METER_JSONL_PATH:
        return
    try:
        _ensure_dir(METER_JSONL_PATH)
        with open(METER_JSONL_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    except Exception:
        pass

def _log_to_json_pretty(obj: dict):
    """
    What: Write a human-readable JSON snapshot.
    Why: Handy for manual audits and ticket attachments.
    """
    if not METER_LOG_JSON_PRETTY or not METER_JSON_PRETTY_PATH:
        return
    try:
        _ensure_dir(METER_JSON_PRETTY_PATH)
        with open(METER_JSON_PRETTY_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False, indent=2) + "\n\n")
    except Exception:
        pass

def privacy_meter_report(
    section_name: str,
    question: str,
    answer_text: str,
    user_role: str | None,
    requested_sensitive_cols: list[str] | None,
    input_scan_results: list | None,
    denylist_hit: bool | None,
    guardrails_blocked: bool,
    trigger_reason: str | None = None,
    role_verified: bool | None = None,
    is_self: bool | None = None,
    admin_bypass: bool | None = None,
) -> str:
    """
    What: Build, log, and render the Privacy Meter for a single response.
    Why:
      - Standardize safety telemetry: stage, gates fired, detectors, input flags.
      - Provide quick risk scoring and any visible output PII.
      - Persist structured logs (CSV, JSONL, pretty JSON) for audits.

    Returns:
      - Markdown table string (compact or full) to display under the answer.
    """
    eid = _event_id()
    flags = _input_flags(input_scan_results)

    # Canonicalize requested sensitive items and dedupe
    req_raw = [str(x) for x in (requested_sensitive_cols or []) if str(x).strip()]
    req = canonicalize_sensitive(req_raw)

    stage, detectors = _parse_trigger_info(trigger_reason)
    out_pii = _detect_output_pii(answer_text)

    status = "Blocked" if guardrails_blocked else "Allowed"
    stage_disp = stage.title() if stage else "-"
    out_pii_display = ", ".join(sorted(out_pii)) if out_pii else "None (0)"
    score = _risk_score_guard_aware(req, bool(role_verified), flags, stage, detectors)
    level = _level(score)

    # Exposure risk: if nothing leaked and it was blocked, exposure is 0; otherwise mirrors score
    exposure_score = 0 if (guardrails_blocked and not out_pii) else score
    exposure_level = _level(exposure_score)

    gates_fired = "-" if not (stage and detectors) else f"{stage}: " + ", ".join(detectors).replace("_", " ")

    # Build CSV row
    now_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    csv_row = {
        "time_utc": now_utc,
        "event_id": eid,
        "section": section_name or "-",
        "status": status,
        "stage": stage_disp,
        "risk": score,
        "risk_level": level,
        "role": (user_role or "-"),
        "role_verified": _bool_str(role_verified),
        "is_self": _bool_str(is_self),
        "admin_bypass": _bool_str(admin_bypass),
        "guardrails_blocked": _bool_str(guardrails_blocked),
        "denylist_hit": _bool_str(denylist_hit),
        "requested_sensitive": ", ".join(req) if req else "-",
        "gates_fired": gates_fired,
        "detectors": ", ".join(detectors) if detectors else "-",
        "trigger_reason": trigger_reason or "-",
        "output_pii_shown": ", ".join(sorted(out_pii)) if out_pii else "-",
        "input_flags": _flags_str(flags),
        "question_preview": _preview(question, METER_LOG_PREVIEW_CHARS),
        "answer_preview": _preview(answer_text, METER_LOG_PREVIEW_CHARS),
    }
    _log_to_csv(csv_row)

    # Build JSON objects (machine- and human-friendly)
    json_obj = {
        "time_utc": now_utc,
        "event_id": eid,
        "section": section_name or None,
        "status": status,
        "stage": stage,  # "input" | "retrieval" | "output" | "policy" | None
        "risk": {"score": score, "level": level},
        "risk_breakdown": {
            "attempt": {"score": score, "level": level},
            "exposure": {"score": exposure_score, "level": exposure_level},
        },
        "role": {"name": user_role or None, "verified": bool(role_verified), "is_self": bool(is_self)},
        "admin_bypass": bool(admin_bypass),
        "guardrails_blocked": bool(guardrails_blocked),
        "denylist_hit": bool(denylist_hit),
        "requested_sensitive": req,
        "detectors": detectors or [],
        "gates_fired": gates_fired if gates_fired != "-" else None,
        "trigger_reason": trigger_reason,
        "output_pii_shown": sorted(list(out_pii)) if out_pii else [],
        "input_flags": flags,
        "question_preview": _preview(question, METER_LOG_PREVIEW_CHARS),
        "answer_preview": _preview(answer_text, METER_LOG_PREVIEW_CHARS),
    }

    # Optional "full dump" for deep-dive audits (kept for backward compat)
    if METER_LOG_FULL:
        json_obj["answer_full"] = answer_text or ""

    _log_to_jsonl(json_obj)
    _log_to_json_pretty(json_obj)

    # Render compact meter (default) for UI/CLI; switch to full via env
    if METER_COMPACT:
        rows = [
            "| Metric | Value |",
            "|---|---|",
            f"| Event ID | {eid} |",
            f"| Status | {status} |",
            f"| Stage | {stage_disp} |",
            f"| Risk | {score} ({level}) |",
        ]
        if SHOW_EXPOSURE_RISK:
            rows.append(f"| Exposure risk | {exposure_score} ({exposure_level}) |")
        rows.append(f"| Input flags | {_flags_str(flags)} |")
        if req:
            rows.append(f"| Requested sensitive | {', '.join(req)} |")
        if gates_fired != "-":
            rows.append(f"| Gates fired | {gates_fired} |")
        if trigger_reason:
            rows.append(f"| Trigger reason | {trigger_reason} |")
        rows.append(f"| Output PII shown | {out_pii_display} |")
        return "\n".join(rows)

    # Full (PRIV_METER_COMPACT=0)
    role_disp = (user_role or "-")
    role_verified_disp = "True" if role_verified else "False"
    admin_bypass_disp = "yes" if admin_bypass else "no"
    rows = [
        "| Metric | Value |",
        "|---|---|",
        f"| Event ID | {eid} |",
        f"| Time | {now_utc} |",
        f"| Status | {status} |",
        f"| Stage | {stage_disp} |",
        f"| Role | {role_disp} (verified: {role_verified_disp}) |",
        f"| Admin bypass | {admin_bypass_disp} |",
        f"| Risk score | {score} ({level}) |",
    ]
    if SHOW_EXPOSURE_RISK:
        rows.append(f"| Exposure risk | {exposure_score} ({exposure_level}) |")
    rows.extend([
        f"| Requested sensitive | {', '.join(req) if req else '-'} |",
        f"| Input flags | prompt_injection:{flags['prompt_injection']}, jailbreak:{flags['jailbreak']}, secrets:{flags['secrets']}, toxicity:{flags['toxicity']} |",
        f"| Gates fired | {gates_fired} |",
        f"| Trigger reason | {trigger_reason or '-'} |",
        f"| Output PII shown | {out_pii_display} |",
    ])
    return "\n".join(rows)