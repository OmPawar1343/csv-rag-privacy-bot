# privacy_report.py
"""
What:
  Generate a simple privacy report from the new Privacy Meter CSV logs
  (logs/privacy_meter.csv) emitted by privacy_utils.

Why:
  - Summarize total events, risk levels, average risk score.
  - Count blocks (guardrails_blocked/denylist_hit) and input flags.
  - Aggregate which PII types were shown in outputs.
  - Provide top requested sensitive fields and a High-risk rows extract.
"""

import os, sys, argparse, datetime
import pandas as pd

DEFAULT_LOG = "logs/privacy_meter.csv"

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def parse_input_flags_to_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert 'input_flags' column (string) into boolean columns:
      flag_prompt_injection, flag_jailbreak, flag_secrets, flag_toxicity
    """
    cols = ["prompt_injection", "jailbreak", "secrets", "toxicity"]
    for c in cols:
        df[f"flag_{c}"] = False

    def parse_row(s: str):
        out = {f"flag_{c}": False for c in cols}
        if not isinstance(s, str) or ":" not in s:
            return out
        try:
            parts = [p.strip() for p in s.split(",")]
            for p in parts:
                if ":" not in p:
                    continue
                k, v = [x.strip().lower() for x in p.split(":", 1)]
                if k in cols:
                    out[f"flag_{k}"] = (v == "true")
        except Exception:
            pass
        return out

    parsed = df["input_flags"].apply(parse_row)
    for c in cols:
        df[f"flag_{c}"] = parsed.apply(lambda d: d.get(f"flag_{c}", False))
    return df

def split_requested_sensitive(df: pd.DataFrame) -> pd.DataFrame:
    """
    Count requested_sensitive values (comma-separated string column) safely.
    Returns a dataframe with columns: field, count (sorted by count desc).
    """
    counts = {}
    # Pick the right series explicitly (avoid 'or' with Series which is ambiguous)
    if "requested_sensitive" in df.columns:
        s = df["requested_sensitive"].fillna("").astype(str)
    else:
        s = pd.Series([], dtype=str)

    for cell in s:
        cell = cell.strip()
        if not cell or cell == "-":
            continue
        for item in [x.strip() for x in cell.split(",") if x.strip()]:
            counts[item] = counts.get(item, 0) + 1

    if not counts:
        return pd.DataFrame(columns=["field", "count"])

    rs = pd.DataFrame(
        [{"field": k, "count": v} for k, v in sorted(counts.items(), key=lambda x: (-x[1], x[0]))]
    )
    return rs

def tally_output_pii(df: pd.DataFrame) -> dict:
    """
    Aggregate counts of common PII types from 'output_pii_shown' column.
    Expected values in CSV: comma-separated kinds (e.g., "Aadhaar, Email, Phone").
    We count a subset for the summary: Email, EmpID, Gender, DOB, Address.
    """
    keys = ["Email", "EmpID", "Gender", "DOB", "Address"]
    counts = {k: 0 for k in keys}

    def parse_list(val: str):
        if not isinstance(val, str) or not val.strip() or val.strip() in ("-", "None (0)"):
            return []
        return [x.strip() for x in val.split(",") if x.strip()]

    series = df["output_pii_shown"] if "output_pii_shown" in df.columns else pd.Series([], dtype=str)
    for kinds in series.apply(parse_list):
        s = set(kinds)
        for k in keys:
            if k in s:
                counts[k] += 1
    return counts

def make_summary_markdown(summary: dict) -> str:
    lines = [
        "| Metric | Value |",
        "|---|---|",
        f"| Total sections | {summary['total']} |",
        f"| Risk levels | Low={summary['low']} • Medium={summary['medium']} • High={summary['high']} |",
        f"| Avg risk score | {summary['avg_score']:.1f} |",
        f"| Denylist hits | {summary['denylist_hit']} |",
        f"| Guardrails blocked | {summary['guardrails_blocked']} |",
        f"| Prompt injection flagged | {summary['flag_prompt_injection']} |",
        f"| Jailbreak flagged | {summary['flag_jailbreak']} |",
        f"| Secrets flagged | {summary['flag_secrets']} |",
        f"| Toxicity flagged | {summary['flag_toxicity']} |",
        f"| Output PII totals | emails={summary['sum_emails']}, empid={summary['sum_empid']}, gender={summary['sum_gender']}, dob={summary['sum_dob']}, address={summary['sum_address']} |",
    ]
    return "\n".join(lines)

def main():
    ap = argparse.ArgumentParser(description="Generate Privacy Meter report from CSV logs (privacy_meter.csv)")
    ap.add_argument("--log", default=DEFAULT_LOG, help="Path to Privacy Meter CSV (default: logs/privacy_meter.csv)")
    ap.add_argument("--out", default="reports", help="Output reports directory")
    ap.add_argument("--since", default=None, help="Filter rows since YYYY-MM-DD (optional; uses time_utc)")
    args = ap.parse_args()

    if not os.path.exists(args.log):
        print(f"[!] Log file not found: {args.log}")
        sys.exit(1)

    # Robust read: skip malformed rows
    df = pd.read_csv(args.log, dtype=str, engine="python", on_bad_lines="skip")

    # Ensure expected columns exist
    for col in ["risk", "risk_level", "denylist_hit", "guardrails_blocked", "input_flags", "output_pii_shown"]:
        if col not in df.columns:
            if col == "risk":
                df[col] = "0"
            elif col == "risk_level":
                df[col] = "-"
            else:
                df[col] = ""

    # Optional date filter (use time_utc from privacy_utils CSV)
    if args.since and "time_utc" in df.columns:
        try:
            df["ts"] = pd.to_datetime(df["time_utc"], errors="coerce", utc=True)
            since_dt = pd.to_datetime(args.since).tz_localize("UTC")
            df = df[df["ts"] >= since_dt]
        except Exception:
            pass

    if df.empty:
        print("[i] No rows to report after filters.")
        sys.exit(0)

    # Parse risk to numeric
    df["risk"] = pd.to_numeric(df["risk"], errors="coerce").fillna(0).astype(int)

    # Booleans: denylist_hit, guardrails_blocked
    for bcol in ["denylist_hit", "guardrails_blocked"]:
        if bcol in df.columns:
            df[bcol] = df[bcol].astype(str).str.lower().isin(["true", "1", "yes"])
        else:
            df[bcol] = False

    # Parse input_flags into booleans
    df["input_flags"] = df["input_flags"].fillna("")
    df = parse_input_flags_to_cols(df)

    # Output dir stamped per run
    stamp = datetime.datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = os.path.join(args.out, stamp)
    ensure_dir(out_dir)

    # Tally common PII kinds from output_pii_shown
    pii_counts = tally_output_pii(df)

    # Build summary
    summary = {
        "total": len(df),
        "low": int((df["risk_level"] == "Low").sum()),
        "medium": int((df["risk_level"] == "Medium").sum()),
        "high": int((df["risk_level"] == "High").sum()),
        "avg_score": float(df["risk"].astype(float).mean()),
        "denylist_hit": int(df["denylist_hit"].sum()),
        "guardrails_blocked": int(df["guardrails_blocked"].sum()),
        # Input flags
        "flag_prompt_injection": int(df["flag_prompt_injection"].sum()),
        "flag_jailbreak": int(df["flag_jailbreak"].sum()),
        "flag_secrets": int(df["flag_secrets"].sum()),
        "flag_toxicity": int(df["flag_toxicity"].sum()),
        # Output PII totals (subset)
        "sum_emails": int(pii_counts.get("Email", 0)),
        "sum_empid": int(pii_counts.get("EmpID", 0)),
        "sum_gender": int(pii_counts.get("Gender", 0)),
        "sum_dob": int(pii_counts.get("DOB", 0)),
        "sum_address": int(pii_counts.get("Address", 0)),
    }

    # Save summary CSV and MD
    pd.DataFrame([summary]).to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write("# Privacy Meter Summary\n\n")
        f.write(make_summary_markdown(summary))
        f.write("\n")

    # Top requested_sensitive fields (if present)
    if "requested_sensitive" in df.columns:
        rs = split_requested_sensitive(df)
        if not rs.empty:
            rs.to_csv(os.path.join(out_dir, "requested_sensitive_counts.csv"), index=False)

    # Flags totals
    flags_totals = pd.DataFrame([
        {"flag": "prompt_injection", "count": summary["flag_prompt_injection"]},
        {"flag": "jailbreak", "count": summary["flag_jailbreak"]},
        {"flag": "secrets", "count": summary["flag_secrets"]},
        {"flag": "toxicity", "count": summary["flag_toxicity"]},
    ])
    flags_totals.to_csv(os.path.join(out_dir, "flags_counts.csv"), index=False)

    # Output PII totals (subset)
    pii_totals = pd.DataFrame([
        {"entity": "Email", "count": summary["sum_emails"]},
        {"entity": "EmpID", "count": summary["sum_empid"]},
        {"entity": "Gender", "count": summary["sum_gender"]},
        {"entity": "DOB", "count": summary["sum_dob"]},
        {"entity": "Address", "count": summary["sum_address"]},
    ])
    pii_totals.to_csv(os.path.join(out_dir, "pii_totals.csv"), index=False)

    # High-risk rows extract
    high = df[df["risk_level"] == "High"].copy()
    if not high.empty:
        # Keep a manager-friendly subset of columns if present
        keep_cols = [
            "time_utc","section","role","requested_sensitive","input_flags",
            "risk","risk_level","denylist_hit","guardrails_blocked",
            "gates_fired","detectors","trigger_reason","output_pii_shown",
            "question_preview","answer_preview"
        ]
        for c in keep_cols:
            if c not in high.columns:
                high[c] = ""
        high[keep_cols].to_csv(os.path.join(out_dir, "high_risk_rows.csv"), index=False)

    # Daily average risk (if time_utc present)
    if "time_utc" in df.columns:
        try:
            df["date"] = pd.to_datetime(df["time_utc"], errors="coerce").dt.date
            daily = df.groupby("date")["risk"].mean().reset_index(name="avg_risk_score")
            daily.to_csv(os.path.join(out_dir, "daily_risk.csv"), index=False)
        except Exception:
            pass

    print(f"[✓] Report generated at: {out_dir}")

if __name__ == "__main__":
    main()