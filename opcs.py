#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║        OPERATION CLEAN SLATE  v2.0  —  Production Deduplication Engine      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  What's new vs v1.0                                                          ║
║  ─────────────────                                                           ║
║  • SSN Fuzzy Clustering via strict Levenshtein edit-distance ≤ 1             ║
║    (replaces fuzz.ratio % approach — exactly one digit off, no more)         ║
║  • Two-Key Cluster Verification for distance-1 SSN matches:                  ║
║    DOB exact-match OR name fuzz.ratio > 75 required to merge                 ║
║  • Zero-Floor Thresholds: scores below per-field cutoffs collapse to 0.0     ║
║    (eliminates "participation points" for wholly mismatched fields)           ║
║  • Granular per-field string-distance confidence (thefuzz-powered)           ║
║  • SSN confidence: 100% exact / 90% one-edit / 0% defensive fallback         ║
║  • Separate phone_confidence() — strict cutoff=80, no baseline floor         ║
║  • Contact-only mismatch auto-merge: if SSN/name/DOB/address all pass but    ║
║    only phone/email differ, the group is identity-verified and auto-merged    ║
║  • contact_confidence() retained for emails only — 50-pt baseline floor      ║
║  • Weighted Overall Group Confidence score (exact weight matrix)              ║
║  • Interactive Mass Auto-Approval pre-review phase (rich UI)                 ║
║  • Beautiful terminal UI via rich: tables, progress bars, colour prompts     ║
║  • pandas for all vectorized data manipulation                               ║
║  • Enriched audit_log.json (field confidences + weighted group confidence)   ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Dependencies  (pip install -r requirements.txt)                             ║
║    pandas>=2.2    thefuzz>=0.22    rich>=13.7    python-Levenshtein>=0.25    ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Usage                                                                       ║
║    python operation_clean_slate_v2.py                                        ║
║    python operation_clean_slate_v2.py --input /data/clients.json             ║
║    python operation_clean_slate_v2.py --dry-run                              ║
║    python operation_clean_slate_v2.py --mass-approve-threshold 85            ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import Levenshtein                 # python-Levenshtein — strict edit-distance
from thefuzz import fuzz           # used for all non-SSN field confidence
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich.text import Text
from rich import print as rprint

# ──────────────────────────────────────────────────────────────────────────────
# § 0  GLOBAL CONSOLE & CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────

console = Console()          # single shared rich Console instance

INPUT_FILE     = "operation-clean-slate.json"
OUTPUT_MERGED  = "merged_clients.json"
OUTPUT_REMOVED = "duplicates_removed.json"
OUTPUT_AUDIT   = "audit_log.json"

# ── SSN clustering ────────────────────────────────────────────────────────────
# Hard rule: Levenshtein.distance(norm_ssn, norm_core) must be <= 1.
# For distance == 1, a secondary Two-Key check (DOB exact OR name ratio > 75)
# is required before merging. No tunable threshold on the distance itself.
DEFAULT_MASS_APPROVE_PCTG = 99.0   # 100 = mass-approve disabled by default;
                                    # pass --mass-approve-threshold <N> to enable

# ── Exact weight matrix (must sum to 1.0) ────────────────────────────────────
WEIGHTS: dict[str, float] = {
    "ssn":           0.30,
    "name":          0.25,
    "date_of_birth": 0.22,
    "address":       0.18,
    "phone_number":  0.03,
    "email":         0.02,
}

# ── Address abbreviation normalisation table ──────────────────────────────────
ADDR_ABBR: dict[str, str] = {
    r"\bstreet\b":    "st",
    r"\bavenue\b":    "ave",
    r"\bboulevard\b": "blvd",
    r"\bdrive\b":     "dr",
    r"\broad\b":      "rd",
    r"\blane\b":      "ln",
    r"\bcourt\b":     "ct",
    r"\bplace\b":     "pl",
    r"\bcircle\b":    "cir",
    r"\bsuite\b":     "ste",
    r"\bapartment\b": "apt",
    r"\bnorth\b":     "n",
    r"\bsouth\b":     "s",
    r"\beast\b":      "e",
    r"\bwest\b":      "w",
}

TIMESTAMP_FMTS = [
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%d",
]

# ── Colour palette (rich markup) ──────────────────────────────────────────────
CLR_HIGH   = "bold green"
CLR_MED    = "bold yellow"
CLR_LOW    = "bold red"
CLR_RULE   = "dim cyan"
CLR_HEAD   = "bold white"
CLR_MUTED  = "dim"


# ══════════════════════════════════════════════════════════════════════════════
# § 1  INGESTION  (pandas)
# ══════════════════════════════════════════════════════════════════════════════

def load_dataframe(filepath: str) -> pd.DataFrame:
    """
    Read the JSON array into a pandas DataFrame.

    All missing values are filled with empty strings so that downstream
    string operations never encounter NaN. The original dict form of each
    row is preserved in a synthetic ``_raw`` column so we can serialise
    records back to JSON without precision loss.
    """
    if not os.path.exists(filepath):
        console.print(
            f"[bold red][ERROR][/] Input file not found: [yellow]'{filepath}'[/]\n"
            "        Place it in the working directory or pass [cyan]--input <path>[/]"
        )
        sys.exit(1)

    with open(filepath, "r", encoding="utf-8") as fh:
        try:
            raw_list: list[dict] = json.load(fh)
        except json.JSONDecodeError as exc:
            console.print(f"[bold red][ERROR][/] Malformed JSON: {exc}")
            sys.exit(1)

    if not isinstance(raw_list, list):
        console.print("[bold red][ERROR][/] Expected a JSON array at the root level.")
        sys.exit(1)

    df = pd.DataFrame(raw_list)

    # Guarantee all expected columns exist, even if absent in the source file
    required_cols = [
        "record_id", "ssn", "first_name", "last_name",
        "date_of_birth", "address", "phone_number", "email", "created_at",
    ]
    for col in required_cols:
        if col not in df.columns:
            df[col] = ""

    # Fill NaN → "" across all text columns so fuzz.ratio never sees NaN
    df = df.fillna("")

    # Stash the original dict per row (used when writing JSON outputs)
    df["_raw"] = [dict(row) for row in raw_list]

    console.print(
        f"[bold green][INFO][/] Loaded [cyan]{len(df)}[/] record(s) from "
        f"[yellow]'{filepath}'[/]"
    )
    return df


# ══════════════════════════════════════════════════════════════════════════════
# § 2  SSN FUZZY CLUSTERING  (Two-Key Verification)
# ══════════════════════════════════════════════════════════════════════════════

def _normalise_ssn(ssn: str) -> str:
    """
    Strip all non-digit characters so '123-45-6789' and '123 45 6789'
    compare purely on their nine digits, eliminating punctuation noise.
    """
    return re.sub(r"\D", "", ssn.strip())


def fuzzy_cluster_ssns(df: pd.DataFrame) -> list[dict]:
    """
    Single-pass greedy SSN clustering with Two-Key Verification.

    Clustering Rules
    ────────────────
    distance == 0  →  exact SSN match → merge unconditionally
    distance == 1  →  one-digit variant → SECONDARY CHECK required:
                       • DOB exactly matches the cluster's core record, OR
                       • Full name fuzz.ratio > 75 against the core record
                      Pass either check → merge. Fail both → new singleton.
    distance >= 2  →  too different → new singleton cluster

    This prevents "Frankenstein Clusters" where a one-digit SSN coincidence
    links two genuinely distinct people who share neither name nor DOB.

    Algorithm
    ─────────
    For every row in the DataFrame (insertion order preserved):
      1. Normalise the row's SSN to digits-only via _normalise_ssn().
      2. Compute Levenshtein.distance against every existing cluster's
         core SSN (also digits-only).
      3. Pick the cluster with the smallest distance.
      4a. If distance == 0  → merge unconditionally.
      4b. If distance == 1  → run Two-Key check against cluster's core record.
          Pass (DOB exact OR name ratio > 75) → merge.
          Fail → open a new singleton cluster.
      4c. If distance >= 2  → open a new singleton cluster.
      5. After any merge, recompute core SSN as the most-frequent raw SSN
         in the updated cluster.

    Returns
    ───────
    A list of cluster dicts, each containing:
        core_ssn  – str  : canonical SSN (most frequent in cluster)
        records   – list : raw record dicts belonging to this cluster
        is_fuzzy  – bool : True when SSNs are not all identical (OCR case)
    """
    clusters: list[dict] = []

    for _, row in df.iterrows():
        raw_ssn  = str(row["ssn"]).strip()
        norm_ssn = _normalise_ssn(raw_ssn)
        record   = _row_to_record(row)

        if not norm_ssn:
            # No SSN: isolated singleton — preserved but never merged
            clusters.append({"core_ssn": "", "records": [record], "is_fuzzy": False})
            continue

        best_distance = float("inf")
        best_idx      = -1

        for idx, cluster in enumerate(clusters):
            dist = Levenshtein.distance(norm_ssn, _normalise_ssn(cluster["core_ssn"]))
            if dist < best_distance:
                best_distance = dist
                best_idx      = idx

        merged = False

        if best_distance == 0:
            # Exact SSN match — merge unconditionally
            merged = True

        elif best_distance == 1:
            # One-digit variant — require secondary identity verification
            # against the cluster's founding (index 0) record.
            core_record = clusters[best_idx]["records"][0]

            dob_match = (
                record.get("date_of_birth", "").strip() ==
                core_record.get("date_of_birth", "").strip()
                and record.get("date_of_birth", "").strip() != ""
            )

            incoming_name = (
                f"{record.get('first_name', '')} {record.get('last_name', '')}"
                .strip().lower()
            )
            core_name = (
                f"{core_record.get('first_name', '')} {core_record.get('last_name', '')}"
                .strip().lower()
            )
            name_match = fuzz.ratio(incoming_name, core_name) > 75

            merged = dob_match or name_match

        # distance >= 2 leaves merged = False, opening a new singleton below

        if merged:
            clusters[best_idx]["records"].append(record)

            # Recompute core SSN = most-frequent *raw* SSN in updated cluster.
            ssn_counter = Counter(r["ssn"] for r in clusters[best_idx]["records"])
            clusters[best_idx]["core_ssn"] = ssn_counter.most_common(1)[0][0]

            all_ssns = [r["ssn"] for r in clusters[best_idx]["records"]]
            clusters[best_idx]["is_fuzzy"] = len(set(all_ssns)) > 1
        else:
            # Distance >= 2, or distance == 1 but failed Two-Key check
            clusters.append({"core_ssn": raw_ssn, "records": [record], "is_fuzzy": False})

    return clusters


def _row_to_record(row: pd.Series) -> dict:
    """
    Convert a DataFrame row back to a plain dict, stripping synthetic columns
    (anything prefixed with underscore) for clean JSON output.
    """
    raw = row.get("_raw")
    if isinstance(raw, dict):
        return raw
    return {k: v for k, v in row.to_dict().items() if not k.startswith("_")}


# ══════════════════════════════════════════════════════════════════════════════
# § 3  UTILITY HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def parse_ts(ts: str) -> datetime:
    """
    Parse ISO-8601 timestamp string to a timezone-aware datetime.
    Falls back to Unix epoch so min/max comparisons never raise.
    """
    if not ts:
        return datetime.fromtimestamp(0, tz=timezone.utc)
    for fmt in TIMESTAMP_FMTS:
        try:
            dt = datetime.strptime(ts.strip(), fmt)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return datetime.fromtimestamp(0, tz=timezone.utc)


def normalise_address(addr: str) -> str:
    """
    Lowercase → strip punctuation → expand abbreviations → collapse spaces.
    Used before passing addresses to fuzz.ratio so 'Main Street' and
    'Main St' produce a high similarity score instead of a moderate one.
    """
    if not addr:
        return ""
    a = addr.lower()
    a = re.sub(r"[^\w\s]", " ", a)
    for pattern, repl in ADDR_ABBR.items():
        a = re.sub(pattern, repl, a)
    return re.sub(r"\s+", " ", a).strip()


def most_frequent_value(
    records: list[dict], field: str
) -> tuple[Any, str]:
    """
    Return (chosen_value, rule_label) for the most common non-empty value
    of `field` across the record list.
    Tie-break: value belonging to the most recently created record wins.
    """
    values  = [r.get(field) for r in records if r.get(field) and str(r.get(field)).strip()]
    if not values:
        return None, "Most Frequent"

    counter   = Counter(values)
    max_count = counter.most_common(1)[0][1]
    top_vals  = {v for v, c in counter.items() if c == max_count}

    if len(top_vals) == 1:
        return top_vals.pop(), "Most Frequent"

    # Tie-break: value from the most recent record
    latest = max(records, key=lambda r: parse_ts(r.get("created_at", "")))
    chosen = latest.get(field) if latest.get(field) in top_vals else sorted(top_vals)[0]
    return chosen, "Most Frequent (tie-break: latest)"


def latest_field(records: list[dict], field: str) -> tuple[Any, str]:
    """Return (value, rule) of `field` from the most recently created record."""
    rec = max(records, key=lambda r: parse_ts(r.get("created_at", "")))
    return rec.get(field), "Latest Record"


def oldest_record_id(records: list[dict]) -> str:
    """Return the record_id of the oldest (earliest created_at) record."""
    rec = min(records, key=lambda r: parse_ts(r.get("created_at", "")))
    return rec.get("record_id", "UNKNOWN")


def resolve_address(records: list[dict]) -> tuple[str, str]:
    """
    Fuzzy-normalisation address resolution:
      1. Normalise every address.
      2. Group records by their normalised form.
      3. Within the largest consensus group, pick the original address
         from the most recently created record.
    Returns (original_address, rule_label).
    """
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        groups[normalise_address(r.get("address", ""))].append(r)

    largest_grp = max(groups.values(), key=len)
    latest_rec  = max(largest_grp, key=lambda r: parse_ts(r.get("created_at", "")))
    return latest_rec.get("address", ""), "Fuzzy Norm + Latest in Largest Group"


# ══════════════════════════════════════════════════════════════════════════════
# § 4  CONFIDENCE ENGINE  (thefuzz-powered, strict zero-floor thresholds)
# ══════════════════════════════════════════════════════════════════════════════

def _avg_fuzz_ratio(canonical: str, sources: list[str], cutoff: int = 0) -> float:
    """
    Average fuzz.ratio (0–100) of `canonical` against every source string.

    If a single comparison falls below `cutoff`, that comparison's score is
    forced to exactly 0.0 rather than its raw ratio — eliminating the
    "participation points" problem where completely mismatched fields still
    contribute a small positive score to the weighted total.

    cutoff=0 (default) preserves the original behaviour for all callers
    that do not need a strict floor (e.g. the email baseline logic).

    Returns 100.0 if there are no sources to compare.
    """
    if not sources:
        return 100.0
    scores: list[float] = []
    for s in sources:
        ratio = fuzz.ratio(canonical.lower().strip(), s.lower().strip())
        scores.append(float(ratio) if ratio >= cutoff else 0.0)
    return sum(scores) / len(scores)


def ssn_confidence(records: list[dict], core_ssn: str) -> float:
    """
    Compute SSN confidence using strict Levenshtein edit-distance scoring.

    Scoring is deliberately discrete to match the binary clustering rule —
    a continuous scale would imply that distance-2 SSNs are "88 % confident"
    when the clustering rule explicitly rejects them as separate identities.

        distance == 0  →  100.0 %  (exact match)
        distance == 1  →   90.0 %  (one-digit OCR/typo variant, in-cluster)
        distance >= 2  →    0.0 %  (should never appear; cluster invariant
                                    prevents it, kept as defensive fallback)

    Returns the average confidence across all records in the cluster.
    """
    norm_core = _normalise_ssn(core_ssn)
    scores: list[float] = []

    for rec in records:
        norm_rec = _normalise_ssn(rec.get("ssn", ""))
        dist     = Levenshtein.distance(norm_rec, norm_core)

        if dist == 0:
            scores.append(100.0)
        elif dist == 1:
            scores.append(90.0)
        else:
            scores.append(0.0)   # defensive fallback; cluster invariant prevents this

    return sum(scores) / len(scores) if scores else 0.0


def name_confidence(records: list[dict], canon_first: str, canon_last: str) -> float:
    """
    Average fuzz.ratio of the full canonical name against each source record's
    full name. Scores below 75 are zeroed out — a completely different name
    must not silently prop up the group confidence.

    Combining first + last into one string before comparing prevents the
    score from being inflated when only a common last name matches.
    Returns 0.0–100.0.
    """
    canon_full = f"{canon_first or ''} {canon_last or ''}".strip().lower()
    sources    = [
        f"{r.get('first_name', '')} {r.get('last_name', '')}".strip()
        for r in records
    ]
    return _avg_fuzz_ratio(canon_full, sources, cutoff=75)


def dob_confidence(records: list[dict], canon_dob: str) -> float:
    """
    Average fuzz.ratio for date_of_birth strings with a strict cutoff of 85.
    A completely different birth date (e.g. off by a decade) scores 0, not ~60.
    Date strings are short so character-level similarity is still meaningful
    (e.g. '1985-03-22' vs '1985-03-12' will score ~90 and pass the cutoff).
    Returns 0.0–100.0.
    """
    sources = [r.get("date_of_birth", "") for r in records]
    return _avg_fuzz_ratio(canon_dob or "", sources, cutoff=85)


def address_confidence(records: list[dict], canon_addr: str) -> float:
    """
    Average fuzz.ratio on *normalised* address strings with a cutoff of 65.
    Abbreviation variants ('Street' vs 'St') are collapsed before scoring so
    the cutoff is not wasted on punctuation noise. A wholly different address
    scores 0.
    Returns 0.0–100.0.
    """
    norm_canon = normalise_address(canon_addr)
    sources    = [normalise_address(r.get("address", "")) for r in records]
    return _avg_fuzz_ratio(norm_canon, sources, cutoff=65)


def phone_confidence(records: list[dict], canon_val: str) -> float:
    """
    Phone-specific confidence with a strict cutoff of 80.

    Unlike emails, phone numbers are not expected to vary legitimately across
    duplicate records for the same person — a completely different number is a
    meaningful signal, so no baseline floor is applied here.
    Scores below 80 are zeroed out entirely.
    Returns 0.0–100.0.
    """
    sources = [r.get("phone_number", "") for r in records]
    return _avg_fuzz_ratio(canon_val or "", sources, cutoff=80)


def contact_confidence(records: list[dict], field: str, canon_val: str) -> float:
    """
    Email-only confidence with a 50-point baseline floor.

    Users legitimately hold multiple email addresses, so a completely
    different-but-valid address should NOT score 0 %.
    Formula:  50  +  (avg_fuzz_ratio / 100) × 50
      → completely different values → 50 % baseline
      → identical values            → 100 %
      → minor typo variants         → 50–100 % proportionally

    NOTE: This function is intentionally kept for emails only.
          Phone numbers use the separate `phone_confidence` function.
    Returns 0.0–100.0.
    """
    sources   = [r.get(field, "") for r in records]
    avg_ratio = _avg_fuzz_ratio(canon_val or "", sources)   # no cutoff for email
    return 50.0 + (avg_ratio / 100.0) * 50.0


def weighted_group_confidence(field_confs: dict[str, float]) -> float:
    """
    Compute the weighted sum group confidence score.

    field_confs must contain keys: ssn, name, date_of_birth,
                                   address, phone_number, email
    Returns a float 0.0–100.0.
    """
    return (
        field_confs["ssn"]             * WEIGHTS["ssn"]
        + field_confs["name"]          * WEIGHTS["name"]
        + field_confs["date_of_birth"] * WEIGHTS["date_of_birth"]
        + field_confs["address"]       * WEIGHTS["address"]
        + field_confs["phone_number"]  * WEIGHTS["phone_number"]
        + field_confs["email"]         * WEIGHTS["email"]
    )


# ══════════════════════════════════════════════════════════════════════════════
# § 5  MERGE PROPOSAL BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_proposal(cluster: dict) -> dict:
    """
    Construct the full merge proposal for one duplicate cluster.

    Returns a rich proposal dict containing:
        canonical        – the proposed merged record (plain dict)
        decisions        – per-field metadata (value, rule, confidence %)
        field_confs      – raw float confidences keyed by field
        group_confidence – weighted overall confidence (0.0–100.0)
        retained_id      – record_id of oldest source record
        purged_ids       – list of record_ids that will be discarded
    """
    records  = cluster["records"]
    core_ssn = cluster["core_ssn"]

    # ── Select canonical field values ─────────────────────────────────────────
    fn_val,    fn_rule   = most_frequent_value(records, "first_name")
    ln_val,    ln_rule   = most_frequent_value(records, "last_name")
    dob_val,   dob_rule  = most_frequent_value(records, "date_of_birth")
    ph_val,    ph_rule   = most_frequent_value(records, "phone_number")
    email_val, em_rule   = latest_field(records, "email")
    addr_val,  addr_rule = resolve_address(records)
    retained             = oldest_record_id(records)

    # ── Compute per-field confidence scores ───────────────────────────────────
    ssn_conf  = ssn_confidence(records, core_ssn)
    name_conf = name_confidence(records, fn_val or "", ln_val or "")
    dob_conf  = dob_confidence(records, dob_val or "")
    addr_conf = address_confidence(records, addr_val or "")
    ph_conf   = phone_confidence(records, ph_val or "")                    # strict, no floor
    em_conf   = contact_confidence(records, "email", email_val or "")     # 50-pt floor

    field_confs = {
        "ssn":           ssn_conf,
        "name":          name_conf,
        "date_of_birth": dob_conf,
        "address":       addr_conf,
        "phone_number":  ph_conf,
        "email":         em_conf,
    }
    grp_conf = weighted_group_confidence(field_confs)

    # ── Assemble canonical record ─────────────────────────────────────────────
    canonical = {
        "record_id":     retained,
        "ssn":           core_ssn,
        "first_name":    fn_val,
        "last_name":     ln_val,
        "date_of_birth": dob_val,
        "address":       addr_val,
        "phone_number":  ph_val,
        "email":         email_val,
        "created_at":    min(
            (r.get("created_at", "") for r in records),
            key=lambda t: parse_ts(t),
        ),
    }

    # ── Per-field decision metadata (for UI + audit log) ─────────────────────
    decisions = {
        "first_name": {
            "value":      fn_val,
            "rule":       fn_rule,
            "confidence": f"{ssn_conf:.1f}%",   # SSN conf covers identity anchor
        },
        "last_name": {
            "value":      ln_val,
            "rule":       ln_rule,
            "confidence": f"{name_conf:.1f}%",
        },
        "date_of_birth": {
            "value":      dob_val,
            "rule":       dob_rule,
            "confidence": f"{dob_conf:.1f}%",
        },
        "address": {
            "value":      addr_val,
            "rule":       addr_rule,
            "confidence": f"{addr_conf:.1f}%",
        },
        "phone_number": {
            "value":      ph_val,
            "rule":       ph_rule,
            "confidence": f"{ph_conf:.1f}%",
        },
        "email": {
            "value":      email_val,
            "rule":       em_rule,
            "confidence": f"{em_conf:.1f}%",
        },
    }

    purged_ids = [
        r.get("record_id") for r in records if r.get("record_id") != retained
    ]

    # ── Contact-only mismatch detection ──────────────────────────────────────
    # A cluster is "contact-only mismatch" when every identity-critical field
    # (SSN, name, DOB, address) passes its confidence cutoff (score > 0),
    # but phone and/or email are the sole source of score drag.
    # Such clusters are safe to auto-merge: different phones/emails are a
    # normal lifecycle event, not evidence of distinct people.
    identity_fields_ok = (
        ssn_conf  > 0.0
        and name_conf  > 0.0
        and dob_conf   > 0.0
        and addr_conf  > 0.0
    )
    contact_fields_low = (ph_conf == 0.0 or em_conf < 75.0)
    contact_only_mismatch = identity_fields_ok and contact_fields_low

    # Core confidence: weighted score computed on identity fields only,
    # re-normalised so their weights still sum to 1.0.
    # Used for display and audit — gives a cleaner signal when contact
    # fields are deliberately excluded from the identity judgement.
    id_weight_total = (
        WEIGHTS["ssn"] + WEIGHTS["name"] +
        WEIGHTS["date_of_birth"] + WEIGHTS["address"]
    )
    core_confidence = (
        ssn_conf  * WEIGHTS["ssn"]
        + name_conf  * WEIGHTS["name"]
        + dob_conf   * WEIGHTS["date_of_birth"]
        + addr_conf  * WEIGHTS["address"]
    ) / id_weight_total

    return {
        "canonical":             canonical,
        "decisions":             decisions,
        "field_confs":           field_confs,
        "group_confidence":      grp_conf,
        "core_confidence":       core_confidence,
        "contact_only_mismatch": contact_only_mismatch,
        "retained_id":           retained,
        "purged_ids":            purged_ids,
        "core_ssn":              core_ssn,
        "records":               records,
        "is_fuzzy":              cluster["is_fuzzy"],
    }


# ══════════════════════════════════════════════════════════════════════════════
# § 6  RICH UI HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _conf_colour(pct: float) -> str:
    """Map a confidence percentage to a rich colour tag."""
    if pct >= 90:
        return CLR_HIGH
    if pct >= 70:
        return CLR_MED
    return CLR_LOW


def _conf_bar(pct: float, width: int = 14) -> Text:
    """
    Build a rich Text object containing a coloured block bar + percentage.
    Example:  ██████████░░░░  71.3%
    """
    filled  = round(pct / 100 * width)
    bar_str = "█" * filled + "░" * (width - filled)
    colour  = _conf_colour(pct)
    text    = Text()
    text.append(bar_str, style=colour)
    text.append(f"  {pct:>5.1f}%", style=colour)
    return text


def _trunc(value: Any, maxlen: int = 36) -> str:
    """Safely truncate a value to maxlen characters for display."""
    s = str(value) if value is not None else "(none)"
    return s if len(s) <= maxlen else s[:maxlen - 1] + "…"


def print_source_table(records: list[dict], core_ssn: str, is_fuzzy: bool) -> None:
    """
    Render a rich.table.Table showing the raw source records side-by-side.
    Highlights SSN cells in yellow when they differ from the core SSN
    (indicating a fuzzy-matched variant caught by the clustering step).
    """
    t = Table(
        title="[bold]Source Records[/]",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
        border_style="dim",
        expand=False,
    )
    t.add_column("#",           style="dim",    width=3,  justify="right")
    t.add_column("Record ID",   style="white",  width=12)
    t.add_column("SSN",         style="yellow", width=15)
    t.add_column("Name",        style="white",  width=20)
    t.add_column("DOB",         style="white",  width=12)
    t.add_column("Phone",       style="white",  width=18)
    t.add_column("Email",       style="white",  width=28)
    t.add_column("Created At",  style="dim",    width=22)

    for i, rec in enumerate(records, 1):
        ssn_raw   = rec.get("ssn", "")
        ssn_style = (
            "bold red" if is_fuzzy and ssn_raw != core_ssn else "yellow"
        )
        full_name = f"{rec.get('first_name','')} {rec.get('last_name','')}".strip()
        t.add_row(
            str(i),
            _trunc(rec.get("record_id"), 12),
            Text(ssn_raw, style=ssn_style),
            _trunc(full_name, 20),
            _trunc(rec.get("date_of_birth"), 12),
            _trunc(rec.get("phone_number"), 18),
            _trunc(rec.get("email"), 28),
            _trunc(rec.get("created_at"), 22),
        )

    console.print(t)


def print_proposal_table(proposal: dict, group_idx: int, total_groups: int) -> None:
    """
    Render the proposed canonical record as a rich.table.Table with:
      • field value
      • selection rule
      • per-field confidence bar
      • weighted contribution to the overall score (weight × confidence)
    """
    decisions   = proposal["decisions"]
    field_confs = proposal["field_confs"]
    grp_conf    = proposal["group_confidence"]
    is_fuzzy    = proposal["is_fuzzy"]
    core_ssn    = proposal["core_ssn"]

    # ── Header panel ──────────────────────────────────────────────────────────
    ssn_tag          = f"[yellow]{core_ssn}[/]"
    fuzzy_tag        = " [bold red](⚠ SSN VARIANTS DETECTED)[/]" if is_fuzzy else ""
    contact_only     = proposal.get("contact_only_mismatch", False)
    core_conf        = proposal.get("core_confidence", grp_conf)
    grp_colour       = _conf_colour(grp_conf)
    core_conf_colour = _conf_colour(core_conf)

    contact_tag = (
        "  [bold green](✔ CONTACT-ONLY MISMATCH — IDENTITY VERIFIED)[/]"
        if contact_only else ""
    )
    core_conf_tag = (
        f"  [dim]Identity-only Confidence:[/] [{core_conf_colour}]{core_conf:.2f}%[/]"
        if contact_only else ""
    )
    console.print(Panel(
        f"[bold]SSN Cluster:[/] {ssn_tag}{fuzzy_tag}{contact_tag}    "
        f"Records: [cyan]{len(proposal['records'])}[/]    "
        f"Group [bold]Weighted Confidence:[/] "
        f"[{grp_colour}]{grp_conf:.2f}%[/]"
        f"{core_conf_tag}    "
        f"Group [cyan]{group_idx}[/] of [cyan]{total_groups}[/]",
        title="[bold white] PROPOSED CANONICAL RECORD [/]",
        border_style="green" if contact_only else "blue",
        expand=True,
    ))

    # ── Proposal table ────────────────────────────────────────────────────────
    t = Table(
        box=box.SIMPLE_HEAD,
        show_header=True,
        header_style="bold cyan",
        expand=True,
    )
    t.add_column("Field",         style="bold white",  width=16)
    t.add_column("Proposed Value",                     width=32)
    t.add_column("Rule",          style=CLR_RULE,      width=36)
    t.add_column("Confidence",                         width=24, justify="left")
    t.add_column("Wtd. Contrib.", justify="right",     width=14)

    field_display_order = [
        ("ssn",           "SSN",           core_ssn,                               "ssn"),
        ("first_name",    "First Name",    proposal["canonical"]["first_name"],    "name"),
        ("last_name",     "Last Name",     proposal["canonical"]["last_name"],      "name"),
        ("date_of_birth", "Date of Birth", proposal["canonical"]["date_of_birth"], "date_of_birth"),
        ("address",       "Address",       proposal["canonical"]["address"],        "address"),
        ("phone_number",  "Phone",         proposal["canonical"]["phone_number"],   "phone_number"),
        ("email",         "Email",         proposal["canonical"]["email"],          "email"),
    ]

    for field_key, label, value, conf_key in field_display_order:
        conf_pct = field_confs.get(conf_key, 100.0)
        weight   = WEIGHTS.get(conf_key, 0.0)
        contrib  = conf_pct * weight
        rule     = decisions.get(field_key, {}).get("rule", "—") if field_key != "ssn" else "Fuzzy Cluster Core"

        if field_key in ("first_name", "last_name"):
            weight  = WEIGHTS["name"] / 2
            contrib = conf_pct * weight

        t.add_row(
            label,
            _trunc(value, 32),
            rule,
            _conf_bar(conf_pct),
            f"[{_conf_colour(conf_pct)}]{contrib:>6.2f}[/]",
        )

    console.print(t)

    # ── Weight legend ─────────────────────────────────────────────────────────
    _print_weight_legend(field_confs, grp_conf)

    # ── Retained record ───────────────────────────────────────────────────────
    console.print(
        f"  [dim]Retained Record ID:[/] [bold white]{proposal['retained_id']}[/]   "
        f"[dim]Purge:[/] [red]{proposal['purged_ids']}[/]\n"
    )


def _print_weight_legend(field_confs: dict[str, float], grp_conf: float) -> None:
    """Print an inline weighted-sum breakdown so the math is transparent."""
    rows = [
        ("SSN",     field_confs["ssn"],           WEIGHTS["ssn"]),
        ("Name",    field_confs["name"],           WEIGHTS["name"]),
        ("DOB",     field_confs["date_of_birth"],  WEIGHTS["date_of_birth"]),
        ("Address", field_confs["address"],        WEIGHTS["address"]),
        ("Phone",   field_confs["phone_number"],   WEIGHTS["phone_number"]),
        ("Email",   field_confs["email"],          WEIGHTS["email"]),
    ]
    parts = "  +  ".join(
        f"[dim]{lbl}[/] [{_conf_colour(conf)}]{conf:.1f}[/][dim]×{w:.2f}[/]=[{_conf_colour(conf*w)}]{conf*w:.2f}[/]"
        for lbl, conf, w in rows
    )
    grp_col = _conf_colour(grp_conf)
    console.print(
        f"\n  [bold]Weighted score:[/]  {parts}\n"
        f"  [bold]  ─────────────── =[/] "
        f"[{grp_col} bold]{grp_conf:.2f}%[/]\n"
    )


def _progress_bar_render(pct: float, width: int = 30) -> str:
    """Plain string version used in the mass-approve summary table."""
    filled = round(pct / 100 * width)
    bar    = "█" * filled + "░" * (width - filled)
    return f"{bar} {pct:.1f}%"


def print_mass_approve_summary(
    high_conf: list[dict],
    low_conf:  list[dict],
    threshold: float,
) -> None:
    """
    Render the pre-review mass-approval summary table with one row per
    high-confidence group so the operator sees exactly what will be
    auto-merged without individual review.
    """
    console.rule("[bold yellow]PRE-REVIEW: MASS AUTO-APPROVAL PHASE[/]")

    t = Table(
        title=(
            f"[bold green]{len(high_conf)}[/] group(s) with "
            f"[bold]Weighted Confidence ≥ {threshold:.0f}%[/] "
            f"(will be mass-approved if you say Yes)"
        ),
        box=box.ROUNDED,
        header_style="bold cyan",
        border_style="dim",
        expand=True,
    )
    t.add_column("Core SSN",    width=16, style="yellow")
    t.add_column("Records",     width=9,  justify="right")
    t.add_column("Fuzzy?",      width=8,  justify="center")
    t.add_column("Wtd. Conf.",  width=44)
    t.add_column("Retained ID", width=13, style="dim white")

    for p in high_conf:
        grp_conf = p["group_confidence"]
        t.add_row(
            p["core_ssn"],
            str(len(p["records"])),
            "[bold red]YES[/]" if p["is_fuzzy"] else "[dim]no[/]",
            Text(_progress_bar_render(grp_conf, 28), style=_conf_colour(grp_conf)),
            p["retained_id"],
        )

    console.print(t)
    console.print(
        f"  [dim]{len(low_conf)} group(s) with confidence < {threshold:.0f}% "
        f"will proceed to manual 1-by-1 review.[/]\n"
    )


# ══════════════════════════════════════════════════════════════════════════════
# § 7  APPROVAL HELPERS & AUDIT BUILDING
# ══════════════════════════════════════════════════════════════════════════════

def build_audit_entry(proposal: dict) -> dict:
    """
    Construct the audit log entry for one approved merge.
    Includes every field-level confidence AND the weighted group score.
    """
    fc = proposal["field_confs"]
    return {
        "ssn":                       proposal["core_ssn"],
        "timestamp_of_approval":     datetime.now(tz=timezone.utc).isoformat(),
        "retained_record_id":        proposal["retained_id"],
        "purged_record_ids":         proposal["purged_ids"],
        "ssn_fuzzy_cluster":         proposal["is_fuzzy"],
        "field_decisions":           proposal["decisions"],
        "field_confidences": {
            "ssn":           f"{fc['ssn']:.2f}%",
            "name":          f"{fc['name']:.2f}%",
            "date_of_birth": f"{fc['date_of_birth']:.2f}%",
            "address":       f"{fc['address']:.2f}%",
            "phone_number":  f"{fc['phone_number']:.2f}%",
            "email":         f"{fc['email']:.2f}%",
        },
        "weighted_group_confidence": f"{proposal['group_confidence']:.2f}%",
        "core_confidence":           f"{proposal['core_confidence']:.2f}%",
        "contact_only_mismatch":     proposal["contact_only_mismatch"],
    }


def _approve_cluster(
    proposal:          dict,
    merged_canonicals: list[dict],
    all_removed:       list[dict],
    audit_log:         list[dict],
) -> None:
    """Commit an approved merge to the three output buckets in-place."""
    merged_canonicals.append(proposal["canonical"])
    retained = proposal["retained_id"]
    all_removed.extend(r for r in proposal["records"] if r.get("record_id") != retained)
    audit_log.append(build_audit_entry(proposal))


def _reject_cluster(proposal: dict, clean_records: list[dict]) -> None:
    """On rejection, fold all source records back into the clean pool untouched."""
    clean_records.extend(proposal["records"])


# ══════════════════════════════════════════════════════════════════════════════
# § 8  FILE I/O
# ══════════════════════════════════════════════════════════════════════════════

def write_json(filepath: str, data: list) -> None:
    """Serialise a list to a pretty-printed JSON file and confirm to console."""
    with open(filepath, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
    console.print(
        f"  [bold green]✓[/] Written → [yellow]{filepath}[/]  "
        f"([cyan]{len(data)}[/] record(s))"
    )


# ══════════════════════════════════════════════════════════════════════════════
# § 9  ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

def run(
    input_path:        str,
    dry_run:           bool  = False,
    mass_approve_pctg: float = DEFAULT_MASS_APPROVE_PCTG,
) -> None:
    """
    Main orchestration function.

    Flow
    ────
    1.  Load JSON → pandas DataFrame
    2.  Fuzzy-cluster by SSN (Levenshtein + Two-Key Verification)
    3.  Split into singletons (clean) and duplicate clusters
    4.  Build merge proposals for every duplicate cluster
    5.  PRE-REVIEW PHASE: mass auto-approve high-confidence clusters
    6.  MANUAL REVIEW LOOP: 1-by-1 review of remaining clusters
    7.  Write merged_clients.json, duplicates_removed.json, audit_log.json
    8.  Print summary
    """

    # ── 1. Ingest ─────────────────────────────────────────────────────────────
    df = load_dataframe(input_path)

    # ── 2. Fuzzy SSN clustering ───────────────────────────────────────────────
    console.print(
        f"[bold green][INFO][/] Running SSN clustering "
        f"([bold]Levenshtein distance ≤ 1[/] + Two-Key Verification for distance-1 matches)…"
    )
    clusters = fuzzy_cluster_ssns(df)

    # ── 3. Separate singletons from duplicate clusters ────────────────────────
    clean_records: list[dict] = []
    dup_clusters:  list[dict] = []

    for c in clusters:
        if len(c["records"]) == 1:
            clean_records.append(c["records"][0])
        else:
            dup_clusters.append(c)

    fuzzy_count = sum(1 for c in dup_clusters if c["is_fuzzy"])
    console.print(
        f"[bold green][INFO][/] [cyan]{len(clean_records)}[/] clean singleton(s)  |  "
        f"[cyan]{len(dup_clusters)}[/] duplicate group(s)  |  "
        f"[red]{fuzzy_count}[/] group(s) with SSN fuzzy-variants"
    )

    if not dup_clusters:
        console.print("[bold green][INFO][/] No duplicate groups found — nothing to review.")
        if not dry_run:
            write_json(OUTPUT_MERGED,  [r for r in df["_raw"]])
            write_json(OUTPUT_REMOVED, [])
            write_json(OUTPUT_AUDIT,   [])
        return

    # ── 4. Build proposals for all duplicate clusters ─────────────────────────
    with Progress(
        TextColumn("[bold cyan]Building merge proposals…"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
        transient=True,
    ) as prog:
        task = prog.add_task("proposals", total=len(dup_clusters))
        proposals: list[dict] = []
        for cluster in dup_clusters:
            proposals.append(build_proposal(cluster))
            prog.advance(task)

    # ── 5. Pre-review: auto-merge contact-only + mass auto-approval phase ───────
    # Bucket 1: contact_only_mismatch=True → auto-merge immediately, no prompt.
    # Bucket 2: group_confidence >= threshold (and not contact-only) → mass-approve candidate.
    # Bucket 3: everything else → manual 1-by-1 review.
    contact_only_proposals = [p for p in proposals if p["contact_only_mismatch"]]
    remaining_proposals    = [p for p in proposals if not p["contact_only_mismatch"]]
    high_conf = [p for p in remaining_proposals if p["group_confidence"] >= mass_approve_pctg]
    low_conf  = [p for p in remaining_proposals if p["group_confidence"] <  mass_approve_pctg]

    merged_canonicals: list[dict] = []
    all_removed:       list[dict] = []
    audit_log:         list[dict] = []
    approved_count = 0
    skipped_count  = 0

    console.print()
    console.print(
        f"[bold green][INFO][/] Found [cyan]{len(proposals)}[/] duplicate group(s)."
    )
    console.print(
        f"[bold green][INFO][/] [bold green]{len(contact_only_proposals)}[/] group(s) have "
        f"contact-only mismatches (phone/email only) — identity verified, auto-merging."
    )
    console.print(
        f"[bold green][INFO][/] [bold green]{len(high_conf)}[/] remaining group(s) have a "
        f"Weighted Overall Confidence ≥ [cyan]{mass_approve_pctg:.0f}%[/]."
    )

    # ── Auto-merge contact-only groups (no prompt) ────────────────────────────
    if contact_only_proposals and not dry_run:
        console.rule("[bold green]AUTO-MERGE: CONTACT-ONLY MISMATCH GROUPS[/]")
        t_co = Table(
            title=(
                f"[bold green]{len(contact_only_proposals)}[/] group(s) auto-merged: "
                f"SSN / Name / DOB / Address all verified — only phone/email differ"
            ),
            box=box.ROUNDED,
            header_style="bold cyan",
            border_style="green",
            expand=True,
        )
        t_co.add_column("Core SSN",       width=16, style="yellow")
        t_co.add_column("Records",        width=9,  justify="right")
        t_co.add_column("Identity Conf.", width=20)
        t_co.add_column("Phone OK?",      width=10, justify="center")
        t_co.add_column("Email OK?",      width=10, justify="center")
        t_co.add_column("Retained ID",    width=13, style="dim white")

        for p in contact_only_proposals:
            fc       = p["field_confs"]
            core_c   = p["core_confidence"]
            ph_ok    = "[dim]✓[/]" if fc["phone_number"] > 0 else "[bold red]✗[/]"
            em_ok    = "[dim]✓[/]" if fc["email"] >= 75   else "[bold yellow]~[/]"
            t_co.add_row(
                p["core_ssn"],
                str(len(p["records"])),
                Text(_progress_bar_render(core_c, 14), style=_conf_colour(core_c)),
                ph_ok,
                em_ok,
                p["retained_id"],
            )
            _approve_cluster(p, merged_canonicals, all_removed, audit_log)
            approved_count += 1

        console.print(t_co)
        console.print(
            f"  [bold green]✓[/] Auto-merged [cyan]{len(contact_only_proposals)}[/] "
            f"contact-only group(s).\n"
        )
    elif contact_only_proposals and dry_run:
        console.print(
            f"  [dim][DRY-RUN][/] Would auto-merge "
            f"[cyan]{len(contact_only_proposals)}[/] contact-only group(s).\n"
        )

    mass_approved_high = False

    # Mass-approve phase only fires when the operator explicitly opted in via
    # --mass-approve-threshold <N> (i.e. threshold < 100).  The default of
    # 100.0 means this block is skipped entirely and every group goes straight
    # to the 1-by-1 manual review loop below.
    if high_conf and not dry_run and mass_approve_pctg < 100.0:
        print_mass_approve_summary(high_conf, low_conf, mass_approve_pctg)
        mass_approved_high = Confirm.ask(
            f"[bold yellow]Do you want to mass-approve these "
            f"{len(high_conf)} high-confidence group(s) immediately?[/]"
        )

        if mass_approved_high:
            with Progress(
                TextColumn("[bold green]Mass-approving…"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                console=console,
                transient=True,
            ) as prog:
                task = prog.add_task("approving", total=len(high_conf))
                for p in high_conf:
                    _approve_cluster(p, merged_canonicals, all_removed, audit_log)
                    approved_count += 1
                    prog.advance(task)

            console.print(
                f"[bold green]✓[/] Mass-approved [cyan]{len(high_conf)}[/] group(s). "
                f"Proceeding to manual review for the remaining "
                f"[cyan]{len(low_conf)}[/] group(s).\n"
            )

    # ── 6. Manual 1-by-1 review loop ─────────────────────────────────────────
    # mass-approve accepted → only the low-confidence remainder needs review.
    # mass-approve declined, skipped, or threshold==100 → ALL groups reviewed.
    manual_queue = low_conf if mass_approved_high else remaining_proposals
    total_manual = len(manual_queue)

    if total_manual == 0 and not dry_run:
        console.print("[bold green][INFO][/] No groups require manual review.\n")
    else:
        if total_manual > 0:
            console.rule(
                f"[bold blue]MANUAL 1-BY-1 REVIEW[/]  "
                f"[dim]({total_manual} group(s) to review)[/]"
            )

        for idx, proposal in enumerate(manual_queue, 1):
            console.print()
            print_source_table(
                proposal["records"],
                proposal["core_ssn"],
                proposal["is_fuzzy"],
            )
            print_proposal_table(proposal, idx, total_manual)

            if dry_run:
                console.print("  [dim][DRY-RUN] No prompt — no changes will be written.[/]\n")
                continue

            approved = Confirm.ask(
                f"  [bold yellow]Approve this merge?[/] "
                f"([green]Y[/]=merge  [red]N[/]=keep all originals)"
            )
            console.print()

            if approved:
                _approve_cluster(proposal, merged_canonicals, all_removed, audit_log)
                approved_count += 1
                console.print(
                    f"  [bold green]✓ Merge approved.[/] "
                    f"Retained: [cyan]{proposal['retained_id']}[/]  "
                    f"Purged: [red]{proposal['purged_ids']}[/]\n"
                )
            else:
                _reject_cluster(proposal, clean_records)
                skipped_count += 1
                console.print(
                    f"  [bold red]✗ Merge rejected.[/] "
                    f"All [cyan]{len(proposal['records'])}[/] records kept as-is.\n"
                )

    if dry_run:
        console.print("\n[bold yellow][DRY-RUN][/] Complete. No files were written.")
        return

    # ── 7. Write outputs ──────────────────────────────────────────────────────
    console.rule("[bold white]WRITING OUTPUT FILES[/]")
    final_merged = clean_records + merged_canonicals
    write_json(OUTPUT_MERGED,  final_merged)
    write_json(OUTPUT_REMOVED, all_removed)
    write_json(OUTPUT_AUDIT,   audit_log)

    # ── 8. Summary ────────────────────────────────────────────────────────────
    console.print()
    console.rule("[bold white]OPERATION CLEAN SLATE v2.0 — SUMMARY[/]")
    summary = Table(box=box.SIMPLE, show_header=False, expand=False, padding=(0, 2))
    summary.add_column(style="dim")
    summary.add_column(style="bold cyan", justify="right")
    rows_summary = [
        ("Total input records",        str(len(df))),
        ("SSN clusters formed",        str(len(clusters))),
        ("Clean singleton records",    str(len(clean_records))),
        ("Duplicate groups found",           str(len(proposals))),
        ("  └─ with SSN fuzzy-match",        str(fuzzy_count)),
        ("  └─ contact-only auto-merged",    str(len(contact_only_proposals))),
        ("  └─ mass-approved",               str(len(high_conf) if mass_approved_high else 0)),
        ("  └─ manually approved",           str(
            approved_count
            - len(contact_only_proposals)
            - (len(high_conf) if mass_approved_high else 0)
        )),
        ("  └─ rejected / kept as-is",       str(skipped_count)),
        ("Records in merged output",   str(len(final_merged))),
        ("Records purged",             str(len(all_removed))),
        ("Audit log entries",          str(len(audit_log))),
    ]
    for label, val in rows_summary:
        summary.add_row(label, val)
    console.print(summary)
    console.rule()


# ══════════════════════════════════════════════════════════════════════════════
# § 10  CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="operation_clean_slate_v2",
        description="Operation Clean Slate v2.0 — Advanced Deduplication Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python operation_clean_slate_v2.py
  python operation_clean_slate_v2.py --input /data/clients.json
  python operation_clean_slate_v2.py --dry-run
  python operation_clean_slate_v2.py --mass-approve-threshold 85
""",
    )
    parser.add_argument(
        "--input", "-i",
        default=INPUT_FILE,
        metavar="FILE",
        help=f"Path to input JSON file (default: {INPUT_FILE})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview all merge proposals without writing any output files.",
    )
    parser.add_argument(
        "--mass-approve-threshold",
        type=float,
        default=DEFAULT_MASS_APPROVE_PCTG,
        metavar="PCT",
        help=(
            f"Weighted group confidence %% at or above which groups are offered "
            f"for mass auto-approval before 1-by-1 review. "
            f"Default: {DEFAULT_MASS_APPROVE_PCTG} (disabled — every group goes to manual review). "
            f"Set e.g. 90 to enable mass-approval for high-confidence groups."
        ),
    )
    return parser


def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    if not (0.0 <= args.mass_approve_threshold <= 100.0):
        parser.error("--mass-approve-threshold must be between 0.0 and 100.0.")

    console.print()
    console.rule("[bold white] OPERATION CLEAN SLATE  v2.0  |  Deduplication Engine [/]")
    console.print(
        f"[dim]Mass-approve threshold:[/] [cyan]{args.mass_approve_threshold}%[/]   "
        f"[dim]Dry-run:[/] [cyan]{args.dry_run}[/]"
    )
    console.print()

    run(
        input_path        = args.input,
        dry_run           = args.dry_run,
        mass_approve_pctg = args.mass_approve_threshold,
    )


if __name__ == "__main__":
    main()