# Operation Clean Slate v2.0
**Production-Grade Client Deduplication Engine**

A terminal-based system for detecting, merging, and auditing duplicate client records in large financial databases. Uses AI-powered fuzzy matching, confidence scoring, and interactive review workflows to safely consolidate records while maintaining data integrity.

---

## 🎯 The Problem

Financial institutions accumulate duplicate client records through:
- Multiple onboarding entries under slightly different names or spellings
- Data migrations and imports with formatting variations
- OCR errors in SSN or address fields
- Address variants (Street vs St, Avenue vs Ave, etc.)

Duplicates cause:
- Incorrect mailings and communication failures
- Split transaction histories
- Compliance and audit failures
- Inaccurate reporting and analytics

**Solution**: Detect all duplicates by SSN, recommend optimal merges with per-field confidence scores, and allow staff to review and approve before executing the consolidation.

---

## 🚀 Key Features

### 1. **Smart Duplicate Detection**
- Exact SSN matching groups records reliably
- **Fuzzy SSN clustering** via Levenshtein edit-distance (distance ≤ 1) for OCR/typo variants
- **Two-Key verification** for distance-1 matches: requires DOB match OR name similarity (75%+) to prevent false merges
- Clusters marked as "fuzzy" when SSN variants are detected

### 2. **Intelligent Merge Recommendations**
Field-level resolution rules with AI confidence scoring:

| Field | Rule | Confidence Metric |
|-------|------|---|
| `first_name` / `last_name` | Most frequent value | Fuzzy name match ratio (75%+ cutoff) |
| `date_of_birth` | Most frequent value | Fuzzy date match (85%+ cutoff) |
| `address` | Fuzzy-normalised, most recent | Normalised address match (65%+ cutoff) |
| `phone_number` | Most frequent value | Exact/fuzzy match (80%+ cutoff, no floor) |
| `email` | Latest record | Contact confidence (50–100% scale with baseline) |

**Address fuzzy-matching** normalises variants:
- `Street` = `St` = `St.` = `street`
- `Avenue` = `Ave` = `Av` = `avenue`
- Cardinal directions: `North` = `N`, etc.

### 3. **AI-Powered Confidence Scoring**
- **Per-field confidence** (0–100%) reflects how certain the system is that each value is correct
- **Weighted group confidence** combines all fields using exact weight matrix
- **Zero-floor strict thresholds** prevent mismatched fields from "participation points"
- **Contact-only mismatch detection**: auto-identifies groups where only email/phone differ (safe to auto-merge)
- **Core confidence** isolates identity-critical fields (SSN, name, DOB, address) separately from contact info

Weight matrix (sums to 1.0):
```
SSN:           30%
Name:          25%
Date of Birth: 22%
Address:       18%
Phone:          3%
Email:          2%
```

### 4. **Interactive Terminal Review Interface**
Rich, colour-coded terminal UI showing:
- **Source Records Table**: all raw records side-by-side with SSN variants highlighted
- **Proposal Table**: recommended canonical values, selection rules, per-field confidence bars
- **Group Confidence**: weighted overall score + identity-only score
- **Approve/Reject buttons** for each group

### 5. **Mass Auto-Approval**
Pre-review phase identifies high-confidence groups (default ≥99%) for:
- Single bulk approval action instead of 1-by-1 review
- Customisable threshold via `--mass-approve-threshold`
- Clear summary of which groups will be auto-merged

### 6. **Complete Audit Trail**
Comprehensive audit log records:
- Timestamp and approval status
- Retained record ID (oldest) vs purged IDs
- Per-field decision metadata (selected value, rule, confidence %)
- Field-level confidences (JSON numeric)
- Weighted group confidence
- Fuzzy SSN detection flag
- Contact-only mismatch flag

---

## 📦 Installation

### Prerequisites
- Python 3.8+
- pip

### Setup
```bash
# Clone or download the repository
cd /path/to/OperationCleanSlate

# Install dependencies
pip install -r requirements.txt
```

**Dependencies:**
- `pandas>=2.2.0` — DataFrames and vectorised operations
- `thefuzz>=0.22.1` — Fuzzy string matching
- `python-Levenshtein>=0.25.0` — Fast edit-distance for SSN clustering (4–10× speed-up)
- `rich>=13.7.0` — Beautiful terminal UI (tables, progress bars, colour)

---

## 🎮 Usage

### Basic Workflow
```bash
# Run with default input file (operation-clean-slate.json)
python opcs.py

# Specify custom input file
python opcs.py --input /data/clients.json

# Dry-run mode (preview changes without writing output files)
python opcs.py --dry-run

# Set mass auto-approval threshold (0–100%)
python opcs.py --mass-approve-threshold 90

# Combine options
python opcs.py --input clients.json --mass-approve-threshold 85 --dry-run
```

### Input Format
JSON array of client records:
```json
[
  {
    "record_id": "REC-1001",
    "ssn": "123-45-6789",
    "first_name": "John",
    "last_name": "Doe",
    "date_of_birth": "1985-03-22",
    "address": "123 Main St, New York, NY 10001",
    "phone_number": "+1-212-555-0100",
    "email": "john.doe@example.com",
    "created_at": "2019-04-10T08:00:00Z"
  },
  { ... }
]
```

Supports ISO-8601 timestamps in multiple formats; missing fields default to empty strings.

---

## 📊 Output Files

After approval, the system generates three files:

### 1. **merged_clients.json**
Deduplicated canonical client list. One record per unique SSN with:
- Oldest `record_id` retained (or can be overridden to generate new ID)
- Consolidated field values per selection rules
- `created_at` set to oldest timestamp in the group

```json
[
  {
    "record_id": "REC-1001",
    "ssn": "123-45-6789",
    "first_name": "John",
    "last_name": "Doe",
    "date_of_birth": "1985-03-22",
    "address": "123 Main St, New York, NY 10001",
    "phone_number": "+1-212-555-0100",
    "email": "john.doe@example.com",
    "created_at": "2019-04-10T08:00:00Z"
  }
]
```

### 2. **duplicates_removed.json**
All redundant records that were purged. Enables recovery if needed.

### 3. **audit_log.json**
Complete merge history with field-level metadata:

```json
[
  {
    "ssn": "123-45-6789",
    "timestamp_of_approval": "2024-06-15T14:22:00Z",
    "retained_record_id": "REC-1001",
    "purged_record_ids": ["REC-1042", "REC-1087"],
    "ssn_fuzzy_cluster": false,
    "field_decisions": {
      "first_name": {
        "value": "John",
        "rule": "Most Frequent",
        "confidence": "91.0%"
      },
      ...
    },
    "field_confidences": {
      "ssn": "100.00%",
      "name": "91.20%",
      "date_of_birth": "96.30%",
      "address": "85.10%",
      "phone_number": "95.80%",
      "email": "75.50%"
    },
    "weighted_group_confidence": "92.34%",
    "core_confidence": "93.10%",
    "contact_only_mismatch": false
  }
]
```

---

## 🔍 Example Walkthrough

**Input**: Three records with SSN `123-45-6789` but name and address variants:

| record_id | first_name | address | phone_number | created_at |
|-----------|------------|---------|--------------|------------|
| REC-1001 | John | 123 Main Street, NY 10001 | +1-212-555-0100 | 2019-04-10 |
| REC-1042 | Jon | 123 Main St., NY 10001 | +1-212-555-0100 | 2023-11-01 |
| REC-1087 | John | 123 Main St, NY 10001 | +1-212-555-0199 | 2024-06-15 |

**System Actions**:
1. **Detects** all three as one group (same SSN) → Exact match, confidence 100%
2. **Fuzzy-matches addresses** → All three normalise to same location
3. **Recommends**:
   - `first_name`: "John" (appears 2/3 times) — confidence 91%
   - `address`: "123 Main Street, NY 10001" (latest in matched group) — confidence 85%
   - `phone_number`: "+1-212-555-0100" (appears 2/3 times) — confidence 95%
4. **Presents** in interactive UI with rule labels, confidence bars, and Approve/Reject
5. **On approval**:
   - Writes single canonical record with `record_id: REC-1001` (oldest)
   - Purges REC-1042 and REC-1087 to `duplicates_removed.json`
   - Records merge metadata with all confidence scores to `audit_log.json`

---

## 🛠️ Technical Highlights

### SSN Fuzzy Clustering with Levenshtein Distance
```
Clustering Invariant:
  distance == 0  →  100% match → merge unconditionally
  distance == 1  →  one-digit variant → Two-Key check required
  distance >= 2  →  too different → new singleton cluster
```

Two-Key verification prevents false merges when a one-digit SSN typo coincidentally links unrelated people:
- Requires **DOB exact match** OR **name similarity > 75%**
- Both checks must fail to reject the merge

### Confidence Scoring Strategy
- **SSN**: Discrete (100% exact / 90% distance-1 / 0% distance≥2)
- **Name**: Fuzzy match ratio with 75% cutoff (scores <75 → 0.0)
- **DOB**: Fuzzy match ratio with 85% cutoff
- **Address**: Match on normalised strings with 65% cutoff
- **Phone**: Fuzzy match with 80% cutoff, **no baseline floor** (strict)
- **Email**: Fuzzy match with **50% baseline floor** (users legitimately hold multiple emails)

### Zero-Floor Thresholds
Scores below per-field cutoffs collapse to 0.0, preventing completely mismatched fields from contaminating the overall confidence via small "participation points".

### Contact-Only Mismatch Detection
If identity fields (SSN, name, DOB, address) all pass their cutoffs but phone/email are the only mismatch source, the group is flagged as "contact-only mismatch" — safe to auto-merge since different contact details are a normal lifecycle event, not evidence of distinct identities.

### Field Resolution Rules are Extensible
Rules are applied in the `build_proposal()` function. Adding a new rule (e.g., "prefer earliest entry") requires only configuration changes to the selection logic, not modifications to the merge or detection pipeline.

---

## ⚙️ Configuration & Command-Line Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--input <path>` | string | `operation-clean-slate.json` | Path to input JSON file |
| `--mass-approve-threshold <N>` | float | `99.0` | Weighted confidence threshold (0–100) for auto-approval; groups at or above this score skip individual review; 100 = disabled |
| `--dry-run` | flag | off | Preview all changes without writing output files |

---

## 📋 Evaluation Criteria

✅ **Duplicate Detection**: All SSN collisions identified; single group per unique SSN  
✅ **Merge Accuracy**: Rules applied correctly; address fuzzy-matching prevents naive string equality  
✅ **Confidence Scoring**: Per-field and weighted scores calculated; zero-floor thresholds applied  
✅ **Review Interface**: Source records, proposal, rules, confidence bars, Approve/Reject shown  
✅ **Merge & Purge**: Canonical record written; duplicates removed; audit log complete  
✅ **Audit Trail**: Timestamp, field decisions, per-field confidences, weighted scores  
✅ **Code Quality**: Extensible rule engine; adding new rules requires configuration only  

---

## 🎓 Credits

**Project**: Operation Clean Slate v2.0  
**Hackathon**: BNY Career Development Program, IIT-BHU  
**Highlights**: AI-powered fuzzy matching, confidence scoring, and interactive approval workflow  

---

## 📝 License

Open source. Use and modify freely for your deduplication needs.
