"""
policy_agent.py
===============
The Policy Agent:
  1. Accepts output from your Clinical Reader Agent (JSON dict OR plain text)
  2. Embeds the query with Gemini
  3. Searches FAISS vector store filtered by payer (and optionally disease)
  4. Sends top policy chunks + patient data to Gemini for a decision
  5. Returns: { decision, confidence, reason, policy_references, appeal_hint }

Usage (standalone test):
    python policy_agent.py
"""

import os
import pickle
import json
import re
import numpy as np
import faiss
import load_gemini_env
import google.generativeai as genai

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

load_gemini_env.load()
GEMINI_API_KEY = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
_GEMINI_CONFIGURED = False


def _ensure_gemini_configured() -> None:
    global _GEMINI_CONFIGURED
    if _GEMINI_CONFIGURED:
        return
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "Missing Gemini API key. Set GEMINI_API_KEY (or GOOGLE_API_KEY) "
            "in FILES/.env or your environment."
        )
    genai.configure(api_key=GEMINI_API_KEY)
    _GEMINI_CONFIGURED = True


EMBED_MODEL  = "models/gemini-embedding-2-preview"
DECISION_MODEL = "models/gemini-3-flash-preview"
TOP_K        = 6
VECTOR_STORE = "./vector_store"

# ICD prefix → disease name mapping
ICD_DISEASE_MAP = {
    "E11": "diabetes_type2",
    "E10": "diabetes_type2",
    "J45": "asthma",
    "J44": "asthma",
    "I10": "hypertension",
    "I11": "hypertension",
    "I12": "hypertension",
}

_DB_CACHE = {}

# ─────────────────────────────────────────────────────────────────────────────
# TOKEN BUDGET
# ─────────────────────────────────────────────────────────────────────────────

# FIX 1: Increased from 800 → 2048 so JSON is never truncated mid-output
_DECISION_MAX_OUTPUT_TOKENS = 2048
_SUMMARY_MAX_CHARS          = 600
_EVIDENCE_MAX_CHARS         = 400
_CHUNK_MAX_CHARS            = 400
_POLICY_MAX_CHUNKS          = 4

# FIX 2: Removed "response_mime_type": "application/json" — this caused silent
#         failures on some Gemini versions. We parse JSON from text instead.
_DECISION_GEN_CONFIG = {
    "temperature":       0.1,
    "max_output_tokens": _DECISION_MAX_OUTPUT_TOKENS,
    # DO NOT set response_mime_type here — it causes API errors with some models
}


# ─────────────────────────────────────────────────────────────────────────────
# LOAD PAYER-SPECIFIC VECTOR STORE
# ─────────────────────────────────────────────────────────────────────────────

def load_payer_db(payer: str, store_dir: str = VECTOR_STORE):
    payer = payer.lower().strip()
    if payer in _DB_CACHE:
        return _DB_CACHE[payer]

    payer_dir  = os.path.join(store_dir, payer)
    index_path = os.path.join(payer_dir, "index.faiss")
    meta_path  = os.path.join(payer_dir, "metadata.pkl")

    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"\n❌ No vector DB found for payer '{payer}' at '{payer_dir}/'.\n"
            f"   Run:  python ingest_policies.py  first!\n"
        )

    index = faiss.read_index(index_path)
    with open(meta_path, "rb") as f:
        metadata = pickle.load(f)

    print(f"✅ Loaded {payer.upper()} DB: {index.ntotal} vectors ({len(metadata)} chunks)")
    _DB_CACHE[payer] = (index, metadata)
    return index, metadata


# ─────────────────────────────────────────────────────────────────────────────
# INPUT NORMALISER
# ─────────────────────────────────────────────────────────────────────────────

def normalise_input(clinical_input) -> dict:
    if isinstance(clinical_input, str):
        return {
            "payer":     extract_payer_from_text(clinical_input),
            "icd_codes": extract_icd_from_text(clinical_input),
            "cpt_code":  extract_cpt_from_text(clinical_input),
            "summary":   clinical_input,
            "evidence":  ""
        }
    return {
        "payer":     clinical_input.get("payer", "unknown").lower().strip(),
        "icd_codes": clinical_input.get("icd_codes", clinical_input.get("icd_code", [])),
        "cpt_code":  clinical_input.get("cpt_code", clinical_input.get("cpt", "")),
        "summary":   clinical_input.get("summary", clinical_input.get("clinical_summary", "")),
        "evidence":  clinical_input.get("evidence", clinical_input.get("supporting_evidence", ""))
    }


def extract_payer_from_text(text: str) -> str:
    text_lower = text.lower()
    for payer in ["uhc", "aetna", "cigna"]:
        if payer in text_lower:
            return payer
    return "unknown"


def extract_icd_from_text(text: str) -> list:
    return re.findall(r'\b[A-Z]\d{2}(?:\.\d+)?\b', text)


def extract_cpt_from_text(text: str) -> str:
    matches = re.findall(r'\b\d{5}\b', text)
    return matches[0] if matches else ""


def detect_disease(icd_codes: list) -> str:
    for code in icd_codes:
        prefix = code[:3].upper()
        if prefix in ICD_DISEASE_MAP:
            return ICD_DISEASE_MAP[prefix]
    return ""


def _available_diseases_from_metadata(metadata) -> list:
    diseases = set()
    for m in metadata or []:
        d = (m.get("disease") or "").strip()
        if d:
            diseases.add(d)
    return sorted(diseases)


# ─────────────────────────────────────────────────────────────────────────────
# RETRIEVAL
# ─────────────────────────────────────────────────────────────────────────────

def embed_query(text: str) -> np.ndarray:
    _ensure_gemini_configured()
    result = genai.embed_content(
        model=EMBED_MODEL,
        content=text,
        task_type="retrieval_query"
    )
    return np.array(result["embedding"], dtype="float32").reshape(1, -1)


def retrieve_policy_chunks(
    index, metadata, query_vec, disease: str = "", top_k: int = TOP_K
) -> list:
    search_k = min(top_k * 8, index.ntotal)
    distances, indices = index.search(query_vec, search_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        chunk_meta = metadata[idx]
        score = float(dist)
        if disease and disease.lower() in chunk_meta.get("disease", "").lower():
            score *= 0.80
        results.append({**chunk_meta, "score": score})

    results.sort(key=lambda x: x["score"])
    return results[:top_k]


# ─────────────────────────────────────────────────────────────────────────────
# JSON REPAIR UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def _trim(text: str, max_chars: int) -> str:
    if not text or len(text) <= max_chars:
        return text or ""
    parts = text[:max_chars].rsplit(" ", 1)
    return (parts[0] if len(parts) > 1 else text[:max_chars]) + "…"


def _strip_markdown_fences(text: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` wrappers."""
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*",     "", text)
    text = re.sub(r"\s*```$",     "", text)
    return text.strip()


def _extract_first_json_object(text: str) -> str:
    """
    Scan for the first complete { ... } block in text.
    Handles nested braces correctly.
    """
    start = text.find("{")
    if start == -1:
        return text

    depth  = 0
    in_str = False
    esc    = False

    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start: i + 1]

    # JSON was truncated — return what we have so the heuristic parser can try
    return text[start:]


def _repair_truncated_json(text: str) -> str:
    """
    FIX 3: Close any open string/array/object brackets caused by truncation.
    This is the key fix for "Unterminated string" / "Expecting ','/" errors.
    """
    # Close any open string first
    in_str = False
    esc    = False
    for ch in text:
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if ch == '"':
            in_str = not in_str

    if in_str:
        text += '"'  # close the open string

    # Count open brackets/braces and close them
    opens = []
    in_str = False
    esc    = False
    for ch in text:
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if not in_str:
            if ch in "{[":
                opens.append(ch)
            elif ch == "}" and opens and opens[-1] == "{":
                opens.pop()
            elif ch == "]" and opens and opens[-1] == "[":
                opens.pop()

    # Close in reverse order
    closers = {"[": "]", "{": "}"}
    for bracket in reversed(opens):
        text += closers[bracket]

    return text


def _full_repair(raw: str) -> str:
    """Full repair pipeline for Gemini JSON output."""
    text = _strip_markdown_fences(raw)

    # If it's a JSON array, pull the first object out
    if text.lstrip().startswith("["):
        text = _extract_first_json_object(text)
    else:
        text = _extract_first_json_object(text)

    # Fix numeric ellipsis: 85,... → 85,
    text = re.sub(r"(\d+(?:\.\d+)?)\s*,\s*\.\.\.", r"\1", text)
    text = re.sub(r"(\d+(?:\.\d+)?)\s*\.\.\.",      r"\1", text)

    # Fix missing values shown as ...
    text = re.sub(r":\s*\.\.\.(\s*[\}\],])", r": null\1", text)

    # Fix trailing commas before } or ]
    text = re.sub(r",\s*([}\]])", r"\1", text)

    # Attempt to close truncated JSON
    text = _repair_truncated_json(text)

    # Remove trailing commas again after repair
    text = re.sub(r",\s*([}\]])", r"\1", text)

    return text.strip()


def _heuristic_parse_decision(text: str) -> dict:
    """Last-resort key/value extraction when JSON is unrecoverable."""
    src = (text or "").strip()

    def _find_str(key: str) -> str:
        m = re.search(rf'"{re.escape(key)}"\s*:\s*"([^"\n\r]*)', src)
        return m.group(1).strip() if m else ""

    def _find_int(key: str) -> int:
        m = re.search(rf'"{re.escape(key)}"\s*:\s*(\d+)', src)
        return int(m.group(1)) if m else 0

    def _find_list(key: str) -> list:
        m = re.search(rf'"{re.escape(key)}"\s*:\s*\[(.*?)\]', src, re.DOTALL)
        if not m:
            return []
        return [i.strip() for i in re.findall(r'"([^"]+)"', m.group(1)) if i.strip()]

    decision   = _find_str("decision")
    confidence = _find_int("confidence")

    # If heuristic found a real decision, trust it
    valid_decisions = {"APPROVED", "DENIED", "PENDING_MORE_INFO"}
    if decision not in valid_decisions:
        decision = "PENDING_MORE_INFO"

    return {
        "decision":          decision,
        "confidence":        confidence,
        "reason":            _find_str("reason"),
        "policy_references": _find_list("policy_references"),
        "criteria_met":      _find_list("criteria_met"),
        "criteria_missing":  _find_list("criteria_missing"),
        "appeal_hint":       _find_str("appeal_hint"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# DECISION PROMPT
# ─────────────────────────────────────────────────────────────────────────────

# FIX 4: Prompt now explicitly forbids arrays and ellipsis, requests short strings
DECISION_PROMPT = """You are a prior-authorization specialist. 
Return ONLY a single JSON object — no markdown, no array wrapper, no trailing text.

Required JSON schema (all fields mandatory, use short values):
{{
  "decision": "APPROVED" or "DENIED" or "PENDING_MORE_INFO",
  "confidence": <integer 0-100>,
  "reason": "<1-2 sentences>",
  "policy_references": ["<ref1>", "<ref2>"],
  "criteria_met": ["<item1>"],
  "criteria_missing": ["<item1>"],
  "appeal_hint": "<one sentence or empty string>"
}}

Patient data:
  Payer     : {payer}
  ICD codes : {icd_codes}
  CPT code  : {cpt_code}
  Summary   : {summary}
  Evidence  : {evidence}

Relevant policy excerpts:
{policy_text}

Decide based ONLY on the policy excerpts above. Output the JSON object now:"""


# ─────────────────────────────────────────────────────────────────────────────
# DECISION ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def make_decision(patient: dict, policy_chunks: list) -> dict:
    summary  = _trim(patient.get("summary",  "") or "", _SUMMARY_MAX_CHARS)
    evidence = _trim(patient.get("evidence", "") or "", _EVIDENCE_MAX_CHARS)

    top_chunks  = policy_chunks[:_POLICY_MAX_CHUNKS]
    policy_text = ""

    for i, chunk in enumerate(top_chunks, 1):
        raw_content = (
            chunk.get("text") or
            chunk.get("content") or
            chunk.get("page_content") or
            ""
        )
        if not raw_content:
            continue
        policy_text += f"[{i}] {_trim(raw_content, _CHUNK_MAX_CHARS)}\n"

    if not policy_text.strip():
        policy_text = "No specific policy criteria found. Evaluate based on standard medical necessity."

    prompt = DECISION_PROMPT.format(
        payer       = patient.get("payer", "UNKNOWN").upper(),
        icd_codes   = ", ".join(patient.get("icd_codes", [])) if patient.get("icd_codes") else "N/A",
        cpt_code    = patient.get("cpt_code") or "N/A",
        summary     = summary   or "Not provided",
        evidence    = evidence  or "Not provided",
        policy_text = policy_text.strip(),
    )

    print(f"   🔢 Decision prompt chars: {len(prompt)} | max output tokens: {_DECISION_MAX_OUTPUT_TOKENS}")

    raw = ""
    try:
        _ensure_gemini_configured()
        model    = genai.GenerativeModel(DECISION_MODEL)
        response = model.generate_content(prompt, generation_config=_DECISION_GEN_CONFIG)
        raw      = (getattr(response, "text", "") or "").strip()

        if not raw:
            raise ValueError("Gemini returned an empty response")

        # ── Multi-stage JSON repair ───────────────────────────────────────────
        result = None

        # Stage 1: full repair pipeline
        try:
            cleaned = _full_repair(raw)
            result  = json.loads(cleaned)
        except (json.JSONDecodeError, Exception):
            pass

        # Stage 2: extract first object, then repair
        if result is None:
            try:
                obj     = _extract_first_json_object(raw)
                cleaned = _full_repair(obj)
                result  = json.loads(cleaned)
            except (json.JSONDecodeError, Exception):
                pass

        # Stage 3: heuristic extraction (regex fallback)
        if result is None:
            print(f"⚠️  JSON parse failed — using heuristic extraction")
            result = _heuristic_parse_decision(raw)

        if not isinstance(result, dict):
            raise ValueError(f"Unexpected type from JSON parse: {type(result)}")

        # Normalise keys
        valid_decisions = {"APPROVED", "DENIED", "PENDING_MORE_INFO"}
        if result.get("decision") not in valid_decisions:
            result["decision"] = "PENDING_MORE_INFO"

        result.setdefault("confidence",        0)
        result.setdefault("reason",            "")
        result.setdefault("policy_references", [])
        result.setdefault("criteria_met",      [])
        result.setdefault("criteria_missing",  [])
        result.setdefault("appeal_hint",       "")

    except Exception as e:
        print(f"❌ Error in Gemini processing: {e}")
        if raw:
            print(f"DEBUG raw (first 200 chars): {raw[:200]}")
        result = {
            "decision":          "PENDING_MORE_INFO",
            "confidence":        0,
            "reason":            f"Gemini processing error: {e}",
            "policy_references": [],
            "criteria_met":      [],
            "criteria_missing":  ["Manual review required due to automated decision failure."],
            "appeal_hint":       "Review the retrieved policy snippets manually.",
        }

    return result


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTIVE DENIAL ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def predict_approval_probability(patient: dict) -> dict:
    score    = 50
    evidence = (patient.get("evidence") or "").lower()
    summary  = (patient.get("summary")  or "").lower()

    positive_signals = [
        "failed conservative", "mri shows", "specialist referral",
        "documented", "prior treatment", "chronic", "severe"
    ]
    negative_signals = [
        "no prior treatment", "first line", "not documented",
        "refused", "non-compliant"
    ]

    for sig in positive_signals:
        if sig in evidence or sig in summary:
            score += 8
    for sig in negative_signals:
        if sig in evidence or sig in summary:
            score -= 10

    if patient.get("icd_codes"):
        score += 10
    if patient.get("cpt_code"):
        score += 5

    score = max(5, min(95, score))

    risk_level = (
        "LOW RISK"    if score >= 70 else
        "MEDIUM RISK" if score >= 45 else
        "HIGH RISK"
    )
    return {
        "approval_probability": score,
        "risk_level":           risk_level,
        "note":                 "Pre-submission estimate based on documentation quality",
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN POLICY AGENT FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def run_policy_agent(clinical_input, vector_store_dir: str = VECTOR_STORE) -> dict:
    patient = normalise_input(clinical_input)

    payer = patient["payer"]
    print(f"\n🏥 Patient payer  : {payer.upper()}")
    print(f"🔬 ICD codes      : {patient['icd_codes']}")
    print(f"🔧 CPT code       : {patient['cpt_code']}")

    if payer == "unknown":
        return {
            "decision": "ERROR",
            "reason":   "Payer not identified. Please specify payer (uhc / cigna / aetna).",
            "pre_submission": {}
        }

    try:
        index, metadata = load_payer_db(payer, vector_store_dir)
    except FileNotFoundError as e:
        return {"decision": "ERROR", "reason": str(e)}

    disease = detect_disease(patient["icd_codes"])
    print(f"🩺 Detected disease: {disease or 'unknown (semantic search will handle it)'}")

    # FIX 5: Don't return early for unknown disease — let semantic search handle it.
    # The old code returned PENDING_MORE_INFO here, killing the pipeline for R52 etc.
    pre_score = predict_approval_probability(patient)
    print(f"\n📊 Pre-submission risk: {pre_score['risk_level']} ({pre_score['approval_probability']}%)")

    query = (
        f"Prior authorization for payer {payer}. "
        f"Diagnosis codes: {', '.join(patient['icd_codes'])}. "
        f"Procedure code: {patient['cpt_code']}. "
        f"Clinical summary: {patient['summary']}. "
        f"Supporting evidence: {patient['evidence']}."
    ).strip()

    print(f"\n🔍 Searching {payer.upper()} policy database...")
    query_vec = embed_query(query)
    chunks    = retrieve_policy_chunks(index, metadata, query_vec, disease)

    if not chunks:
        return {
            "decision":          "ERROR",
            "reason":            f"No relevant policy chunks found in {payer.upper()} database.",
            "pre_submission":    pre_score,
        }

    sources = set(c["source"] for c in chunks)
    print(f"   Found {len(chunks)} chunks from: {sources}")

    print(f"\n🤖 Running Gemini decision engine ({DECISION_MODEL})...")
    decision = make_decision(patient, chunks)

    decision["pre_submission_risk"] = pre_score
    decision["payer"]               = payer
    decision["icd_codes"]           = patient["icd_codes"]
    decision["cpt_code"]            = patient["cpt_code"]
    decision["policy_sources"]      = list(sources)

    return decision


# ─────────────────────────────────────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sample_dict = {
        "payer":     "uhc",
        "icd_codes": ["E11.9", "E11.65"],
        "cpt_code":  "99213",
        "summary":   "Patient with uncontrolled Type 2 Diabetes Mellitus. HbA1c 9.2%. "
                     "On metformin but inadequate glycemic control.",
        "evidence":  "Failed metformin monotherapy for 6 months. "
                     "Endocrinologist referral documented. Recent labs show HbA1c 9.2%."
    }

    sample_text = """
    Patient: John Doe | Payer: Aetna
    Diagnosis: Hypertension (I10) — uncontrolled despite lifestyle modification.
    CPT: 99213
    Clinical note: Patient has documented hypertension for 2 years.
    Failed first-line medication (lisinopril) due to side effects.
    Blood pressure readings consistently above 160/100 mmHg.
    Requesting prior authorization for combination antihypertensive therapy.
    """

    print("=" * 60)
    print("TEST 1 — Dict Input (UHC Diabetes)")
    print("=" * 60)
    result1 = run_policy_agent(sample_dict)
    print(json.dumps(result1, indent=2))

    print("\n" + "=" * 60)
    print("TEST 2 — Plain Text Input (Aetna Hypertension)")
    print("=" * 60)
    result2 = run_policy_agent(sample_text)
    print(json.dumps(result2, indent=2))