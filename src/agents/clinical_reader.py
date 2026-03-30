"""
clinical_reader_agent.py
========================
MERGED Clinical Reader Agent — works in TWO modes:

  MODE 1 — CSV Patient (your teammate's approach)
    Input : patient_id (int)
    Source: patients.csv
    Flow  : CSV row → Gemini extraction → structured bundle

  MODE 2 — EHR Text Note (first agent approach)
    Input : raw EHR text (str)
    Source: free-text clinical note
    Flow  : LLM extraction → structured bundle (with regex fallback)

BOTH modes output the SAME unified dict that feeds directly into policy_agent.py

FALLBACK LOGIC (when Gemini API fails / exhausted):
  → Rule-based ICD mapping from disease name / symptoms
  → Pre-built code tables for Diabetes T2, Asthma, Hypertension
  → Regex extraction from text

CODE TYPES EXTRACTED:
  • ICD-10  — diagnosis codes
  • CPT     — procedure codes
  • HCPCS   — durable medical equipment / drugs
  • LOINC   — lab test codes
  • NDC     — drug codes (from medication names)
  • SNOMED  — clinical concept codes (inferred)
"""

import json
import os
import re
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# GEMINI SETUP  (optional — graceful fallback if key missing)
# ─────────────────────────────────────────────────────────────────────────────
try:
    import google.generativeai as genai
    _GENAI_AVAILABLE = True
except ImportError:
    genai = None
    _GENAI_AVAILABLE = False

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL   = "models/gemini-3-flash-preview"
CSV_PATH       = "patients.csv"

# ─────────────────────────────────────────────────────────────────────────────
# STATIC FALLBACK CODE TABLES
# Used when Gemini API is unavailable / exhausted
# ─────────────────────────────────────────────────────────────────────────────

DISEASE_FALLBACK: Dict[str, Dict] = {
    # ── Diabetes ──────────────────────────────────────────────────────────────
    "diabetes": {
        "icd10_codes":  ["E11.9", "E11.65", "E11.649"],
        "cpt_codes":    ["99213", "83036"],        # office visit + HbA1c
        "hcpcs_codes":  ["A4253", "E2100"],        # glucose strips + monitor
        "loinc_codes":  ["4548-4", "2339-0"],      # HbA1c + glucose
        "ndc_hint":     "metformin / insulin",
        "snomed_codes": ["44054006"],              # Type 2 diabetes
        "description":  "Type 2 Diabetes Mellitus — unspecified / with hyperglycemia",
    },
    "diabetes type 2": {
        "icd10_codes":  ["E11.9", "E11.65", "E11.649"],
        "cpt_codes":    ["99213", "83036"],
        "hcpcs_codes":  ["A4253", "E2100"],
        "loinc_codes":  ["4548-4", "2339-0"],
        "ndc_hint":     "metformin / insulin",
        "snomed_codes": ["44054006"],
        "description":  "Type 2 Diabetes Mellitus",
    },
    # ── Asthma ────────────────────────────────────────────────────────────────
    "asthma": {
        "icd10_codes":  ["J45.51", "J45.901", "J45.41"],
        "cpt_codes":    ["94640", "94060", "99213"],  # nebulizer + spirometry + visit
        "hcpcs_codes":  ["A7003", "A7005"],           # nebulizer supplies
        "loinc_codes":  ["19926-5", "20150-9"],       # FEV1 + peak flow
        "ndc_hint":     "albuterol / budesonide",
        "snomed_codes": ["195967001"],                # asthma
        "description":  "Asthma — severe persistent / exacerbation",
    },
    # ── Hypertension ──────────────────────────────────────────────────────────
    "hypertension": {
        "icd10_codes":  ["I10", "I11.9", "I13.10"],
        "cpt_codes":    ["99213", "93000"],           # office visit + ECG
        "hcpcs_codes":  ["A4670"],                   # BP cuff
        "loinc_codes":  ["8480-6", "8462-4"],        # systolic + diastolic BP
        "ndc_hint":     "lisinopril / amlodipine",
        "snomed_codes": ["38341003"],                # hypertensive disorder
        "description":  "Essential (primary) hypertension",
    },
    # ── Additional common diseases ────────────────────────────────────────────
    "pneumonia": {
        "icd10_codes":  ["J18.9", "J15.9"],
        "cpt_codes":    ["99214", "71046"],
        "hcpcs_codes":  [],
        "loinc_codes":  ["24336-0"],
        "ndc_hint":     "amoxicillin / azithromycin",
        "snomed_codes": ["233604007"],
        "description":  "Pneumonia, unspecified organism",
    },
    "copd": {
        "icd10_codes":  ["J44.1", "J44.0"],
        "cpt_codes":    ["94060", "94640", "99214"],
        "hcpcs_codes":  ["A7003"],
        "loinc_codes":  ["19926-5"],
        "ndc_hint":     "tiotropium / salmeterol",
        "snomed_codes": ["13645005"],
        "description":  "COPD with acute exacerbation",
    },
    "heart failure": {
        "icd10_codes":  ["I50.9", "I50.32"],
        "cpt_codes":    ["93306", "99214"],
        "hcpcs_codes":  [],
        "loinc_codes":  ["33762-6"],
        "ndc_hint":     "furosemide / carvedilol",
        "snomed_codes": ["84114007"],
        "description":  "Heart failure, unspecified",
    },
}

# Symptom → ICD hint mapping (last resort fallback)
SYMPTOM_ICD_MAP = {
    "Has_Fever":       ("R50.9",  "Fever, unspecified"),
    "Has_Cough":       ("R05.9",  "Cough, unspecified"),
    "Has_Fatigue":     ("R53.83", "Other fatigue"),
    "Has_Pain":        ("R52",    "Pain, unspecified"),
    "Has_Hypertension":("I10",    "Essential hypertension"),
    "Has_Diabetes":    ("E11.9",  "Type 2 diabetes mellitus"),
}

# Medicine name → rough NDC class mapping
MEDICINE_NDC_HINTS = {
    "metformin":   "NDC: metformin HCl tablet class",
    "insulin":     "NDC: insulin glargine / aspart class",
    "lisinopril":  "NDC: lisinopril tablet class",
    "amlodipine":  "NDC: amlodipine besylate tablet class",
    "albuterol":   "NDC: albuterol sulfate inhaler class",
    "budesonide":  "NDC: budesonide inhalation class",
    "furosemide":  "NDC: furosemide tablet class",
    "atorvastatin":"NDC: atorvastatin calcium tablet class",
}


# ─────────────────────────────────────────────────────────────────────────────
# REGEX PATTERNS  (used in EHR text mode fallback)
# ─────────────────────────────────────────────────────────────────────────────
_ICD_RE      = re.compile(r'\b([A-Z]\d{2}(?:\.\d{1,4})?)\b')
_CPT_RE      = re.compile(r'(?:CPT\s*(?:code\s*)?)?(\b\d{5}\b)')
_HCPCS_RE    = re.compile(r'\b([A-Z]\d{4})\b')
_LOINC_RE    = re.compile(r'\b(\d{3,5}-\d)\b')
_PAYER_RE    = re.compile(r'\b(uhc|aetna|cigna|united\s*health|blue\s*cross)\b', re.I)
_NECESSITY_RE= re.compile(r'(?:medical\s*necessity|necessity)[:\s]*([^.]+)', re.I | re.DOTALL)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _clean(val) -> str:
    s = str(val).strip()
    return "" if s.lower() in ("nan", "none", "null", "") else s


def _lookup_disease(name: str) -> Optional[Dict]:
    """Case-insensitive lookup in DISEASE_FALLBACK table."""
    key = name.lower().strip()
    # Exact match first
    if key in DISEASE_FALLBACK:
        return DISEASE_FALLBACK[key]
    # Partial match
    for k, v in DISEASE_FALLBACK.items():
        if k in key or key in k:
            return v
    return None


def _ndc_from_medicines(meds: List[Dict]) -> List[str]:
    hints = []
    for m in meds:
        name = m.get("medicine_name", "").lower()
        for drug, hint in MEDICINE_NDC_HINTS.items():
            if drug in name:
                hints.append(hint)
    return list(dict.fromkeys(hints))


def _build_medications(row: Dict) -> List[Dict]:
    meds = []
    for i in range(1, 4):
        name = _clean(row.get(f"Medicine_{i}", ""))
        if not name:
            continue
        meds.append({
            "medicine_name": name,
            "dosage":        _clean(row.get(f"Dosage_{i}",       "")),
            "frequency":     _clean(row.get(f"Frequency_{i}",    "")),
            "duration":      _clean(row.get(f"Duration_{i}",     "")),
            "instructions":  _clean(row.get(f"Instructions_{i}", "")),
        })
    return meds


def _build_symptoms(row: Dict) -> Dict:
    active, absent = [], []
    for col in ["Has_Fever", "Has_Cough", "Has_Fatigue", "Has_Pain"]:
        try:
            (active if int(row.get(col, 0)) == 1 else absent).append(col)
        except Exception:
            absent.append(col)
    return {"active": active, "absent": absent}


def _build_comorbidities(row: Dict) -> Dict:
    result = {
        "hypertension": bool(int(row.get("Has_Hypertension", 0) or 0)),
        "diabetes":     bool(int(row.get("Has_Diabetes",     0) or 0)),
        "other":        []
    }
    return result


def _symptom_icd_fallback(row: Dict) -> List[str]:
    codes = []
    for col, (code, _) in SYMPTOM_ICD_MAP.items():
        try:
            if int(row.get(col, 0)) == 1:
                codes.append(code)
        except Exception:
            pass
    return codes


# ─────────────────────────────────────────────────────────────────────────────
# GEMINI EXTRACTION  (shared by both modes)
# ─────────────────────────────────────────────────────────────────────────────

def _configure_gemini() -> Optional[Any]:
    if not _GENAI_AVAILABLE or not GEMINI_API_KEY or "YOUR_" in GEMINI_API_KEY:
        return None
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        return genai.GenerativeModel(GEMINI_MODEL)
    except Exception:
        return None


def _gemini_extract_from_text(model, note: str) -> Dict:
    """
    Ask Gemini to extract ALL code types from an EHR note.
    Returns a dict — never raises (caller handles exceptions).
    """
    prompt = f"""You are a clinical coding assistant for prior authorization.
Extract the following from the EHR note below.
Return ONLY strict JSON — no markdown, no backticks, no explanation.

Schema:
{{
  "icd10_codes":   ["<ICD-10 codes — explicit or inferred>"],
  "cpt_codes":     ["<CPT procedure codes>"],
  "hcpcs_codes":   ["<HCPCS supply/drug codes>"],
  "loinc_codes":   ["<LOINC lab/observation codes>"],
  "snomed_codes":  ["<SNOMED CT concept codes>"],
  "ndc_hints":     ["<drug name + NDC class hints>"],
  "medical_necessity_summary": "<2-4 sentence summary>",
  "supporting_evidence": ["<evidence snippet 1>", "...up to 8"],
  "patient_summary": "<one line>",
  "payer": "<uhc|aetna|cigna|unknown>",
  "confidence_score": <0.0-1.0>
}}

Rules:
- ICD-10: 1-5 codes (infer if not explicit)
- CPT: include if a procedure/test is ordered or done
- HCPCS: include for equipment, supplies, injected drugs
- LOINC: include for any lab tests
- SNOMED: include primary diagnosis concept
- If uncertain about a code, omit it — do not guess rare codes.
- payer: extract from text or return "unknown"

EHR NOTE:
{note}"""

    response = model.generate_content(
        prompt,
        generation_config={"temperature": 0.1, "response_mime_type": "application/json"}
    )
    raw = (getattr(response, "text", "") or "").strip()
    if not raw:
        try:
            raw = response.candidates[0].content.parts[0].text.strip()
        except Exception:
            raw = ""
    raw = re.sub(r"^```[a-z]*\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    data = json.loads(raw)
    return data


def _gemini_extract_from_csv_row(model, disease: str, causes: str, tips: str) -> Dict:
    """
    Ask Gemini to assign codes from CSV disease label + causes text.
    Returns a dict — never raises (caller handles exceptions).
    """
    prompt = f"""You are a clinical extraction agent for a Prior Authorization system.
Given the patient diagnosis below, return ONLY a valid JSON object.
No explanation. No markdown. No backticks. Just raw JSON.

Disease: {disease}
Disease Causes: {causes}
Health Tips: {tips}

Return exactly this structure:
{{
  "icd10_code": "<most accurate ICD-10 code>",
  "icd10_codes": ["<up to 3 ICD-10 codes>"],
  "icd10_description": "<full description>",
  "cpt_codes": ["<relevant CPT codes>"],
  "hcpcs_codes": ["<relevant HCPCS codes>"],
  "loinc_codes": ["<relevant LOINC codes>"],
  "snomed_codes": ["<SNOMED concept codes>"],
  "ndc_hints": ["<drug NDC class hints>"],
  "disease_causes_list": ["<cause1>", "<cause2>"],
  "health_tips_list": ["<tip1>", "<tip2>"],
  "confidence_score": <float 0.0 to 1.0>
}}"""

    response = model.generate_content(
        prompt,
        generation_config={"temperature": 0.1, "response_mime_type": "application/json"}
    )
    raw = (getattr(response, "text", "") or "").strip()
    raw = re.sub(r"^```[a-z]*\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    # Robust extraction of first {...} block
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end > start:
        raw = raw[start:end+1]
    return json.loads(raw)


# ─────────────────────────────────────────────────────────────────────────────
# UNIFIED OUTPUT BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def _build_unified_bundle(
    *,
    mode: str,                        # "csv" or "text"
    payer: str,
    icd10_codes: List[str],
    cpt_codes: List[str],
    hcpcs_codes: List[str],
    loinc_codes: List[str],
    snomed_codes: List[str],
    ndc_hints: List[str],
    medical_necessity_summary: str,
    supporting_evidence: List[str],
    patient_summary: str,
    confidence_score: float,
    extraction_method: str,           # "gemini" / "fallback_table" / "regex"
    # Optional CSV-specific fields
    patient_row: Optional[Dict] = None,
    disease_label: str = "",
    icd10_description: str = "",
    medications: Optional[List[Dict]] = None,
    vitals: Optional[Dict] = None,
    comorbidities: Optional[Dict] = None,
    symptoms: Optional[Dict] = None,
    polypharmacy: Optional[Dict] = None,
    health_tips: Optional[List[str]] = None,
) -> Dict:

    # Primary ICD-10 code
    primary_icd = icd10_codes[0] if icd10_codes else "UNKNOWN"

    bundle = {
        "bundle_id":          str(uuid.uuid4()),
        "generated_at":       datetime.utcnow().isoformat() + "Z",
        "mode":               mode,
        "extraction_method":  extraction_method,
        "confidence_score":   round(confidence_score, 3),
        "requires_human_review": confidence_score < 0.75,

        # ── Core fields for policy_agent.py ──────────────────────────────────
        # These are what policy_agent.normalise_input() reads
        "payer":      payer,
        "icd_codes":  icd10_codes,        # ← policy_agent key
        "cpt_code":   cpt_codes[0] if cpt_codes else "",
        "summary":    medical_necessity_summary,
        "evidence":   " | ".join(supporting_evidence),

        # ── Extended codes block ──────────────────────────────────────────────
        "codes": {
            "icd10":  icd10_codes,
            "cpt":    cpt_codes,
            "hcpcs":  hcpcs_codes,
            "loinc":  loinc_codes,
            "snomed": snomed_codes,
            "ndc":    ndc_hints,
        },

        # ── Diagnosis ─────────────────────────────────────────────────────────
        "diagnosis": {
            "primary_icd10":      primary_icd,
            "icd10_description":  icd10_description or "",
            "predicted_disease":  disease_label,
            "all_icd10_codes":    icd10_codes,
        },

        # ── Patient narrative ─────────────────────────────────────────────────
        "patient_summary":         patient_summary,
        "medical_necessity":       medical_necessity_summary,
        "supporting_evidence":     supporting_evidence,

        # ── PA context (filled by user / UI) ─────────────────────────────────
        "pa_context": {
            "payer_id":   None,
            "payer_name": payer if payer != "unknown" else None,
            "plan_type":  None,
            "member_id":  None,
            "pa_required": True,
        },
    }

    # Add CSV-specific fields if available
    if patient_row:
        pid = patient_row.get("Patient_ID", "")
        bundle["patient"] = {
            "patient_id":  int(pid) if str(pid).isdigit() else pid,
            "age":         int(patient_row.get("Age",        0) or 0),
            "gender":      _clean(patient_row.get("Gender",       "")),
            "blood_group": _clean(patient_row.get("Blood_Group",  "")),
            "weight_kg":   float(patient_row.get("Weight_kg",     0) or 0),
        }
    if vitals:
        bundle["vitals"] = vitals
    if symptoms:
        bundle["symptoms"] = symptoms
    if comorbidities:
        bundle["comorbidities"] = comorbidities
    if medications:
        bundle["medications"] = medications
        bundle["ndc_from_medications"] = _ndc_from_medicines(medications)
    if polypharmacy:
        bundle["polypharmacy"] = polypharmacy
    if health_tips:
        bundle["health_tips"] = health_tips

    return bundle


# ─────────────────────────────────────────────────────────────────────────────
# MODE 1 — CSV PATIENT
# ─────────────────────────────────────────────────────────────────────────────

def read_from_csv(patient_id: int, payer: str = "unknown", csv_path: str = CSV_PATH) -> Dict:
    """
    Extract clinical data for a patient from patients.csv.

    Args:
        patient_id : Patient_ID value in the CSV
        payer      : "uhc" / "aetna" / "cigna" (set by user in UI)
        csv_path   : path to patients.csv

    Returns:
        Unified bundle dict ready for policy_agent.run_policy_agent()
    """
    # ── Load CSV ──────────────────────────────────────────────────────────────
    try:
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
    except FileNotFoundError:
        return _error_bundle(f"CSV not found: {csv_path}", payer)

    row_df = df[df["Patient_ID"] == patient_id]
    if row_df.empty:
        # Common user behavior: entering 1, 2, 3... as a row selector.
        # If the CSV Patient_ID values are not sequential, fall back to 1-based row index.
        try:
            idx = int(patient_id) - 1
        except Exception:
            idx = -1
        if 0 <= idx < len(df):
            row_df = df.iloc[[idx]]
        else:
            return _error_bundle(f"Patient ID {patient_id} not found in CSV", payer)

    row = row_df.iloc[0].to_dict()

    disease = _clean(row.get("Predicted_Disease", ""))
    causes  = _clean(row.get("Disease_Causes",    ""))
    tips    = _clean(row.get("Personalized_Health_Tips", ""))
    meds    = _build_medications(row)
    symptoms     = _build_symptoms(row)
    comorbidities= _build_comorbidities(row)
    vitals = {
        "temperature_c": float(row.get("Temperature_C", 0) or 0),
        "heart_rate":    int(row.get("Heart_Rate",    0) or 0),
        "bp_systolic":   int(row.get("BP_Systolic",   0) or 0),
        "wbc_count":     float(row.get("WBC_Count",   0) or 0),
        "glucose_level": int(row.get("Glucose_Level", 0) or 0),
    }
    polypharmacy = {
        "risk_level":      _clean(row.get("Polypharmacy_Risk",           "")),
        "recommendation":  _clean(row.get("Polypharmacy_Recommendation", "")),
    }

    # ── Try Gemini ────────────────────────────────────────────────────────────
    model = _configure_gemini()
    gemini_data   = None
    extraction_method = "fallback_table"

    if model and disease:
        try:
            gemini_data = _gemini_extract_from_csv_row(model, disease, causes, tips)
            extraction_method = "gemini"
            print(f"✅ Gemini extraction succeeded for patient {patient_id}")
        except Exception as e:
            print(f"⚠️  Gemini failed for patient {patient_id}: {e} — using fallback table")

    # ── Fallback: static table + symptom mapping ──────────────────────────────
    if gemini_data is None:
        fb = _lookup_disease(disease) or {}
        # Also check comorbidities
        extra_icd = _symptom_icd_fallback(row)
        gemini_data = {
            "icd10_codes":       fb.get("icd10_codes", extra_icd) or extra_icd,
            "icd10_code":        (fb.get("icd10_codes", ["UNKNOWN"]) or ["UNKNOWN"])[0],
            "icd10_description": fb.get("description", disease),
            "cpt_codes":         fb.get("cpt_codes",   []),
            "hcpcs_codes":       fb.get("hcpcs_codes", []),
            "loinc_codes":       fb.get("loinc_codes", []),
            "snomed_codes":      fb.get("snomed_codes",[]),
            "ndc_hints":         [fb.get("ndc_hint",   "")] if fb.get("ndc_hint") else [],
            "disease_causes_list": [causes] if causes else [],
            "health_tips_list":    [tips]   if tips   else [],
            "confidence_score":  0.60,
        }

    icd10_codes = (
        gemini_data.get("icd10_codes") or
        ([gemini_data.get("icd10_code")] if gemini_data.get("icd10_code") else [])
    )
    icd10_codes = [c for c in icd10_codes if c and c != "UNKNOWN"]

    # Auto-add comorbidity codes if not already present
    if comorbidities.get("hypertension") and not any(c.startswith("I1") for c in icd10_codes):
        icd10_codes.append("I10")
    if comorbidities.get("diabetes") and not any(c.startswith("E1") for c in icd10_codes):
        icd10_codes.append("E11.9")

    # Build necessity summary from causes
    necessity = (
        f"Patient presents with {disease}. "
        f"Contributing factors: {causes[:200] if causes else 'documented clinically'}. "
        f"Prior authorization requested for medically necessary treatment."
    )

    evidence_list = []
    if vitals["bp_systolic"] > 140:
        evidence_list.append(f"Elevated BP: {vitals['bp_systolic']} mmHg systolic")
    if vitals["glucose_level"] > 126:
        evidence_list.append(f"Elevated glucose: {vitals['glucose_level']} mg/dL")
    if vitals["heart_rate"] > 100:
        evidence_list.append(f"Tachycardia: HR {vitals['heart_rate']} bpm")
    if symptoms["active"]:
        evidence_list.append(f"Active symptoms: {', '.join(symptoms['active'])}")
    for m in meds:
        evidence_list.append(f"Medication: {m['medicine_name']} {m['dosage']} {m['frequency']}")

    confidence = float(gemini_data.get("confidence_score", 0.60))

    return _build_unified_bundle(
        mode               = "csv",
        payer              = payer.lower().strip(),
        icd10_codes        = list(dict.fromkeys(icd10_codes)),
        cpt_codes          = gemini_data.get("cpt_codes",   []),
        hcpcs_codes        = gemini_data.get("hcpcs_codes", []),
        loinc_codes        = gemini_data.get("loinc_codes", []),
        snomed_codes       = gemini_data.get("snomed_codes",[]),
        ndc_hints          = gemini_data.get("ndc_hints",   []),
        medical_necessity_summary = necessity,
        supporting_evidence       = evidence_list,
        patient_summary    = f"Patient {patient_id} — {disease}",
        confidence_score   = confidence,
        extraction_method  = extraction_method,
        patient_row        = row,
        disease_label      = disease,
        icd10_description  = gemini_data.get("icd10_description", ""),
        medications        = meds,
        vitals             = vitals,
        comorbidities      = comorbidities,
        symptoms           = symptoms,
        polypharmacy       = polypharmacy,
        health_tips        = gemini_data.get("health_tips_list", []),
    )


# ─────────────────────────────────────────────────────────────────────────────
# MODE 2 — EHR TEXT NOTE
# ─────────────────────────────────────────────────────────────────────────────

def read_from_ehr_note(ehr_note: str, payer: str = "unknown") -> Dict:
    """
    Extract clinical data from a free-text EHR note.

    Args:
        ehr_note : raw clinical note string
        payer    : "uhc" / "aetna" / "cigna" or "unknown" (auto-detected if not given)

    Returns:
        Unified bundle dict ready for policy_agent.run_policy_agent()
    """
    note = ehr_note.strip()

    # Auto-detect payer from text if not provided
    if payer == "unknown":
        m = _PAYER_RE.search(note)
        if m:
            raw_payer = m.group(1).lower()
            payer = "uhc" if "united" in raw_payer else raw_payer

    # ── Try Gemini ────────────────────────────────────────────────────────────
    model = _configure_gemini()
    extraction_method = "regex"
    data: Optional[Dict] = None

    if model:
        try:
            data = _gemini_extract_from_text(model, note)
            extraction_method = "gemini"
            if payer == "unknown" and data.get("payer"):
                payer = data["payer"].lower()
            print("✅ Gemini EHR extraction succeeded")
        except Exception as e:
            print(f"⚠️  Gemini failed: {e} — using regex fallback")

    # ── Regex fallback ────────────────────────────────────────────────────────
    if data is None:
        icd10_codes  = list(dict.fromkeys(_ICD_RE.findall(note)))
        cpt_codes    = list(dict.fromkeys(_CPT_RE.findall(note)))
        hcpcs_codes  = list(dict.fromkeys(_HCPCS_RE.findall(note)))
        loinc_codes  = list(dict.fromkeys(_LOINC_RE.findall(note)))

        # Remove CPTs that look like ICD codes (5-digit numeric only)
        cpt_codes = [c for c in cpt_codes if c.isdigit()]

        # If no ICD codes found, try disease keyword lookup
        if not icd10_codes:
            for disease_key in DISEASE_FALLBACK:
                if disease_key in note.lower():
                    fb = DISEASE_FALLBACK[disease_key]
                    icd10_codes  = fb["icd10_codes"]
                    cpt_codes    = cpt_codes  or fb["cpt_codes"]
                    hcpcs_codes  = hcpcs_codes or fb["hcpcs_codes"]
                    loinc_codes  = loinc_codes or fb["loinc_codes"]
                    break

        # Necessity
        m_nec = _NECESSITY_RE.search(note)
        necessity = m_nec.group(1).strip()[:400] if m_nec else (
            "Clinical documentation supports medical necessity per standard of care."
        )

        # Evidence: keyword-rich lines
        evidence_keywords = [
            "mri", "imaging", "failed", "conservative", "therapy", "pain",
            "radiculopathy", "herniat", "compression", "positive", "hba1c",
            "glucose", "bp", "blood pressure", "spirometry", "fev1", "peak flow",
            "x-ray", "ecg", "specialist", "referral", "documented", "chronic"
        ]
        evidence_list = [
            line.strip()[:200]
            for line in note.split("\n")
            if len(line.strip()) > 20 and any(kw in line.lower() for kw in evidence_keywords)
        ][:8]

        data = {
            "icd10_codes":              icd10_codes,
            "cpt_codes":                cpt_codes,
            "hcpcs_codes":              hcpcs_codes,
            "loinc_codes":              loinc_codes,
            "snomed_codes":             [],
            "ndc_hints":                [],
            "medical_necessity_summary":necessity,
            "supporting_evidence":      evidence_list,
            "patient_summary":          note.split("\n")[0][:120],
            "confidence_score":         0.55,
        }

    return _build_unified_bundle(
        mode               = "text",
        payer              = payer,
        icd10_codes        = data.get("icd10_codes",  []),
        cpt_codes          = data.get("cpt_codes",    []),
        hcpcs_codes        = data.get("hcpcs_codes",  []),
        loinc_codes        = data.get("loinc_codes",  []),
        snomed_codes       = data.get("snomed_codes", []),
        ndc_hints          = data.get("ndc_hints",    []),
        medical_necessity_summary = data.get("medical_necessity_summary", ""),
        supporting_evidence       = data.get("supporting_evidence", []),
        patient_summary    = data.get("patient_summary", ""),
        confidence_score   = float(data.get("confidence_score", 0.55)),
        extraction_method  = extraction_method,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CONVENIENCE: AUTO-DETECT MODE
# ─────────────────────────────────────────────────────────────────────────────

def extract(input_data, payer: str = "unknown", csv_path: str = CSV_PATH) -> Dict:
    """
    Smart entry point — auto-detects mode.

    Args:
        input_data : int (patient_id) → CSV mode
                     str              → EHR text mode
                     dict             → already structured, just normalise
        payer      : "uhc" / "aetna" / "cigna" (optional, auto-detected if possible)
    """
    if isinstance(input_data, int):
        return read_from_csv(input_data, payer=payer, csv_path=csv_path)

    if isinstance(input_data, str):
        # Check if it's a patient_id string
        if input_data.strip().isdigit():
            return read_from_csv(int(input_data), payer=payer, csv_path=csv_path)
        return read_from_ehr_note(input_data, payer=payer)

    if isinstance(input_data, dict):
        # Already structured — just ensure all keys exist and return
        return {
            "payer":      input_data.get("payer", payer).lower(),
            "icd_codes":  input_data.get("icd_codes", input_data.get("icd10_codes", [])),
            "cpt_code":   input_data.get("cpt_code",  input_data.get("cpt_codes", [""])[0] if input_data.get("cpt_codes") else ""),
            "summary":    input_data.get("summary",   input_data.get("medical_necessity_summary", "")),
            "evidence":   input_data.get("evidence",  " | ".join(input_data.get("supporting_evidence", []))),
            "codes":      input_data.get("codes", {}),
            "extraction_method": "passthrough",
            "bundle_id":  str(uuid.uuid4()),
            "generated_at": datetime.utcnow().isoformat() + "Z",
        }

    return _error_bundle(f"Unsupported input type: {type(input_data)}", payer)


def _error_bundle(reason: str, payer: str) -> Dict:
    return {
        "bundle_id":          str(uuid.uuid4()),
        "generated_at":       datetime.utcnow().isoformat() + "Z",
        "error":              reason,
        "payer":              payer,
        "icd_codes":          [],
        "cpt_code":           "",
        "summary":            reason,
        "evidence":           "",
        "extraction_method":  "error",
        "confidence_score":   0.0,
        "requires_human_review": True,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SAVE HELPER
# ─────────────────────────────────────────────────────────────────────────────

def save_bundle(bundle: Dict, output_dir: str = "output") -> str:
    os.makedirs(output_dir, exist_ok=True)
    bid  = bundle.get("bundle_id", str(uuid.uuid4()))[:8]
    mode = bundle.get("mode", "unknown")
    pid  = bundle.get("patient", {}).get("patient_id", "ehr")
    fname = f"{output_dir}/bundle_{mode}_{pid}_{bid}.json"
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2)
    print(f"💾 Saved: {fname}")
    return fname


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  CLINICAL READER AGENT — Standalone Test")
    print("=" * 60)

    # ── Test A: CSV mode ──────────────────────────────────────────────────────
    print("\n[TEST A] CSV MODE — Patient ID from patients.csv")
    try:
        pid   = int(input("Enter Patient ID (or press Enter to skip): ").strip() or "0")
        payer = input("Enter payer (uhc/aetna/cigna): ").strip() or "uhc"
        if pid > 0:
            bundle_a = read_from_csv(pid, payer=payer)
            save_bundle(bundle_a)
            print("\n📦 Bundle keys:", list(bundle_a.keys()))
            print("   ICD codes :", bundle_a["icd_codes"])
            print("   CPT code  :", bundle_a["cpt_code"])
            print("   Payer     :", bundle_a["payer"])
            print("   Method    :", bundle_a["extraction_method"])
            print("   Confidence:", bundle_a["confidence_score"])
    except Exception as e:
        print(f"   Skipped CSV test: {e}")

    # ── Test B: EHR text mode ─────────────────────────────────────────────────
    print("\n[TEST B] EHR TEXT MODE — Sample note")
    sample_ehr = """
    PATIENT: Jane Doe | DOB: 1968-04-12 | Payer: Aetna | Member ID: AET-9921
    Chief Complaint: Uncontrolled hypertension and Type 2 diabetes mellitus.

    History: Patient has documented hypertension (I10) for 3 years.
    Current BP readings: 168/104 mmHg, 172/108 mmHg on two separate visits.
    Failed first-line lisinopril therapy due to persistent cough.
    Fasting glucose 185 mg/dL. HbA1c 8.9% (LOINC 4548-4).

    Medications tried:
    - Lisinopril 10mg (failed — ACE inhibitor cough)
    - Metformin 500mg BID (partial response)

    Plan: Requesting PA for amlodipine 10mg + SGLT2 inhibitor (empagliflozin).
    CPT: 99214 (office visit), 83036 (HbA1c lab), 93000 (ECG)
    Medical Necessity: Uncontrolled BP despite first-line therapy. Dual comorbidity
    increases cardiovascular risk. SGLT2 inhibitor provides both glycemic and
    cardioprotective benefit per ADA guidelines.
    """
    bundle_b = read_from_ehr_note(sample_ehr, payer="aetna")
    save_bundle(bundle_b)
    print("\n📦 EHR Bundle:")
    print("   ICD codes  :", bundle_b["icd_codes"])
    print("   CPT code   :", bundle_b["cpt_code"])
    print("   HCPCS      :", bundle_b["codes"].get("hcpcs", []))
    print("   LOINC      :", bundle_b["codes"].get("loinc", []))
    print("   SNOMED     :", bundle_b["codes"].get("snomed", []))
    print("   Payer      :", bundle_b["payer"])
    print("   Method     :", bundle_b["extraction_method"])
    print("   Confidence :", bundle_b["confidence_score"])
    print("   Evidence   :", bundle_b["supporting_evidence"][:3])

    print("\n✅ Done. Check output/ folder for saved bundles.")
    print("\n🔗 To pipe into policy agent:")
    print("   from policy_agent import run_policy_agent")
    print("   decision = run_policy_agent(bundle_b)")
