"""
appeal_agent.py
===============
Takes a DENIED decision from policy_agent.py and generates
a professional appeal letter using Gemini.

Usage:
    from appeal_agent import generate_appeal
    letter = generate_appeal(patient_data, denial_result)
"""

"""
appeal_agent.py
===============
Generates a professional Prior Authorization Appeal Letter.
 
- NO Gemini / LLM dependency — fully rule-based template engine
- Uses real patient data from Clinical Reader Agent output
- Falls back to a complete default letter if patient data is missing
- Output: plain text letter ready to print / send
 
Usage:
    from appeal_agent import generate_appeal
 
    letter = generate_appeal(patient_data, denial_result)
    print(letter)
"""
 
from datetime import date
 
# ─────────────────────────────────────────────────────────────────────────────
# PAYER ADDRESS BOOK  (used in letter header)
# ─────────────────────────────────────────────────────────────────────────────
 
PAYER_INFO = {
    "uhc": {
        "name":    "UnitedHealthcare",
        "dept":    "Prior Authorization Appeals Department",
        "address": "P.O. Box 31364, Salt Lake City, UT 84131-0364",
        "phone":   "1-866-889-4149",
    },
    "aetna": {
        "name":    "Aetna Health Plans",
        "dept":    "Clinical Appeals Unit",
        "address": "P.O. Box 14463, Lexington, KY 40512-4463",
        "phone":   "1-800-537-9384",
    },
    "cigna": {
        "name":    "Cigna Healthcare",
        "dept":    "Appeals and Grievances Department",
        "address": "P.O. Box 188004, Chattanooga, TN 37422-8004",
        "phone":   "1-800-244-6224",
    },
}
 
DEFAULT_PAYER = {
    "name":    "Health Insurance Appeals Department",
    "dept":    "Prior Authorization Review Unit",
    "address": "P.O. Box 00000, [City, State, ZIP]",
    "phone":   "1-800-000-0000",
}
 
# ─────────────────────────────────────────────────────────────────────────────
# ICD-10 DESCRIPTION MAP  (for readable letter text)
# ─────────────────────────────────────────────────────────────────────────────
 
ICD_DESCRIPTIONS = {
    "E11.9":  "Type 2 Diabetes Mellitus, uncontrolled",
    "E11.65": "Type 2 Diabetes Mellitus with hyperglycemia",
    "E10.9":  "Type 1 Diabetes Mellitus",
    "I10":    "Essential (Primary) Hypertension",
    "I11.9":  "Hypertensive Heart Disease",
    "J45.51": "Severe Persistent Asthma with acute exacerbation",
    "J45.41": "Moderate Persistent Asthma with acute exacerbation",
    "J45.901":"Unspecified Asthma with acute exacerbation",
    "J44.1":  "Chronic Obstructive Pulmonary Disease with acute exacerbation",
    "J18.9":  "Pneumonia, unspecified organism",
    "I50.9":  "Heart Failure, unspecified",
    "R50.9":  "Fever, unspecified",
    "R05.9":  "Cough, unspecified",
}
 
CPT_DESCRIPTIONS = {
    "99213": "Office or outpatient visit, established patient (moderate complexity)",
    "99214": "Office or outpatient visit, established patient (high complexity)",
    "83036": "Hemoglobin A1c (HbA1c) laboratory test",
    "94640": "Pressurized or non-pressurized inhalation treatment",
    "94060": "Spirometry with bronchodilator",
    "93000": "Electrocardiogram (ECG), routine",
    "93306": "Echocardiography with Doppler",
    "71046": "Radiologic examination, chest, 2 views",
}
 
# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
 
def _today() -> str:
    return date.today().strftime("%B %d, %Y")
 
 
def _payer_block(payer_key: str) -> dict:
    return PAYER_INFO.get(payer_key.lower().strip(), DEFAULT_PAYER)
 
 
def _describe_icd(codes: list) -> str:
    parts = []
    for c in codes:
        desc = ICD_DESCRIPTIONS.get(c, "")
        parts.append(f"{c} ({desc})" if desc else c)
    return ", ".join(parts) if parts else "N/A"
 
 
def _describe_cpt(code: str) -> str:
    desc = CPT_DESCRIPTIONS.get(code, "")
    return f"{code} — {desc}" if desc else (code or "N/A")
 
 
def _has_real_data(patient_data: dict) -> bool:
    """Return True only if the patient dict has enough real data to build a letter."""
    if not patient_data:
        return False
    has_icd     = bool(patient_data.get("icd_codes") or patient_data.get("icd_code"))
    has_summary = bool((patient_data.get("summary") or "").strip())
    has_payer   = bool((patient_data.get("payer") or "").strip().lower() not in ("", "unknown"))
    return has_icd and has_summary and has_payer
 
 
def _criteria_paragraph(criteria_missing: list) -> str:
    """Build a paragraph addressing each missing criterion directly."""
    if not criteria_missing:
        return (
            "We respectfully contend that all required clinical criteria have been satisfied "
            "based on the documentation provided. Our clinical team is prepared to supply "
            "any additional records upon request."
        )
    lines = []
    for criterion in criteria_missing:
        lines.append(
            f"  • Regarding '{criterion}': Our clinical records confirm this criterion "
            f"has been addressed. Supporting documentation is enclosed and available upon request."
        )
    return "\n".join(lines)
 
 
def _evidence_paragraph(evidence: str, supporting_evidence: list) -> str:
    """Format supporting evidence into readable paragraphs."""
    items = []
    if supporting_evidence:
        items = supporting_evidence[:6]
    elif evidence:
        items = [e.strip() for e in evidence.split("|") if e.strip()][:6]
 
    if not items:
        return (
            "Clinical documentation supporting this request is on file and available "
            "for review by your medical director upon request."
        )
    return "\n".join(f"  • {item}" for item in items)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# MAIN LETTER BUILDER  (real patient data)
# ─────────────────────────────────────────────────────────────────────────────
 
def _build_patient_letter(patient_data: dict, denial_result: dict) -> str:
    payer_key   = (patient_data.get("payer") or denial_result.get("payer") or "").lower()
    payer       = _payer_block(payer_key)
    icd_codes   = patient_data.get("icd_codes") or []
    cpt_code    = patient_data.get("cpt_code") or denial_result.get("cpt_code") or ""
    summary     = (patient_data.get("summary") or "").strip()
    evidence    = (patient_data.get("evidence") or "").strip()
    sup_ev      = patient_data.get("supporting_evidence", [])
 
    denial_reason     = (denial_result.get("reason") or "Reason not specified.").strip()
    criteria_missing  = denial_result.get("criteria_missing", [])
    criteria_met      = denial_result.get("criteria_met", [])
    appeal_hint       = (denial_result.get("appeal_hint") or "").strip()
    confidence        = denial_result.get("confidence", "N/A")
 
    # Patient identifiers from nested dict (CSV mode) or flat (text mode)
    patient_block = patient_data.get("patient", {})
    patient_id    = patient_block.get("patient_id") or patient_data.get("patient_id") or "N/A"
    patient_name  = patient_block.get("name") or patient_data.get("patient_name") or "Patient on File"
    member_id     = (patient_data.get("pa_context") or {}).get("member_id") or "N/A"
 
    # Disease label for readability
    disease_label = (
        patient_data.get("diagnosis", {}).get("predicted_disease")
        or patient_data.get("disease_label")
        or _describe_icd(icd_codes[:1])
        or "documented medical condition"
    )
 
    criteria_para = _criteria_paragraph(criteria_missing)
    evidence_para = _evidence_paragraph(evidence, sup_ev)
 
    met_text = ""
    if criteria_met:
        met_text = (
            "\nThe following criteria have already been confirmed as met:\n"
            + "\n".join(f"  • {c}" for c in criteria_met)
            + "\n"
        )
 
    appeal_basis_text = ""
    if appeal_hint:
        appeal_basis_text = f"\nBasis for Appeal:\n{appeal_hint}\n"
 
    letter = f"""
{_today()}
 
{payer['name']}
{payer['dept']}
{payer['address']}
 
Re: Prior Authorization Appeal
Patient Name     : {patient_name}
Patient ID       : {patient_id}
Member ID        : {member_id}
ICD-10 Code(s)   : {_describe_icd(icd_codes)}
Procedure (CPT)  : {_describe_cpt(cpt_code)}
Payer            : {payer['name']}
Date of Letter   : {_today()}
 
To Whom It May Concern,
 
We are writing to formally appeal the denial of the prior authorization request submitted for
the above-referenced patient in connection with the treatment of {disease_label}.
 
─────────────────────────────────────────────────────────────
I. REASON FOR DENIAL — OUR RESPONSE
─────────────────────────────────────────────────────────────
 
Your denial indicated the following: "{denial_reason}"
 
We respectfully disagree with this determination. The clinical evidence on file clearly
supports the medical necessity of the requested service, and we believe the denial was
issued without full consideration of the patient's complete clinical history.
 
─────────────────────────────────────────────────────────────
II. CLINICAL SUMMARY
─────────────────────────────────────────────────────────────
 
{summary}
 
─────────────────────────────────────────────────────────────
III. CRITERIA AND DOCUMENTATION
─────────────────────────────────────────────────────────────
{met_text}
The following unmet criteria cited in the denial are addressed below:
 
{criteria_para}
{appeal_basis_text}
─────────────────────────────────────────────────────────────
IV. SUPPORTING CLINICAL EVIDENCE
─────────────────────────────────────────────────────────────
 
The following objective clinical findings substantiate this appeal:
 
{evidence_para}
 
─────────────────────────────────────────────────────────────
V. REQUEST FOR RECONSIDERATION
─────────────────────────────────────────────────────────────
 
In light of the foregoing, we respectfully request that your medical director conduct
a peer-to-peer review and reconsider the denial within 30 days. We are available to
provide any additional records, imaging, or specialist notes required to support
this appeal.
 
Please direct correspondence to the treating provider of record. You may also reach
our office at the contact information on file with your network directory.
 
Sincerely,
 
[Treating Physician / Authorized Representative]
[Provider Name]
[Provider NPI]
[Practice Name & Address]
[Phone / Fax]
 
— Prepared by AutoAuth Prior Authorization System —
    """.strip()
 
    return letter
 
 
# ─────────────────────────────────────────────────────────────────────────────
# FALLBACK LETTER  (no patient data available)
# ─────────────────────────────────────────────────────────────────────────────
 
def _build_fallback_letter(denial_result: dict) -> str:
    payer_key = (denial_result.get("payer") or "").lower()
    payer     = _payer_block(payer_key)
    icd_codes = denial_result.get("icd_codes", [])
    cpt_code  = denial_result.get("cpt_code", "")
    reason    = (denial_result.get("reason") or "Insufficient clinical documentation.").strip()
    criteria  = denial_result.get("criteria_missing", [])
    hint      = (denial_result.get("appeal_hint") or "").strip()
 
    criteria_para = _criteria_paragraph(criteria)
 
    letter = f"""
{_today()}
 
{payer['name']}
{payer['dept']}
{payer['address']}
 
Re: Prior Authorization Appeal
ICD-10 Code(s)  : {_describe_icd(icd_codes) if icd_codes else "On file with your office"}
Procedure (CPT) : {_describe_cpt(cpt_code) if cpt_code else "On file with your office"}
Payer           : {payer['name']}
Date of Letter  : {_today()}
 
To Whom It May Concern,
 
We are writing to formally appeal the denial of a prior authorization request for the
patient identified in your denial notice. We believe this denial was issued without full
consideration of the complete clinical documentation supporting medical necessity.
 
─────────────────────────────────────────────────────────────
I. REASON FOR DENIAL — OUR RESPONSE
─────────────────────────────────────────────────────────────
 
Your denial cited the following: "{reason}"
 
We respectfully challenge this determination. The patient's clinical condition meets
the established medical necessity criteria required for the requested procedure or
service under your plan's coverage policy.
 
─────────────────────────────────────────────────────────────
II. CLINICAL JUSTIFICATION
─────────────────────────────────────────────────────────────
 
The patient has a documented diagnosis requiring the requested treatment. Conservative
and first-line therapies have been attempted where applicable, and the treating physician
has determined that the requested service is the medically necessary next step in the
patient's plan of care.
 
All clinical records, laboratory results, imaging studies, and specialist notes are
available for review upon request.
 
─────────────────────────────────────────────────────────────
III. ADDRESSING UNMET CRITERIA
─────────────────────────────────────────────────────────────
 
{criteria_para}
 
{"─" * 61}
IV. APPEAL BASIS
{"─" * 61}
 
{hint if hint else "We request that a board-certified physician in the relevant specialty conduct a peer-to-peer review of the complete clinical record before a final determination is made."}
 
─────────────────────────────────────────────────────────────
V. REQUEST FOR RECONSIDERATION
─────────────────────────────────────────────────────────────
 
We respectfully request that your medical director reconsider this denial within
30 days of this letter. Continued denial of this medically necessary service may
result in deterioration of the patient's health and potential legal and regulatory
review under applicable state and federal law.
 
Please contact our office for any additional documentation or to schedule a
peer-to-peer review.
 
Sincerely,
 
[Treating Physician / Authorized Representative]
[Provider Name]
[Provider NPI]
[Practice Name & Address]
[Phone / Fax]
 
— Prepared by AutoAuth Prior Authorization System —
    """.strip()
 
    return letter
 
 
# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
 
def generate_appeal(patient_data: dict, denial_result: dict) -> str:
    """
    Generate a prior authorization appeal letter.
 
    Args:
        patient_data  : Output from clinical_reader_agent.extract() / read_from_csv()
        denial_result : Output from policy_agent.run_policy_agent()
 
    Returns:
        Plain text appeal letter (str)
 
    Behavior:
        - If decision is not DENIED → returns a short note (no letter needed)
        - If patient_data has real info → builds a personalised letter
        - If patient_data is empty / missing → builds a complete generic letter
    """
    decision = denial_result.get("decision", "")
 
    if decision != "DENIED":
        return (
            f"No appeal letter required.\n"
            f"Current authorization status: {decision}\n"
            f"Reason: {denial_result.get('reason', 'N/A')}"
        )
 
    if _has_real_data(patient_data):
        print("📝 Building personalised appeal letter from patient data...")
        return _build_patient_letter(patient_data, denial_result)
    else:
        print("📝 Patient data unavailable — generating complete appeal letter...")
        return _build_fallback_letter(denial_result)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE TEST
# ─────────────────────────────────────────────────────────────────────────────
 
if __name__ == "__main__":
    print("=" * 60)
    print("  APPEAL AGENT — Standalone Test")
    print("=" * 60)
 
    # ── Test A: With full patient data ────────────────────────────────────────
    print("\n[TEST A] Full patient data (UHC Diabetes)")
    patient_a = {
        "payer":     "uhc",
        "icd_codes": ["E11.9", "E11.65"],
        "cpt_code":  "99213",
        "summary":   (
            "Patient presents with uncontrolled Type 2 Diabetes Mellitus. "
            "HbA1c is 9.2% despite 6 months of metformin monotherapy. "
            "Endocrinologist referral is documented."
        ),
        "evidence":  "Failed metformin 6 months | HbA1c 9.2% | Endocrinologist referral on file",
        "supporting_evidence": [
            "HbA1c 9.2% — above target of 7.0% per ADA guidelines",
            "Metformin monotherapy failed after 6 months of documented use",
            "Endocrinologist referral placed and documented in chart",
            "Patient compliant with diet and lifestyle modifications",
        ],
        "patient": {"patient_id": 101, "name": "Jane Doe"},
        "diagnosis": {"predicted_disease": "Type 2 Diabetes Mellitus"},
    }
 
    denial_a = {
        "payer":            "uhc",
        "decision":         "DENIED",
        "reason":           "Step therapy requirements not met. No documented failure of GLP-1 agonist.",
        "criteria_missing": [
            "GLP-1 agonist trial and failure",
            "Endocrinologist specialist note within 90 days",
        ],
        "criteria_met": [
            "HbA1c above 8.0%",
            "Metformin monotherapy documented",
        ],
        "appeal_hint": (
            "Provide GLP-1 agonist contraindication documentation or prior trial records. "
            "Attach endocrinologist note dated within last 90 days."
        ),
        "icd_codes": ["E11.9", "E11.65"],
        "cpt_code":  "99213",
    }
 
    letter_a = generate_appeal(patient_a, denial_a)
    print(letter_a)
 
    # ── Test B: Missing patient data (fallback) ───────────────────────────────
    print("\n\n" + "=" * 60)
    print("[TEST B] Missing patient data (Cigna Asthma — fallback letter)")
    print("=" * 60)
 
    denial_b = {
        "payer":            "cigna",
        "decision":         "DENIED",
        "reason":           "Step therapy not documented. No evidence of failed step 2 medications.",
        "criteria_missing": ["Step 2 therapy failure", "Pulmonologist referral"],
        "appeal_hint":      "Provide documentation of prior failed medications and pulmonologist note.",
        "icd_codes":        ["J45.51"],
        "cpt_code":         "94640",
    }
 
    letter_b = generate_appeal({}, denial_b)
    print(letter_b)
 
    # ── Test C: Approved case (no letter needed) ──────────────────────────────
    print("\n\n" + "=" * 60)
    print("[TEST C] Approved case — no letter needed")
    print("=" * 60)
 
    result_c = generate_appeal(patient_a, {"decision": "APPROVED", "reason": "All criteria met."})
    print(result_c)