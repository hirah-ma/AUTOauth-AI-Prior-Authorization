# -*- coding: utf-8 -*-
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

from src.agents.appeal_generator import generate_appeal
from src.agents.clinical_reader import extract as clinical_extract
from src.agents.policy_engine import run_policy_agent


# ─── Helpers ────────────────────────────────────────────────────────────────

def _now_ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


OUTPUT_DIR = Path("output")
DASHBOARD_METRICS_PATH = OUTPUT_DIR / "dashboard_metrics.json"
RUN_HISTORY_PATH = OUTPUT_DIR / "run_history.json"


_SAMPLE_SCENARIOS = {
    "Asthma Exacerbation (nebulizer)": {
        "ehr": (
            "PATIENT: John Smith | DOB: 11/02/1981 | MRN: JS-2024-1102\n"
            "CHIEF COMPLAINT: Severe asthma exacerbation\n"
            "HPI: Severe persistent asthma with acute exacerbation. Two ER visits in past 90 days. Daily SABA use.\n"
            "DX: J45.51\nCPT: 94640 (nebulizer treatment), 94060 (spirometry)\n"
            "Evidence: Peak flow 52% predicted. Failed inhaled corticosteroid step-up (budesonide). Pulmonology referral documented.\n"
            "Medical Necessity: Requires escalation of therapy per payer criteria.\n"
        ),
        "csv_patient": 2,
    },
    "Surgical — Lumbar Laminectomy / Discectomy (CPT 63030)": {
        "ehr": (
            "AutoAuth Agent - Prior Authorization Demo\n\n"
            "PATIENT: Jane Doe | DOB: 03/15/1968 | MRN: JD-2024-001\n"
            "CHIEF COMPLAINT: Chronic low back pain with radiculopathy\n"
            "HISTORY OF PRESENT ILLNESS:\n"
            "Patient presents with 12-month history of chronic lumbar radiculopathy.\n"
            "MRI from 02/2024 demonstrates L4-L5 disc herniation with nerve root compression.\n"
            "Failed conservative treatment including 8+ weeks physical therapy.\n"
            "Also trialed NSAIDs and epidural steroid injection x2. VAS pain score 8/10.\n"
            "Positive straight leg raise. Sensory deficit in L5 distribution.\n\n"
            "ICD-10: M54.16, M51.06\nCPT: 63030\n"
            "Medical Necessity: Refractory pain with neurologic deficit and imaging correlation.\n"
        ),
        "csv_patient": 1,
    },
}


def _resolve_payer(payer_label: str) -> str:
    payer = (payer_label or "").strip().lower()
    if payer in {"payer_a", "payer a", "payer-a", "payer_a (demo)", "payer_a"}:
        return "aetna"
    return payer


def _load_metrics() -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not DASHBOARD_METRICS_PATH.is_file():
        return {"total_runs": 0, "approvals": 0, "denials": 0,
                "pending_more_info": 0, "errors": 0, "cumulative_days_saved": 0.0}
    try:
        return json.loads(DASHBOARD_METRICS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"total_runs": 0, "approvals": 0, "denials": 0,
                "pending_more_info": 0, "errors": 0, "cumulative_days_saved": 0.0}


def _save_metrics(m: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DASHBOARD_METRICS_PATH.write_text(json.dumps(m, indent=2), encoding="utf-8")


def _load_history() -> List[Dict[str, Any]]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not RUN_HISTORY_PATH.is_file():
        return []
    try:
        data = json.loads(RUN_HISTORY_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _append_history(record: Dict[str, Any]) -> None:
    history = _load_history()
    history.append(record)
    RUN_HISTORY_PATH.write_text(json.dumps(history, indent=2), encoding="utf-8")


def compute_payer_behavior(history: List[Dict[str, Any]]) -> Dict[str, Any]:
    stats: Dict[str, Dict[str, Any]] = {}
    for r in history:
        payer_id = str(r.get("payer_id") or "")
        cpt_codes = r.get("cpt_codes") or []
        if not payer_id or not cpt_codes:
            continue
        cpt = str(cpt_codes[0])
        key = f"{payer_id}::{cpt}"
        if key not in stats:
            stats[key] = {"runs": 0, "approved": 0}
        stats[key]["runs"] += 1
        if str(r.get("status") or "").lower() == "approved":
            stats[key]["approved"] += 1
    rates: Dict[str, Any] = {}
    for key, v in stats.items():
        runs = int(v.get("runs") or 0)
        approved = int(v.get("approved") or 0)
        rates[key] = {"runs": runs, "approved": approved,
                      "approval_rate": (approved / runs) if runs else 0.0}
    return rates


def compute_payer_behavior_by_payer(history: List[Dict[str, Any]]) -> Dict[str, Any]:
    stats: Dict[str, Dict[str, Any]] = {}
    total_runs = 0
    total_approved = 0
    for r in history:
        payer_id = str(r.get("payer_id") or "")
        status = str(r.get("status") or "")
        if not payer_id:
            continue
        if payer_id not in stats:
            stats[payer_id] = {"runs": 0, "approved": 0}
        stats[payer_id]["runs"] += 1
        total_runs += 1
        if status.lower() == "approved":
            stats[payer_id]["approved"] += 1
            total_approved += 1
    out: Dict[str, Any] = {}
    for payer_id, v in stats.items():
        runs = int(v.get("runs") or 0)
        approved = int(v.get("approved") or 0)
        out[payer_id] = {
            "runs": runs, "approved": approved,
            "approval_rate": (approved / runs) if runs else 0.0,
            "run_share": (runs / total_runs) if total_runs else 0.0,
            "approval_share": (approved / total_approved) if total_approved else 0.0,
        }
    out["__all__"] = {
        "runs": total_runs, "approved": total_approved,
        "approval_rate": (total_approved / total_runs) if total_runs else 0.0,
        "run_share": 1.0 if total_runs else 0.0,
        "approval_share": 1.0 if total_approved else 0.0,
    }
    return out


def _estimate_days_saved(decision: str) -> float:
    if decision == "APPROVED":        return 3.0
    if decision == "DENIED":          return 1.0
    if decision == "PENDING_MORE_INFO": return 0.5
    return 0.0


def _decision_drivers(clinical_bundle: dict, policy_decision: dict) -> list:
    drivers = []
    codes    = (clinical_bundle or {}).get("codes", {}) or {}
    icd      = codes.get("icd10") or []
    cpt      = codes.get("cpt") or []
    evidence = ((clinical_bundle or {}).get("evidence") or "").lower()
    summary  = ((clinical_bundle or {}).get("summary") or "").lower()

    drivers.append("ICD-10 codes present"  if icd else "ICD-10 codes missing")
    drivers.append("Procedure code present" if (cpt or (clinical_bundle or {}).get("cpt_code")) else "Procedure code missing")

    signals = [
        ("mri",             "Imaging evidence present (MRI)"),
        ("physical therapy","Conservative therapy documented (PT)"),
        ("failed",          "Treatment failure documented"),
        ("radicul",         "Radicular symptoms documented"),
        ("neurolog",        "Neurologic deficit mentioned"),
        ("specialist",      "Specialist involvement documented"),
        ("er visit",        "Recent ER utilization documented"),
    ]
    blob = f"{summary} {evidence}"
    for sig, label in signals:
        if sig in blob:
            drivers.append(label)

    missing = (policy_decision or {}).get("criteria_missing") or []
    if missing:
        drivers.append("Missing policy criteria identified")

    return drivers[:6]


# ─── CSS injection from template ────────────────────────────────────────────

TEMPLATE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&family=Fraunces:ital,wght@0,300;0,400;0,600;1,300;1,400&display=swap');

:root {
  --bg:#0A0E17; --surface:#0F141F; --surface2:#141923; --surface3:#1A2030;
  --border:#1E2738; --border2:#2A3547;
  --accent:#00D4AA; --accent-d:rgba(0,212,170,.12); --accent-b:rgba(0,212,170,.35);
  --green:#00F5B8; --green-d:rgba(0,245,184,.10); --green-b:rgba(0,245,184,.30);
  --red:#FF5370; --red-d:rgba(255,83,112,.10); --red-b:rgba(255,83,112,.30);
  --yellow:#FFCB6B; --yellow-d:rgba(255,203,107,.10); --yellow-b:rgba(255,203,107,.28);
  --blue:#82AAFF; --blue-d:rgba(130,170,255,.10); --blue-b:rgba(130,170,255,.30);
  --cyan:#89DDFF; --orange:#F78C6C;
  --muted:#4F5B66; --text:#E8EDF2; --text2:#8B99A8;
  --font:'Inter',sans-serif; --mono:'JetBrains Mono',monospace; --serif:'Fraunces',serif;
  --r:12px; --rl:16px;
}

/* ── Reset Streamlit chrome ── */
html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"],
[data-testid="block-container"] {
  background: var(--bg) !important;
  color: var(--text) !important;
  font-family: var(--font) !important;
}
[data-testid="stSidebar"] {
  background: var(--surface) !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }
[data-testid="stHeader"] { background: var(--surface) !important; border-bottom: 1px solid var(--border) !important; }
section[data-testid="stSidebar"] > div { padding-top: 0 !important; }

/* ── Streamlit widget overrides ── */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
  background: var(--surface2) !important;
  border: 1px solid var(--border2) !important;
  border-radius: var(--r) !important;
  color: var(--text) !important;
  font-family: var(--font) !important;
}
div[data-baseweb="select"] > div:focus-within,
div[data-baseweb="input"] > div:focus-within {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px var(--accent-d) !important;
}
textarea {
  background: var(--surface2) !important;
  border: 1px solid var(--border2) !important;
  border-radius: var(--r) !important;
  color: var(--text) !important;
  font-family: var(--mono) !important;
  font-size: .8rem !important;
}
textarea:focus { border-color: var(--accent) !important; box-shadow: 0 0 0 3px var(--accent-d) !important; }

/* primary button = Run All Agents */
div[data-testid="stButton"] > button[kind="primary"] {
  background: linear-gradient(135deg, var(--accent) 0%, var(--blue) 100%) !important;
  color: #0A0E17 !important;
  font-family: var(--font) !important;
  font-weight: 800 !important;
  font-size: .92rem !important;
  border: none !important;
  border-radius: var(--r) !important;
  padding: 14px !important;
  text-transform: uppercase !important;
  letter-spacing: .04em !important;
  box-shadow: 0 4px 20px rgba(0,212,170,.3) !important;
  transition: all .3s ease !important;
  width: 100% !important;
}
div[data-testid="stButton"] > button[kind="primary"]:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 6px 28px rgba(0,212,170,.5) !important;
}
div[data-testid="stButton"] > button:not([kind="primary"]) {
  background: var(--surface2) !important;
  border: 1px solid var(--border2) !important;
  border-radius: var(--r) !important;
  color: var(--text2) !important;
  font-family: var(--font) !important;
  font-weight: 600 !important;
  transition: all .25s ease !important;
}
div[data-testid="stButton"] > button:not([kind="primary"]):hover {
  border-color: var(--accent) !important;
  color: var(--accent) !important;
  background: var(--accent-d) !important;
}

/* toggle / radio */
div[data-testid="stRadio"] label,
div[data-testid="stToggle"] label { color: var(--text2) !important; font-family: var(--font) !important; }
div[role="radiogroup"] label[data-baseweb="radio"] > div:first-child { border-color: var(--accent) !important; }

/* metric cards */
div[data-testid="stMetric"] {
  background: var(--surface2) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--r) !important;
  padding: 12px 14px !important;
  transition: all .3s ease !important;
}
div[data-testid="stMetric"]:hover {
  border-color: var(--border2) !important;
  transform: translateY(-2px) !important;
  box-shadow: 0 2px 8px rgba(0,0,0,.15) !important;
}
div[data-testid="stMetricValue"] {
  font-family: var(--mono) !important;
  font-size: 1.4rem !important;
  font-weight: 700 !important;
  color: var(--blue) !important;
}
div[data-testid="stMetricLabel"] {
  font-family: var(--mono) !important;
  font-size: .58rem !important;
  color: var(--muted) !important;
  text-transform: uppercase !important;
  letter-spacing: .1em !important;
}

/* tabs */
div[data-testid="stTabs"] button[role="tab"] {
  font-family: var(--mono) !important;
  font-size: .72rem !important;
  color: var(--text2) !important;
  background: var(--surface2) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px 8px 0 0 !important;
  font-weight: 600 !important;
}
div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
  color: var(--accent) !important;
  border-bottom-color: var(--surface) !important;
  background: var(--surface) !important;
}
div[data-testid="stTabs"] div[role="tabpanel"] {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: 0 var(--r) var(--r) var(--r) !important;
  padding: 20px !important;
}

/* expander */
div[data-testid="stExpander"] {
  background: var(--surface2) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--r) !important;
}
div[data-testid="stExpander"] summary { color: var(--text2) !important; font-family: var(--mono) !important; font-size: .8rem !important; }

/* dataframe */
div[data-testid="stDataFrame"] { border: 1px solid var(--border) !important; border-radius: var(--r) !important; overflow: hidden !important; }

/* spinner */
div[data-testid="stSpinner"] p { color: var(--accent) !important; font-family: var(--mono) !important; font-size: .8rem !important; }

/* scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--surface); }
::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent); }

/* ── Template component classes ── */
@keyframes pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:.4;transform:scale(1.5)} }
@keyframes scan  { from{background-position:-300px 0} to{background-position:300px 0} }
@keyframes statusPulse { 0%,100%{opacity:1} 50%{opacity:.7} }

.aa-topbar {
  background: var(--surface);
  border-bottom: 1px solid var(--border);
  padding: 14px 24px;
  display: flex;
  align-items: center;
  gap: 12px;
  margin: -1rem -1rem 1.5rem -1rem;
  box-shadow: 0 2px 12px rgba(0,0,0,.08);
}
.aa-brand { font-family: var(--serif); font-style: italic; font-size: 1.55rem;
  background: linear-gradient(135deg,var(--accent) 0%,var(--blue) 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 400; }
.aa-topbar-sub { font-family: var(--mono); font-size: .62rem; color: var(--text2); font-weight: 500; }
.ai-pill { display: inline-flex; align-items: center; gap: 6px; padding: 5px 12px;
  background: var(--accent-d); border: 1px solid var(--accent-b); border-radius: 24px;
  font-family: var(--mono); font-size: .62rem; color: var(--accent); font-weight: 600;
  box-shadow: 0 0 20px rgba(0,212,170,.15); margin-left: auto; }
.ai-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--accent);
  animation: pulse 1.8s infinite; box-shadow: 0 0 8px var(--accent); display: inline-block; }

.sb-brand { font-family: var(--serif); font-style: italic; font-size: 1.55rem;
  background: linear-gradient(135deg,var(--accent) 0%,var(--blue) 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 400; letter-spacing: -.02em; }
.sb-brand-sub { font-family: var(--mono); font-size: .58rem; color: var(--muted);
  letter-spacing: .16em; text-transform: uppercase; margin-top: 2px; }
.sb-sec-title { font-family: var(--mono); font-size: .6rem; color: var(--muted);
  text-transform: uppercase; letter-spacing: .14em; margin-bottom: 10px; font-weight: 600; }

.panel { background: var(--surface); border: 1px solid var(--border); border-radius: var(--rl);
  padding: 22px; box-shadow: 0 2px 8px rgba(0,0,0,.15); transition: all .3s ease; }
.panel:hover { box-shadow: 0 4px 16px rgba(0,0,0,.25); }
.panel-title { font-family: var(--serif); font-style: italic; font-size: 1.45rem;
  color: var(--text); letter-spacing: -.02em; margin-bottom: 4px; }
.panel-sub { font-family: var(--mono); font-size: .58rem; color: var(--muted);
  letter-spacing: .14em; text-transform: uppercase; margin-bottom: 18px; }

.dh { border-radius: var(--rl); padding: 24px; position: relative; overflow: hidden;
  box-shadow: 0 4px 16px rgba(0,0,0,.25); margin-bottom: 16px; }
.dh::before { content:''; position: absolute; top:0; left:0; right:0; height: 3px; background: currentColor; }
.dh.approved  { background: var(--green-d);  border: 1px solid var(--green-b);  color: var(--green); }
.dh.denied    { background: var(--red-d);    border: 1px solid var(--red-b);    color: var(--red); }
.dh.pending   { background: var(--yellow-d); border: 1px solid var(--yellow-b); color: var(--yellow); }
.dh.error     { background: var(--surface);  border: 1px solid var(--border);   color: var(--muted); }
.dh-top { display: flex; align-items: flex-start; justify-content: space-between; flex-wrap: wrap; gap: 16px; margin-bottom: 14px; }
.dh-verdict { font-family: var(--serif); font-size: 2.2rem; font-weight: 700; line-height: .95; letter-spacing: -.03em; }
.dh-lbl { font-family: var(--mono); font-size: .6rem; color: var(--muted); text-transform: uppercase; letter-spacing: .15em; font-weight: 600; }
.dh-kpis { display: flex; gap: 10px; flex-wrap: wrap; }
.dh-kpi { background: rgba(0,0,0,.3); border: 1px solid rgba(255,255,255,.08); border-radius: var(--r);
  padding: 10px 14px; min-width: 82px; backdrop-filter: blur(8px); transition: all .3s ease; }
.dh-kpi:hover { transform: translateY(-2px); }
.dh-kpi-val { font-family: var(--mono); font-size: 1.2rem; font-weight: 700; color: var(--text); letter-spacing: -.02em; }
.dh-kpi-lbl { font-family: var(--mono); font-size: .54rem; color: var(--muted); text-transform: uppercase; letter-spacing: .1em; margin-top: 3px; }
.conf-bar-bg { height: 4px; background: rgba(255,255,255,.1); border-radius: 2px; overflow: hidden; margin-top: 6px; }
.conf-bar    { height: 100%; border-radius: 2px; transition: width .8s ease; box-shadow: 0 0 12px currentColor; }
.dh-reason   { padding: 12px 16px; background: rgba(0,0,0,.25); border-radius: var(--r);
  font-size: .84rem; line-height: 1.7; color: var(--text); border-left: 3px solid; font-weight: 500; }
.dh-reason.approved { border-left-color: var(--green); }
.dh-reason.denied   { border-left-color: var(--red); }
.dh-reason.pending  { border-left-color: var(--yellow); }

.step-card { background: var(--surface); border: 1px solid var(--border); border-radius: var(--rl);
  overflow: hidden; margin-bottom: 16px; box-shadow: 0 2px 8px rgba(0,0,0,.15); transition: all .3s ease; }
.step-card:hover { box-shadow: 0 4px 16px rgba(0,0,0,.25); border-color: var(--border2); }
.step-header { display: flex; align-items: center; gap: 12px; padding: 14px 18px; border-bottom: 1px solid var(--border); }
.step-num  { font-family: var(--mono); font-size: .58rem; color: var(--muted); letter-spacing: .1em; font-weight: 600; }
.step-name { font-size: .88rem; font-weight: 700; color: var(--text); letter-spacing: -.01em; }
.step-status { font-family: var(--mono); font-size: .6rem; padding: 4px 10px; border-radius: 6px; font-weight: 700; letter-spacing: .05em; margin-left: auto; }
.ss-done { background: var(--green-d);  color: var(--green);  border: 1px solid var(--green-b); }
.ss-run  { background: var(--blue-d);   color: var(--blue);   border: 1px solid var(--blue-b); animation: statusPulse 2s infinite; }
.ss-wait { background: var(--surface2); color: var(--muted);  border: 1px solid var(--border); }
.ss-err  { background: var(--red-d);    color: var(--red);    border: 1px solid var(--red-b); }
.step-body { padding: 18px; }

.chip-group { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 6px; }
.chip { font-family: var(--mono); font-size: .68rem; padding: 5px 10px; border-radius: 6px; border: 1px solid; font-weight: 600; cursor: default; }
.chip.icd   { background: rgba(130,170,255,.08); color: #8CB4FF; border-color: rgba(140,180,255,.3); }
.chip.cpt   { background: rgba(0,245,184,.08);   color: var(--green); border-color: rgba(0,245,184,.3); }
.chip.hcpcs { background: rgba(255,203,107,.08); color: var(--yellow); border-color: rgba(255,203,107,.3); }
.chip.loinc { background: rgba(200,120,255,.08); color: #D088FF; border-color: rgba(200,120,255,.3); }
.code-lbl   { font-family: var(--mono); font-size: .58rem; color: var(--muted); text-transform: uppercase; letter-spacing: .12em; margin: 10px 0 5px; font-weight: 600; }

.ev { background: var(--surface2); border-left: 3px solid var(--accent); border-radius: 0 var(--r) var(--r) 0;
  padding: 7px 12px; font-size: .78rem; color: var(--text2); margin-bottom: 5px; line-height: 1.6; transition: all .25s ease; }
.ev:hover { background: var(--surface3); border-left-color: var(--cyan); transform: translateX(3px); }

.crit-box { border-radius: var(--r); padding: 14px; transition: all .3s ease; }
.crit-box:hover { transform: translateY(-2px); }
.crit-box.met     { background: var(--green-d); border: 1px solid var(--green-b); }
.crit-box.missing { background: var(--red-d);   border: 1px solid var(--red-b); }
.crit-title { font-family: var(--mono); font-size: .6rem; text-transform: uppercase; letter-spacing: .12em; font-weight: 700; margin-bottom: 8px; }
.crit-box.met     .crit-title { color: var(--green); }
.crit-box.missing .crit-title { color: var(--red); }
.crit-item { font-size: .78rem; line-height: 1.65; color: var(--text2); padding: 2px 0; display: flex; gap: 8px; font-weight: 500; }

.psource { font-family: var(--mono); font-size: .65rem; padding: 6px 12px; border-radius: 8px;
  background: var(--surface3); border: 1px solid var(--border2); color: var(--text2); display: inline-flex;
  align-items: center; gap: 6px; cursor: pointer; transition: all .25s ease; font-weight: 600; margin: 4px 4px 0 0; }
.psource:hover { border-color: var(--accent); color: var(--accent); background: var(--accent-d); transform: translateY(-2px); }
.pdf-badge { font-family: var(--mono); font-size: .58rem; font-weight: 800; color: var(--red);
  background: var(--red-d); padding: 2px 6px; border-radius: 4px; border: 1px solid var(--red-b); }

.hint-box { background: var(--yellow-d); border: 1px solid var(--yellow-b); border-radius: var(--r);
  padding: 12px 16px; font-size: .8rem; color: var(--text); line-height: 1.7; margin-top: 12px;
  display: flex; gap: 10px; font-weight: 500; box-shadow: 0 0 20px rgba(255,203,107,.15); }

.driver-chip { font-family: var(--mono); font-size: .66rem; padding: 4px 10px; border-radius: 6px;
  background: var(--surface3); border: 1px solid var(--border2); color: var(--text2);
  display: inline-block; font-weight: 600; margin: 3px 3px 0 0; }

.appeal-box { background: var(--surface2); border: 1px solid var(--border); border-radius: var(--r);
  padding: 18px; font-family: var(--mono); font-size: .72rem; line-height: 1.9; color: var(--text2);
  white-space: pre-wrap; max-height: 320px; overflow-y: auto; margin-top: 10px; }

.sub-approved { background: var(--green-d);  border: 1px solid var(--green-b);  color: var(--green);  border-radius: var(--r); padding: 12px 16px; font-size: .84rem; font-weight: 600; }
.sub-denied   { background: var(--red-d);    border: 1px solid var(--red-b);    color: var(--red);    border-radius: var(--r); padding: 12px 16px; font-size: .84rem; font-weight: 600; }
.sub-pending  { background: var(--yellow-d); border: 1px solid var(--yellow-b); color: var(--yellow); border-radius: var(--r); padding: 12px 16px; font-size: .84rem; font-weight: 600; }
.sub-error    { background: var(--red-d);    border: 1px solid var(--red-b);    color: var(--red);    border-radius: var(--r); padding: 12px 16px; font-size: .84rem; font-weight: 600; }

.req-id { font-family: var(--mono); font-size: .72rem; color: var(--text2); margin-bottom: 8px; }
.req-id code { background: var(--surface3); border: 1px solid var(--border2); border-radius: 5px; padding: 2px 8px; color: var(--accent); }

.empty-state { display: flex; flex-direction: column; align-items: center; justify-content: center;
  padding: 80px 20px; text-align: center; color: var(--muted); }
.empty-icon  { font-size: 3.2rem; opacity: .2; margin-bottom: 12px; }
.empty-title { font-family: var(--serif); font-style: italic; font-size: 1.7rem; color: var(--border2); font-weight: 300; }
.empty-sub   { font-family: var(--mono); font-size: .62rem; letter-spacing: .12em; margin-top: 6px; }

.run-row { display: flex; align-items: center; gap: 8px; padding: 6px 0; border-bottom: 1px solid rgba(30,39,56,.5); font-size: .72rem; }
.run-row:last-child { border-bottom: none; }
.run-payer { font-family: var(--mono); font-size: .62rem; color: var(--text2); min-width: 40px; font-weight: 600; }
.run-cpt   { font-family: var(--mono); font-size: .6rem; color: var(--muted); min-width: 40px; }
.run-badge { margin-left: auto; font-family: var(--mono); font-size: .56rem; padding: 3px 8px; border-radius: 5px; font-weight: 700; }
.rb-a { background: var(--green-d);  color: var(--green);  border: 1px solid var(--green-b); }
.rb-d { background: var(--red-d);    color: var(--red);    border: 1px solid var(--red-b); }
.rb-p { background: var(--yellow-d); color: var(--yellow); border: 1px solid var(--yellow-b); }

.rate-bar-bg   { flex: 1; height: 4px; background: var(--border); border-radius: 2px; overflow: hidden; }
.rate-bar-fill { height: 100%; border-radius: 2px; transition: width .6s ease; }
</style>
"""


def _render_topbar():
    st.markdown("""
    <div class="aa-topbar">
      <span class="aa-brand">AutoAuth</span>
      <span class="aa-topbar-sub" style="color:#2A3547">/</span>
      <span class="aa-topbar-sub">Prior Authorization Pipeline</span>
      <div class="ai-pill"><span class="ai-dot"></span>AI Agents Active</div>
    </div>
    """, unsafe_allow_html=True)


def _chips_html(codes: dict) -> str:
    html = ""
    for kind, cls in [("icd10","icd"),("cpt","cpt"),("hcpcs","hcpcs"),("loinc","loinc")]:
        vals = codes.get(kind) or []
        if vals:
            lbl = kind.upper().replace("10","10")
            html += f'<div class="code-lbl">{lbl}</div><div class="chip-group">'
            html += "".join(f'<span class="chip {cls}">{c}</span>' for c in vals)
            html += "</div>"
    return html


def _decision_hero_html(decision_value: str, pa_decision: dict, est_days: float) -> str:
    cls = {"APPROVED":"approved","DENIED":"denied","PENDING_MORE_INFO":"pending"}.get(decision_value,"error")
    verdict = {"APPROVED":"✓ APPROVED","DENIED":"✗ DENIED","PENDING_MORE_INFO":"⚑ PENDING"}.get(decision_value,"⚠ ERROR")
    bcolor = {"APPROVED":"var(--green)","DENIED":"var(--red)","PENDING_MORE_INFO":"var(--yellow)"}.get(decision_value,"var(--muted)")
    pre = pa_decision.get("pre_submission_risk") or {}
    ap  = pre.get("approval_probability","—")
    risk= pre.get("risk_level","—")
    conf= pa_decision.get("confidence",0)
    reason = pa_decision.get("reason","")

    return f"""
    <div class="dh {cls}">
      <div class="dh-top">
        <div>
          <div class="dh-lbl">Authorization Decision</div>
          <div class="dh-verdict" style="color:{bcolor}">{verdict}</div>
        </div>
        <div class="dh-kpis">
          <div class="dh-kpi">
            <div class="dh-kpi-val">{ap}{"%" if ap != "—" else ""}</div>
            <div class="dh-kpi-lbl">Pre-Score</div>
          </div>
          <div class="dh-kpi">
            <div class="dh-kpi-val">{conf}%</div>
            <div class="dh-kpi-lbl">AI Confidence</div>
            <div class="conf-bar-bg"><div class="conf-bar" style="width:{conf}%;background:{bcolor}"></div></div>
          </div>
          <div class="dh-kpi">
            <div class="dh-kpi-val" style="color:{('var(--green)' if risk=='LOW' else 'var(--red)' if risk=='HIGH' else 'var(--yellow)')}">{risk}</div>
            <div class="dh-kpi-lbl">Risk Level</div>
          </div>
          <div class="dh-kpi">
            <div class="dh-kpi-val">{est_days:.1f}d</div>
            <div class="dh-kpi-lbl">Days Saved</div>
          </div>
        </div>
      </div>
      <div class="dh-reason {cls}">{reason}</div>
    </div>"""


def _step_header_html(num: str, name: str, status: str, status_cls: str) -> str:
    return f"""
    <div class="step-card">
      <div class="step-header">
        <div>
          <div class="step-num">{num}</div>
          <div class="step-name">{name}</div>
        </div>
        <div class="step-status {status_cls}">{status}</div>
      </div>
      <div class="step-body">"""


def _submission_html(decision_value: str, req_id: str) -> str:
    if decision_value == "APPROVED":
        cls, msg = "sub-approved", f"✓ Submit electronically (mock). Case is <strong>APPROVED</strong>."
    elif decision_value == "DENIED":
        cls, msg = "sub-denied", "✗ Case is <strong>DENIED</strong>. Generate appeal letter and attach missing documentation."
    elif decision_value == "PENDING_MORE_INFO":
        cls, msg = "sub-pending", "⚑ Manual check required. Provide missing documentation and re-run."
    else:
        cls, msg = "sub-error", "⚠ Automated processing failed. Route to manual review."
    return f'<div class="req-id">Request ID: <code>{req_id}</code></div><div class="{cls}">{msg}</div>'


# ─── Page config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AutoAuth Agent — Prior Authorization Demo",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject template CSS
st.markdown(TEMPLATE_CSS, unsafe_allow_html=True)

# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<div class="sb-brand">AutoAuth</div><div class="sb-brand-sub">Prior Authorization Intelligence</div>', unsafe_allow_html=True)
    st.markdown("---")

    # Revenue Dashboard
    st.markdown('<div class="sb-sec-title">Revenue Acceleration Dashboard</div>', unsafe_allow_html=True)
    metrics = _load_metrics()

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Runs",    metrics.get("total_runs", 0))
        st.metric("Approvals",     metrics.get("approvals", 0))
        st.metric("Denials",       metrics.get("denials", 0))
    with col2:
        st.metric("Pending",       metrics.get("pending_more_info", 0))
        st.metric("Errors",        metrics.get("errors", 0))
        st.metric("Days Saved",    f"{metrics.get('cumulative_days_saved', 0.0):.1f}d")

    with st.expander("Reset dashboard metrics"):
        if st.button("Reset", use_container_width=True):
            _save_metrics({"total_runs":0,"approvals":0,"denials":0,
                           "pending_more_info":0,"errors":0,"cumulative_days_saved":0.0})
            st.rerun()

    st.markdown("---")

    # Payer Behavior Model
    st.markdown('<div class="sb-sec-title">Payer Behavior Model</div>', unsafe_allow_html=True)
    history      = _load_history()
    payer_stats  = compute_payer_behavior_by_payer(history)
    rates        = compute_payer_behavior(history)

    payer_rows = []
    for payer_id, v in payer_stats.items():
        if payer_id == "__all__":
            continue
        payer_rows.append({
            "payer_id": payer_id,
            "runs":     int(v.get("runs") or 0),
            "approved": int(v.get("approved") or 0),
            "approval_rate": float(v.get("approval_rate") or 0.0),
            "run_share":     float(v.get("run_share") or 0.0),
        })
    payer_rows = sorted(payer_rows, key=lambda r: (-r["runs"], -r["approval_rate"], r["payer_id"]))

    all_stats = payer_stats.get("__all__", {})
    m1, m2, m3 = st.columns(3)
    with m1: st.metric("All runs",      int(all_stats.get("runs") or 0))
    with m2: st.metric("All approvals", int(all_stats.get("approved") or 0))
    with m3: st.metric("Overall %",     f"{float(all_stats.get('approval_rate') or 0.0):.0%}")

    # Payer approval rate visual bars
    if payer_rows:
        bars_html = ""
        for r in payer_rows:
            rate_pct = int(r["approval_rate"] * 100)
            bcolor = "var(--green)" if rate_pct >= 70 else "var(--yellow)" if rate_pct >= 40 else "var(--red)"
            bars_html += f"""
            <div style="margin-bottom:10px">
              <div style="display:flex;justify-content:space-between;margin-bottom:4px">
                <span style="font-family:var(--mono);font-size:.62rem;color:var(--text2);font-weight:600">{r['payer_id'].upper()}</span>
                <span style="font-family:var(--mono);font-size:.62rem;color:{bcolor};font-weight:700">{rate_pct}%</span>
              </div>
              <div class="rate-bar-bg"><div class="rate-bar-fill" style="width:{rate_pct}%;background:{bcolor};box-shadow:0 0 8px {bcolor}"></div></div>
            </div>"""
        st.markdown(f'<div class="sb-sec-title" style="margin-top:12px">Approval Rate by Payer</div>{bars_html}', unsafe_allow_html=True)

    # CPT breakdown table
    rows = []
    for key, v in rates.items():
        pid, cpt = (key.split("::", 1) + [""])[:2]
        rows.append({
            "payer_id": pid, "cpt_code": cpt,
            "runs": int(v.get("runs") or 0),
            "approved": int(v.get("approved") or 0),
            "approval_rate": float(v.get("approval_rate") or 0.0),
        })
    rows = sorted(rows, key=lambda r: (-r["runs"], -r["approval_rate"]))
    if rows:
        st.markdown('<div class="sb-sec-title" style="margin-top:14px">CPT Breakdown</div>', unsafe_allow_html=True)
        st.dataframe(rows, use_container_width=True, hide_index=True,
                     column_config={"approval_rate": st.column_config.NumberColumn("Appr. Rate", format="%.2f")})

    # Recent runs
    st.markdown("---")
    st.markdown('<div class="sb-sec-title">Recent Runs</div>', unsafe_allow_html=True)
    recent = [{"payer_id": r.get("payer_id",""), "cpt_codes": r.get("cpt_codes",[""]),
               "status": r.get("status","")} for r in history[-7:]][::-1]
    if recent:
        runs_html = ""
        for r in recent:
            cpt = (r["cpt_codes"] or [""])[0]
            s   = r["status"]
            cls = "rb-a" if s=="APPROVED" else "rb-d" if s=="DENIED" else "rb-p"
            lbl = "✓ APPR" if s=="APPROVED" else "✗ DENIED" if s=="DENIED" else "⚑ PEND"
            runs_html += f"""<div class="run-row">
              <span class="run-payer">{r['payer_id'].upper()}</span>
              <span class="run-cpt">{cpt}</span>
              <span class="run-badge {cls}">{lbl}</span>
            </div>"""
        st.markdown(runs_html, unsafe_allow_html=True)
    else:
        st.caption("No runs yet this session.")

# ─── Main content ─────────────────────────────────────────────────────────────

# Topbar
_render_topbar()

left, right = st.columns([0.42, 0.58], gap="large")

with left:
    st.markdown("""
    <div class="panel-title">Prior Auth</div>
    <div class="panel-sub">v3.0 · Autonomous Agent Pipeline</div>
    """, unsafe_allow_html=True)

    with st.container():
        payer_label = st.selectbox("Payer", ["aetna", "cigna", "uhc", "PAYER_A (Demo → Aetna)"], index=0)
        payer = _resolve_payer(payer_label)

        input_mode = st.radio("Clinical Reader Input", ["CSV Patient", "EHR Note"], horizontal=True)

        col_t1, col_t2 = st.columns(2)
        with col_t1:
            force_denial = st.toggle("Force Denial (demo)", value=False)
        with col_t2:
            save_outputs = st.toggle("Save Outputs", value=True)

        scenario_fallback = st.selectbox(
            "Sample Scenario (used if no input provided)",
            list(_SAMPLE_SCENARIOS.keys()), index=0,
        )
        scenario = _SAMPLE_SCENARIOS[scenario_fallback]

        ehr_note   = ""
        csv_patient = None
        if input_mode == "EHR Note":
            ehr_note = st.text_area(
                "Unstructured Provider Documentation",
                value="", height=220,
                placeholder="Paste EHR note here. Leave blank to use sample scenario.",
            )
        else:
            csv_patient = st.number_input("Patient ID / Row", min_value=1, value=1, step=1)

        run_btn = st.button("▶  Run All Agents", type="primary", use_container_width=True)

with right:
    if "last_result" not in st.session_state:
        st.session_state.last_result = None
        st.session_state.last_appeal = None

    if run_btn:
        if input_mode == "EHR Note":
            input_data = (ehr_note or "").strip() or scenario.get("ehr", "")
        else:
            input_data = int(csv_patient) if csv_patient is not None else int(scenario.get("csv_patient", 1))

        with st.spinner("🔬 Running Clinical Reader Agent…"):
            clinical_bundle = clinical_extract(input_data, payer=payer)

        result = {
            "clinical_bundle": clinical_bundle,
            "pa_decision": None,
            "appeal_letter": None,
            "pipeline_summary": {},
        }

        if "error" in clinical_bundle:
            result["pa_decision"] = {"decision": "ERROR", "reason": clinical_bundle.get("error"), "confidence": 0}
            result["pipeline_summary"] = {"status": "FAILED_AT_CLINICAL_READER", "payer": payer}
        else:
            with st.spinner("🏛️ Running Policy Intelligence Agent…"):
                try:
                    pa_decision = run_policy_agent(clinical_bundle, vector_store_dir="./vector_store")
                except Exception as e:
                    pa_decision = {"decision": "ERROR", "reason": f"Policy agent failed: {e}", "confidence": 0}

            if force_denial:
                forced = dict(pa_decision or {})
                forced["decision"] = "DENIED"
                forced.setdefault("confidence", 0)
                forced.setdefault("reason", "Forced denial for demo.")
                forced.setdefault("criteria_missing", ["Demo: missing documentation"])
                forced.setdefault("appeal_hint", "Provide documentation and re-submit.")
                pa_decision = forced

            result["pa_decision"] = pa_decision
            decision = (pa_decision or {}).get("decision", "ERROR")
            pre = (pa_decision or {}).get("pre_submission_risk") or {}
            result["pipeline_summary"] = {
                "status": decision,
                "payer": clinical_bundle.get("payer"),
                "icd_codes": clinical_bundle.get("icd_codes"),
                "cpt_code": clinical_bundle.get("cpt_code"),
                "all_codes": clinical_bundle.get("codes", {}),
                "confidence": (pa_decision or {}).get("confidence"),
                "pre_approval_score": pre.get("approval_probability"),
                "risk_level": pre.get("risk_level"),
                "extraction_method": clinical_bundle.get("extraction_method"),
                "policy_sources": (pa_decision or {}).get("policy_sources", []),
            }

            # Update metrics
            metrics = _load_metrics()
            metrics["total_runs"] = int(metrics.get("total_runs", 0)) + 1
            if decision == "APPROVED":         metrics["approvals"]        = int(metrics.get("approvals",0)) + 1
            elif decision == "DENIED":         metrics["denials"]          = int(metrics.get("denials",0)) + 1
            elif decision == "PENDING_MORE_INFO": metrics["pending_more_info"] = int(metrics.get("pending_more_info",0)) + 1
            else:                              metrics["errors"]           = int(metrics.get("errors",0)) + 1
            metrics["cumulative_days_saved"] = float(metrics.get("cumulative_days_saved",0.0)) + _estimate_days_saved(decision)
            if save_outputs:
                _save_metrics(metrics)
                _append_history({
                    "ts": _now_ts(), "payer_id": payer,
                    "cpt_codes": [clinical_bundle.get("cpt_code") or ""],
                    "icd_codes": clinical_bundle.get("icd_codes") or [],
                    "status": decision,
                    "confidence": (pa_decision or {}).get("confidence"),
                })

        st.session_state.last_result = result
        st.session_state.last_appeal = None
        st.rerun()

    # ── Output panel ────────────────────────────────────────────────────────
    if st.session_state.last_result is None:
        st.markdown("""
        <div class="empty-state">
          <div class="empty-icon">⚕</div>
          <div class="empty-title">Configure &amp; run agents →</div>
          <div class="empty-sub">Select payer · choose scenario · click Run</div>
        </div>""", unsafe_allow_html=True)
    else:
        result         = st.session_state.last_result
        clinical       = result.get("clinical_bundle") or {}
        pa_decision    = result.get("pa_decision") or {}
        decision_value = pa_decision.get("decision", "")
        est_days       = _estimate_days_saved(decision_value)
        req_id         = f"PA-{uuid.uuid4().hex[:8].upper()}"

        # ── STEP 1 card: Clinical Reader ─────────────────────────────────
        codes    = clinical.get("codes", {}) or {}
        summary  = clinical.get("patient_summary") or clinical.get("summary") or ""
        evidence = clinical.get("supporting_evidence") or []
        if not isinstance(evidence, list):
            evidence = [str(evidence)]
        conf_pct = int((clinical.get("confidence") or 0) * 100)
        method   = clinical.get("extraction_method", "EHR NLP extraction")

        if "error" in clinical:
            step1_body = f'<div style="color:var(--red);font-family:var(--mono);font-size:.8rem">{clinical["error"]}</div>'
            step1_cls  = "ss-err"
            step1_lbl  = "Error"
        else:
            ev_html = "".join(f'<div class="ev">{e}</div>' for e in evidence[:5])
            step1_body = f"""
              <div style="display:flex;align-items:center;gap:8px;margin-bottom:12px">
                <span style="background:var(--accent-d);border:1px solid var(--accent-b);border-radius:24px;
                  font-family:var(--mono);font-size:.62rem;color:var(--accent);padding:4px 12px;font-weight:600">
                  <span class="ai-dot" style="display:inline-block;margin-right:6px"></span>
                  AI Confidence: {conf_pct}% · {method}
                </span>
              </div>
              <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px">
                <div>
                  <div class="code-lbl">Extracted Medical Codes</div>
                  {_chips_html(codes)}
                </div>
                <div>
                  <div class="code-lbl">Clinical Summary</div>
                  <div class="ev" style="border-left-color:var(--accent)">{summary}</div>
                  <div class="code-lbl">Supporting Evidence</div>
                  {ev_html}
                </div>
              </div>"""
            step1_cls = "ss-done"
            step1_lbl = "Complete"

        st.markdown(f"""
        <div class="step-card">
          <div class="step-header">
            <div><div class="step-num">STEP 01 / 03</div><div class="step-name">🔬 Clinical Reader Agent</div></div>
            <div class="step-status {step1_cls}">{step1_lbl}</div>
          </div>
          <div class="step-body">{step1_body}</div>
        </div>""", unsafe_allow_html=True)

        # ── STEP 2 card: Policy Intelligence ────────────────────────────
        crit_met     = pa_decision.get("criteria_met") or []
        crit_missing = pa_decision.get("criteria_missing") or []
        sources      = pa_decision.get("policy_sources") or []
        hint         = pa_decision.get("appeal_hint")
        drivers      = _decision_drivers(clinical, pa_decision)

        met_html = "".join(f'<div class="crit-item"><span style="color:var(--green);font-weight:700;margin-right:4px">✓</span>{c}</div>' for c in crit_met)
        mis_html = "".join(f'<div class="crit-item"><span style="color:var(--red);font-weight:700;margin-right:4px">✗</span>{c}</div>' for c in crit_missing)
        crit_html = ""
        if met_html or mis_html:
            crit_html = '<div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:14px">'
            if met_html:     crit_html += f'<div class="crit-box met"><div class="crit-title">✓ Criteria Met</div>{met_html}</div>'
            else:            crit_html += "<div></div>"
            if mis_html:     crit_html += f'<div class="crit-box missing"><div class="crit-title">✗ Criteria Missing</div>{mis_html}</div>'
            else:            crit_html += "<div></div>"
            crit_html += "</div>"

        drivers_html = ""
        if drivers:
            chips = "".join(f'<span class="driver-chip">▸ {d}</span>' for d in drivers)
            drivers_html = f'<div class="code-lbl" style="margin-top:14px">Decision Drivers</div><div>{chips}</div>'

        src_html = ""
        if sources:
            src_html = '<div class="code-lbl" style="margin-top:14px">Policy Sources Matched</div><div>'
            src_html += "".join(f'<span class="psource"><span class="pdf-badge">PDF</span>{s}</span>' for s in sources)
            src_html += "</div>"

        hint_html = f'<div class="hint-box">💡 <div><strong style="color:var(--yellow)">Appeal Guidance</strong><br>{hint}</div></div>' if hint else ""

        step2_cls = "ss-err" if decision_value == "ERROR" else "ss-done"
        step2_body = f"""
          {_decision_hero_html(decision_value, pa_decision, est_days)}
          {crit_html}
          {drivers_html}
          {src_html}
          {hint_html}"""

        st.markdown(f"""
        <div class="step-card">
          <div class="step-header">
            <div><div class="step-num">STEP 02 / 03</div><div class="step-name">🏛️ Policy Intelligence Agent</div></div>
            <div class="step-status {step2_cls}">Complete</div>
          </div>
          <div class="step-body">{step2_body}</div>
        </div>""", unsafe_allow_html=True)

        # ── STEP 3 card: Submission & Appeal ────────────────────────────
        can_appeal = decision_value == "DENIED"
        step3_title = "📝 Appeal Letter Agent" if can_appeal else "📬 Submission Agent"
        step3_cls   = "ss-done" if can_appeal or decision_value == "APPROVED" else "ss-wait"

        st.markdown(f"""
        <div class="step-card">
          <div class="step-header">
            <div><div class="step-num">STEP 03 / 03</div><div class="step-name">{step3_title}</div></div>
            <div class="step-status {step3_cls}">Complete</div>
          </div>
          <div class="step-body">
            {_submission_html(decision_value, req_id)}
          </div>
        </div>""", unsafe_allow_html=True)

        if can_appeal:
            st.markdown('<div class="code-lbl" style="margin-top:4px">Appeal Letter</div>', unsafe_allow_html=True)
            if st.button("⚡ Generate Appeal Letter", use_container_width=True):
                patient_for_appeal = {
                    "payer": clinical.get("payer", payer),
                    "icd_codes": clinical.get("icd_codes", []),
                    "cpt_code": clinical.get("cpt_code", ""),
                    "summary": clinical.get("summary", ""),
                    "evidence": clinical.get("evidence", ""),
                }
                with st.spinner("Generating appeal letter…"):
                    try:
                        st.session_state.last_appeal = generate_appeal(patient_for_appeal, pa_decision)
                    except Exception as e:
                        st.session_state.last_appeal = f"Appeal generation failed: {e}"

            if st.session_state.last_appeal:
                st.markdown(f'<div class="appeal-box">{st.session_state.last_appeal}</div>', unsafe_allow_html=True)

        # Raw JSON tab
        with st.expander("Raw JSON output"):
            st.markdown(f'<span style="font-family:var(--mono);font-size:.62rem;color:var(--muted)">Last run: {_now_ts()}</span>', unsafe_allow_html=True)
            st.code(json.dumps(result, indent=2), language="json")