"""
Lease Intelligence Platform — v2.0
Attorney-grade lease analysis with professional PDF reporting.

Key fixes over v1:
- Gemini now extracts RAW TEXT first via pdfplumber, then sends text to API
  (avoids File API upload failures that caused silent mock fallback)
- Claude prompt completely rewritten to attorney-level analysis depth
- PDF report rebuilt to law-firm quality: cover page, clause citations,
  negotiation scripts, per-tenant cost breakdown
- Debug logging so you can see exactly what's happening at each step
"""

import os, json, uuid, hashlib, time, logging, base64, io
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Optional
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("leaseintel")

app = FastAPI(title="Lease Intelligence Platform", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

static_path = Path(__file__).parent.parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")


# ── STEP 0: EXTRACT TEXT FROM PDF ────────────────────────────────────────────
# This is the critical fix. Instead of uploading binary PDF to Gemini File API
# (which times out or fails silently), we extract text first with pdfplumber,
# then send the text directly. Much more reliable.

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract raw text from PDF using pdfplumber. Returns full lease text."""
    try:
        import pdfplumber
        text_pages = []
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:
                    text_pages.append(f"--- PAGE {i+1} ---\n{text}")
        full_text = "\n\n".join(text_pages)
        logger.info(f"Extracted {len(full_text)} characters from {len(text_pages)} pages")
        return full_text
    except ImportError:
        logger.warning("pdfplumber not installed — using basic text extraction")
        # Fallback: try to decode any readable text from PDF bytes
        try:
            decoded = pdf_bytes.decode('latin-1', errors='ignore')
            # Extract text between stream markers
            import re
            texts = re.findall(r'BT\s*(.*?)\s*ET', decoded, re.DOTALL)
            return ' '.join(texts)[:50000]
        except:
            return ""


# ── GEMINI EXTRACTION ─────────────────────────────────────────────────────────

GEMINI_SYSTEM = """You are a residential lease extraction engine for a legal technology platform.
Your job is to extract structured data from lease agreements with high precision.

RULES:
1. Return ONLY valid JSON. No markdown, no explanation, no preamble.
2. Every field must have: {"value": ..., "confidence": 0.0-1.0}
3. If a field is not found: {"value": null, "confidence": 0.0, "null_reason": "explanation"}
4. Dates must be ISO format: YYYY-MM-DD
5. Dollar amounts must be decimal strings: "1650.00" not "$1,650"
6. Booleans must be true/false not "yes"/"no"
7. Arrays must be JSON arrays: ["electricity", "gas"]
8. If a clause is unusual or potentially illegal, add: "flag_for_reasoning": true
9. Extract tenant names as a comma-separated string if multiple tenants
10. For entry notice: if NO hours are specified, set value to null and flag_for_reasoning: true

CONFIDENCE GUIDE:
- 0.95-1.0: Exact verbatim number/date found
- 0.80-0.94: Clear statement, minor inference
- 0.60-0.79: Inferred from context
- 0.40-0.59: Uncertain, could be interpreted differently
- 0.0-0.39: Not found or extremely ambiguous

FLAG THESE CLAUSES (set flag_for_reasoning: true):
- Entry notice under 24 hours OR not specified at all
- Renewal window under 30 days OR not specified
- Security deposit over 1.5x monthly rent
- Late fee escalation clauses (fees that increase over time)
- Parental guarantee / co-signer requirement
- Mandatory binding arbitration (waives right to sue)
- Joint and several liability across multiple tenants
- Deposit deductions above 100% of actual cost
- Holdover rent increases
- Automatic renewal clauses"""


def build_gemini_prompt(state: str, lease_text: str) -> str:
    return f"""Extract all fields from this {state} residential lease agreement text.

LEASE TEXT:
{lease_text[:40000]}

Return a JSON object with EXACTLY these fields:

{{
  "tenant_full_name": {{"value": "comma-separated if multiple", "confidence": 0.0}},
  "landlord_full_name": {{"value": "...", "confidence": 0.0}},
  "landlord_entity_type": {{"value": "Individual|LLC|Corp|Trust", "confidence": 0.0}},
  "property_address": {{"value": "street address only", "confidence": 0.0}},
  "property_city": {{"value": "...", "confidence": 0.0}},
  "property_state": {{"value": "2-letter state", "confidence": 0.0}},
  "property_zip": {{"value": "...", "confidence": 0.0}},
  "unit_number": {{"value": "...", "confidence": 0.0}},
  "property_type": {{"value": "House|Apartment|Condo|Townhouse|Room", "confidence": 0.0}},
  "square_footage": {{"value": null, "confidence": 0.0}},
  "furnished": {{"value": false, "confidence": 0.0}},
  "monthly_rent": {{"value": "decimal string", "confidence": 0.0}},
  "number_of_tenants": {{"value": 1, "confidence": 0.0}},
  "security_deposit": {{"value": "decimal string", "confidence": 0.0}},
  "last_month_deposit": {{"value": null, "confidence": 0.0}},
  "pet_deposit": {{"value": null, "confidence": 0.0}},
  "other_fees_monthly": {{"value": null, "confidence": 0.0}},
  "late_fee_structure": {{"value": "describe the full late fee structure verbatim", "confidence": 0.0}},
  "late_fee_initial_amount": {{"value": null, "confidence": 0.0}},
  "late_fee_initial_pct": {{"value": null, "confidence": 0.0, "null_reason": "use if percentage not dollar"}},
  "late_fee_grace_days": {{"value": null, "confidence": 0.0}},
  "late_fee_escalates": {{"value": false, "confidence": 0.0, "flag_for_reasoning": false}},
  "nsf_fee": {{"value": null, "confidence": 0.0}},
  "lease_type": {{"value": "fixed_term|month_to_month|unknown", "confidence": 0.0}},
  "lease_start_date": {{"value": "YYYY-MM-DD", "confidence": 0.0}},
  "lease_end_date": {{"value": "YYYY-MM-DD", "confidence": 0.0}},
  "rent_due_day": {{"value": 1, "confidence": 0.0}},
  "renewal_notice_days_required": {{"value": null, "confidence": 0.0}},
  "renewal_deadline_explicit": {{"value": null, "confidence": 0.0, "null_reason": "specific date if stated"}},
  "holdover_rent_amount": {{"value": null, "confidence": 0.0}},
  "early_termination_fee": {{"value": null, "confidence": 0.0}},
  "early_termination_notice_days": {{"value": null, "confidence": 0.0}},
  "landlord_entry_notice_hours": {{"value": null, "confidence": 0.0}},
  "landlord_entry_clause_verbatim": {{"value": "exact quoted text of entry clause", "confidence": 0.0}},
  "utilities_tenant_responsible": {{"value": [], "confidence": 0.0}},
  "utilities_landlord_responsible": {{"value": [], "confidence": 0.0}},
  "all_utilities_tenant": {{"value": false, "confidence": 0.0}},
  "tenant_maintenance_obligations": {{"value": "describe", "confidence": 0.0}},
  "repair_deduction_multiplier": {{"value": null, "confidence": 0.0, "null_reason": "e.g. 1.5 if 150% of cost"}},
  "parental_guarantee_required": {{"value": false, "confidence": 0.0, "flag_for_reasoning": false}},
  "joint_several_liability": {{"value": false, "confidence": 0.0}},
  "mandatory_arbitration": {{"value": false, "confidence": 0.0, "flag_for_reasoning": false}},
  "pets_allowed": {{"value": false, "confidence": 0.0}},
  "subletting_allowed": {{"value": false, "confidence": 0.0}},
  "smoking_prohibited": {{"value": true, "confidence": 0.0}},
  "guest_policy": {{"value": "describe", "confidence": 0.0}},
  "police_violation_fee": {{"value": null, "confidence": 0.0, "flag_for_reasoning": false}},
  "attorneys_fees_clause": {{"value": false, "confidence": 0.0}},
  "lease_signed_date": {{"value": null, "confidence": 0.0}},
  "lease_document_pages": {{"value": null, "confidence": 0.0}}
}}"""


async def call_gemini_with_text(lease_text: str, state: str) -> dict:
    """Send lease text directly to Gemini. Much more reliable than File API."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.warning("No GEMINI_API_KEY configured")
        return None

    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)

        model = genai.GenerativeModel(
            model_name=os.getenv("GEMINI_MODEL", "gemini-1.5-pro"),
            system_instruction=GEMINI_SYSTEM,
            generation_config=genai.types.GenerationConfig(
                temperature=0.05,
                response_mime_type="application/json",
                max_output_tokens=8192,
            )
        )

        prompt = build_gemini_prompt(state, lease_text)
        logger.info(f"Sending {len(prompt)} char prompt to Gemini")

        response = model.generate_content(prompt)
        raw = response.text.strip()

        # Strip any accidental markdown
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"): raw = raw[4:]
        raw = raw.strip()

        parsed = json.loads(raw)
        logger.info(f"Gemini returned {len(parsed)} fields")
        return parsed

    except json.JSONDecodeError as e:
        logger.error(f"Gemini returned invalid JSON: {e}")
        return None
    except Exception as e:
        logger.error(f"Gemini API call failed: {e}")
        return None


# ── PARSE GEMINI RESPONSE ─────────────────────────────────────────────────────

def parse_field(raw_field, field_type="str"):
    """Extract value and confidence from a Gemini field dict."""
    if not isinstance(raw_field, dict):
        return raw_field, 0.5, False
    value = raw_field.get("value")
    confidence = float(raw_field.get("confidence", 0.0))
    flag = bool(raw_field.get("flag_for_reasoning", False))
    if value is None:
        return None, confidence, flag
    if field_type == "decimal":
        try: value = Decimal(str(value).replace(",","").replace("$","").strip())
        except: value = None
    elif field_type == "int":
        try: value = int(float(str(value)))
        except: value = None
    elif field_type == "float":
        try: value = float(str(value))
        except: value = None
    elif field_type == "date":
        try: value = date.fromisoformat(str(value))
        except: value = None
    elif field_type == "bool":
        if isinstance(value, bool): pass
        elif isinstance(value, str): value = value.lower() in ("true","yes","1")
        else: value = bool(value)
    elif field_type == "list":
        if not isinstance(value, list):
            value = [v.strip() for v in str(value).split(",") if v.strip()]
    return value, confidence, flag


def parse_gemini_response(raw: dict) -> tuple:
    """Parse Gemini response into clean layer1 dict with confidence scores."""
    if not raw:
        return {}, {}, []

    layer1 = {}
    confidence_scores = {}
    reasoning_flags = []

    FIELD_MAP = {
        "tenant_full_name": "str",
        "landlord_full_name": "str",
        "landlord_entity_type": "str",
        "property_address": "str",
        "property_city": "str",
        "property_state": "str",
        "property_zip": "str",
        "unit_number": "str",
        "property_type": "str",
        "square_footage": "float",
        "furnished": "bool",
        "monthly_rent": "decimal",
        "number_of_tenants": "int",
        "security_deposit": "decimal",
        "last_month_deposit": "decimal",
        "pet_deposit": "decimal",
        "other_fees_monthly": "decimal",
        "late_fee_structure": "str",
        "late_fee_initial_amount": "decimal",
        "late_fee_initial_pct": "float",
        "late_fee_grace_days": "int",
        "late_fee_escalates": "bool",
        "nsf_fee": "decimal",
        "lease_type": "str",
        "lease_start_date": "date",
        "lease_end_date": "date",
        "rent_due_day": "int",
        "renewal_notice_days_required": "int",
        "renewal_deadline_explicit": "date",
        "holdover_rent_amount": "decimal",
        "early_termination_fee": "decimal",
        "early_termination_notice_days": "int",
        "landlord_entry_notice_hours": "int",
        "landlord_entry_clause_verbatim": "str",
        "utilities_tenant_responsible": "list",
        "utilities_landlord_responsible": "list",
        "all_utilities_tenant": "bool",
        "tenant_maintenance_obligations": "str",
        "repair_deduction_multiplier": "float",
        "parental_guarantee_required": "bool",
        "joint_several_liability": "bool",
        "mandatory_arbitration": "bool",
        "pets_allowed": "bool",
        "subletting_allowed": "bool",
        "smoking_prohibited": "bool",
        "guest_policy": "str",
        "police_violation_fee": "decimal",
        "attorneys_fees_clause": "bool",
        "lease_signed_date": "date",
        "lease_document_pages": "int",
    }

    for field, ftype in FIELD_MAP.items():
        raw_field = raw.get(field, {"value": None, "confidence": 0.0})
        value, confidence, flag = parse_field(raw_field, ftype)
        layer1[field] = value
        confidence_scores[field] = confidence
        if flag:
            reasoning_flags.append(field)

    return layer1, confidence_scores, reasoning_flags


# ── LAYER 2: COMPUTED FIELDS ──────────────────────────────────────────────────

def compute_layer2(l1: dict) -> dict:
    today = date.today()
    l2 = {}

    rent = l1.get("monthly_rent") or Decimal("0")
    sec  = l1.get("security_deposit") or Decimal("0")
    last = l1.get("last_month_deposit") or Decimal("0")
    pet  = l1.get("pet_deposit") or Decimal("0")
    fees = l1.get("other_fees_monthly") or Decimal("0")
    num_tenants = l1.get("number_of_tenants") or 1

    l2["effective_monthly_cost"] = rent + fees
    l2["total_upfront_cost"] = rent + sec + last + pet
    l2["annualized_rent"] = rent * 12

    if num_tenants > 1 and rent > 0:
        l2["rent_per_tenant"] = rent / num_tenants
        l2["deposit_per_tenant"] = sec / num_tenants
        l2["upfront_per_tenant"] = l2["total_upfront_cost"] / num_tenants

    start = l1.get("lease_start_date")
    end   = l1.get("lease_end_date")
    if start and end and isinstance(start, date) and isinstance(end, date):
        delta = end - start
        l2["lease_term_days"]   = delta.days
        l2["lease_term_months"] = round(delta.days / 30.44)
        if rent > 0:
            l2["total_liability_term"] = rent * l2["lease_term_months"]
        l2["days_until_lease_end"] = (end - today).days

        notice = l1.get("renewal_notice_days_required")
        explicit_dl = l1.get("renewal_deadline_explicit")
        if explicit_dl and isinstance(explicit_dl, date):
            l2["renewal_deadline_date"] = explicit_dl
            l2["days_until_renewal_deadline"] = (explicit_dl - today).days
        elif notice:
            dl = end - timedelta(days=notice)
            l2["renewal_deadline_date"] = dl
            l2["days_until_renewal_deadline"] = (dl - today).days

    sqft = l1.get("square_footage")
    eff  = l2.get("effective_monthly_cost", Decimal("0"))
    if sqft and sqft > 0 and eff > 0:
        l2["implied_cost_per_sqft_monthly"] = float(eff) / sqft

    if sec > 0 and rent > 0:
        l2["deposit_as_months_rent"] = float(sec) / float(rent)

    late_amt = l1.get("late_fee_initial_amount")
    late_pct = l1.get("late_fee_initial_pct")
    if late_amt and rent > 0:
        l2["late_fee_as_pct_rent"] = float(late_amt) / float(rent)
    elif late_pct:
        l2["late_fee_as_pct_rent"] = late_pct / 100
        if rent > 0:
            l2["late_fee_dollar_equivalent"] = rent * Decimal(str(late_pct / 100))

    holdover = l1.get("holdover_rent_amount")
    if holdover and rent > 0:
        l2["holdover_rent_increase"] = holdover - rent
        l2["holdover_rent_increase_pct"] = float(holdover - rent) / float(rent)

    return l2


# ── ATTORNEY-GRADE CLAUDE PROMPT ──────────────────────────────────────────────

CLAUDE_SYSTEM = """You are a senior residential tenants' rights attorney with 20 years of experience 
reviewing residential leases in all 50 US states. You specialize in protecting tenants from 
one-sided, non-compliant, and predatory lease clauses.

Your analysis must be:
- LEGALLY PRECISE: Cite specific state statutes when a clause violates or approaches the limit of state law
- PRACTICALLY ACTIONABLE: Give tenants specific language to request changes and exactly what to say
- HONEST ABOUT SEVERITY: Don't inflate risk scores. A well-written lease should score GREEN even if it has some tenant-unfavorable clauses that are still legal
- TENANT-FOCUSED: Written for a first-time renter who has never had a lawyer review a lease

SCORING RUBRIC:
- 0-25 = GREEN: Tenant-favorable or balanced. Standard, enforceable terms.
- 26-50 = YELLOW: A few concerning clauses worth negotiating, but nothing illegal or severely one-sided.
- 51-75 = ORANGE: Multiple significant issues. Negotiate specific clauses before signing.
- 76-100 = RED: Serious concerns. Potentially illegal clauses or severely one-sided terms. Consider walking away.

SEVERITY DEFINITIONS:
- CRITICAL: Clause is potentially illegal under state law, or causes severe irreversible harm if enforced
- HIGH: Clause is legal but significantly landlord-favorable, materially affects tenant's rights or finances
- MEDIUM: Clause is below industry standard or creates meaningful financial exposure worth negotiating
- LOW: Minor issue or informational — worth knowing but not a deal-breaker

ALWAYS CHECK:
1. Entry notice vs. state minimum (most states: 24 hours)
2. Security deposit vs. state cap (VA: 2 months, CA: 2 months, NY: 1 month)
3. Late fee structure — escalating fees, short grace periods
4. Renewal/notice deadlines — short windows trap tenants
5. Joint and several liability — especially for student group leases
6. Parental guarantees — parents often don't understand they're primary obligors
7. Mandatory arbitration — waives constitutional right to jury trial
8. Repair deduction multipliers above 100%
9. Holdover rent increases — automatic rent jumps for staying one extra day
10. Attorney fee clauses — one-sided vs. mutual

Return ONLY valid JSON. No markdown, no preamble."""


def build_claude_prompt(layer1: dict, layer2: dict, reasoning_flags: list, state: str) -> str:
    STATE_LAW = {
        "VA": {
            "name": "Virginia",
            "entry_notice_hours": 24,
            "entry_statute": "VA Code § 55.1-1229",
            "deposit_max_months": 2,
            "deposit_statute": "VA Code § 55.1-1226",
            "late_fee_note": "Virginia does not cap late fees by statute but courts apply reasonableness standard",
            "key_statutes": "Virginia Residential Landlord and Tenant Act (VRLTA), VA Code §§ 55.1-1200 et seq."
        },
        "CA": {
            "name": "California",
            "entry_notice_hours": 24,
            "entry_statute": "CA Civil Code § 1954",
            "deposit_max_months": 2,
            "deposit_statute": "CA Civil Code § 1950.5",
            "late_fee_note": "Late fees must be a reasonable estimate of actual damages",
            "key_statutes": "California Civil Code §§ 1940-1954.1"
        },
        "NY": {
            "name": "New York",
            "entry_notice_hours": 24,
            "entry_statute": "NY Real Property Law § 235-b",
            "deposit_max_months": 1,
            "deposit_statute": "NY General Obligations Law § 7-108",
            "late_fee_note": "Late fees capped at $50 or 5% of monthly rent, whichever is less",
            "key_statutes": "New York Real Property Law"
        },
        "TX": {
            "name": "Texas",
            "entry_notice_hours": 24,
            "entry_statute": "TX Property Code § 92.0081",
            "deposit_max_months": None,
            "deposit_statute": "TX Property Code § 92.103 (no statutory cap)",
            "late_fee_note": "Late fees must be reasonable; no specific cap",
            "key_statutes": "Texas Property Code Chapter 92"
        },
        "FL": {
            "name": "Florida",
            "entry_notice_hours": 12,
            "entry_statute": "FL Statute § 83.53",
            "deposit_max_months": None,
            "deposit_statute": "FL Statute § 83.49 (no cap, but must be held in separate account)",
            "late_fee_note": "Late fees must be reasonable",
            "key_statutes": "Florida Residential Landlord and Tenant Act, FL Stat. § 83.40"
        },
    }

    law = STATE_LAW.get(state.upper(), {
        "name": state,
        "entry_notice_hours": 24,
        "entry_statute": "state law (check your state's landlord-tenant act)",
        "deposit_max_months": 2,
        "deposit_statute": "your state's security deposit statute",
        "late_fee_note": "Check your state's late fee rules",
        "key_statutes": f"{state} Residential Landlord and Tenant Act"
    })

    def s(v):
        if isinstance(v, (date, datetime)): return v.isoformat()
        if isinstance(v, Decimal): return str(v)
        return v

    # Build a rich context for Claude
    data_summary = {
        # Parties
        "tenants": s(layer1.get("tenant_full_name")),
        "number_of_tenants": layer1.get("number_of_tenants"),
        "landlord": s(layer1.get("landlord_full_name")),
        "landlord_type": s(layer1.get("landlord_entity_type")),
        "property": f"{s(layer1.get('property_address'))}, {s(layer1.get('property_city'))}, {state}",
        # Financials
        "monthly_rent": s(layer1.get("monthly_rent")),
        "rent_per_tenant": s(layer2.get("rent_per_tenant")),
        "security_deposit": s(layer1.get("security_deposit")),
        "deposit_per_tenant": s(layer2.get("deposit_per_tenant")),
        "deposit_as_months_rent": layer2.get("deposit_as_months_rent"),
        "total_upfront_cost": s(layer2.get("total_upfront_cost")),
        "upfront_per_tenant": s(layer2.get("upfront_per_tenant")),
        "total_liability_term": s(layer2.get("total_liability_term")),
        "annualized_rent": s(layer2.get("annualized_rent")),
        # Term
        "lease_start": s(layer1.get("lease_start_date")),
        "lease_end": s(layer1.get("lease_end_date")),
        "lease_term_months": layer2.get("lease_term_months"),
        "days_until_lease_end": layer2.get("days_until_lease_end"),
        # Renewal
        "renewal_notice_days": layer1.get("renewal_notice_days_required"),
        "renewal_deadline_explicit": s(layer1.get("renewal_deadline_explicit")),
        "renewal_deadline_computed": s(layer2.get("renewal_deadline_date")),
        "days_until_renewal_deadline": layer2.get("days_until_renewal_deadline"),
        "holdover_rent": s(layer1.get("holdover_rent_amount")),
        "holdover_increase": s(layer2.get("holdover_rent_increase")),
        # Fees
        "late_fee_structure_verbatim": layer1.get("late_fee_structure"),
        "late_fee_initial_amount": s(layer1.get("late_fee_initial_amount")),
        "late_fee_initial_pct": layer1.get("late_fee_initial_pct"),
        "late_fee_grace_days": layer1.get("late_fee_grace_days"),
        "late_fee_escalates": layer1.get("late_fee_escalates"),
        "late_fee_as_pct_rent": layer2.get("late_fee_as_pct_rent"),
        "nsf_fee": s(layer1.get("nsf_fee")),
        "police_violation_fee": s(layer1.get("police_violation_fee")),
        # Entry
        "entry_notice_hours": layer1.get("landlord_entry_notice_hours"),
        "entry_clause_verbatim": layer1.get("landlord_entry_clause_verbatim"),
        "state_minimum_entry_hours": law["entry_notice_hours"],
        "entry_statute": law["entry_statute"],
        # Deposit law
        "deposit_max_months_state": law["deposit_max_months"],
        "deposit_statute": law["deposit_statute"],
        "repair_deduction_multiplier": layer1.get("repair_deduction_multiplier"),
        # Risk flags
        "parental_guarantee_required": layer1.get("parental_guarantee_required"),
        "joint_several_liability": layer1.get("joint_several_liability"),
        "mandatory_arbitration": layer1.get("mandatory_arbitration"),
        "attorneys_fees_clause": layer1.get("attorneys_fees_clause"),
        # Utilities
        "utilities_tenant": layer1.get("utilities_tenant_responsible"),
        "utilities_landlord": layer1.get("utilities_landlord_responsible"),
        "all_utilities_tenant": layer1.get("all_utilities_tenant"),
        # Other
        "early_termination_fee": s(layer1.get("early_termination_fee")),
        "early_termination_notice_days": layer1.get("early_termination_notice_days"),
        "pets_allowed": layer1.get("pets_allowed"),
        "subletting_allowed": layer1.get("subletting_allowed"),
        "guest_policy": layer1.get("guest_policy"),
        "tenant_maintenance_obligations": layer1.get("tenant_maintenance_obligations"),
        # Reasoning flags from extraction
        "fields_flagged_during_extraction": reasoning_flags,
        # State law context
        "state_law": law,
    }

    return f"""Analyze this {law['name']} residential lease for tenant risk. 
Governing law: {law['key_statutes']}

EXTRACTED LEASE DATA:
{json.dumps(data_summary, indent=2, default=str)}

Return this JSON structure:

{{
  "flags": [
    {{
      "flag_id": "snake_case_id",
      "severity": "CRITICAL|HIGH|MEDIUM|LOW",
      "category": "entry_rights|financial|renewal|termination|maintenance|liability|privacy|arbitration|utilities",
      "short_description": "One sentence — what is the problem",
      "detailed_explanation": "2-3 sentences explaining why this matters practically to the tenant",
      "raw_clause_citation": "exact quoted language from the lease, or null if inferred",
      "statute_citation": "specific code section if applicable, e.g. VA Code § 55.1-1229",
      "jurisdiction_note": "how state law applies to this specific clause",
      "what_to_negotiate": "Exact language to request: 'Please amend Section X to read: ...'",
      "if_they_refuse": "What it means for you if landlord won't change this"
    }}
  ],
  "overall_lease_risk_score": 0,
  "renewal_risk_score": 0,
  "financial_burden_score": 0,
  "landlord_access_risk_score": 0,
  "termination_risk_score": 0,
  "liability_risk_score": 0,
  "risk_tier": "GREEN|YELLOW|ORANGE|RED",
  "score_rationale": "2-3 sentences explaining the overall score and what drives it",
  "lease_compared_to_market": "1-2 sentences: is this lease better, worse, or typical vs. comparable leases in this market?",
  "tenant_summary_biggest_risk": "One direct sentence naming the single most important risk",
  "tenant_summary_biggest_positive": "One direct sentence naming the best clause for the tenant",
  "tenant_summary_key_action": "One specific action the tenant should take before signing",
  "negotiation_priority_order": ["flag_id_1", "flag_id_2", "flag_id_3"],
  "walk_away_triggers": ["describe conditions under which tenant should refuse to sign"],
  "tenant_disclaimer": "This analysis is for informational purposes only and does not constitute legal advice. Consult a licensed attorney for guidance specific to your situation."
}}"""


async def call_claude(layer1, layer2, reasoning_flags, state):
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        logger.warning("No ANTHROPIC_API_KEY — returning null reasoning")
        return None

    try:
        from anthropic import Anthropic
        client = Anthropic(api_key=api_key)

        prompt = build_claude_prompt(layer1, layer2, reasoning_flags, state)
        logger.info(f"Sending {len(prompt)} char prompt to Claude")

        response = client.messages.create(
            model=os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6"),
            max_tokens=6000,
            system=CLAUDE_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )

        raw = response.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"): raw = raw[4:]
        result = json.loads(raw.strip())
        logger.info(f"Claude returned {len(result.get('flags', []))} flags, score={result.get('overall_lease_risk_score')}")
        return result

    except json.JSONDecodeError as e:
        logger.error(f"Claude returned invalid JSON: {e}")
        return None
    except Exception as e:
        logger.error(f"Claude API call failed: {e}")
        return None


# ── PROFESSIONAL PDF REPORT ───────────────────────────────────────────────────

def generate_professional_pdf(layer1, layer2, claude_output, extraction_id):
    """Generate a law-firm quality PDF report."""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.colors import HexColor, white, black
        from reportlab.lib.units import inch
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
        from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                         TableStyle, HRFlowable, PageBreak, KeepTogether)

        NAVY  = HexColor("#065A82"); TEAL  = HexColor("#1C7293"); MINT  = HexColor("#02C39A")
        LGRAY = HexColor("#F1F5F9"); MGRAY = HexColor("#CBD5E1"); DGRAY = HexColor("#475569")
        RED   = HexColor("#DC2626"); ORANGE= HexColor("#EA580C"); YELLOW= HexColor("#D97706")
        GREEN = HexColor("#16A34A"); CREAM = HexColor("#FFFBF5"); SLATE = HexColor("#1E293B")
        SCORE_C = {"GREEN": GREEN, "YELLOW": YELLOW, "ORANGE": ORANGE, "RED": RED}
        SEV_C   = {"CRITICAL": RED, "HIGH": ORANGE, "MEDIUM": YELLOW, "LOW": GREEN}

        def s(v):
            if isinstance(v, (date, datetime)): return v.isoformat()
            if isinstance(v, Decimal): return f"{float(str(v)):,.2f}"
            if v is None: return "—"
            return str(v)

        def fmt_dollar(v):
            if v is None: return "—"
            try: return f"${float(str(v)):,.0f}"
            except: return str(v)

        score = claude_output.get("overall_lease_risk_score", 0) if claude_output else 0
        tier  = claude_output.get("risk_tier", "YELLOW") if claude_output else "YELLOW"
        flags = claude_output.get("flags", []) if claude_output else []
        tier_color = SCORE_C.get(tier, YELLOW)
        tier_labels = {"GREEN":"Favorable","YELLOW":"Review Recommended",
                       "ORANGE":"Negotiate Before Signing","RED":"Serious Concerns — Seek Legal Advice"}

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter,
                                leftMargin=0.85*inch, rightMargin=0.85*inch,
                                topMargin=0.75*inch, bottomMargin=0.75*inch,
                                title="Lease Intelligence Analysis Report")

        # ── STYLES ──
        body_style = ParagraphStyle("body", fontName="Helvetica", fontSize=9.5,
                                    textColor=SLATE, leading=15, spaceAfter=6)
        label_style = ParagraphStyle("label", fontName="Helvetica-Bold", fontSize=8,
                                     textColor=DGRAY, leading=12, spaceBefore=2)
        value_style = ParagraphStyle("value", fontName="Helvetica", fontSize=11,
                                     textColor=SLATE, leading=16)
        section_style = ParagraphStyle("section", fontName="Helvetica-Bold", fontSize=13,
                                       textColor=NAVY, leading=18, spaceBefore=18, spaceAfter=8)
        flag_title_style = ParagraphStyle("flagtitle", fontName="Helvetica-Bold", fontSize=10,
                                          textColor=SLATE, leading=15)
        flag_body_style  = ParagraphStyle("flagbody", fontName="Helvetica", fontSize=9,
                                          textColor=DGRAY, leading=14)
        cite_style = ParagraphStyle("cite", fontName="Helvetica-Oblique", fontSize=8.5,
                                    textColor=TEAL, leading=13)
        action_style = ParagraphStyle("action", fontName="Helvetica-Bold", fontSize=9,
                                      textColor=NAVY, leading=14)
        small_style = ParagraphStyle("small", fontName="Helvetica", fontSize=8,
                                     textColor=DGRAY, leading=12)

        story = []

        # ── COVER HEADER ──
        hdr = Table([[
            Paragraph('<font color="white"><b>LEASE INTELLIGENCE</b></font>',
                      ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=18, textColor=white)),
            Paragraph(f'<font color="#CBD5E1">Analysis Report · {datetime.today().strftime("%B %d, %Y")}</font>',
                      ParagraphStyle("h2", fontName="Helvetica", fontSize=10, textColor=white,
                                     alignment=TA_RIGHT)),
        ]], colWidths=[4*inch, 2.8*inch])
        hdr.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,-1),NAVY),
            ("TOPPADDING",(0,0),(-1,-1),16),("BOTTOMPADDING",(0,0),(-1,-1),16),
            ("LEFTPADDING",(0,0),(0,0),20),("RIGHTPADDING",(-1,0),(-1,0),20),
            ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ]))
        story.append(hdr)

        # Property address bar
        addr = f"{s(layer1.get('property_address'))}, {s(layer1.get('property_city'))}, {s(layer1.get('property_state'))} {s(layer1.get('property_zip'))}"
        tenants = s(layer1.get("tenant_full_name"))
        addr_bar = Table([[
            Paragraph(f'<b>{addr}</b>', ParagraphStyle("addr", fontName="Helvetica-Bold",
                                                       fontSize=12, textColor=NAVY)),
            Paragraph(f'Tenant(s): {tenants}', ParagraphStyle("ten", fontName="Helvetica",
                                                               fontSize=9, textColor=DGRAY, alignment=TA_RIGHT)),
        ]], colWidths=[4*inch, 2.8*inch])
        addr_bar.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,-1),LGRAY),
            ("TOPPADDING",(0,0),(-1,-1),10),("BOTTOMPADDING",(0,0),(-1,-1),10),
            ("LEFTPADDING",(0,0),(0,0),20),("RIGHTPADDING",(-1,0),(-1,0),20),
            ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
            ("LINEBELOW",(0,0),(-1,-1),2,MINT),
        ]))
        story.append(addr_bar)
        story.append(Spacer(1, 14))

        # ── RISK SCORE PANEL ──
        score_bar = Table([[
            Paragraph(f'<font color="white"><b>{score}</b><br/><font size="10">/ 100</font></font>',
                      ParagraphStyle("sc", fontName="Helvetica-Bold", fontSize=32, textColor=white,
                                     alignment=TA_CENTER, leading=38)),
            [Paragraph(f'<font color="white"><b>{tier}</b></font>',
                       ParagraphStyle("tier", fontName="Helvetica-Bold", fontSize=20, textColor=white)),
             Paragraph(f'<font color="#E2E8F0">{tier_labels.get(tier,"")}</font>',
                       ParagraphStyle("tl", fontName="Helvetica", fontSize=11, textColor=white, leading=16)),
             Spacer(1,4),
             Paragraph(f'<font color="#CBD5E1">{claude_output.get("score_rationale","") if claude_output else ""}</font>',
                       ParagraphStyle("sr", fontName="Helvetica", fontSize=9, textColor=white,
                                      leading=14, alignment=TA_JUSTIFY))],
        ]], colWidths=[1.3*inch, 5.5*inch])
        score_bar.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,-1),tier_color),
            ("TOPPADDING",(0,0),(-1,-1),16),("BOTTOMPADDING",(0,0),(-1,-1),16),
            ("LEFTPADDING",(0,0),(0,0),16),("LEFTPADDING",(1,0),(1,0),16),
            ("RIGHTPADDING",(-1,0),(-1,0),16),
            ("VALIGN",(0,0),(0,0),"MIDDLE"),("VALIGN",(1,0),(1,0),"TOP"),
            ("LINEAFTER",(0,0),(0,0),1,HexColor("#FFFFFF33")),
        ]))
        story.append(score_bar)
        story.append(Spacer(1, 14))

        # ── SUB-SCORES ──
        sub_fields = [
            ("Financial Burden",   "financial_burden_score"),
            ("Renewal Risk",       "renewal_risk_score"),
            ("Access Rights",      "landlord_access_risk_score"),
            ("Termination Risk",   "termination_risk_score"),
            ("Liability Risk",     "liability_risk_score"),
        ]
        def score_color(v):
            if v is None: return MGRAY
            if v >= 75: return RED
            if v >= 50: return ORANGE
            if v >= 25: return YELLOW
            return GREEN

        sub_cells = []
        for label, key in sub_fields:
            val = claude_output.get(key) if claude_output else None
            sc = SCORE_C.get(tier, YELLOW) if val is None else score_color(val)
            sub_cells.append(Table([[
                Paragraph(f'<font color="white"><b>{val if val is not None else "—"}</b></font>',
                          ParagraphStyle("sv", fontName="Helvetica-Bold", fontSize=18, textColor=white,
                                         alignment=TA_CENTER)),
                Paragraph(label, ParagraphStyle("sl", fontName="Helvetica", fontSize=8,
                                                textColor=DGRAY, alignment=TA_CENTER)),
            ]], colWidths=[1.3*inch]))
            sub_cells[-1].setStyle(TableStyle([
                ("BACKGROUND",(0,0),(0,0),sc),
                ("TOPPADDING",(0,0),(0,0),8),("BOTTOMPADDING",(0,0),(0,0),6),
                ("BACKGROUND",(0,1),(0,1),LGRAY),
                ("TOPPADDING",(0,1),(0,1),4),("BOTTOMPADDING",(0,1),(0,1),6),
            ]))

        sub_row = Table([sub_cells], colWidths=[1.32*inch]*5)
        sub_row.setStyle(TableStyle([("LEFTPADDING",(0,0),(-1,-1),0),("RIGHTPADDING",(0,0),(-1,-1),2)]))
        story.append(sub_row)
        story.append(Spacer(1, 16))

        # ── FINANCIAL SUMMARY ──
        story.append(Paragraph("FINANCIAL SUMMARY", section_style))
        story.append(HRFlowable(width="100%", thickness=1, color=MGRAY))
        story.append(Spacer(1, 8))

        num_tenants = layer1.get("number_of_tenants") or 1
        fin_data = [
            ["Monthly Rent", fmt_dollar(layer1.get("monthly_rent")),
             "Security Deposit", fmt_dollar(layer1.get("security_deposit"))],
            ["Total Upfront Cost", fmt_dollar(layer2.get("total_upfront_cost")),
             "Full-Term Exposure", fmt_dollar(layer2.get("total_liability_term"))],
        ]
        if num_tenants > 1:
            fin_data.append([
                f"Cost Per Tenant ({num_tenants} people)", fmt_dollar(layer2.get("rent_per_tenant")),
                "Deposit Per Tenant", fmt_dollar(layer2.get("deposit_per_tenant")),
            ])
        fin_data += [
            ["Deposit Ratio", f"{layer2.get('deposit_as_months_rent', 0):.1f}x rent" if layer2.get('deposit_as_months_rent') else "—",
             "Late Fee", f"{layer2.get('late_fee_as_pct_rent', 0)*100:.1f}% of rent" if layer2.get('late_fee_as_pct_rent') else "—"],
            ["Lease Term", f"{layer2.get('lease_term_months', '—')} months",
             "Holdover Rent Increase", fmt_dollar(layer2.get("holdover_rent_increase")) if layer2.get("holdover_rent_increase") else "None stated"],
        ]
        if layer2.get("renewal_deadline_date"):
            fin_data.append([
                "Renewal Deadline", s(layer2.get("renewal_deadline_date")),
                "Days Until Deadline", str(layer2.get("days_until_renewal_deadline", "—"))
            ])

        fin_table = Table(fin_data, colWidths=[2.0*inch, 1.5*inch, 2.0*inch, 1.3*inch])
        fin_style = [
            ("FONTNAME",(0,0),(0,-1),"Helvetica-Bold"),
            ("FONTNAME",(2,0),(2,-1),"Helvetica-Bold"),
            ("FONTSIZE",(0,0),(-1,-1),9),
            ("TEXTCOLOR",(0,0),(0,-1),DGRAY),
            ("TEXTCOLOR",(2,0),(2,-1),DGRAY),
            ("TEXTCOLOR",(1,0),(1,-1),SLATE),
            ("TEXTCOLOR",(3,0),(3,-1),SLATE),
            ("TOPPADDING",(0,0),(-1,-1),7),("BOTTOMPADDING",(0,0),(-1,-1),7),
            ("LEFTPADDING",(0,0),(-1,-1),8),
            ("BACKGROUND",(0,0),(0,-1),LGRAY),("BACKGROUND",(2,0),(2,-1),LGRAY),
            ("LINEBELOW",(0,0),(-1,-2),0.5,MGRAY),
            ("BOX",(0,0),(-1,-1),0.5,MGRAY),
        ]
        fin_table.setStyle(TableStyle(fin_style))
        story.append(fin_table)
        story.append(Spacer(1, 16))

        # ── RISK FLAGS ──
        story.append(Paragraph(f"RISK FLAGS  ({len(flags)} identified)", section_style))
        story.append(HRFlowable(width="100%", thickness=1, color=MGRAY))
        story.append(Spacer(1, 8))

        if not flags:
            story.append(Paragraph("No significant risk flags identified. This lease appears to be within normal parameters.", body_style))
        else:
            # Sort by severity
            sev_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
            sorted_flags = sorted(flags, key=lambda f: sev_order.get(f.get("severity","LOW"), 4))

            for flag in sorted_flags:
                sev = flag.get("severity", "LOW")
                sev_c = SEV_C.get(sev, MGRAY)

                rows = [
                    # Title row
                    [Paragraph(f'<font color="white"><b> {sev} </b></font>',
                               ParagraphStyle("sev_lbl", fontName="Helvetica-Bold", fontSize=8,
                                              textColor=white, alignment=TA_CENTER)),
                     Paragraph(f'<b>{flag.get("short_description","")}</b>', flag_title_style)],
                    # Explanation
                    [Paragraph(flag.get("category","").upper().replace("_"," "),
                               ParagraphStyle("cat", fontName="Helvetica-Bold", fontSize=7,
                                              textColor=white, alignment=TA_CENTER)),
                     Paragraph(flag.get("detailed_explanation",""), flag_body_style)],
                ]

                # Add citation if present
                citation = flag.get("raw_clause_citation") or flag.get("statute_citation")
                if citation:
                    rows.append([
                        Paragraph("", small_style),
                        Paragraph(f'<i>"{citation}"</i>', cite_style)
                    ])

                # Jurisdiction note
                jnote = flag.get("jurisdiction_note") or flag.get("statute_citation")
                if jnote:
                    rows.append([
                        Paragraph("", small_style),
                        Paragraph(f'⚖ {jnote}', ParagraphStyle("jn", fontName="Helvetica-Oblique",
                                                                   fontSize=8.5, textColor=TEAL, leading=13))
                    ])

                # Negotiation language
                negotiate = flag.get("what_to_negotiate")
                if negotiate:
                    rows.append([
                        Paragraph("", small_style),
                        Paragraph(f'<b>What to request:</b> {negotiate}',
                                  ParagraphStyle("neg", fontName="Helvetica", fontSize=9,
                                                 textColor=NAVY, leading=14))
                    ])

                flag_table = Table(rows, colWidths=[0.9*inch, 5.9*inch])
                flag_style = [
                    ("BACKGROUND",(0,0),(0,-1),sev_c),
                    ("VALIGN",(0,0),(0,0),"MIDDLE"),
                    ("VALIGN",(0,1),(0,-1),"TOP"),
                    ("TOPPADDING",(0,0),(-1,-1),7),("BOTTOMPADDING",(0,0),(-1,-1),7),
                    ("LEFTPADDING",(0,0),(0,-1),6),("LEFTPADDING",(1,0),(1,-1),10),
                    ("RIGHTPADDING",(-1,0),(-1,-1),10),
                    ("BOX",(0,0),(-1,-1),0.5,MGRAY),
                    ("LINEBELOW",(0,0),(-1,-2),0.5,HexColor("#E2E8F0")),
                    ("BACKGROUND",(1,0),(1,-1),white),
                ]
                story.append(KeepTogether([flag_table, Spacer(1, 8)]))

        story.append(Spacer(1, 8))

        # ── SUMMARY TABLE ──
        story.append(Paragraph("WHAT THIS MEANS FOR YOU", section_style))
        story.append(HRFlowable(width="100%", thickness=1, color=MGRAY))
        story.append(Spacer(1, 8))

        if claude_output:
            summary_data = [
                ["🚨  Biggest Risk",
                 Paragraph(claude_output.get("tenant_summary_biggest_risk","—"), body_style)],
                ["✅  Best Clause",
                 Paragraph(claude_output.get("tenant_summary_biggest_positive","—"), body_style)],
                ["➡  Action Item",
                 Paragraph(f'<b>{claude_output.get("tenant_summary_key_action","—")}</b>', body_style)],
                ["📊  vs. Market",
                 Paragraph(claude_output.get("lease_compared_to_market","—"), body_style)],
            ]
            walk_away = claude_output.get("walk_away_triggers", [])
            if walk_away:
                wa_text = " / ".join(walk_away[:2])
                summary_data.append(["⛔  Walk Away If",
                                      Paragraph(wa_text, ParagraphStyle("wa", fontName="Helvetica",
                                                                        fontSize=9, textColor=RED, leading=14))])

            sum_table = Table(summary_data, colWidths=[1.1*inch, 5.7*inch])
            sum_table.setStyle(TableStyle([
                ("FONTNAME",(0,0),(0,-1),"Helvetica-Bold"),
                ("FONTSIZE",(0,0),(0,-1),9),
                ("TEXTCOLOR",(0,0),(0,-1),DGRAY),
                ("BACKGROUND",(0,0),(0,-1),LGRAY),
                ("TOPPADDING",(0,0),(-1,-1),9),("BOTTOMPADDING",(0,0),(-1,-1),9),
                ("LEFTPADDING",(0,0),(0,-1),8),("LEFTPADDING",(1,0),(1,-1),12),
                ("BOX",(0,0),(-1,-1),0.5,MGRAY),
                ("LINEBELOW",(0,0),(-1,-2),0.5,MGRAY),
                ("VALIGN",(0,0),(-1,-1),"TOP"),
            ]))
            story.append(sum_table)

        # ── NEGOTIATION PRIORITIES ──
        priorities = claude_output.get("negotiation_priority_order", []) if claude_output else []
        if priorities:
            story.append(Spacer(1, 12))
            story.append(Paragraph("NEGOTIATION PRIORITIES", section_style))
            story.append(HRFlowable(width="100%", thickness=1, color=MGRAY))
            story.append(Spacer(1, 6))
            story.append(Paragraph("Address these in order before signing:", body_style))
            for i, fid in enumerate(priorities[:5], 1):
                matching = [f for f in flags if f.get("flag_id") == fid]
                if matching:
                    f = matching[0]
                    story.append(Paragraph(
                        f'<b>{i}. {f.get("short_description","")}</b> — {f.get("what_to_negotiate","")}',
                        ParagraphStyle("pri", fontName="Helvetica", fontSize=9,
                                       textColor=SLATE, leading=14, leftIndent=12, spaceAfter=4)
                    ))

        # ── DISCLAIMER ──
        story.append(Spacer(1, 20))
        story.append(HRFlowable(width="100%", thickness=0.5, color=MGRAY))
        story.append(Spacer(1, 6))
        disclaimer = claude_output.get("tenant_disclaimer","") if claude_output else ""
        story.append(Paragraph(disclaimer,
                                ParagraphStyle("disc", fontName="Helvetica-Oblique", fontSize=8,
                                               textColor=DGRAY, leading=12, alignment=TA_CENTER)))
        story.append(Paragraph(
            f"Report ID: {extraction_id} · Generated by Lease Intelligence Platform · lease-intelligence.onrender.com",
            ParagraphStyle("rid", fontName="Helvetica", fontSize=7.5, textColor=MGRAY,
                           alignment=TA_CENTER, spaceBefore=4)
        ))

        doc.build(story)
        buffer.seek(0)
        return buffer.read()

    except Exception as e:
        logger.error(f"PDF generation failed: {e}")
        import traceback; traceback.print_exc()
        return b""


# ── QUALITY GATE ──────────────────────────────────────────────────────────────

CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.70"))
CRITICAL_FIELDS = {"monthly_rent", "lease_start_date", "lease_end_date", "property_address"}

def run_quality_gate(layer1, confidence_scores, reasoning_flags):
    requires_review, reasons = False, []
    missing = [f for f in CRITICAL_FIELDS if not layer1.get(f)]
    if missing:
        requires_review = True
        reasons.append(f"Missing critical fields: {', '.join(missing)}")
    low_conf = [f for f, v in confidence_scores.items() if f in CRITICAL_FIELDS and v < CONFIDENCE_THRESHOLD]
    if low_conf:
        requires_review = True
        reasons.append(f"Low confidence on critical fields: {', '.join(low_conf)}")
    rent = layer1.get("monthly_rent")
    if rent and (rent < 50 or rent > 100000):
        requires_review = True
        reasons.append(f"Unusual rent amount: ${rent}")
    return requires_review, reasons


# ── SUPABASE INSERT ───────────────────────────────────────────────────────────

def insert_to_supabase(extraction_id, layer1, layer2, metadata):
    try:
        from supabase import create_client
        client = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])
        def s(v):
            if isinstance(v, (date, datetime)): return v.isoformat()
            if isinstance(v, Decimal): return str(v)
            return v
        row = {
            "extraction_id": extraction_id,
            "created_at": datetime.utcnow().isoformat(),
            "pipeline_version": "2.0.0",
            **{k: s(v) for k, v in layer1.items()},
            **{f"computed_{k}": s(v) for k, v in layer2.items()},
            "zip_code": layer1.get("property_zip"),
            "requires_human_review": metadata.get("requires_human_review", False),
        }
        client.table("leases").insert(row).execute()
        logger.info(f"Supabase insert OK: {extraction_id}")
    except Exception as e:
        logger.error(f"Supabase insert failed (non-fatal): {e}")


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def landing():
    html_path = Path(__file__).parent.parent / "static" / "index.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text())
    return HTMLResponse("<h1>Lease Intelligence v2.0</h1>")

@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": "2.0.0",
        "gemini_configured": bool(os.getenv("GEMINI_API_KEY")),
        "claude_configured": bool(os.getenv("ANTHROPIC_API_KEY")),
        "supabase_configured": bool(os.getenv("SUPABASE_URL")),
    }

@app.post("/analyze")
async def analyze_lease(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    state: str = Form("VA"),
    tier: str = Form("paid"),
    user_id: Optional[str] = Form(None),
):
    """Full pipeline: text extraction → Gemini → Claude → PDF report."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files accepted")

    pdf_bytes = await file.read()
    if len(pdf_bytes) < 100:
        raise HTTPException(400, "File appears empty")
    if len(pdf_bytes) > 20 * 1024 * 1024:
        raise HTTPException(413, "File too large — maximum 20MB")

    extraction_id = str(uuid.uuid4())
    doc_hash = hashlib.sha256(pdf_bytes).hexdigest()
    t_start = time.time()
    logger.info(f"[{extraction_id}] Starting: {file.filename} ({len(pdf_bytes)//1024}KB) state={state}")

    # Step 1: Extract text from PDF
    lease_text = extract_text_from_pdf(pdf_bytes)
    if len(lease_text) < 200:
        logger.warning(f"[{extraction_id}] Very short text extracted ({len(lease_text)} chars) — may be scanned PDF")

    # Step 2: Gemini field extraction
    raw_gemini = await call_gemini_with_text(lease_text, state)
    if raw_gemini is None:
        logger.warning(f"[{extraction_id}] Gemini returned None — check API key and billing")
        return JSONResponse({"error": "Extraction failed. Check GEMINI_API_KEY configuration.",
                             "extraction_id": extraction_id}, status_code=500)

    layer1, confidence_scores, reasoning_flags = parse_gemini_response(raw_gemini)

    # Step 3: Layer 2 computation
    layer2 = compute_layer2(layer1)

    # Step 4: Quality gate
    requires_review, review_reasons = run_quality_gate(layer1, confidence_scores, reasoning_flags)

    # Step 5: Claude reasoning
    claude_output = await call_claude(layer1, layer2, reasoning_flags, state)
    if claude_output is None:
        logger.warning(f"[{extraction_id}] Claude returned None — check API key and billing")

    # Step 6: Generate PDF
    pdf_report = generate_professional_pdf(layer1, layer2, claude_output, extraction_id)
    pdf_b64 = base64.b64encode(pdf_report).decode() if pdf_report else None

    def s(v):
        if isinstance(v, (date, datetime)): return v.isoformat()
        if isinstance(v, Decimal): return str(v)
        return v

    result = {
        "extraction_id": extraction_id,
        "document_filename": file.filename,
        "document_hash_sha256": doc_hash,
        "created_at": datetime.utcnow().isoformat(),
        "pipeline_version": "2.0.0",
        "processing_time_seconds": round(time.time() - t_start, 3),
        "analysis_tier": tier,
        "text_extracted_chars": len(lease_text),
        "layer1": {k: s(v) for k, v in layer1.items()},
        "layer2": {k: s(v) for k, v in layer2.items()},
        "layer3_flags": claude_output.get("flags", []) if claude_output else [],
        "layer4": {
            "overall_lease_risk_score": claude_output.get("overall_lease_risk_score") if claude_output else None,
            "renewal_risk_score": claude_output.get("renewal_risk_score") if claude_output else None,
            "financial_burden_score": claude_output.get("financial_burden_score") if claude_output else None,
            "landlord_access_risk_score": claude_output.get("landlord_access_risk_score") if claude_output else None,
            "termination_risk_score": claude_output.get("termination_risk_score") if claude_output else None,
            "liability_risk_score": claude_output.get("liability_risk_score") if claude_output else None,
            "risk_tier": claude_output.get("risk_tier") if claude_output else None,
            "score_rationale": claude_output.get("score_rationale") if claude_output else None,
            "lease_compared_to_market": claude_output.get("lease_compared_to_market") if claude_output else None,
            "tenant_summary_biggest_risk": claude_output.get("tenant_summary_biggest_risk") if claude_output else None,
            "tenant_summary_biggest_positive": claude_output.get("tenant_summary_biggest_positive") if claude_output else None,
            "tenant_summary_key_action": claude_output.get("tenant_summary_key_action") if claude_output else None,
            "negotiation_priority_order": claude_output.get("negotiation_priority_order") if claude_output else [],
            "walk_away_triggers": claude_output.get("walk_away_triggers") if claude_output else [],
            "tenant_disclaimer": claude_output.get("tenant_disclaimer") if claude_output else "",
        },
        "confidence_scores": confidence_scores,
        "reasoning_flags": reasoning_flags,
        "requires_human_review": requires_review,
        "human_review_reasons": review_reasons,
        "pdf_base64": pdf_b64,
    }

    if os.getenv("SUPABASE_URL"):
        background_tasks.add_task(insert_to_supabase, extraction_id, layer1, layer2,
                                   {"requires_human_review": requires_review})

    logger.info(f"[{extraction_id}] Complete in {result['processing_time_seconds']}s | "
                f"flags={len(result['layer3_flags'])} | score={result['layer4']['overall_lease_risk_score']} | "
                f"review={requires_review}")
    return JSONResponse(result)

@app.get("/demo")
async def demo():
    """Returns realistic demo data for UI testing."""
    return JSONResponse({
        "extraction_id": "demo-" + str(uuid.uuid4())[:8],
        "document_filename": "sample_lease_demo.pdf",
        "is_demo": True,
        "layer1": {
            "tenant_full_name": "Harrison Cerone, Thomas Denton, Dalton Jobe",
            "landlord_full_name": "Mitchell D. Shaner",
            "property_address": "19 Winding Way",
            "property_city": "Lexington",
            "property_state": "VA",
            "property_zip": "24450",
            "monthly_rent": "3900.00",
            "security_deposit": "7800.00",
            "number_of_tenants": 3,
            "lease_start_date": "2025-06-08",
            "lease_end_date": "2026-05-30",
            "landlord_entry_notice_hours": None,
            "landlord_entry_clause_verbatim": "Landlord and Landlord's agents shall have the right at all reasonable times during the term of this Agreement to enter the Premises",
            "late_fee_escalates": True,
            "late_fee_structure": "10% after 13th, increases to 15% on 20th, possession demand on 25th",
            "parental_guarantee_required": True,
            "joint_several_liability": True,
            "holdover_rent_amount": "4000.00",
            "all_utilities_tenant": True,
            "utilities_tenant_responsible": ["electricity","gas","water","internet","all utilities"],
            "utilities_landlord_responsible": [],
            "police_violation_fee": "200.00",
            "subletting_allowed": False,
            "pets_allowed": False,
        },
        "layer2": {
            "effective_monthly_cost": "3900.00",
            "total_upfront_cost": "11700.00",
            "total_liability_term": "46800.00",
            "lease_term_months": 12,
            "deposit_as_months_rent": 2.0,
            "rent_per_tenant": "1300.00",
            "deposit_per_tenant": "2600.00",
            "upfront_per_tenant": "3900.00",
            "renewal_deadline_date": None,
            "days_until_renewal_deadline": None,
            "holdover_rent_increase": "100.00",
        },
        "layer3_flags": [
            {"flag_id": "entry_notice_unspecified", "severity": "CRITICAL", "category": "entry_rights",
             "short_description": "Lease specifies no entry notice period — Virginia law requires 24 hours minimum.",
             "detailed_explanation": "Paragraph 13 grants access 'at all reasonable times' with no hours defined. Virginia Code § 55.1-1229 requires landlords to give at least 24 hours advance notice. This clause is unenforceable as written and gives you no protection against surprise entries.",
             "raw_clause_citation": "Landlord and Landlord's agents shall have the right at all reasonable times during the term of this Agreement to enter the Premises",
             "statute_citation": "VA Code § 55.1-1229",
             "jurisdiction_note": "Virginia requires minimum 24-hour notice except in emergencies. 'Reasonable times' without a defined hour minimum does not satisfy this requirement.",
             "what_to_negotiate": "Request Landlord add: 'Except in case of emergency, Landlord shall provide at least 24 hours advance written notice before entering the Premises.'",
             "if_they_refuse": "Landlord can legally argue they gave 'reasonable' notice even with 1-2 hours. Document any entries in writing."},
            {"flag_id": "escalating_late_fees", "severity": "CRITICAL", "category": "financial",
             "short_description": "Escalating late fee structure and possession demand by the 25th may be legally excessive.",
             "detailed_explanation": "The fee escalates from 10% ($390) on the 13th to 15% ($585) on the 20th, then landlord demands immediate possession on the 25th — all within a single month. The possession demand without court process on day 25 is particularly aggressive and likely unenforceable without proper legal proceedings.",
             "raw_clause_citation": "10% of the total amount due after the 13th day... increased by 15% on the 20th day... If rent and all late charges are not paid in full by the 25th, Tenant will immediately turn over possession",
             "statute_citation": "VA Code § 55.1-1245",
             "jurisdiction_note": "Virginia requires proper unlawful detainer proceedings before a landlord can remove a tenant. Self-help eviction (demanding possession without court process) violates Virginia law.",
             "what_to_negotiate": "Request: 'Late fee shall be a flat 5% of monthly rent if rent is not received within 5 days of the due date. Landlord's sole remedy for nonpayment is legal proceedings per VRLTA.'",
             "if_they_refuse": "This clause is likely unenforceable in its current form, but fighting it requires hiring an attorney."},
            {"flag_id": "parental_guarantee", "severity": "HIGH", "category": "liability",
             "short_description": "Parental guarantee makes parents unconditionally and primarily liable — before the landlord even pursues tenants.",
             "detailed_explanation": "The guarantee states parents' liability is 'continuing, absolute and unconditional' and the landlord is not required to pursue remedies against tenants before going after parents. Both parents must sign. This means your parents are primary obligors, not co-signers — the landlord can sue them first without ever contacting you.",
             "raw_clause_citation": "The liability of the parent or legal guardian shall be continuing, absolute and unconditional. The Landlord shall not be required to exercise remedies against the Tenant(s) before proceeding against the person signing the Guarantee.",
             "statute_citation": None,
             "jurisdiction_note": "This guarantee is legally enforceable under Virginia contract law. Parents signing this are full primary obligors.",
             "what_to_negotiate": "Request: 'Guarantor liability is secondary. Landlord must first demand payment from Tenant and allow 30 days to cure before pursuing Guarantor.'",
             "if_they_refuse": "Ensure parents fully understand they can be sued directly and first, before the landlord pursues you."},
            {"flag_id": "joint_several_liability", "severity": "HIGH", "category": "liability",
             "short_description": "Joint and several liability means each tenant is 100% responsible for the entire rent if others don't pay.",
             "detailed_explanation": "With 3+ tenants, if one person can't pay their share, the others are legally obligated to cover the full amount. The landlord can pursue any single tenant for the entire $3,900/month rent, regardless of individual shares.",
             "raw_clause_citation": "Tenants acknowledge that each of them shall be jointly and severally responsible for all terms of this Lease including the payment of rent.",
             "statute_citation": None,
             "jurisdiction_note": "Standard in Virginia group leases. Legally enforceable.",
             "what_to_negotiate": "This is standard and difficult to negotiate out. Instead, ensure you have written agreements between co-tenants about individual shares.",
             "if_they_refuse": "Accept it as standard, but document rent contributions carefully."},
            {"flag_id": "holdover_auto_increase", "severity": "MEDIUM", "category": "renewal",
             "short_description": "Staying even one day past May 30 automatically converts to month-to-month at $4,000/month.",
             "detailed_explanation": "If you remain in the property after May 30, 2026, rent automatically increases to $4,000/month (a $100/month increase) and you're locked into a month-to-month tenancy requiring 30 days notice to exit. One day of holdover could cost you an extra $100.",
             "raw_clause_citation": "rent shall then be due and owing at FOUR THOUSAND and N0/100 ($4,000.00) DOLLARS per month",
             "statute_citation": None,
             "jurisdiction_note": "Standard holdover provision. Legally enforceable.",
             "what_to_negotiate": "Request holdover rent remain at $3,900 for a 30-day grace period.",
             "if_they_refuse": "Mark your calendar: vacate by May 30, 2026."},
            {"flag_id": "all_utilities_tenant", "severity": "MEDIUM", "category": "utilities",
             "short_description": "Tenant is responsible for ALL utilities — budget $300-600/month additional for a house this size.",
             "detailed_explanation": "Paragraph 10 transfers complete utility responsibility to tenants, including electricity, gas, water, and sewer. For a multi-bedroom house in Lexington, VA, expect $300-600/month in utilities split among tenants, significantly raising the true monthly cost above $3,900.",
             "raw_clause_citation": "Tenant shall be responsible for arranging for and paying for all utility services required on the Premises.",
             "statute_citation": None,
             "jurisdiction_note": "Standard in Virginia leases for single-family homes.",
             "what_to_negotiate": "Ask for landlord to cover water/sewer at minimum, as these can spike unpredictably.",
             "if_they_refuse": "Budget accordingly. Request copies of prior year utility bills before signing."},
            {"flag_id": "police_violation_fee", "severity": "MEDIUM", "category": "financial",
             "short_description": "$200 automatic charge for any police-issued violation, including noise and trash.",
             "detailed_explanation": "Paragraph 11(l) imposes an automatic $200 charge on the household for any police-issued violation — including noise ordinance violations from a neighbor's complaint. In a college-area rental, this is a meaningful liability particularly around social events.",
             "raw_clause_citation": "TWO HUNDRED and NO/100 ($200.00) DOLLARS will be charged to any household if the police issue any violation, including but not limited to noise or trash",
             "statute_citation": None,
             "jurisdiction_note": "Legal in Virginia as a contractual penalty clause.",
             "what_to_negotiate": "Request removal, or limit to violations where tenant is found guilty.",
             "if_they_refuse": "Factor this into risk assessment for your specific living situation."},
        ],
        "layer4": {
            "overall_lease_risk_score": 72,
            "renewal_risk_score": 40,
            "financial_burden_score": 65,
            "landlord_access_risk_score": 82,
            "termination_risk_score": 55,
            "liability_risk_score": 78,
            "risk_tier": "ORANGE",
            "score_rationale": "This lease scores ORANGE primarily due to two serious issues: the complete absence of an entry notice period (which may violate Virginia law) and an aggressive escalating late fee structure with a legally dubious self-help possession demand. The parental guarantee and joint-and-several liability are standard for student group leases but carry real financial exposure.",
            "lease_compared_to_market": "Above average risk for Lexington, VA student housing. The escalating late fee structure and unspecified entry notice are more aggressive than typical local leases. The Lexprop/Borden Road lease format in this market is generally more tenant-favorable.",
            "tenant_summary_biggest_risk": "No entry notice hours specified — landlord can argue 'reasonable' means any time, and the possession demand on day 25 of late payment is legally aggressive.",
            "tenant_summary_biggest_positive": "Lease is clearly structured with defined terms, property condition requirements, and transparent party obligations.",
            "tenant_summary_key_action": "Before signing, demand in writing: (1) entry notice amended to 24 hours, (2) late fee escalation removed in favor of flat 5%, (3) ensure all co-tenants understand their joint liability exposure.",
            "negotiation_priority_order": ["entry_notice_unspecified", "escalating_late_fees", "parental_guarantee"],
            "walk_away_triggers": ["Landlord refuses to specify a minimum entry notice period", "Landlord refuses to remove the self-help possession demand on day 25"],
            "tenant_disclaimer": "This analysis is for informational purposes only and does not constitute legal advice. Laws vary by jurisdiction and change over time. Consult a licensed Virginia attorney or contact the Virginia Poverty Law Center (1-888-LEGLAID) for advice specific to your situation.",
        }
    })
