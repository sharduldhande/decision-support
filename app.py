import streamlit as st
import google.genai as genai
from google.genai import types
import json
import os
from sentence_transformers import SentenceTransformer
import chromadb


@st.cache_resource
def load_embed_model():
    return SentenceTransformer("NeuML/pubmedbert-base-embeddings")


@st.cache_resource
def load_chroma():
    client = chromadb.PersistentClient(path="./chroma_db")
    return client.get_collection("guidelines")


embed_model = load_embed_model()
collection = load_chroma()

DEFAULT_GEMINI_API_KEY = None


def resolve_gemini_api_key():
    env_keys = [
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "GOOGLE_GENAI_API_KEY",
    ]
    for key_name in env_keys:
        value = os.environ.get(key_name)
        if value:
            return value

    try:
        for key_name in env_keys:
            if key_name in st.secrets and st.secrets[key_name]:
                return st.secrets[key_name]
    except Exception:
        pass

    if DEFAULT_GEMINI_API_KEY:
        return DEFAULT_GEMINI_API_KEY

    st.error(
        "No Gemini API key found. Set GEMINI_API_KEY in your environment, "
        "add it to Streamlit secrets, or set DEFAULT_GEMINI_API_KEY in app.py."
    )
    st.stop()


gemini_client = genai.Client(api_key=resolve_gemini_api_key())

st.title("Coronary Intervention Decision Support")
st.caption("Based on 2021 ACC/AHA/SCAI Guidelines")

EXTRACTION_PROMPT = """Extract clinical findings from this cardiac catheterization or CCTA report.
Return ONLY valid JSON, no markdown backticks, matching this exact schema:

{
  "clinical_context": {
    "comorbidities": [],
    "cardiac_diagnosis": null,
    "valvular_disease": null,
    "rhythm": null,
    "lv_function": null
  },
  "coronary_anatomy": {
    "dominance": "right | left | codominant | unknown",
    "left_main": {
      "ostial":   { "stenosis_pct": null, "stenosis_qualifier": "not_described", "significant": false },
      "mid":      { "stenosis_pct": null, "stenosis_qualifier": "not_described", "significant": false },
      "distal":   { "stenosis_pct": null, "stenosis_qualifier": "not_described", "significant": false }
    },
    "lad": {
      "proximal": { "stenosis_pct": null, "stenosis_qualifier": "not_described", "significant": false },
      "mid":      { "stenosis_pct": null, "stenosis_qualifier": "not_described", "significant": false },
      "distal":   { "stenosis_pct": null, "stenosis_qualifier": "not_described", "significant": false }
    },
    "lcx": {
      "dominant": false,
      "proximal": { "stenosis_pct": null, "stenosis_qualifier": "not_described", "significant": false },
      "mid":      { "stenosis_pct": null, "stenosis_qualifier": "not_described", "significant": false },
      "distal":   { "stenosis_pct": null, "stenosis_qualifier": "not_described", "significant": false }
    },
    "rca": {
      "dominant": false,
      "proximal": { "stenosis_pct": null, "stenosis_qualifier": "not_described", "significant": false },
      "mid":      { "stenosis_pct": null, "stenosis_qualifier": "not_described", "significant": false },
      "distal":   { "stenosis_pct": null, "stenosis_qualifier": "not_described", "significant": false }
    },
    "branches": {
      "diagonal_1": { "stenosis_pct": null, "stenosis_qualifier": "not_described", "significant": false },
      "diagonal_2": { "stenosis_pct": null, "stenosis_qualifier": "not_described", "significant": false },
      "om_1":       { "stenosis_pct": null, "stenosis_qualifier": "not_described", "significant": false },
      "om_2":       { "stenosis_pct": null, "stenosis_qualifier": "not_described", "significant": false },
      "ramus":      { "stenosis_pct": null, "stenosis_qualifier": "not_described", "significant": false },
      "pda":        { "stenosis_pct": null, "stenosis_qualifier": "not_described", "significant": false },
      "plb":        { "stenosis_pct": null, "stenosis_qualifier": "not_described", "significant": false }
    }
  },
  "angiographic_diagnosis": {
    "summary": null,
    "vessel_disease_count": null,
    "significant_lesions": []
  }
}

FIELD RULES:

stenosis_pct:
  - Array of integers when a percentage is stated: [30] for a single value, [30, 40] for a range like "30-40%"
  - null if no percentage is mentioned (use stenosis_qualifier to capture qualitative descriptions instead)

stenosis_qualifier:
  - Must be one of: "normal", "mild", "mild_plaque", "moderate", "severe", "total_occlusion", "not_described"
  - Use "not_described" only when a percentage is given without a qualitative description
  - Use "normal" when the report explicitly states normal or no disease

significant:
  - true if stenosis >= 70% for any vessel, or >= 50% for left main
  - true if the report describes total occlusion, critical stenosis, or uses the word "significant"
  - false otherwise

dominant (on lcx and rca):
  - Set based on the dominance stated in the report
  - Right dominant: rca.dominant = true, lcx.dominant = false
  - Left dominant: rca.dominant = false, lcx.dominant = true
  - Codominant: both true

clinical_context:
  - comorbidities: array of strings for non-cardiac conditions (e.g., ["HTN", "DM", "CKD"]). Empty array if none listed.
  - cardiac_diagnosis: the primary cardiac indication for the procedure (e.g., "CAD - ACS - STEMI"). null if not stated.
  - valvular_disease: any valvular findings mentioned (e.g., "SEVERE MR"). null if not stated.
  - rhythm: cardiac rhythm if mentioned (e.g., "SR", "AF"). null if not stated.
  - lv_function: LV function description if mentioned (e.g., "MILD LV DYSFUNCTION", "LVEF 35%"). null if not stated.

angiographic_diagnosis:
  - summary: the final diagnosis line from the report (e.g., "CAD - DVD", "CAD - TVD"). null if not stated.
  - vessel_disease_count: number of diseased vessels (1, 2, 3) derived from the diagnosis (SVD=1, DVD=2, TVD=3). null if not stated.
  - significant_lesions: array of objects for each significant lesion:
      { "vessel": "LCx", "segment": "mid", "finding": "total_occlusion", "stenosis_pct": [100] }
    Empty array if no significant lesions.

GENERAL RULES:
  - Use null for any field not mentioned in the report.
  - Do not guess or infer values not explicitly stated.
  - If the report says "normal" for an entire vessel without segment-level detail, mark all segments as stenosis_qualifier "normal".
  - If the report describes disease without specifying a segment (e.g., "LCx total occlusion" after "proximal normal"), assign to the most logical next segment.
  - Preserve stenosis ranges exactly as stated. Do not round or pick a single value from a range.
"""

DECISION_SYSTEM_PROMPT = """\
You are a clinical decision support tool based on the 2021 ACC/AHA/SCAI Guideline \
for Coronary Artery Revascularization.

A NSTEMI patient has undergone diagnostic ICA. Ad hoc PCI was NOT performed. \
The Heart Team must now decide: complete revascularization via PCI or CABG?

Use ONLY the guideline context provided below. For each recommendation cite the \
exact section title, Class of Recommendation, and Level of Evidence.

Respond ONLY with valid JSON (no markdown backticks) matching this schema:
{
    "recommendation": "PCI" | "CABG" | "PCI preferred" | "CABG preferred" | "Either acceptable" | "Medical management" | "Insufficient data",
    "confidence": "strong" | "moderate" | "weak",
    "primary_rationale": "2–3 sentence plain-language explanation",
    "supporting_citations": [
        {
          "section_title": "exact title as found in the guidelines",
          "class_of_recommendation": "I" | "IIa" | "IIb" | "III",
          "level_of_evidence": "A" | "B-R" | "B-NR" | "C-LD" | "C-EO",
          "guideline_text": "brief neutral summary of what the guideline states",
          "patient_relevance": "why this specific clause applies to this patient"
        }
    ],
    "conditions_favouring_pci": ["..."],
    "conditions_favouring_cabg": ["..."],
    "conditions_favouring_medical_management": ["..."],
    "heart_team_considerations": "additional note for the multidisciplinary team",
    "urgent_issues": null
}"""


def clean_json_response(text):
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()

def build_findings_summary(findings):
    lines = []
    anatomy = findings.get("coronary_anatomy", {})
    clinical = findings.get("clinical_context", {})
    diagnosis = findings.get("angiographic_diagnosis", {})

    # Clinical context
    if clinical.get("cardiac_diagnosis"):
        lines.append(f"Diagnosis: {clinical['cardiac_diagnosis']}")
    if clinical.get("comorbidities"):
        lines.append(f"Comorbidities: {', '.join(clinical['comorbidities'])}")
    if clinical.get("valvular_disease"):
        lines.append(f"Valvular: {clinical['valvular_disease']}")
    if clinical.get("lv_function"):
        lines.append(f"LV function: {clinical['lv_function']}")

    lines.append(f"Coronary dominance: {anatomy.get('dominance', 'unknown')}")

    # Angiographic diagnosis
    if diagnosis.get("summary"):
        lines.append(f"Angiographic diagnosis: {diagnosis['summary']}")
    if diagnosis.get("vessel_disease_count") is not None:
        lines.append(f"Vessel disease count: {diagnosis['vessel_disease_count']}")

    vessels = {
        "left_main": "Left Main",
        "lad": "LAD",
        "lcx": "LCx",
        "rca": "RCA",
    }

    for key, name in vessels.items():
        vessel = anatomy.get(key, {})
        for segment, data in vessel.items():
            if not isinstance(data, dict):
                continue
            qualifier = data.get("stenosis_qualifier", "not_described")
            pct = data.get("stenosis_pct")
            significant = data.get("significant", False)

            if qualifier in ("normal", "not_described") and not pct and not significant:
                continue

            parts = []
            if pct:
                parts.append(f"{pct}% stenosis")
            if qualifier not in ("not_described",):
                parts.append(qualifier.replace("_", " "))
            if significant:
                parts.append("(significant)")
            desc = ", ".join(parts) if parts else qualifier
            lines.append(f"{name} {segment}: {desc}")

    branches = anatomy.get("branches", {})
    for branch, data in branches.items():
        if not isinstance(data, dict):
            continue
        qualifier = data.get("stenosis_qualifier", "not_described")
        pct = data.get("stenosis_pct")
        significant = data.get("significant", False)

        if qualifier in ("normal", "not_described") and not pct and not significant:
            continue

        parts = []
        if pct:
            parts.append(f"{pct}% stenosis")
        if qualifier not in ("not_described",):
            parts.append(qualifier.replace("_", " "))
        if significant:
            parts.append("(significant)")
        desc = ", ".join(parts) if parts else qualifier
        lines.append(f"{branch.replace('_', ' ').title()}: {desc}")

    return "\n".join(lines)


def count_diseased_vessels(findings):
    dx_count = findings.get("angiographic_diagnosis", {}).get("vessel_disease_count")
    if dx_count is not None:
        return dx_count

    anatomy = findings.get("coronary_anatomy", {})
    count = 0
    for vessel in ["left_main", "lad", "lcx", "rca"]:
        vessel_data = anatomy.get(vessel, {})
        for segment, data in vessel_data.items():
            if isinstance(data, dict) and data.get("significant"):
                count += 1
                break
    return count


def retrieve_guidelines(query_text, n_results=10):
    query_embedding = embed_model.encode(query_text).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
    )
    sections = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        sections.append({"title": meta["title"], "content": doc})
    return sections


def build_decision_prompt(findings, guideline_sections):
    findings_json = json.dumps(findings, indent=2)

    guidelines_text = "\n\n---\n\n".join(
        f"### {s['title']}\n{s['content']}" for s in guideline_sections
    )

    return f"""## Patient Clinical Findings (full structured extraction from ICA)

```json
{findings_json}
```

## Relevant Guideline Sections

{guidelines_text}

## Task

Based on the complete clinical findings above and ONLY the guideline sections provided, produce your \
JSON recommendation for this NSTEMI patient."""


def render_decision(decision):
    rec = decision.get("recommendation", "Unknown")
    confidence = decision.get("confidence", "unknown")
    icon = {
        "PCI": "🟢", "PCI preferred": "🟢",
        "CABG": "🔵", "CABG preferred": "🔵",
        "Either acceptable": "🟡",
        "Medical management": "🟠",
        "Insufficient data": "🔴",
    }.get(rec, "⚪")

    st.subheader(f"{icon} Recommendation: {rec}")
    st.markdown(f"**Confidence:** {confidence}")
    st.markdown(f"**Rationale:** {decision.get('primary_rationale', 'N/A')}")

    ht = decision.get("heart_team_considerations")
    if ht:
        st.info(f"**Heart Team Note:** {ht}")

    urgent = decision.get("urgent_issues")
    if urgent:
        st.error(f"**Urgent Issues:** {urgent}")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Factors Favouring PCI**")
        for item in decision.get("conditions_favouring_pci", []):
            st.markdown(f"- {item}")
    with col2:
        st.markdown("**Factors Favouring CABG**")
        for item in decision.get("conditions_favouring_cabg", []):
            st.markdown(f"- {item}")
    with col3:
        st.markdown("**Factors Favouring Medical Mgmt**")
        for item in decision.get("conditions_favouring_medical_management", []):
            st.markdown(f"- {item}")

    citations = decision.get("supporting_citations", [])
    if citations:
        st.subheader("Supporting Guideline Citations")
        for i, cite in enumerate(citations, 1):
            with st.expander(f"Citation {i}: {cite.get('section_title', 'N/A')}"):
                st.markdown(f"**Class:** {cite.get('class_of_recommendation', '?')} "
                            f"| **Level:** {cite.get('level_of_evidence', '?')}")
                st.markdown(f"**Guideline:** {cite.get('guideline_text', '')}")
                st.markdown(f"**Patient Relevance:** {cite.get('patient_relevance', '')}")



uploaded_file = st.file_uploader("Upload ICA/CCTA Report (PDF)", type="pdf")

if uploaded_file:
    pdf_bytes = uploaded_file.read()
    pdf_part = types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf")
    prompt_part = types.Part.from_text(text=EXTRACTION_PROMPT)
    message = types.Content(role="user", parts=[pdf_part, prompt_part])

    with st.spinner(text="Extracting clinical findings..."):
        response = gemini_client.models.generate_content(
            model="gemini-2.5-pro",
            # model="gemini-2.5-flash",
            contents=[message],
        )

    try:
        findings = json.loads(clean_json_response(response.text))
        st.subheader("Extracted Findings")
        st.text(build_findings_summary(findings))
    except json.JSONDecodeError:
        st.error("Failed to parse findings. Raw response:")
        st.code(response.text)
        st.stop()

    summary = build_findings_summary(findings)
    n_vessels = count_diseased_vessels(findings)

    queries = [
        summary,
        f"NSTEMI revascularization {n_vessels}-vessel disease",
        "PCI versus CABG multivessel coronary artery disease",
    ]

    seen_titles = set()
    all_sections = []
    for q in queries:
        for s in retrieve_guidelines(q):
            if s["title"] not in seen_titles:
                seen_titles.add(s["title"])
                all_sections.append(s)

    with st.spinner("Generating guideline-based recommendation..."):
        decision_prompt = build_decision_prompt(findings, all_sections)
        decision_response = gemini_client.models.generate_content(
            # model="gemini-2.5-pro",
            model="gemini-2.5-flash",
            contents=[
                types.Content(role="user", parts=[
                    types.Part.from_text(text=decision_prompt)
                ]),
            ],
            config=types.GenerateContentConfig(
                system_instruction=DECISION_SYSTEM_PROMPT,
                temperature=0.2,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )

    try:
        decision = json.loads(clean_json_response(decision_response.text))
    except json.JSONDecodeError:
        st.error("Failed to parse decision response. Raw output:")
        st.code(decision_response.text)
        st.stop()

    st.divider()
    render_decision(decision)

    st.divider()
    st.warning(
        "⚠️ This tool is for clinical decision *support* only. "
        "It does not replace clinical judgment or Heart Team discussion."
    )
