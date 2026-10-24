import os
import base64
from typing import NamedTuple
from functools import lru_cache
from pathlib import Path
from langchain.chat_models import init_chat_model
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable
import chromadb
from schemas import Findings, Decision, Segment, VESSEL_NAMES

MODEL = os.environ.get("LLM_MODEL", "anthropic:claude-haiku-5-5")

model = init_chat_model(MODEL, max_tokens=8000)

DB_PATH = Path(__file__).parent / "chroma_db"
COLLECTION = "guidelines_v2"
EMBED_MODEL = "NeuML/pubmedbert-base-embeddings"
CHUNKS_PER_QUERY = 8

@lru_cache(maxsize=1)
def get_retriever():
    store = Chroma(
        client= chromadb.PersistentClient(path=str(DB_PATH)),
        collection_name=COLLECTION,
        embedding_function=HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    )
    return store.as_retriever(search_kwargs = {"k": CHUNKS_PER_QUERY})

EXTRACTION_PROMPT = """\
Extract the clinical findings from this cardiac catheterization or CCTA report.

- Use null (or the default) for anything the report does not mention. Do not guess or infer values \
that are not explicitly stated.
- If the report describes disease without naming a segment (e.g. "LCx total occlusion" after \
"proximal normal"), assign it to the most logical next segment.
- If a whole vessel is described as normal without segment detail, mark every segment of it as normal.
"""


def extraction_messages(inputs):
    return [HumanMessage([
        {"type": "file", "base64": inputs["pdf64"], "mime_type":"application/pdf"},
        {"type": "text", "text": EXTRACTION_PROMPT}
    ])]

extract = extraction_messages | model.with_structured_output(Findings)


def describe_lesion(lesion):
    location = f"{lesion.vessel} {lesion.segment}".strip()
    pct = "-".join(map(str, lesion.stenosis_pct)) + "%" if lesion.stenosis_pct else lesion.stenosis_qualifier
    return f"{location} {pct} ({lesion.severity})"

def search_queries(findings):
    ctx = findings.clinical_context
    summary = "; ".join(
        [describe_lesion(l) for l in findings.lesions]
        + ctx.comorbidities
        + [v for v in (ctx.valvular_disease, ctx.lv_function) if v]
    )
    n_vessels = findings.angiographic_diagnosis.vessel_disease_count or len(
        {l.vessel for l in findings.lesions if l.severity == "significant"}
    )
    queries = [
        summary,
        f"NSTEMI revascularization {n_vessels}-vessel disease",
        "PCI versus CABG multivessel coronary artery disease",
    ]
    if any(c.upper() in ("DM", "T2DM", "T1DM") or "DIABET" in c.upper() for c in ctx.comorbidities):
        queries.append("diabetes multivessel CAD involving LAD CABG over PCI")
    if any(l.severity == "intermediate" for l in findings.lesions):
        queries.append("intermediate stenosis physiological assessment FFR iFR to guide revascularization")
    return queries

def retrieve(inputs):
    results = get_retriever().batch(search_queries(inputs["findings"]))
    seen, chunks = set(), []
    for doc in (doc for docs in results for doc in docs):
        if doc.id not in seen:
            seen.add(doc.id)
            chunks.append(f"### {doc.page_content}")
    return "\n\n---\n\n".join(chunks)


DECISION_PROMPT = """\
You are a clinical decision support tool based on the 2021 ACC/AHA/SCAI Guideline for Coronary \
Artery Revascularization.

A NSTEMI patient has undergone diagnostic ICA. Ad hoc PCI was NOT performed. The Heart Team must now \
decide: complete revascularization via PCI or CABG?

Use ONLY the guideline sections provided. Prefer citing the formal recommendations, marked \
"Recommendation (COR ..., LOE ...)", and copy their class and level exactly. When citing other \
guideline text, leave class and level null.

Lesion severity in the findings was computed from the guideline's angiographic thresholds \
(>=70%, or >=50% for left main, is significant; 40-69% is intermediate). Intermediate lesions have \
not had physiological assessment (FFR/iFR). Base the anatomy on this lesion analysis. If it disagrees \
with the report's own diagnosis line (e.g. "DVD"), say so in the rationale.
"""


def anatomy_summary(findings):
    lesions = findings.lesions
    intermediate = [describe_lesion(l) for l in lesions if l.severity == "intermediate"]
    vessels = findings.significant_vessels
    return (
        f"Report's diagnosis line: {findings.angiographic_diagnosis.summary or 'not stated'}\n"
        f"Lesion analysis: {len(vessels)} significant vessel(s): {', '.join(vessels) or 'none'}\n"
        f"Intermediate lesions (no FFR/iFR): {'; '.join(intermediate) or 'none'}"
    )


def decision_messages(inputs):
    findings = inputs["findings"]
    findings_json = findings.model_dump_json(indent=2, exclude_defaults=True)
    return [
        SystemMessage(DECISION_PROMPT),
        HumanMessage(
            f"## Anatomy summary\n\n{anatomy_summary(findings)}\n\n"
            f"## Patient findings\n\n{findings_json}\n\n"
            f"## Guideline sections\n\n{inputs['guidelines']}"
        ),
    ]


decide = decision_messages | model.with_structured_output(Decision)


SCOPE_MESSAGES = {
    "STEMI": "Primary PCI of the culprit artery is the standard of care for STEMI. This tool covers NSTEMI only.",
    "posterior_MI": "This tool does not support posterior wall MI cases.",
    "stable": "This tool covers acute presentations (NSTEMI) only.",
    "non_coronary": "This doesn't appear to be a coronary angiography report.",
}

ASSUMPTION_NOTES = {
    "unstable_angina": "Unstable angina: assessed as NSTE-ACS, which shares NSTEMI revascularization guidance.",
    "ACS_unspecified": "ACS type not stated; assumed NSTEMI.",
    "not_stated": "Presentation not stated; assumed NSTEMI.",
}


class Result(NamedTuple):
    findings: Findings | None
    decision: Decision | None
    message: str | None
    note: str | None = None


def has_coronary_findings(findings):
    anatomy = findings.coronary_anatomy
    segments = [seg for name in VESSEL_NAMES for _, seg in getattr(anatomy, name) if isinstance(seg, Segment)]
    segments += [seg for _, seg in anatomy.branches]
    return findings.angiographic_diagnosis.summary is not None or any(
        seg.stenosis_pct or seg.stenosis_qualifier != "not_described" for seg in segments
    )


@traceable(name="analyze_report")
def run_steps(pdf_bytes):
    yield {"stage": "reading"}
    findings = extract.invoke({"pdf64": base64.b64encode(pdf_bytes).decode()})
    if not has_coronary_findings(findings):
        yield {"result": Result(None, None, SCOPE_MESSAGES["non_coronary"])}
        return
    presentation = findings.clinical_context.presentation
    if presentation in SCOPE_MESSAGES:
        yield {"result": Result(findings, None, SCOPE_MESSAGES[presentation])}
        return

    yield {"stage": "searching"}
    inputs = {"findings": findings, "guidelines": retrieve({"findings": findings})}
    yield {"stage": "writing"}
    decision = decide.invoke(inputs)
    yield {"result": Result(findings, decision, None, ASSUMPTION_NOTES.get(presentation))}


def run(pdf_bytes):
    result = None
    for event in run_steps(pdf_bytes):
        result = event.get("result", result)
    return result