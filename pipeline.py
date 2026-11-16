import os
import re
import base64
from collections import defaultdict
from typing import NamedTuple
from functools import lru_cache
from pathlib import Path
from langchain.chat_models import init_chat_model
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langsmith import traceable
import chromadb
from pydantic import ValidationError
from schemas import Findings, Decision, Segment, VESSEL_NAMES

MODEL = os.environ.get("LLM_MODEL", "anthropic:claude-haiku-5-5")

model = init_chat_model(MODEL, max_tokens=8000)

DB_PATH = Path(__file__).parent / "chroma_db"
COLLECTION = "guidelines_v2"
EMBED_MODEL = "NeuML/pubmedbert-base-embeddings"
CHUNKS_PER_QUERY = 8
CHUNKS_PER_CONDITIONAL_QUERY = 4
PARENTS_MAX = 8
FULL_SECTION_MAX_CHARS = 6000   # longer sections send only the matched chunks plus their neighbours

DIABETES = re.compile(r"diabet|\b(?:N?ID|T[12]\s?|TYPE\s?[12]\s?)?DM(?:\s?[12])?\b", re.I)
WEAK_LV = re.compile(r"dysfunction|reduced|depressed|poor|impaired", re.I)
STEMI_SECTION = re.compile(r"^5\.(\d+\.)?\s")   # section 5 is STEMI-only; STEMI cases exit before retrieval

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
    if "text" in inputs:
        report = {"type": "text", "text": f"<report>\n{inputs['text']}\n</report>"}
    else:
        report = {"type": "file", "base64": inputs["pdf64"], "mime_type": "application/pdf"}
    return [HumanMessage([report, {"type": "text", "text": EXTRACTION_PROMPT}])]

# Structured output occasionally comes back malformed at random; one retry almost always succeeds.
RETRY = {"retry_if_exception_type": (ValidationError,), "stop_after_attempt": 2}

extract = (extraction_messages | model.with_structured_output(Findings)).with_retry(**RETRY)


def describe_lesion(lesion):
    location = f"{lesion.vessel} {lesion.segment}".strip()
    pct = "-".join(map(str, lesion.stenosis_pct)) + "%" if lesion.stenosis_pct else lesion.stenosis_qualifier
    return f"{location} {pct} ({lesion.severity})"

def search_queries(findings):
    """(question, k) pairs; conditional questions fetch fewer chunks."""
    ctx = findings.clinical_context
    lesions = findings.lesions
    vessels = findings.significant_vessels
    left_main = "Left Main" in vessels
    n_vessels = len([v for v in vessels if v != "Left Main"])
    k, k_cond = CHUNKS_PER_QUERY, CHUNKS_PER_CONDITIONAL_QUERY

    queries = [("NSTE-ACS PCI versus CABG multivessel coronary artery disease", k)]
    if lesions:
        queries.append(("; ".join(describe_lesion(l) for l in lesions), k))
    if ctx.comorbidities:
        queries.append(("; ".join(ctx.comorbidities) + " coronary revascularization", k))
    if n_vessels:
        queries.append((f"NSTE-ACS revascularization {n_vessels}-vessel disease", k))

    if left_main:
        queries.append(("significant left main stenosis CABG versus PCI survival anatomic complexity", k_cond))
    if not vessels:
        queries.append(("no anatomic or physiological criteria for revascularization should not be performed", k_cond))
    if any(l.vessel == "LAD" and l.segment == "proximal" and l.severity == "significant" for l in lesions):
        queries.append(("proximal LAD stenosis revascularization survival", k_cond))
    low_ef = (ctx.lvef_pct is not None and ctx.lvef_pct <= 50) or (
        ctx.lvef_pct is None and ctx.lv_function and WEAK_LV.search(ctx.lv_function)
    )
    if low_ef and (n_vessels >= 2 or left_main):
        queries.append(("multivessel CAD left ventricular systolic dysfunction ejection fraction CABG survival", k_cond))
    if ctx.prior_cabg:
        queries.append(("prior CABG patent LIMA graft repeat revascularization PCI", k_cond))
    if ctx.age is not None and ctx.age >= 75:
        queries.append(("older adults high surgical risk PCI versus CABG", k_cond))
    if ctx.valvular_disease:
        queries.append((f"{ctx.valvular_disease} valve surgery significant CAD concomitant CABG", k_cond))
    if any(DIABETES.search(c) for c in ctx.comorbidities):
        queries.append(("diabetes multivessel CAD involving LAD CABG over PCI", k_cond))
    if any(l.severity == "intermediate" for l in lesions):
        queries.append(("intermediate stenosis physiological assessment FFR iFR to guide revascularization", k_cond))
    return queries


def base_title(title):
    return title.split(" — ")[0]      # "7.1. … — Synopsis" -> "7.1. …"

def strip_title(text):
    return text.split("\n", 1)[-1]    # every stored chunk starts with its title line

@lru_cache(maxsize=1)
def guideline_index():
    """All recommendation rows by section title, and all prose chunks by section number."""
    data = get_retriever().vectorstore.get(include=["documents", "metadatas"])
    recs, chunks = defaultdict(list), defaultdict(dict)
    for text, meta in zip(data["documents"], data["metadatas"]):
        if meta.get("kind") == "recommendation":
            recs[meta["title"]].append(text)
        else:
            chunks[meta["section"]][meta["chunk"]] = text
    return recs, chunks


EXTRA_QUERIES = os.environ.get("EXTRA_QUERIES", "1") != "0"
EXTRA_QUERIES_MAX = 4

EXTRA_QUERIES_PROMPT = PromptTemplate.from_template("""\
You write search queries for the 2021 ACC/AHA/SCAI Guideline for Coronary Artery Revascularization.

Below are a patient's extracted findings and the queries already being run. Write up to 4 additional \
short queries (5-12 words) for patient factors in the findings that could change the choice between \
PCI, CABG or medical therapy and that the existing queries do not already cover. Examples: kidney \
disease or dialysis, anticoagulation or bleeding risk, inability to take dual antiplatelet therapy, \
lesion complexity (heavy calcification, chronic total occlusion, bifurcation, diffuse disease), \
hemodynamic instability, frailty, or other cardiac surgery needed.

Use the guideline's vocabulary and only cover factors present in the findings. Write one query per \
line, with no numbering or other text. Write nothing if nothing is left to cover.

{question}""")


def extra_queries_input(findings, existing):
    existing = "\n".join(f"- {q}" for q in existing)
    return (f"## Findings\n\n{findings.model_dump_json(indent=2, exclude_defaults=True)}\n\n"
            f"## Existing queries\n\n{existing}")


@lru_cache(maxsize=1)
def get_multi_query():
    return MultiQueryRetriever.from_llm(retriever=get_retriever(), llm=model, prompt=EXTRA_QUERIES_PROMPT)


def all_queries(findings):
    """Rule-based questions, plus a few written by the model for whatever the rules don't cover."""
    queries = search_queries(findings)
    if not EXTRA_QUERIES:
        return queries
    try:
        text = extra_queries_input(findings, [q for q, _ in queries])
        extra = get_multi_query().llm_chain.invoke({"question": text})
    except Exception:
        return queries          # retrieval still works on the rules alone
    seen = {q.lower() for q, _ in queries}
    extra = [q.strip() for q in extra if q.strip() and q.strip().lower() not in seen]
    return queries + [(q, CHUNKS_PER_CONDITIONAL_QUERY) for q in extra[:EXTRA_QUERIES_MAX]]


def retrieve(inputs, queries=None):
    retriever = get_retriever()
    hits, must_keep = {}, []
    if queries is None:
        queries = search_queries(inputs["findings"])
    for q, (query, k) in enumerate(queries):
        for pos, doc in enumerate(retriever.invoke(query, k=k)):
            parent = base_title(doc.metadata["title"])
            if STEMI_SECTION.match(parent):
                continue
            hit = hits.setdefault(parent, {"queries": set(), "best": pos, "sections": defaultdict(set)})
            hit["queries"].add(q)
            hit["best"] = min(hit["best"], pos)
            if doc.metadata.get("kind") != "recommendation":
                hit["sections"][doc.metadata["section"]].add(doc.metadata["chunk"])
            if pos == 0 and parent not in must_keep:
                must_keep.append(parent)          # every question keeps its best match

    ranked = sorted(hits, key=lambda p: (-len(hits[p]["queries"]), hits[p]["best"]))
    top = must_keep + [p for p in ranked if p not in must_keep][:max(0, PARENTS_MAX - len(must_keep))]

    recs, chunks = guideline_index()
    blocks = []
    for parent in top:
        if recs.get(parent):
            blocks.append(f"### {parent} — Recommendations\n" + "\n".join(strip_title(t) for t in recs[parent]))
        for section, hit_chunks in sorted(hits[parent]["sections"].items()):
            pieces = chunks[section]
            title = pieces[min(pieces)].split("\n", 1)[0]
            if sum(map(len, pieces.values())) <= FULL_SECTION_MAX_CHARS:
                keep = sorted(pieces)
            else:
                keep = sorted({c + d for c in hit_chunks for d in (-1, 0, 1)} & pieces.keys())
            body, prev = "", None
            for c in keep:
                body += ("\n[...]\n" if prev is not None and c != prev + 1 else "\n") + strip_title(pieces[c])
                prev = c
            blocks.append(f"### {title}{body}")
    return "\n\n---\n\n".join(blocks)


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

Recommendations written specifically for STEMI (section 5) do not apply to this patient; do not cite them.
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


decide = (decision_messages | model.with_structured_output(Decision)).with_retry(**RETRY)


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
    queries: list | None = None
    guidelines: str | None = None


def has_coronary_findings(findings):
    anatomy = findings.coronary_anatomy
    segments = [seg for name in VESSEL_NAMES for _, seg in getattr(anatomy, name) if isinstance(seg, Segment)]
    segments += [seg for _, seg in anatomy.branches]
    return findings.angiographic_diagnosis.summary is not None or any(
        seg.stenosis_pct or seg.stenosis_qualifier != "not_described" for seg in segments
    )


@traceable(name="analyze_report")
def run_steps(report, text=False):
    """report is PDF bytes, or the report as a string when text=True."""
    yield {"stage": "reading"}
    inputs = {"text": report} if text else {"pdf64": base64.b64encode(report).decode()}
    findings = extract.invoke(inputs)
    if not has_coronary_findings(findings):
        yield {"result": Result(None, None, SCOPE_MESSAGES["non_coronary"])}
        return
    presentation = findings.clinical_context.presentation
    if presentation in SCOPE_MESSAGES:
        yield {"result": Result(findings, None, SCOPE_MESSAGES[presentation])}
        return

    yield {"stage": "searching"}
    queries = all_queries(findings)
    guidelines = retrieve({"findings": findings}, queries)
    yield {"stage": "writing"}
    decision = decide.invoke({"findings": findings, "guidelines": guidelines})
    yield {"result": Result(findings, decision, None, ASSUMPTION_NOTES.get(presentation), queries, guidelines)}


def run(report, text=False):
    result = None
    for event in run_steps(report, text):
        result = event.get("result", result)
    return result