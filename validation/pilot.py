"""Run the real pipeline on a small, balanced set of clean MIMIC excerpts.

Picks cases the Haiku review marked clean (one cath, usable anatomy, no problem, no
leak) from data/excerpts_blind.jsonl, sends each excerpt through pipeline.run with
text=True, and writes data/pilot.jsonl. Prints counts only, never note text or
model output that could quote it.

Checks, per case:
- extraction: the significant vessels the pipeline found vs a simple regex reading of
  the excerpt (max stenosis per vessel, >=70%, or >=50% for left main);
- retrieval: how many guideline chunks came back, and whether every citation's
  section title appears in them (grounding);
- decision: the recommendation against the treatment the patient actually received.

Usage: LANGSMITH_TRACING=false python validation/pilot.py [per_label]   (default 25)
"""

import hashlib
import json
import os
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

ROOT = Path(__file__).parent.parent
DATA = Path(__file__).parent / "data"
sys.path.insert(0, str(ROOT))

if os.environ.get("LANGSMITH_TRACING", "").lower() != "false":
    raise SystemExit("Set LANGSMITH_TRACING=false: MIMIC text must not be sent to LangSmith.")
load_dotenv(ROOT / ".env")
import pipeline  # noqa: E402  (needs the API key loaded first)

VESSELS = {
    "LM": r"LMCA|left main|\bLM\b",
    "LAD": r"\bLAD\b|left anterior descending",
    "LCX": r"\bLCx\b|circumflex|\bOM\d?\b",
    "RCA": r"\bRCA\b|right coronary|\bPDA\b",
}


GRAFTS = re.compile(r"\b(SVG|LIMA|RIMA|LITA|RITA|grafts?|prior CABG|s/p CABG)\b", re.I)


def regex_vessels(text):
    found = set()
    for name, pattern in VESSELS.items():
        pcts = [int(m.group(2)) for m in re.finditer(rf"({pattern})[^\n%]{{0,60}}?(\d{{2,3}})\s?%", text, re.I)]
        pcts = [p for p in pcts if p <= 100]
        if max(pcts, default=0) >= (50 if name == "LM" else 70):
            found.add(name)
    return found


def pipeline_vessels(findings):
    # Same four names as the regex; the regex has no ramus, so leave it out.
    names = {"Left Main": "LM", "LAD": "LAD", "LCx": "LCX", "RCA": "RCA"}
    return {names[v] for v in findings.significant_vessels if v in names}


def side(recommendation):
    for label in ("PCI", "CABG"):
        if recommendation.startswith(label):
            return label
    return "other"


def pick(per_label):
    excerpts = {r["note_id"]: r for r in map(json.loads, open(DATA / "excerpts_blind.jsonl"))}
    clean = [r["note_id"] for r in map(json.loads, open(DATA / "review_excerpts_blind.jsonl"))
             if r["leak"] == "none" and r["anatomy_usable"] and r["describes_one_cath"] and r["problem"] == "none"]
    order = sorted(clean, key=lambda n: hashlib.sha256(n.encode()).hexdigest())
    chosen = []
    for label in ("CABG", "PCI"):
        chosen += [excerpts[n] for n in order if excerpts[n]["label"] == label][:per_label]
    return chosen


def run_case(case):
    out = {"note_id": case["note_id"], "label": case["label"]}
    try:
        result = pipeline.run(case["excerpt"], text=True)
    except Exception as e:  # keep going; record the error type and failing field paths, never values
        fields = [".".join(map(str, err["loc"])) for err in e.errors()] if hasattr(e, "errors") else []
        return {**out, "status": "error", "error": type(e).__name__, "error_fields": fields}
    if result.findings is None:
        return {**out, "status": "no_coronary_findings"}
    ref, got = regex_vessels(case["excerpt"]), pipeline_vessels(result.findings)
    ctx = result.findings.clinical_context
    out.update(regex_vessels=sorted(ref), pipeline_vessels=sorted(got), presentation=ctx.presentation,
               age_true=case.get("age"), age_capped=case.get("age_capped"), age_got=ctx.age,
               sex_true=case.get("sex"), sex_got=ctx.sex, prior_cabg_got=ctx.prior_cabg,
               grafts_in_excerpt=bool(GRAFTS.search(case.get("excerpt_raw", case["excerpt"]))))
    if result.decision is None:
        return {**out, "status": "out_of_scope"}

    guidelines = result.guidelines        # the exact text the decision was given
    cites = result.decision.supporting_citations
    out.update(
        status="ok",
        queries=len(result.queries),
        extra_queries=len(result.queries) - len(pipeline.search_queries(result.findings)),
        sections=[line[4:] for line in guidelines.splitlines() if line.startswith("### ")],
        recommendation=result.decision.recommendation,
        confidence=result.decision.confidence,
        chunks=guidelines.count("### "),
        citations=len(cites),
        formal_citations=sum(c.class_of_recommendation is not None for c in cites),
        grounded=all(c.section_title.lower() in guidelines.lower() for c in cites),
    )
    return out


def main(per_label=25):
    cases = pick(int(per_label))
    pipeline.get_retriever()   # load the embedding model once, before the threads start
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(tqdm(pool.map(run_case, cases), total=len(cases), unit="case"))
    with open(DATA / "pilot.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    print(f"cases: {len(rows)} ({dict(Counter(r['label'] for r in rows))})")
    print(f"status: {dict(Counter(r['status'] for r in rows))}")
    print(f"errors: {dict(Counter(r['error'] for r in rows if r['status'] == 'error'))}")
    print(f"error fields: {dict(Counter(f for r in rows for f in r.get('error_fields', [])))}")
    print(f"presentation: {dict(Counter(r.get('presentation') for r in rows if 'presentation' in r))}")

    extracted = [r for r in rows if "pipeline_vessels" in r]
    exact = sum(r["pipeline_vessels"] == r["regex_vessels"] for r in extracted)
    count = sum(len(r["pipeline_vessels"]) == len(r["regex_vessels"]) for r in extracted)
    print(f"extraction vs regex: same vessels {exact}/{len(extracted)}, same count {count}/{len(extracted)}")
    per_vessel = Counter()
    for r in extracted:
        for v in VESSELS:
            per_vessel[(v, v in r["regex_vessels"], v in r["pipeline_vessels"])] += 1
    for v in VESSELS:
        agree = per_vessel[(v, True, True)] + per_vessel[(v, False, False)]
        print(f"  {v:3} agree {agree}/{len(extracted)}, regex-only {per_vessel[(v, True, False)]}, "
              f"pipeline-only {per_vessel[(v, False, True)]}")

    exact_age = [r for r in extracted if r.get("age_true") is not None and not r.get("age_capped")]
    capped = [r for r in extracted if r.get("age_capped")]
    sexed = [r for r in extracted if r.get("sex_true")]
    print(f"age: exact {sum(r['age_got'] == r['age_true'] for r in exact_age)}/{len(exact_age)}, "
          f"capped (>89) left null {sum(r['age_got'] is None for r in capped)}/{len(capped)}")
    print(f"sex: correct {sum(r['sex_got'] == {'M': 'male', 'F': 'female'}[r['sex_true']] for r in sexed)}/{len(sexed)}")
    grafts = Counter((r["grafts_in_excerpt"], bool(r["prior_cabg_got"])) for r in extracted)
    print(f"prior CABG vs graft words in excerpt: both {grafts[(True, True)]}, words only {grafts[(True, False)]}, "
          f"pipeline only {grafts[(False, True)]}, neither {grafts[(False, False)]}")

    ok = [r for r in rows if r["status"] == "ok"]
    if ok:
        print(f"retrieval: chunks median {sorted(r['chunks'] for r in ok)[len(ok) // 2]}, "
              f"all citations grounded {sum(r['grounded'] for r in ok)}/{len(ok)}, "
              f"with a formal recommendation {sum(r['formal_citations'] > 0 for r in ok)}/{len(ok)}")
        print("recommendation by actual treatment:")
        for label in ("CABG", "PCI"):
            c = Counter(r["recommendation"] for r in ok if r["label"] == label)
            print(f"  {label:4} {dict(c)}")
        match = sum(side(r["recommendation"]) == r["label"] for r in ok)
        print(f"recommendation matches actual treatment: {match}/{len(ok)}")


if __name__ == "__main__":
    main(*sys.argv[1:])
