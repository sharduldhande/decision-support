"""Cut the cath findings out of each discharge note in the validation sample.

Reads validation/data/sample.csv (exported from ds_validation_sample) and writes
validation/data/excerpts.jsonl. Prints counts only, never note text.

The excerpt is the cluster of stenosis mentions that best looks like a cath report,
trimmed of lines that describe what was done afterwards (stents, bypass), since
those give the label away.
"""

import csv
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

DATA = Path(__file__).parent / "data"

STENOSIS = re.compile(r"(LAD|LMCA|left main|RCA|LCx|circumflex)[^\n]{0,60}?\d{2,3}\s?%", re.I)
SECTION = re.compile(r"^\s*([A-Z][A-Za-z /&\-]{2,40}):", re.M)

# Headers that start a structured cath report inside Pertinent Results.
CATH_HEADERS = re.compile(
    r"^\s*(cardiac cath(eterization)?|coronary angiography|dominance|lmca|comments|impression)\s*:",
    re.I | re.M,
)
# Narrative sections: usable as a fallback, but they mix in later events.
NARRATIVE = {"BRIEF HOSPITAL COURSE", "HISTORY OF PRESENT ILLNESS", "HOSPITAL COURSE"}

# Anything that says what treatment followed the cath.
LEAK = re.compile(
    r"\b(CABG|bypass|surgery|surgical|cardiac surgery|CT surgery|stent(s|ed|ing)?|DES|BMS|PCI|"
    r"PTCA|angioplasty|balloon|POBA|rotablat\w*|atherectomy|intervention|intervened|"
    # results of a PCI done during the same cath
    r"residual|final (angiograph\w*|injection|result)|TIMI|wire|dissection|"
    r"angiographically apparent|success(ful|fully)?|deploy\w*|post-?dilat\w*|pre-?dilat\w*|"
    r"kissing|thrombectomy|guide catheter|crossed|"
    # pre-surgery workup and antiplatelet plans
    r"carotid|aspirin|plavix|clopidogrel|prasugrel|ticagrelor|"
    # referrals and plans
    r"c-?surg\w*|CT ?surg\w*|cardiothoracic|cardiac surg\w*|consult\w*|referr?(ed|al)|"
    r"(?-i:OR)|s/p|candidate|evaluation for|revasculari[sz]ation)\b",   # "OR" only in capitals
    re.I,
)

# The excerpt ends where another study or the medication list begins.
STOP = re.compile(
    r"^\s*(carotid|echo\w*|TTE|TEE|ECG|EKG|CXR|chest|abdom\w*|renal|ultrasound|US |CT |MRI|"
    r"discharge (labs|medications|diagnosis)|medications|brief hospital course|"
    r"pertinent results|admission labs|interventional details|hemodynamics)\b",
    re.I | re.M,
)
MEDICATION = re.compile(r"\b\d+(\.\d+)?\s*(mg|mcg|units?)\b|\b(daily|bid|tid|qhs|prn)\b", re.I)

CLUSTER_GAP = 800    # stenosis mentions closer than this belong to one block
LOOKBACK = 1500      # how far before a block to look for a cath header
MAX_CHARS = 3000


def section_at(text, pos):
    heads = [m for m in SECTION.finditer(text, 0, pos)]
    return heads[-1].group(1).strip().upper() if heads else ""


def clusters(text):
    hits = [m.start() for m in STENOSIS.finditer(text)]
    groups, current = [], [hits[0]]
    for h in hits[1:]:
        if h - current[-1] > CLUSTER_GAP:
            groups.append(current)
            current = []
        current.append(h)
    groups.append(current)
    return groups


def best_block(text):
    """Return (start, end, source) for the most cath-report-like cluster."""
    scored = []
    for group in clusters(text):
        first, last = group[0], group[-1]
        window = text[max(0, first - LOOKBACK):first]
        headers = list(CATH_HEADERS.finditer(window))
        section = section_at(text, first)
        score = len(group) + (5 if headers else 0) - (5 if section in NARRATIVE else 0)
        if headers:
            start = max(0, first - LOOKBACK) + headers[0].start()
        else:
            para = text.rfind("\n\n", 0, first)
            start = para + 2 if para != -1 else first
        end = text.find("\n\n", last)
        end = len(text) if end == -1 else end
        stop = STOP.search(text, last, end)
        end = stop.start() if stop else end
        source = "cath_block" if headers else ("narrative" if section in NARRATIVE else "other")
        scored.append((score, start, min(end, start + MAX_CHARS), source))
    score, start, end, source = max(scored)
    return start, end, source


def unwrap(block):
    """Rejoin lines the note hard-wrapped at about 70 characters.

    Without this, a sentence that wraps onto a second line is checked as two pieces,
    and a treatment word on the second line slips past the sentence filter.
    """
    lines = []
    for line in block.splitlines():
        line = line.strip()
        if not line:
            continue
        starts_item = re.match(r"(\d+\.|\*|-|[A-Z][A-Za-z /]{2,40}:)", line)
        if lines and not starts_item and not re.search(r"[.:;]$", lines[-1]):
            lines[-1] += " " + line
        else:
            lines.append(line)
    return lines


def strip_leaks(block):
    """Drop sentences that mention treatment; keep the rest verbatim."""
    kept, dropped = [], 0
    for line in unwrap(block):
        if MEDICATION.search(line):
            dropped += 1
            continue
        # Report headers ("Pertinent Results:", "Date: ___ Place: ___") say where the cath
        # was done, which tracks transfers for surgery. Keep the findings, drop the header.
        line = re.sub(r"(?i)\b(pertinent results|cardiac catheterization|catheterization|"
                      r"coronary angiography|date|place|study date|comments|impression)\s*:\s*(___\s*)?",
                      "", line).strip()
        if not line:
            continue
        parts = re.split(r"(?<=[.;])\s+", line)
        clean = [p for p in parts if not LEAK.search(p)]
        dropped += len(parts) - len(clean)
        if clean:
            kept.append(" ".join(clean))
    return "\n".join(kept).strip(), dropped


def main():
    csv.field_size_limit(sys.maxsize)
    rows = list(csv.DictReader(open(DATA / "sample.csv")))
    out, sources, lengths, dropped_by_label = [], Counter(), [], Counter()
    still_has_stenosis = 0

    for r in rows:
        text = r["text"]
        start, end, source = best_block(text)
        block = text[start:end]
        excerpt, dropped = strip_leaks(block)

        sources[(r["label"], source)] += 1
        lengths.append(len(excerpt))
        dropped_by_label[r["label"]] += dropped
        still_has_stenosis += bool(STENOSIS.search(excerpt))
        out.append({
            "note_id": r["note_id"], "hadm_id": r["hadm_id"], "label": r["label"],
            "source": source, "sentences_dropped": dropped, "excerpt": excerpt,
        })

    with open(DATA / "excerpts.jsonl", "w") as f:
        for row in out:
            f.write(json.dumps(row) + "\n")

    print(f"notes: {len(rows)}")
    print(f"excerpts still containing a stenosis %: {still_has_stenosis}")
    print(f"excerpt length (chars): median {int(statistics.median(lengths))}, "
          f"max {max(lengths)}, under 200: {sum(l < 200 for l in lengths)}")
    print("source by label:")
    for (label, source), n in sorted(sources.items()):
        print(f"  {label:5} {source:11} {n}")
    print("treatment sentences dropped:", dict(dropped_by_label))
    print(f"wrote {DATA / 'excerpts.jsonl'}")


if __name__ == "__main__":
    main()
