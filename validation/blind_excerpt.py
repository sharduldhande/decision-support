"""Build cath-findings excerpts by classifying sentences blind.

The regex cut in excerpt.py is crude, but letting a model pick sentences from the
whole note is worse: it would see the treatment that followed and could keep the
anatomy that fits it. Here each candidate sentence is classified on its own, mixed
into batches with sentences from other patients, so the model never sees which note
a sentence came from or what happened next.

Steps:
1. Find every block of stenosis mentions in each note (label-blind regex).
2. Split those blocks into sentences.
3. Shuffle all sentences across notes and send them to Haiku in batches of 30;
   each comes back as finding / treatment / other.
4. Rebuild each note's excerpt from its finding sentences, in original order, using
   the block with the most findings.

Reads validation/data/sample.csv, writes validation/data/excerpts_blind.jsonl and
validation/data/sentence_labels.jsonl. Prints counts only.

Usage: python validation/blind_excerpt.py [limit_notes] [rebuild]
  rebuild reuses data/sentence_labels.jsonl instead of calling the API.
"""

import csv
import json
import random
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from pathlib import Path

import anthropic
from pydantic import BaseModel
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from excerpt import LEAK, LOOKBACK, MEDICATION, STOP, clusters, unwrap  # noqa: E402
from review import api_key  # noqa: E402

DATA = Path(__file__).parent / "data"
MODEL = "claude-haiku-5-5"
BATCH = 30


class Kind(str, Enum):
    finding = "finding"
    treatment = "treatment"
    other = "other"


class Label(BaseModel):
    n: int
    kind: Kind


class Labels(BaseModel):
    labels: list[Label]


PROMPT = """\
Each numbered line below is one sentence from a hospital discharge note. The sentences \
come from many different patients and are in random order; judge each one on its own.

Label each sentence:
- finding: describes what a coronary angiogram or cardiac catheterization showed, such \
as dominance, a vessel's stenosis or occlusion, collaterals, flow, LV function or \
pressures. Mentions of stents or bypass grafts from earlier procedures count as \
findings when they describe the anatomy (for example "patent LIMA to LAD").
- treatment: says what was done or planned for this patient now: a stent placed, \
balloon or PCI performed or its result, bypass surgery, a surgery consult or referral, \
or any recommended treatment.
- other: anything else (history, symptoms, labs, medications, other tests, headers).

Return a label for every number.

{lines}"""


def candidate_blocks(text):
    """Every stenosis block, widened to its paragraph and cut at the next other study."""
    blocks = []
    for group in clusters(text):
        first, last = group[0], group[-1]
        para = text.rfind("\n\n", max(0, first - LOOKBACK), first)
        start = para + 2 if para != -1 else max(0, first - LOOKBACK)
        end = text.find("\n\n", last)
        end = len(text) if end == -1 else end
        stop = STOP.search(text, last, end)
        blocks.append(text[start:stop.start() if stop else end])
    return blocks


def sentences(block):
    out = []
    for line in unwrap(block):
        if MEDICATION.search(line):
            continue
        out += [s for s in re.split(r"(?<=[.;])\s+", line) if len(s.strip()) > 3]
    return out


def classify(client, batch):
    lines = "\n".join(f"{i + 1}. {s['text']}" for i, s in enumerate(batch))
    response = client.messages.parse(
        model=MODEL,
        max_tokens=8000,
        messages=[{"role": "user", "content": PROMPT.format(lines=lines)}],
        output_format=Labels,
    )
    kinds = {l.n: l.kind.value for l in response.parsed_output.labels}
    u = response.usage
    return [(s, kinds.get(i + 1, "missing")) for i, s in enumerate(batch)], u.input_tokens, u.output_tokens


def main(limit=0, mode=""):
    rebuild = mode == "rebuild"
    csv.field_size_limit(sys.maxsize)
    notes = list(csv.DictReader(open(DATA / "sample.csv")))
    if int(limit):
        notes = notes[: int(limit)]

    pool_ = []
    for note in notes:
        for b, block in enumerate(candidate_blocks(note["text"])):
            for k, text in enumerate(sentences(block)):
                pool_.append({"note_id": note["note_id"], "block": b, "pos": k, "text": text})
    random.Random(0).shuffle(pool_)
    batches = [pool_[i:i + BATCH] for i in range(0, len(pool_), BATCH)]
    print(f"notes: {len(notes)}, candidate sentences: {len(pool_)}, batches: {len(batches)}")

    labelled, tokens_in, tokens_out = [], 0, 0
    if rebuild:
        # Reuse the saved labels, e.g. after changing the safety net. No API calls.
        for line in open(DATA / "sentence_labels.jsonl"):
            s = json.loads(line)
            labelled.append((s, s.pop("kind")))
    else:
        client = anthropic.Anthropic(api_key=api_key())
        with ThreadPoolExecutor(max_workers=8) as ex:
            for result, tin, tout in tqdm(ex.map(lambda b: classify(client, b), batches),
                                          total=len(batches), unit="batch"):
                labelled += result
                tokens_in += tin
                tokens_out += tout
        with open(DATA / "sentence_labels.jsonl", "w") as f:
            for s, kind in labelled:
                f.write(json.dumps({**s, "kind": kind}) + "\n")

    kinds = Counter(k for _, k in labelled)
    by_note = defaultdict(lambda: defaultdict(list))
    regex_overrides = 0
    for s, kind in labelled:
        if kind != "finding":
            continue
        if LEAK.search(s["text"]):      # label-blind safety net
            regex_overrides += 1
            continue
        by_note[s["note_id"]][s["block"]].append((s["pos"], s["text"]))

    written, empty = 0, 0
    with open(DATA / "excerpts_blind.jsonl", "w") as f:
        for note in notes:
            blocks = by_note.get(note["note_id"], {})
            if not blocks:
                empty += 1
            best = max(blocks.values(), key=len) if blocks else []
            excerpt = "\n".join(text for _, text in sorted(best))
            f.write(json.dumps({
                "note_id": note["note_id"], "hadm_id": note["hadm_id"], "label": note["label"],
                "source": "blind", "sentences_kept": len(best), "excerpt": excerpt,
            }) + "\n")
            written += 1

    print(f"sentence labels: {dict(kinds)}")
    print(f"findings dropped by the regex safety net: {regex_overrides}")
    print(f"excerpts written: {written}, with no findings: {empty}")
    cost = tokens_in * 0.10 / 1e6 + tokens_out * 0.50 / 1e6
    print(f"tokens in {tokens_in}, out {tokens_out}, about ${cost:.3f}")


if __name__ == "__main__":
    main(*sys.argv[1:])
