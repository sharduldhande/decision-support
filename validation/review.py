"""Have Claude Haiku grade each excerpt for quality and treatment leakage.

Reads validation/data/excerpts.jsonl, writes validation/data/review.jsonl and prints
counts. The model answers in fixed categories only, so nothing from the notes is
echoed back. Calls the Anthropic API directly (no LangChain, no tracing).

Usage: python validation/review.py [limit] [name]
  name is the excerpts file in validation/data without .jsonl (default: excerpts);
  0 as the limit means all.
"""

import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from pathlib import Path

import anthropic
from pydantic import BaseModel
from tqdm import tqdm

ROOT = Path(__file__).parent.parent
DATA = Path(__file__).parent / "data"
MODEL = "claude-haiku-5-5"


class Problem(str, Enum):
    none = "none"
    not_a_cath = "not_a_cath"            # a different test, or no angiography described
    history_only = "history_only"        # an old cath mentioned in passing
    truncated = "truncated"              # cut off before the anatomy is clear
    garbled = "garbled"                  # broken sentences from trimming
    several_procedures = "several_procedures"


class Leak(str, Enum):
    none = "none"
    surgery_referral = "surgery_referral"    # consult, referral or plan for bypass surgery
    pci_performed = "pci_performed"          # stent, balloon or PCI result in this cath
    other_plan = "other_plan"                # any other statement of treatment chosen


class Review(BaseModel):
    describes_one_cath: bool
    anatomy_usable: bool
    problem: Problem
    leak: Leak


PROMPT = """\
Below is an excerpt meant to contain only the findings of one cardiac catheterization \
(coronary anatomy and stenoses), cut from a hospital discharge note. It will be used to \
test whether a system can choose between PCI and CABG from the anatomy alone, so the \
excerpt must not reveal which treatment the patient actually received or was referred for.

Judge the excerpt:
- describes_one_cath: does it describe the findings of a single catheterization?
- anatomy_usable: could a cardiologist tell which vessels are diseased and how severely?
- problem: the main quality problem, or "none".
- leak: does any wording reveal or strongly hint at the treatment chosen? Prior stents or \
grafts from earlier procedures are history, not a leak.

<excerpt>
{excerpt}
</excerpt>"""


def api_key():
    for line in open(ROOT / ".env"):
        if line.startswith("ANTHROPIC_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"')
    raise SystemExit("ANTHROPIC_API_KEY not found in .env")


def review(client, row):
    response = client.messages.parse(
        model=MODEL,
        max_tokens=4000,
        messages=[{"role": "user", "content": PROMPT.format(excerpt=row["excerpt"])}],
        output_format=Review,
    )
    usage = response.usage
    return row, response.parsed_output, usage.input_tokens, usage.output_tokens


def main(limit=0, name="excerpts"):
    rows = [json.loads(line) for line in open(DATA / f"{name}.jsonl")]
    if int(limit):
        rows = rows[: int(limit)]
    out_path = DATA / ("review.jsonl" if name == "excerpts" else f"review_{name}.jsonl")
    client = anthropic.Anthropic(api_key=api_key())

    tokens_in = tokens_out = 0
    counts = {k: Counter() for k in ("problem", "leak", "one_cath", "usable")}
    leak_by = Counter()
    with ThreadPoolExecutor(max_workers=8) as pool, open(out_path, "w") as out:
        results = pool.map(lambda x: review(client, x), rows)
        for row, r, tin, tout in tqdm(results, total=len(rows), unit="note"):
            tokens_in += tin
            tokens_out += tout
            counts["problem"][r.problem.value] += 1
            counts["leak"][r.leak.value] += 1
            counts["one_cath"][r.describes_one_cath] += 1
            counts["usable"][r.anatomy_usable] += 1
            leak_by[(row["label"], row["source"], r.leak.value)] += 1
            out.write(json.dumps({"note_id": row["note_id"], **r.model_dump(mode="json")}) + "\n")

    print(f"reviewed: {len(rows)}")
    for k, c in counts.items():
        print(f"{k}: {dict(c)}")
    print("leaks by label/source:")
    for (label, source, leak), n in sorted(leak_by.items()):
        if leak != "none":
            print(f"  {label:5} {source:11} {leak:17} {n}")
    cost = tokens_in * 0.10 / 1e6 + tokens_out * 0.50 / 1e6
    print(f"tokens in {tokens_in}, out {tokens_out}, about ${cost:.3f}")


if __name__ == "__main__":
    main(*sys.argv[1:])
