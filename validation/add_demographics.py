"""Add an age and sex header line to each excerpt, as a real angiogram report header would have.

Reads data/demographics.csv (exported with demographics.sql) and rewrites data/excerpts_blind.jsonl
and data/excerpts.jsonl in place. The excerpt text itself stays word for word: the original is kept
in `excerpt_raw`, and `excerpt` becomes the header line plus the original. Safe to rerun, since it
always rebuilds from `excerpt_raw`. Also stores `age`, `age_capped` and `sex` on each row, as the
answer key for checking extraction. Prints counts only.

Usage: python -I validation/add_demographics.py
"""

import csv
import json
from pathlib import Path

DATA = Path(__file__).parent / "data"
FILES = ("excerpts_blind.jsonl", "excerpts.jsonl")
SEX = {"M": "Male", "F": "Female"}


def header(person):
    age = ">89" if person["age_capped"] == "true" else person["age"]
    return f"Age: {age} years   Sex: {SEX[person['sex']]}"


def main():
    people = {r["note_id"]: r for r in csv.DictReader(open(DATA / "demographics.csv"))}
    for name in FILES:
        rows = [json.loads(line) for line in open(DATA / name)]
        missing = 0
        for row in rows:
            raw = row.setdefault("excerpt_raw", row["excerpt"])
            person = people.get(row["note_id"])
            if person is None:
                missing += 1
                row["excerpt"] = raw
                continue
            row["excerpt"] = f"{header(person)}\n{raw}"
            row.update(age=int(person["age"]), age_capped=person["age_capped"] == "true", sex=person["sex"])
        with open(DATA / name, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        print(f"{name}: {len(rows) - missing} with a header, {missing} without demographics")


if __name__ == "__main__":
    main()
