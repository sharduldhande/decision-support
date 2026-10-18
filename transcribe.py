import base64
import json
from pathlib import Path
from typing import Literal

import ebooklib
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from ebooklib import epub
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

EPUB_PATH = "ACC_AHA_Guidelines.epub"
OUT_PATH = Path("guideline_recommendations.json")
MODEL = "anthropic:claude-sonnet-5-5"

PROMPT = """\
This image is a recommendation table from the 2021 ACC/AHA/SCAI Guideline for Coronary Artery \
Revascularization. Transcribe it exactly: the table title, then one row per numbered recommendation \
with its Class of Recommendation (COR), Level of Evidence (LOE) and full text. Keep the wording \
verbatim; drop superscript reference numbers. If one COR or LOE cell spans several rows, repeat it \
for each row."""


class Recommendation(BaseModel):
    number: int | None = Field(description="The recommendation's number in the table, if shown")
    cor: Literal["1", "2a", "2b", "3: No Benefit", "3: Harm"] = Field(description="Class of Recommendation")
    loe: Literal["A", "B-R", "B-NR", "C-LD", "C-EO"] = Field(description="Level of Evidence")
    text: str = Field(description="The recommendation text, verbatim")


class RecommendationTable(BaseModel):
    title: str = Field(description="The table's title line, e.g. 'Recommendations for ...'")
    recommendations: list[Recommendation]


def find_tables(path):
    book = epub.read_epub(path)
    images = {item.get_name().rsplit("/", 1)[-1]: item.content
              for item in book.get_items_of_type(ebooklib.ITEM_IMAGE)}
    sections = {}
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        for img in BeautifulSoup(item.content, "html.parser").find_all("img"):
            name = img.get("src", "").rsplit("/", 1)[-1]
            heading = img.find_previous(["h2", "h3", "h4"])
            if ".utab" in name and heading and name not in sections:
                sections[name] = heading.get_text(" ", strip=True)
    return [(name, sections[name], images[name]) for name in sorted(sections)]


def table_messages(image):
    return [HumanMessage([
        {"type": "image", "base64": base64.b64encode(image).decode(), "mime_type": "image/jpeg"},
        {"type": "text", "text": PROMPT},
    ])]


def main():
    load_dotenv()
    done = json.loads(OUT_PATH.read_text()) if OUT_PATH.exists() else {}
    todo = [t for t in find_tables(EPUB_PATH) if t[0] not in done]
    print(f"{len(done)} already transcribed, {len(todo)} to do")

    model = init_chat_model(MODEL, max_tokens=8000).with_structured_output(RecommendationTable)
    results = model.batch([table_messages(image) for _, _, image in todo],
                          config={"max_concurrency": 8}, return_exceptions=True)

    for (name, section, _), result in zip(todo, results):
        if isinstance(result, Exception):
            print(f"FAILED {name}: {str(result)[:150]}")
            continue
        done[name] = {"section": section, **result.model_dump()}
        print(f"{name}: {len(result.recommendations)} recommendations | {section[:60]}")

    OUT_PATH.write_text(json.dumps(dict(sorted(done.items())), indent=2, ensure_ascii=False))
    print(f"Saved {len(done)} tables to {OUT_PATH}")


if __name__ == "__main__":
    main()
