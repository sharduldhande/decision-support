"""Build the guideline index in app/chroma_db using Bedrock Titan embeddings.

Usage: python ingest.py
Runs on this machine with your default AWS credentials; each chunk is one Bedrock call.
"""
import json
import os
import time
from pathlib import Path

import bs4
import chromadb
from dotenv import load_dotenv
from ebooklib import ITEM_DOCUMENT, epub
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

load_dotenv()

import sys  # noqa: E402
sys.path.insert(0, str(Path(__file__).parent / "app"))
from pipeline import COLLECTION, DB_PATH, embeddings  # noqa: E402

DATA = Path(__file__).parent / "data"

ebook = epub.read_epub(str(DATA / "ACC_AHA_Guidelines.epub"))

sections = []
for item in ebook.get_items_of_type(ITEM_DOCUMENT):
    soup = bs4.BeautifulSoup(item.content, "html.parser")
    for heading in soup.find_all(["h1", "h2", "h3", "h4"]):
        content_parts = []
        sibling = heading.find_next_sibling()
        while sibling and sibling.name not in ["h1", "h2", "h3", "h4"]:
            text = sibling.get_text(" ", strip=True)
            if text:
                content_parts.append(text)
            sibling = sibling.find_next_sibling()
        content = "\n".join(content_parts)
        if len(content) > 50:
            sections.append({"title": heading.get_text(" ", strip=True), "content": content})

# Give generic subsection titles their parent's name.
parent_title = ""
for s in sections:
    if s["title"] in ["Synopsis", "Recommendation-Specific Supportive Text",
                      "Recommendation-Specific Supporting Text",
                      "Recommendation-Specific Supporting Tex"]:
        s["title"] = parent_title + " — " + s["title"]
    else:
        parent_title = s["title"]

seen = set()
sections = [s for s in sections if not (s["title"] in seen or seen.add(s["title"]))]

junk = {"Contents", "ACC/AHA Joint Committee Members", "Presidents and Staff",
        "Article Information", "Affiliations", "References", "Sections",
        "Guide", "List of Illustrations", "1.6. Abbreviations"}
sections = [s for s in sections if s["title"] not in junk and not s["title"].startswith("Appendix")]
print("Found sections:", len(sections))

# Character-based now that there is no local tokenizer: ~1600 chars is about the
# 400 PubMedBERT tokens the VPS index used.
splitter = RecursiveCharacterTextSplitter(chunk_size=1600, chunk_overlap=200)

docs = []
for i, section in enumerate(sections):
    for j, piece in enumerate(splitter.split_text(section["content"])):
        docs.append(Document(
            page_content=f"{section['title']}\n{piece}",
            metadata={"title": section["title"], "section": i, "chunk": j},
            id=f"{i}-{j}",
        ))
print(len(sections), "sections ->", len(docs), "chunks")

with open(DATA / "guideline_recommendations.json", encoding="utf-8") as f:
    tables = json.load(f)

for image_name, table in tables.items():
    table_id = image_name.rsplit(".", 2)[-2]
    for rec in table["recommendations"]:
        docs.append(Document(
            page_content=(
                f"{table['section']}\n"
                f"Recommendation (COR {rec['cor']}, LOE {rec['loe']}): {rec['text']}"
            ),
            metadata={"title": table["section"], "kind": "recommendation",
                      "cor": rec["cor"], "loe": rec["loe"]},
            id=f"rec-{table_id}-{rec['number']}",
        ))
print(len(docs), "documents including recommendations")

# Drop any other collection (e.g. an older PubMedBERT one). Keep a partly
# built Titan collection so a rerun resumes instead of re-embedding everything.
client = chromadb.PersistentClient(path=str(DB_PATH))
for collection in client.list_collections():
    if collection.name != COLLECTION:
        client.delete_collection(collection.name)

store = Chroma(
    client=client,
    collection_name=COLLECTION,
    embedding_function=embeddings,
    collection_metadata={"hnsw:space": "cosine"},
)
done = set(store.get(include=[])["ids"])
todo = [d for d in docs if d.id not in done]
print(f"{len(done)} already indexed, {len(todo)} to embed")

# The account may call Titan at most 60 times a minute (one call per text, not adjustable),
# so add one document at a time, about one per second.
for doc in tqdm(todo, unit="doc"):
    started = time.time()
    store.add_documents([doc], ids=[doc.id])
    time.sleep(max(0, 1.05 - (time.time() - started)))

print("Indexed", store._collection.count(), "documents into", COLLECTION)
