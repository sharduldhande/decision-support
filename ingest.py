from ebooklib import epub
import bs4
import json
import chromadb
from tqdm import tqdm
from transformers import AutoTokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

ebook = epub.read_epub("ACC_AHA_Guidelines.epub")

items = ebook.get_items_of_type(9)

sections = []

for item in items:
    soup = bs4.BeautifulSoup(item.content, "html.parser")
    headings = soup.find_all(["h1","h2", "h3", "h4"])
    for heading in headings:
        title = heading.get_text(" ", strip=True)
        content_parts = []

        sibling = heading.find_next_sibling()
        while sibling and sibling.name not in ["h1","h2", "h3", "h4"]:
            text = sibling.get_text(" ", strip=True)
            if text:
                content_parts.append(text)
            sibling = sibling.find_next_sibling()

        content = "\n".join(content_parts)
        if len(content) > 50:
            sections.append({
                "title": title,
                "content": content,
            })

parent_title = ""
for s in sections:
    if s["title"] in ["Synopsis", "Recommendation-Specific Supportive Text",
                       "Recommendation-Specific Supporting Text",
                       "Recommendation-Specific Supporting Tex"]:
        s["title"] = parent_title + " — " + s["title"]
    else:
        parent_title = s["title"]

seen = set()
deduped = []

for s in sections:
    if s["title"] not in seen:
        seen.add(s["title"])
        deduped.append(s)

sections = deduped


junk = {"Contents", "ACC/AHA Joint Committee Members", "Presidents and Staff",
        "Article Information", "Affiliations", "References", "Sections",
        "Guide", "List of Illustrations", "1.6. Abbreviations"}
sections = [s for s in sections if s["title"] not in junk
            and not s["title"].startswith("Appendix")]
print("Found sections:", len(sections))




tokenizer = AutoTokenizer.from_pretrained("NeuML/pubmedbert-base-embeddings")
splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
    tokenizer, chunk_size=400, chunk_overlap=50)

docs = []

for i, section in enumerate(tqdm(sections)):

    pieces = splitter.split_text(section["content"])
    for j, piece in enumerate(pieces):
        docs.append(Document(
            page_content =f"{section['title']}\n{piece}",
            metadata={"title": section["title"], "section":i, "chunk":j},
            id=f"{i}-{j}"))

print(len(sections), "sections ->", len(docs), "chunks")

with open("guideline_recommendations.json", encoding="utf-8") as f:
    tables = json.load(f)

for image_name, table in tables.items():
    table_id = image_name.rsplit(".", 2)[-2]
    for rec in table["recommendations"]:
        docs.append(Document(
            page_content=(
                f"{table['section']}\n"
                f"Recommendation (COR {rec['cor']}, LOE {rec['loe']}): {rec['text']}"
            ),
            metadata={
                "title": table["section"],
                "kind": "recommendation",
                "cor": rec["cor"],
                "loe": rec["loe"],
            },
            id=f"rec-{table_id}-{rec['number']}",
        ))

print(len(docs), "documents including recommendations")

emb = HuggingFaceEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")
client = chromadb.PersistentClient(path="./chroma_db")

if "guidelines_v2" in [c.name for c in client.list_collections()]:
    client.delete_collection("guidelines_v2")


store = Chroma.from_documents(docs, emb,
                              ids = [d.id for d in docs],
                              client=client,
                              collection_name="guidelines_v2",
                              collection_metadata={"hnsw:space": "cosine"}
                              )