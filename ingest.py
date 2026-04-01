from ebooklib import epub
import bs4
from sentence_transformers import SentenceTransformer
import chromadb
from tqdm import tqdm


ebook = epub.read_epub("ACC_AHA_Guidelines.epub")

items = ebook.get_items_of_type(9)

sections = []

for item in items:
    soup = bs4.BeautifulSoup(item.content, "html.parser")
    headings = soup.find_all(["h1","h2", "h3", "h4"])
    for heading in headings:
        title = heading.get_text()
        content_parts = []

        sibling = heading.find_next_sibling()
        while sibling and sibling.name not in ["h1","h2", "h3", "h4"]:
            text = sibling.get_text()
            if text:
                content_parts.append(text)
            sibling = sibling.find_next_sibling()

        content = "\n".join(content_parts)
        if len(content) > 50:
            sections.append({
                "title": title,
                "content": content,
            })

seen = set()
deduped = []

for s in sections:
    if s["title"] not in seen:
        seen.add(s["title"])
        deduped.append(s)

sections = deduped

parent_title = ""
for s in sections:
    if s["title"] in ["Synopsis", "Recommendation-Specific Supportive Text",
                       "Recommendation-Specific Supporting Text",
                       "Recommendation-Specific Supporting Tex"]:
        s["title"] = parent_title + " — " + s["title"]
    else:
        parent_title = s["title"]

junk = {"Contents", "ACC/AHA Joint Committee Members", "Presidents and Staff",
        "Article Information", "Affiliations", "References", "Sections",
        "Guide", "List of Illustrations", "1.6. Abbreviations"}
sections = [s for s in sections if s["title"] not in junk
            and not s["title"].startswith("Appendix")]
print("Found sections:", len(sections))

x = 0
for section in sections:
    print(x)
    x+=1
    print("Title:", section["title"])
    print("Content:", len(section["content"]))

embed_model = SentenceTransformer("NeuML/pubmedbert-base-embeddings")

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("guidelines")



for i, section in enumerate(tqdm(sections, desc="Embedding")):
    embedding = embed_model.encode(section["content"]).tolist()
    collection.add(
        ids=[str(i)],
        documents=[section["content"]],
        metadatas=[{"title": section["title"]}],
        embeddings=[embedding]
    )
    # print(f"Embedded {i}/{len(sections)}: {section['title'][:60]}")

print("Done. Stored:", collection.count(), "sections")


