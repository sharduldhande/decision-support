# Coronary Decision Support System

> *This is a demo and should not be used for real world  decision making.*

An AI-powered clinical decision support tool for evidence-based PCI vs. CABG recommendations, grounded in the **2021 ACC/AHA/SCAI Coronary Revascularization Guidelines**.

---

## Live Demo

**Project page:** [decisionsupport.sharduldhande.com](https://decisionsupport.sharduldhande.com)

Upload any ICA or CCTA report (PDF) to see a guideline-grounded recommendation in seconds.

---

## Overview

Choosing between percutaneous coronary intervention (PCI) and coronary artery bypass grafting (CABG) for a patient with multi-vessel coronary artery disease is one of the most consequential decisions in cardiology. The optimal choice depends on dozens of interacting variables — coronary anatomy, lesion complexity, comorbidities, LV function — and must align with class-of-evidence recommendations from clinical guidelines.

This tool automates that reasoning. A cardiologist uploads a cath lab or CCTA report, and the system:

1. **Extracts** structured clinical findings from the PDF (scans included) with Claude, into a validated Pydantic schema
2. **Retrieves** the most relevant ACC/AHA guideline text and graded recommendations via semantic search
3. **Generates** a recommendation (PCI, CABG, or medical management) with guideline citations, confidence level, and a breakdown of factors favoring each approach

Recommendations are grounded in the actual guideline text — not general medical knowledge — which reduces the risk of hallucination and ensures every recommendation is traceable to a specific Class/Level-of-Evidence statement.

---

## How It Works

```
Input PDF ICA/CCTA report
    ↓
Extract   Claude + Pydantic schema → findings; lesion severity computed in code
    ↓
Scope     STEMI, posterior MI, stable and non-coronary reports are declined
    ↓
Retrieve  PubMedBERT + ChromaDB → guideline chunks and graded recommendations
    ↓
Decide    Claude + Pydantic schema → strategy, confidence, factors, COR/LOE citations
    ↓
Stream    FastAPI streams each stage, then the result, to the web page
```


### Guideline Retrieval

For each case, several queries are issued to ChromaDB and the results are merged and de-duplicated:
1. A summary of the lesions and clinical context
2. A disease-specific query (e.g., *"NSTEMI revascularization 2-vessel disease"*)
3. A generic decision query (*"PCI versus CABG multivessel coronary artery disease"*)
4. A physiology query (FFR/iFR) when intermediate lesions are present

The guideline's recommendation tables are images in the source EPUB, so `transcribe.py` reads them once with Claude vision into `guideline_recommendations.json` (135 recommendations with Class of Recommendation and Level of Evidence), which `ingest.py` indexes alongside the guideline text.

---

## Tech Stack

| Component | Technology |
|---|---|
| Front end | Static web page, Cloudflare Turnstile |
| API | FastAPI (streaming NDJSON) |
| LLM | Claude Haiku 5.5 (extraction and recommendation), Claude Sonnet 5.5 (one-time table transcription) |
| Orchestration | LangChain (structured output, retrievers, LCEL), LangSmith tracing |
| Embeddings | PubMedBERT (`NeuML/pubmedbert-base-embeddings`) |
| Vector Database | ChromaDB |
| Guideline Source | 2021 ACC/AHA/SCAI Coronary Revascularization Guidelines |
| Runtime | Python 3.12+ |

---

## Key Technical Features

- **RAG (Retrieval-Augmented Generation):** Guideline sections are retrieved at runtime via vector similarity search, not hard-coded rules. The system can surface any relevant section across the full guideline document.
- **Domain-specific embeddings:** PubMedBERT, trained on biomedical literature, produces significantly more accurate embeddings for clinical text than general-purpose models.
- **Structured extraction with schema validation:** Pydantic classes define the 60+ field findings schema; the model fills what the report states, and code derives lesion severity (significant ≥70%, ≥50% left main; intermediate 40–69%) and the significant-vessel count from the guideline's thresholds.
- **Citation transparency:** Every recommendation includes the specific guideline section title, Class of Recommendation, Level of Evidence, and the verbatim guideline text that supports it.

---

## Getting Started

### Prerequisites

- Python 3.10+
- An [Anthropic API](https://platform.claude.com) key
- A [Cloudflare Turnstile](https://developers.cloudflare.com/turnstile/) widget (for the public API)

### Installation

```bash
# Clone the repository
git clone https://github.com/sharduldhande/decision-support.git
cd decision-support

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Create a `.env` file (git-ignored):

```bash
ANTHROPIC_API_KEY=...
TURNSTILE_SECRET=...
# optional LangSmith tracing
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=...
LANGSMITH_PROJECT=decision-support-dev
```

### First-Time Setup

Ingest the ACC/AHA guidelines into the vector database (run once):

```bash
python ingest.py
```

This parses the guideline EPUB, splits it into ≤400-token chunks, adds the transcribed recommendations, embeds everything with PubMedBERT and stores it in ChromaDB.

### Run the App

```bash
uvicorn app:app --port 8501
```

`POST /analyze` takes a PDF and a Turnstile token and streams progress stages followed by the result.

---

## Project Structure

```
├── app.py                    # FastAPI: /analyze (Turnstile, limits, streaming)
├── pipeline.py               # LangChain pipeline: extract → retrieve → decide
├── schemas.py                # Pydantic schemas for findings and decisions
├── ingest.py                 # Guideline ingestion script
├── transcribe.py             # One-time transcription of recommendation tables
├── guideline_recommendations.json  # Transcribed recommendations (COR/LOE)
├── requirements.txt          # Python dependencies
├── ACC_AHA_Guidelines.epub   # Source guideline document
└── chroma_db/                # Persistent vector database (created by ingest.py)
```

---

## Clinical Background

**NSTEMI** (Non-ST-Elevation Myocardial Infarction) is a type of heart attack where one or more coronary arteries are partially blocked. Many NSTEMI patients have multi-vessel coronary artery disease (CAD), requiring a decision about revascularization strategy:

- **PCI (Percutaneous Coronary Intervention):** A catheter-based procedure that opens blocked arteries using balloons and stents. Less invasive, shorter recovery.
- **CABG (Coronary Artery Bypass Grafting):** Open-heart surgery that reroutes blood flow around blocked vessels using grafts. Generally preferred for complex multi-vessel disease or left main stenosis.

The 2021 ACC/AHA/SCAI guidelines provide class-of-recommendation and level-of-evidence ratings for specific anatomical and clinical scenarios. Applying these guidelines correctly requires integrating many variables simultaneously — a task well-suited to AI-assisted decision support.

---
