# Coronary Decision Support System

> *This is a demo and should not be used for real world  decision making.*

An AI-powered clinical decision support tool for evidence-based PCI vs. CABG recommendations, grounded in the **2021 ACC/AHA/SCAI Coronary Revascularization Guidelines**.

---

## Live Demo

**URL:** [decisionsupport.sharduldhande.com](https://decisionsupport.sharduldhande.com)

| Field | Value |
|---|---|
| Username | `admin` |
| Password | `gatech` |

Upload any ICA or CCTA report (PDF) to see a guideline-grounded recommendation in seconds.

---

## Overview

Choosing between percutaneous coronary intervention (PCI) and coronary artery bypass grafting (CABG) for a patient with multi-vessel coronary artery disease is one of the most consequential decisions in cardiology. The optimal choice depends on dozens of interacting variables — coronary anatomy, lesion complexity, comorbidities, LV function — and must align with class-of-evidence recommendations from clinical guidelines.

This tool automates that reasoning. A cardiologist uploads a cath lab or CCTA report, and the system:

1. **Extracts** structured clinical findings from the PDF using a large language model
2. **Retrieves** the most relevant ACC/AHA guideline sections via semantic search
3. **Generates** a recommendation (PCI, CABG, or medical management) with guideline citations, confidence level, and a breakdown of factors favoring each approach

Recommendations are grounded in the actual guideline text — not general medical knowledge — which reduces the risk of hallucination and ensures every recommendation is traceable to a specific Class/Level-of-Evidence statement.

---

## How It Works

```
Input PDF ICA/CCTA Report
    ↓
Extract structured JSON (anatomy, stenosis, comorbidities)
    ↓
PubMedBERT + ChromaDB → Retrieve relevant guideline chunks
    ↓
Prepare Recommendation based on Extract and JSON (strategy, confidence, PCI vs CABG)
    ↓
Formatted Output with Citations
```


### Three-Pronged Guideline Retrieval

For each case, three distinct queries are issued to ChromaDB and the results are merged:
1. A summary of the clinical findings
2. A disease-specific query (e.g., *"NSTEMI revascularization 2-vessel disease"*)
3. A generic decision query (e.g., *"PCI versus CABG multivessel coronary artery disease"*)

This ensures both patient-specific and general guideline sections are surfaced.

---

## Tech Stack

| Component | Technology |
|---|---|
| UI | Streamlit |
| Extraction LLM | Google Gemini 2.5 Flash |
| Recommendation LLM | Google Gemini 2.5 Flash |
| Embeddings | PubMedBERT (`NeuML/pubmedbert-base-embeddings`) |
| Vector Database | ChromaDB |
| Guideline Source | 2021 ACC/AHA/SCAI Coronary Revascularization Guidelines |
| Runtime | Python 3.13 |

---

## Key Technical Features

- **RAG (Retrieval-Augmented Generation):** Guideline sections are retrieved at runtime via vector similarity search, not hard-coded rules. The system can surface any relevant section across the full guideline document.
- **Domain-specific embeddings:** PubMedBERT, trained on biomedical literature, produces significantly more accurate embeddings for clinical text than general-purpose models.
- **Structured extraction with schema validation:** The extraction prompt defines a 60+ field JSON schema with explicit clinical rules (e.g., how to interpret qualitative stenosis descriptors like "mild" or "severe", how to determine vessel dominance, what counts as a "significant" lesion).
- **Citation transparency:** Every recommendation includes the specific guideline section title, Class of Recommendation, Level of Evidence, and the verbatim guideline text that supports it.

---

## Getting Started

### Prerequisites

- Python 3.10+
- A [Google AI Studio](https://aistudio.google.com) API key (free tier works)

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

```bash
export GEMINI_API_KEY=your_api_key_here
```

### First-Time Setup

Ingest the ACC/AHA guidelines into the vector database (run once):

```bash
python ingest.py
```

This parses the guideline EPUB, splits it into sections, embeds each section with PubMedBERT, and stores them in ChromaDB.

### Run the App

```bash
streamlit run app.py
```

---

## Project Structure

```
├── app.py                    # Streamlit app + full AI pipeline
├── ingest.py                 # One-time guideline ingestion script
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
