Markdown
# 🍏 Apple 10-K Financial Production RAG Pipeline

A production-grade, containerized Retrieval-Augmented Generation (RAG) system engineered to parse, index, and query complex financial SEC 10-K filings using Hybrid Search (BM25 + Vector Embeddings), Cross-Encoder Reranking, FastAPI, and Google Gemini.

---

## 🏗️ Architecture Overview

[ PDF Document (Apple 10-K) ]
│
▼
[ PyMuPDF Chunking ]
│
┌───────┴───────┐
▼               ▼
[ BM25 Sparse ]  [ Vector Dense (ChromaDB) ]
│               │
└───────┬───────┘
▼
[ Cross-Encoder Reranker ] ── (Top-K Chunks)
│
▼
[ Gemini LLM Synthesis ]
│
▼
[ FastAPI Backend / UI ]


---

## 🚀 Key Features

* **Hybrid Search Strategy:** Combines BM25 lexical keyword search with dense ChromaDB embeddings (`all-MiniLM-L6-v2`) for accurate financial context retrieval.
* **Cross-Encoder Reranking:** Two-stage retrieval pipeline using cross-encoders to ensure high precision in top-3 chunks.
* **FastAPI Microservice:** Asynchronous API with interactive OpenAPI Swagger documentation at `/docs`.
* **Streamlit Dashboard:** Analyst interface with page citations and structured financial breakdowns.
* **Docker Containerized:** Production-ready container environment for isolated and reliable deployment.

---

## 📊 Retrieval Evaluation Benchmark

Evaluated using a 10-query ground truth golden dataset across key financial tables (Net Sales, Operating Income, R&D Expenses, Segment Revenue).

| Metric | Result |
| :--- | :--- |
| **Total Test Queries** | `10` |
| **Top-K Retrieved Chunks** | `3` |
| **Retrieval Recall@3 (Hit Rate)** | **`80.00%`** |
| **Reranker Pipeline** | `Cross-Encoder` |
| **Primary LLM Engine** | `Google Gemini 3.5 Flash Lite` |

---

## 🛠️ Quickstart Guide

### 1. Local Setup

```bash
git clone [https://github.com/](https://github.com/)<your-username>/sec-10k-rag.git
cd sec-10k-rag

python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt