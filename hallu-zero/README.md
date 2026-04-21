# HalluZero — Local Anti-Hallucination LLM Stack

Full-stack application combining RAG, dual-verifier, and RLHF feedback loop — 100% local via Ollama.

## Architecture

```
User → FastAPI Backend → Multi-stage RAG → Ollama LLM → Dual Verifier → Calibrated Response
                                                              ↑
                                              Feedback DB (SQLite) ← RLHF Loop
```

## Stack

| Layer | Technology |
|-------|-----------|
| LLM | Ollama (llama3, mistral, phi3) |
| Embeddings | nomic-embed-text (Ollama) |
| Vector DB | ChromaDB (local) |
| Backend | FastAPI + Python 3.11 |
| Frontend | React 18 + Vite + TailwindCSS |
| Feedback DB | SQLite via SQLAlchemy |

## Prerequisites

1. **Ollama** installed: https://ollama.com/download
2. **Python 3.11+**
3. **Node.js 18+**

## Quick Start

### 1. Pull required Ollama models
```bash
ollama pull llama3
ollama pull nomic-embed-text
```

### 2. Backend setup
```bash
cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --port 8000
```

### 3. Frontend setup
```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173

## Ingesting Documents

```bash
# Drop files into backend/data/documents/ then run:
cd backend
python -m app.rag.ingest --path ./data/documents
```

Supported: `.txt`, `.pdf`, `.md`, `.json`

## Features

- **Multi-stage RAG**: BM25 sparse + ChromaDB dense retrieval + reranking
- **Dual Verifier**: Claim-level NLI checker + factuality critic (both local via Ollama)
- **Uncertainty head**: Token-level confidence scoring
- **RLHF feedback loop**: Thumbs up/down → SQLite → periodic fine-tune prompts
- **Source attribution**: Every claim linked to its source document
- **Re-generation**: Auto-retry on low-confidence responses (max 2 rounds)
