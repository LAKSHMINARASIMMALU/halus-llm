"""
Multi-stage RAG pipeline:
  Stage 1: BM25 sparse retrieval (high recall)
  Stage 2: ChromaDB dense retrieval (semantic)
  Stage 3: Score-based reranking (fast, no extra Ollama call)

BM25 is restored from ChromaDB on every startup so data persists across restarts.
"""
import os
import json
import hashlib
from pathlib import Path
from typing import Optional

import chromadb
from rank_bm25 import BM25Okapi
from chromadb.config import Settings as ChromaSettings

from app.core.ollama_client import get_ollama_client
from config.settings import get_settings

settings = get_settings()


# --- Document model -----------------------------------------------------------

class Document:
    def __init__(self, content: str, metadata: dict):
        self.content = content
        self.metadata = metadata
        self.doc_id = hashlib.md5(content.encode()).hexdigest()[:12]


class RetrievedChunk:
    def __init__(self, content: str, metadata: dict, score: float, source: str):
        self.content = content
        self.metadata = metadata
        self.score = score
        self.source = source

    def to_dict(self):
        return {
            "content": self.content,
            "source": self.source,
            "score": round(self.score, 4),
            "metadata": self.metadata,
        }


# --- ChromaDB vector store ----------------------------------------------------

class VectorStore:
    def __init__(self):
        os.makedirs(settings.chroma_db_path, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=settings.chroma_db_path,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name="hallu_zero_docs",
            metadata={"hnsw:space": "cosine"},
        )
        self.ollama = get_ollama_client()

    async def add_documents(self, documents: list[Document]) -> int:
        if not documents:
            return 0
        texts = [d.content for d in documents]
        embeddings = await self.ollama.embed_batch(texts)
        self.collection.add(
            ids=[d.doc_id for d in documents],
            embeddings=embeddings,
            documents=texts,
            metadatas=[d.metadata for d in documents],
        )
        return len(documents)

    async def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        q_embed = await self.ollama.embed(query)
        results = self.collection.query(
            query_embeddings=[q_embed],
            n_results=min(top_k, self.collection.count() or 1),
            include=["documents", "metadatas", "distances"],
        )
        chunks = []
        if results["documents"] and results["documents"][0]:
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                score = 1.0 - dist
                chunks.append(RetrievedChunk(
                    content=doc,
                    metadata=meta,
                    score=score,
                    source=meta.get("source", "unknown"),
                ))
        return chunks

    def get_all(self) -> tuple[list[str], list[dict]]:
        """Return all (documents, metadatas) stored in ChromaDB."""
        count = self.collection.count()
        if count == 0:
            return [], []
        result = self.collection.get(
            include=["documents", "metadatas"],
            limit=count,
        )
        return result.get("documents", []), result.get("metadatas", [])

    def get_existing_ids(self) -> set[str]:
        result = self.collection.get(include=[])
        return set(result.get("ids", []))

    def count(self) -> int:
        return self.collection.count()


# --- BM25 sparse retriever ----------------------------------------------------

class BM25Retriever:
    def __init__(self):
        self._corpus: list[str] = []
        self._metadata: list[dict] = []
        self._bm25: Optional[BM25Okapi] = None

    def index(self, documents: list[Document]):
        self._corpus = [d.content for d in documents]
        self._metadata = [d.metadata for d in documents]
        tokenized = [doc.lower().split() for doc in self._corpus]
        self._bm25 = BM25Okapi(tokenized)

    def index_raw(self, texts: list[str], metadatas: list[dict]):
        """Index from raw text+metadata lists (used when restoring from ChromaDB)."""
        self._corpus = texts
        self._metadata = metadatas
        tokenized = [t.lower().split() for t in texts]
        self._bm25 = BM25Okapi(tokenized)

    def search(self, query: str, top_k: int = 10) -> list[RetrievedChunk]:
        if not self._bm25 or not self._corpus:
            return []
        scores = self._bm25.get_scores(query.lower().split())
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        results = []
        for i in top_indices:
            if scores[i] > 0:
                meta = self._metadata[i]
                results.append(RetrievedChunk(
                    content=self._corpus[i],
                    metadata=meta,
                    score=float(scores[i]),
                    source=meta.get("source", "unknown"),
                ))
        return results


# --- Reranker -----------------------------------------------------------------

class OllamaReranker:
    async def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int = 3) -> list[RetrievedChunk]:
        # Score-based sort only — no extra Ollama call needed
        return sorted(chunks, key=lambda c: c.score, reverse=True)[:top_k]


# --- Main RAG pipeline --------------------------------------------------------

class RAGPipeline:
    def __init__(self):
        self.vector_store = VectorStore()
        self.bm25 = BM25Retriever()
        self.reranker = OllamaReranker()
        # Restore BM25 from ChromaDB on startup — survives restarts
        self._restore_bm25()

    def _restore_bm25(self):
        """Load all docs from ChromaDB into BM25 index on startup."""
        try:
            texts, metadatas = self.vector_store.get_all()
            if texts:
                self.bm25.index_raw(texts, metadatas)
                print(f"[RAG] Restored {len(texts)} docs into BM25 from ChromaDB")
            else:
                print("[RAG] No existing docs found in ChromaDB")
        except Exception as e:
            print(f"[RAG] BM25 restore warning (non-fatal): {e}")

    async def ingest(self, documents: list[Document]) -> dict:
        # Skip duplicates already in ChromaDB
        existing_ids = self.vector_store.get_existing_ids()
        new_docs = [d for d in documents if d.doc_id not in existing_ids]

        if new_docs:
            await self.vector_store.add_documents(new_docs)

        # Rebuild BM25 from full ChromaDB (old + new)
        self._restore_bm25()
        total = self.vector_store.count()
        return {"ingested": len(new_docs), "total": total}

    async def retrieve(self, query: str) -> list[RetrievedChunk]:
        top_k = settings.top_k_retrieval
        rerank_k = settings.rerank_top_k

        bm25_results = self.bm25.search(query, top_k=top_k)
        dense_results = await self.vector_store.search(query, top_k=top_k)

        # Merge — prefer higher score per unique chunk
        seen: dict[str, RetrievedChunk] = {}
        for chunk in bm25_results + dense_results:
            key = chunk.content[:50]
            if key not in seen or chunk.score > seen[key].score:
                seen[key] = chunk
        merged = list(seen.values())

        return await self.reranker.rerank(query, merged, top_k=rerank_k)

    def status(self) -> dict:
        return {
            "vector_store_docs": self.vector_store.count(),
            "bm25_docs": len(self.bm25._corpus),
        }


# --- Singleton ----------------------------------------------------------------
_pipeline: RAGPipeline | None = None


def get_rag_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline