"""
API routes:
  POST /api/chat           — main generation endpoint
  POST /api/feedback       — submit thumbs up/down
  POST /api/ingest         — upload & ingest documents
  GET  /api/stats          — feedback + system stats
  GET  /api/health         — Ollama health check
  GET  /api/training/pairs — export training pairs
  DELETE /api/knowledge    — wipe ChromaDB
"""
import uuid
import shutil
from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel, Field

from app.core.generation import get_generation_engine
from app.feedback.rlhf import get_feedback_db
from app.rag.pipeline import get_rag_pipeline
from app.core.ollama_client import get_ollama_client
from config.settings import get_settings

settings = get_settings()
router = APIRouter()


# ─────────────────────────────────────────────────────────────
# Request / Response models
# ─────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)
    session_id: Optional[str] = None
    mode: Optional[str] = "auto"   # "auto" | "rag" | "direct"


class ChatResponse(BaseModel):
    session_id: str
    query: str
    response: str
    sources: List[Dict[str, Any]]
    verification: Dict[str, Any]
    confidence: float
    model: str
    attempts: int
    mode: str = "direct"

    class Config:
        extra = "allow"   # 🔥 prevents crashes if new fields appear


class FeedbackRequest(BaseModel):
    session_id: str
    query: str
    response: str
    rating: int = Field(..., ge=-1, le=1)
    comment: Optional[str] = ""
    confidence_score: Optional[float] = 0.0
    verification_passed: Optional[bool] = False
    context_sources: Optional[List[str]] = []


# ─────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────

@router.get("/health")
async def health():
    ollama = get_ollama_client()
    try:
        status = await ollama.health_check()
        rag = get_rag_pipeline()
        rag_status = rag.status()

        return {
            "status": "ok",
            "ollama": status,
            "rag": rag_status,
            "config": {
                "llm_model": settings.ollama_llm_model,
                "embed_model": settings.ollama_embed_model,
                "verifier_model": settings.ollama_verifier_model,
                "confidence_threshold": settings.confidence_threshold,
            },
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=503, detail=f"Ollama unavailable: {str(e)}")


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    engine = get_generation_engine()

    try:
        session_id = req.session_id or str(uuid.uuid4())

        result = await engine.generate(
            query=req.query,
            session_id=session_id,
            mode=req.mode or "auto",
        )

        data = result.to_dict()

        # 🔍 Debug log (very useful)
        print("CHAT RESULT:", data)

        # 🛡 Ensure required fields exist
        data.setdefault("session_id", session_id)
        data.setdefault("query", req.query)
        data.setdefault("sources", [])
        data.setdefault("verification", {})
        data.setdefault("confidence", 0.0)
        data.setdefault("model", settings.ollama_llm_model)
        data.setdefault("attempts", 1)
        data.setdefault("mode", req.mode or "auto")

        return ChatResponse(**data)

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/feedback")
async def submit_feedback(req: FeedbackRequest):
    db = get_feedback_db()

    try:
        db.save_feedback(
            feedback_id=str(uuid.uuid4()),
            query=req.query,
            response=req.response,
            rating=req.rating,
            confidence_score=req.confidence_score or 0.0,
            verification_passed=req.verification_passed or False,
            context_sources=req.context_sources or [],
            comment=req.comment or "",
            model_used=settings.ollama_llm_model,
        )

        stats = db.get_stats()

        # 🔁 Auto-generate training pairs every 10 feedbacks
        if stats["total_feedback"] % 10 == 0:
            db.generate_training_pairs()

        return {"status": "ok", "message": "Feedback recorded"}

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_stats():
    db = get_feedback_db()
    rag = get_rag_pipeline()

    return {
        "feedback": db.get_stats(),
        "recent_feedback": db.get_recent_feedback(10),
        "rag": rag.status(),
    }


@router.post("/ingest")
async def ingest_files(files: List[UploadFile] = File(...)):
    from app.rag.ingest import load_file

    upload_dir = Path("./data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)

    rag = get_rag_pipeline()
    all_docs = []
    saved_files = []

    for upload in files:
        ext = Path(upload.filename).suffix.lower()

        if ext not in {".txt", ".pdf", ".md", ".json"}:
            continue

        dest = upload_dir / upload.filename

        with dest.open("wb") as f:
            shutil.copyfileobj(upload.file, f)

        saved_files.append(str(dest))

        docs = load_file(dest)
        all_docs.extend(docs)

    if not all_docs:
        raise HTTPException(status_code=400, detail="No processable documents found")

    result = await rag.ingest(all_docs)

    return {
        "status": "ok",
        "files_processed": len(saved_files),
        "chunks_ingested": result["ingested"],
        "total_in_store": result["total"],
    }


@router.get("/training/pairs")
async def get_training_pairs():
    db = get_feedback_db()
    pairs = db.export_training_data()

    return {
        "count": len(pairs),
        "pairs": pairs,
        "format": "DPO (chosen/rejected)",
    }


@router.delete("/knowledge")
async def clear_knowledge_base():
    """Wipe all documents from ChromaDB and reset BM25."""
    try:
        rag = get_rag_pipeline()

        rag.vector_store.client.delete_collection("hallu_zero_docs")

        rag.vector_store.collection = rag.vector_store.client.get_or_create_collection(
            name="hallu_zero_docs",
            metadata={"hnsw:space": "cosine"},
        )

        rag.bm25._corpus = []
        rag.bm25._metadata = []
        rag.bm25._bm25 = None

        return {"status": "ok", "message": "Knowledge base cleared"}

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))