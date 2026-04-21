"""
HalluZero — FastAPI entry point
"""

import os
import traceback
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from config.settings import get_settings

settings = get_settings()


# ─────────────────────────────────────────────
# Lifespan (startup / shutdown)
# ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        # Ensure directories exist
        for path in ["./data/documents", "./data/chroma_db", "./data/uploads"]:
            os.makedirs(path, exist_ok=True)

        print("🚀 HalluZero API starting...")
        print(f"LLM Model: {settings.ollama_llm_model}")
        print(f"Embed Model: {settings.ollama_embed_model}")

    except Exception as e:
        print("❌ Startup error:")
        traceback.print_exc()

    yield

    print("🛑 HalluZero API shutting down")


# ─────────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────────
app = FastAPI(
    title="HalluZero API",
    description="Anti-hallucination LLM stack with RAG + dual verifier + RLHF",
    version="1.0.1",
    lifespan=lifespan,
)


# ─────────────────────────────────────────────
# CORS
# ─────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list or ["*"],  # fallback safe
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
app.include_router(router, prefix="/api")


# ─────────────────────────────────────────────
# Root endpoint
# ─────────────────────────────────────────────
@app.get("/")
async def root():
    return {
        "name": "HalluZero",
        "version": "1.0.1",
        "status": "running",
        "llm_model": settings.ollama_llm_model,
        "embed_model": settings.ollama_embed_model,
    }