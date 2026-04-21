"""
Core generation engine — optimized for large documents (40+ pages).
- Higher retrieval top_k to find specific topics in big files
- Context limited to 4000 chars to avoid Ollama overflow
- Includes page numbers in source attribution
"""
import uuid, json, re
from dataclasses import dataclass
from typing import AsyncIterator, Literal

from app.core.ollama_client import get_ollama_client
from app.rag.pipeline import get_rag_pipeline, RetrievedChunk
from config.settings import get_settings

settings = get_settings()

MAX_CONTEXT_CHARS = 4000   # safe limit for llama3 8B
MAX_CHUNKS        = 6      # max chunks to include in context

RAG_SYSTEM_PROMPT = """You are a precise document assistant.
Answer ONLY from the provided context passages below.
Be specific — if a topic is mentioned, explain it fully from the context.
Always mention the page number if available.
If the topic is NOT in the context, say: "I couldn't find information about that in the document. Try rephrasing your question."

ALWAYS end your response with:
<verification>
{"confidence": 0.85, "passed": true, "claims": [{"claim": "brief claim", "verdict": "supported"}], "feedback": "summary"}
</verification>"""

DIRECT_SYSTEM_PROMPT = """You are a helpful, precise AI assistant.
Answer from your training knowledge. Be specific and accurate.
Say "I'm not sure" if uncertain.

ALWAYS end your response with:
<verification>
{"confidence": 0.80, "passed": true, "claims": [{"claim": "brief claim", "verdict": "supported"}], "feedback": "summary"}
</verification>"""


@dataclass
class GenerationResult:
    query: str
    response: str
    sources: list[dict]
    verification: dict
    confidence: float
    model: str
    attempts: int
    mode: str = "direct"
    session_id: str = ""

    def to_dict(self):
        return {
            "session_id": self.session_id,
            "query": self.query,
            "response": self.response,
            "sources": self.sources,
            "verification": self.verification,
            "confidence": round(self.confidence, 3),
            "model": self.model,
            "attempts": self.attempts,
            "mode": self.mode,
        }


def _build_context(chunks: list[RetrievedChunk]) -> str:
    """Build context string with page numbers, capped at MAX_CONTEXT_CHARS."""
    parts = []
    total = 0
    for i, c in enumerate(chunks[:MAX_CHUNKS]):
        page = c.metadata.get("page", "")
        page_str = f" | Page {page}" if page else ""
        header = f"[Source {i+1}: {c.source}{page_str}]"
        content = c.content[:600]  # max 600 chars per chunk
        part = f"{header}\n{content}"
        if total + len(part) > MAX_CONTEXT_CHARS:
            break
        parts.append(part)
        total += len(part)
    return "\n\n".join(parts)


def _parse_verification(raw: str) -> tuple[str, dict]:
    default = {
        "passed": True, "overall_confidence": 0.75, "critic_score": 0.75,
        "critic_feedback": "Answer generated.", "claim_verdicts": [],
        "needs_regeneration": False, "regeneration_hint": "",
    }
    match = re.search(r'<verification>(.*?)</verification>', raw, re.DOTALL)
    if not match:
        jm = re.search(r'\{\s*"confidence".*?\}', raw, re.DOTALL)
        if jm:
            try:
                return raw[:jm.start()].strip(), _norm(json.loads(jm.group()))
            except: pass
        return raw.strip(), default
    try:
        return raw[:match.start()].strip(), _norm(json.loads(match.group(1).strip()))
    except:
        return raw[:match.start()].strip(), default


def _norm(data: dict) -> dict:
    c = float(data.get("confidence", 0.75))
    passed = bool(data.get("passed", c >= settings.confidence_threshold))
    return {
        "passed": passed, "overall_confidence": c, "critic_score": c,
        "critic_feedback": data.get("feedback", ""),
        "claim_verdicts": [
            {"claim": x.get("claim", ""), "verdict": x.get("verdict", "unverifiable"),
             "confidence": c, "evidence": ""}
            for x in data.get("claims", [])
        ],
        "needs_regeneration": not passed,
        "regeneration_hint": "" if passed else data.get("feedback", ""),
    }


class GenerationEngine:
    def __init__(self):
        self.ollama = get_ollama_client()
        self.rag = get_rag_pipeline()

    def _rag_messages(self, query: str, chunks: list[RetrievedChunk]) -> list[dict]:
        ctx = _build_context(chunks)
        return [
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {"role": "user", "content": f"CONTEXT:\n{ctx}\n\nQUESTION: {query}"},
        ]

    def _direct_messages(self, query: str) -> list[dict]:
        return [
            {"role": "system", "content": DIRECT_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]

    async def generate(
        self, query: str, session_id: str = "",
        mode: Literal["auto", "rag", "direct"] = "auto"
    ) -> GenerationResult:
        if not session_id:
            session_id = str(uuid.uuid4())

        chunks = []

        if mode == "direct":
            actual_mode = "direct"
            messages = self._direct_messages(query)

        elif mode == "rag":
            chunks = await self.rag.retrieve(query)
            if not chunks:
                return GenerationResult(
                    query=query,
                    response="No documents found. Please go to **Ingest** and upload your document first, or switch to **Direct** mode.",
                    sources=[], verification={
                        "passed": False, "overall_confidence": 0.0, "critic_score": 0.0,
                        "critic_feedback": "No documents.", "claim_verdicts": [],
                        "needs_regeneration": False, "regeneration_hint": "",
                    },
                    confidence=0.0, model=settings.ollama_llm_model,
                    attempts=1, mode="rag", session_id=session_id,
                )
            actual_mode = "rag"
            messages = self._rag_messages(query, chunks)

        else:  # auto
            chunks = await self.rag.retrieve(query)
            if chunks:
                actual_mode = "rag"
                messages = self._rag_messages(query, chunks)
            else:
                actual_mode = "direct"
                messages = self._direct_messages(query)

        raw = await self.ollama.chat(
            model=settings.ollama_llm_model,
            messages=messages,
            temperature=0.2,
        )
        clean, verification = _parse_verification(raw)

        return GenerationResult(
            query=query, response=clean,
            sources=[c.to_dict() for c in chunks],
            verification=verification,
            confidence=verification["overall_confidence"],
            model=settings.ollama_llm_model,
            attempts=1, mode=actual_mode,
            session_id=session_id,
        )

    async def generate_stream(self, query: str, mode: str = "auto") -> AsyncIterator[str]:
        chunks = []
        if mode != "direct":
            chunks = await self.rag.retrieve(query)
        if chunks:
            messages = self._rag_messages(query, chunks)
        else:
            messages = self._direct_messages(query)
        async for token in self.ollama.chat_stream(
            model=settings.ollama_llm_model, messages=messages,
        ):
            yield token


_engine: GenerationEngine | None = None


def get_generation_engine() -> GenerationEngine:
    global _engine
    if _engine is None:
        _engine = GenerationEngine()
    return _engine