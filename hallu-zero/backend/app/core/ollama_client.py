"""
Thin async wrapper around the Ollama HTTP API.
Handles: chat completions, embeddings, streaming, model health check.
"""

import httpx
import json
from typing import AsyncIterator, List
from config.settings import get_settings

settings = get_settings()


class OllamaClient:
    def __init__(self):
        self.base_url = settings.ollama_base_url
        self.timeout = httpx.Timeout(120.0, connect=10.0)

    # ─────────────────────────────────────────────
    # Health Check
    # ─────────────────────────────────────────────
    async def health_check(self) -> dict:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            r = await client.get(f"{self.base_url}/api/tags")
            r.raise_for_status()
            data = r.json()
            models = [m["name"] for m in data.get("models", [])]
            return {"status": "ok", "models": models}

    # ─────────────────────────────────────────────
    # Chat (non-streaming)
    # ─────────────────────────────────────────────
    async def chat(
        self,
        model: str,
        messages: List[dict],
        temperature: float = 0.3,
        stream: bool = False,
    ) -> str:
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "options": {"temperature": temperature},
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            r = await client.post(f"{self.base_url}/api/chat", json=payload)
            r.raise_for_status()

            data = r.json()
            return data.get("message", {}).get("content", "")

    # ─────────────────────────────────────────────
    # Chat Streaming
    # ─────────────────────────────────────────────
    async def chat_stream(
        self,
        model: str,
        messages: List[dict],
        temperature: float = 0.3,
    ) -> AsyncIterator[str]:

        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": {"temperature": temperature},
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST", f"{self.base_url}/api/chat", json=payload
            ) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if not line:
                        continue

                    try:
                        data = json.loads(line)
                        if not data.get("done"):
                            yield data.get("message", {}).get("content", "")
                    except Exception:
                        continue

    # ─────────────────────────────────────────────
    # Embedding (AUTO VERSION SAFE ✅)
    # ─────────────────────────────────────────────
    async def embed(self, text: str) -> List[float]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:

            # 🔥 Try NEW API first (/api/embed)
            try:
                r = await client.post(
                    f"{self.base_url}/api/embed",
                    json={
                        "model": settings.ollama_embed_model,
                        "input": text
                    }
                )
                r.raise_for_status()
                return r.json()["embeddings"][0]

            except httpx.HTTPStatusError:
                # 🔁 Fallback to OLD API (/api/embeddings)
                r = await client.post(
                    f"{self.base_url}/api/embeddings",
                    json={
                        "model": settings.ollama_embed_model,
                        "prompt": text
                    }
                )
                r.raise_for_status()
                return r.json()["embedding"]

    # ─────────────────────────────────────────────
    # Batch Embeddings (AUTO SAFE ✅)
    # ─────────────────────────────────────────────
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:

            # 🔥 Try NEW API (fast batch)
            try:
                r = await client.post(
                    f"{self.base_url}/api/embed",
                    json={
                        "model": settings.ollama_embed_model,
                        "input": texts
                    }
                )
                r.raise_for_status()
                return r.json()["embeddings"]

            except httpx.HTTPStatusError:
                # 🔁 Fallback to OLD API (loop)
                embeddings = []
                for t in texts:
                    r = await client.post(
                        f"{self.base_url}/api/embeddings",
                        json={
                            "model": settings.ollama_embed_model,
                            "prompt": t
                        }
                    )
                    r.raise_for_status()
                    embeddings.append(r.json()["embedding"])

                return embeddings


# ─────────────────────────────────────────────
# Singleton
# ─────────────────────────────────────────────
_client: OllamaClient | None = None


def get_ollama_client() -> OllamaClient:
    global _client
    if _client is None:
        _client = OllamaClient()
    return _client