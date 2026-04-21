from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache


class Settings(BaseSettings):
    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_llm_model: str = "llama3"
    ollama_embed_model: str = "nomic-embed-text"
    ollama_verifier_model: str = "mistral"

    # RAG
    chroma_db_path: str = "./data/chroma_db"
    documents_path: str = "./data/documents"
    top_k_retrieval: int = 5
    rerank_top_k: int = 3

    # Verifier
    confidence_threshold: float = 0.65
    max_regeneration_attempts: int = 2

    # Feedback
    feedback_db_path: str = "./data/feedback.db"

    # API
    cors_origins: str = "http://localhost:5173,http://localhost:3000"
    api_secret_key: str = "change-me"

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",")]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
