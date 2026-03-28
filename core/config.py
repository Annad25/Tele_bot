import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root (rag_telegram_bot/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Config:
    # --- Telegram ---
    telegram_bot_token: str = field(
        default_factory=lambda: os.environ["TELEGRAM_BOT_TOKEN"]
    )

    # --- OpenAI ---
    openai_api_key: str = field(
        default_factory=lambda: os.environ["OPENAI_API_KEY"]
    )
    openai_model: str = "gpt-5-mini"
    intent_model: str = "gpt-4o-mini"

    # --- Embedding ---
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384

    # --- Qdrant ---
    qdrant_path: str = field(
        default_factory=lambda: str(_PROJECT_ROOT / "qdrant_data")
    )
    qdrant_collection: str = "rag_knowledge"

    # --- RAG ---
    chunk_size: int = 1500
    chunk_overlap: int = 100
    top_k: int = 3

    # --- Memory ---
    max_history: int = 3

    # --- Paths ---
    default_docs_dir: str = field(
        default_factory=lambda: str(_PROJECT_ROOT / "data" / "default_docs")
    )

    # --- Upload limits ---
    max_upload_size_bytes: int = 1_048_576  # 1 MB
