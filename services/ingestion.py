"""Handles reading, chunking, embedding and upserting documents."""

import logging
from pathlib import Path

from services.embedding import EmbeddingService
from services.qdrant_service import QdrantService
from utils.chunker import chunk_text

logger = logging.getLogger(__name__)


async def ingest_directory(
    docs_dir: str,
    embedding_service: EmbeddingService,
    qdrant_service: QdrantService,
    chunk_size: int = 300,
    chunk_overlap: int = 50,
) -> int:
    """Read every .md / .txt file in *docs_dir*, chunk, embed, and upsert.

    All chunks are tagged with ``user_id="global"``.
    Returns the total number of chunks ingested.
    """
    docs_path = Path(docs_dir)
    files = sorted(docs_path.glob("*.md")) + sorted(docs_path.glob("*.txt"))

    if not files:
        logger.warning("No .md or .txt files found in %s", docs_dir)
        return 0

    total_chunks = 0

    for filepath in files:
        text = filepath.read_text(encoding="utf-8")
        chunks = chunk_text(
            text, source=filepath.name, chunk_size=chunk_size, overlap=chunk_overlap
        )

        if not chunks:
            logger.warning("Skipping empty file: %s", filepath.name)
            continue

        texts = [c["text"] for c in chunks]
        embeddings = await embedding_service.embed(texts)
        await qdrant_service.upsert_chunks(chunks, embeddings, user_id="global")

        total_chunks += len(chunks)
        logger.info("Ingested %s → %d chunks", filepath.name, len(chunks))

    logger.info("Total chunks ingested from default docs: %d", total_chunks)
    return total_chunks


async def ingest_text(
    text: str,
    source: str,
    user_id: str,
    embedding_service: EmbeddingService,
    qdrant_service: QdrantService,
    chunk_size: int = 300,
    chunk_overlap: int = 50,
) -> int:
    """Chunk and ingest arbitrary text (e.g. a user-uploaded file).

    Returns the number of chunks created.
    """
    chunks = chunk_text(
        text, source=source, chunk_size=chunk_size, overlap=chunk_overlap
    )

    if not chunks:
        return 0

    texts = [c["text"] for c in chunks]
    embeddings = await embedding_service.embed(texts)
    await qdrant_service.upsert_chunks(chunks, embeddings, user_id=str(user_id))

    logger.info(
        "Ingested user upload '%s' (user=%s) → %d chunks", source, user_id, len(chunks)
    )
    return len(chunks)
