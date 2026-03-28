import asyncio
import logging
from functools import partial

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Wraps a SentenceTransformer model with an async interface.

    The model itself is synchronous and CPU-bound, so every call is
    offloaded to a thread-pool executor to keep the Telegram event
    loop responsive.
    """

    def __init__(self, model_name: str) -> None:
        logger.info("Loading embedding model: %s …", model_name)
        self._model = SentenceTransformer(model_name)
        self._dimension = self._model.get_sentence_embedding_dimension()
        logger.info("Embedding model loaded (dim=%d).", self._dimension)

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Return normalised embeddings for *texts* (async-safe)."""
        loop = asyncio.get_running_loop()
        embeddings = await loop.run_in_executor(
            None,
            partial(self._model.encode, texts, normalize_embeddings=True),
        )
        return embeddings.tolist()

    async def embed_single(self, text: str) -> list[float]:
        """Convenience: embed a single string and return its vector."""
        vectors = await self.embed([text])
        return vectors[0]
