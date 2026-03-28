"""Simple in-memory cache: normalised query text → embedding vector."""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)


def _normalise(text: str) -> str:
    """Lowercase, strip, and collapse whitespace."""
    return re.sub(r"\s+", " ", text.strip().lower())


class EmbeddingCache:
    def __init__(self) -> None:
        self._cache: dict[str, list[float]] = {}

    def get(self, query: str) -> list[float] | None:
        key = _normalise(query)
        hit = self._cache.get(key)
        if hit is not None:
            logger.debug("Cache HIT for: %s", key[:60])
        return hit

    def put(self, query: str, vector: list[float]) -> None:
        key = _normalise(query)
        self._cache[key] = vector

    @property
    def size(self) -> int:
        return len(self._cache)
