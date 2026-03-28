import logging
import uuid

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

logger = logging.getLogger(__name__)


class QdrantService:
    """Async wrapper around a file-backed Qdrant instance."""

    def __init__(self, path: str, collection: str, vector_size: int) -> None:
        self._client = AsyncQdrantClient(path=path)
        self._collection = collection
        self._vector_size = vector_size

    @staticmethod
    def _build_filter(
        user_id: str,
        source_filter: str | None = None,
        include_global: bool = True,
    ) -> Filter:
        """Build a payload filter for either shared+user docs or user-only docs."""
        must_conditions = []
        if source_filter:
            must_conditions.append(
                FieldCondition(key="source", match=MatchValue(value=source_filter))
            )

        if include_global:
            return Filter(
                must=must_conditions or None,
                should=[
                    FieldCondition(key="user_id", match=MatchValue(value="global")),
                    FieldCondition(
                        key="user_id", match=MatchValue(value=str(user_id))
                    ),
                ],
            )

        must_conditions.append(
            FieldCondition(key="user_id", match=MatchValue(value=str(user_id)))
        )
        return Filter(must=must_conditions)

    async def ensure_collection(self) -> None:
        """Create the collection if it does not already exist."""
        collections = await self._client.get_collections()
        existing = {c.name for c in collections.collections}

        if self._collection in existing:
            logger.info("Qdrant collection '%s' already exists.", self._collection)
            return

        await self._client.create_collection(
            collection_name=self._collection,
            vectors_config=VectorParams(
                size=self._vector_size, distance=Distance.COSINE
            ),
        )
        logger.info("Created Qdrant collection '%s'.", self._collection)

    async def upsert_chunks(
        self,
        chunks: list[dict],
        embeddings: list[list[float]],
        user_id: str,
    ) -> None:
        """Insert (or overwrite) embedded chunks into the collection."""
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=emb,
                payload={
                    "user_id": user_id,
                    "source": chunk["source"],
                    "chunk_text": chunk["text"],
                    "chunk_index": chunk["chunk_index"],
                },
            )
            for chunk, emb in zip(chunks, embeddings)
        ]

        await self._client.upsert(
            collection_name=self._collection, points=points
        )
        logger.info(
            "Upserted %d chunks (user_id=%s) into '%s'.",
            len(points),
            user_id,
            self._collection,
        )

    async def search(
        self,
        query_embedding: list[float],
        user_id: str,
        top_k: int = 3,
        source_filter: str | None = None,
        include_global: bool = True,
    ) -> list[dict]:
        """Retrieve the top-k most relevant chunks for *user_id*.

        Results include both global documents and the user's own uploads.
        If *source_filter* is set, only chunks from that source file are returned.
        """
        query_filter = self._build_filter(
            user_id=user_id,
            source_filter=source_filter,
            include_global=include_global,
        )

        results = await self._client.query_points(
            collection_name=self._collection,
            query=query_embedding,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )

        return [
            {
                "source": hit.payload["source"],
                "chunk_text": hit.payload["chunk_text"],
                "chunk_index": hit.payload["chunk_index"],
                "score": hit.score,
            }
            for hit in results.points
        ]

    async def fetch_source_chunks(
        self,
        user_id: str,
        source: str,
        include_global: bool = False,
        batch_size: int = 128,
    ) -> list[dict]:
        """Return all chunks for a source file, ordered by chunk index."""
        scroll_filter = self._build_filter(
            user_id=user_id,
            source_filter=source,
            include_global=include_global,
        )

        offset = None
        records = []

        while True:
            page, offset = await self._client.scroll(
                collection_name=self._collection,
                scroll_filter=scroll_filter,
                limit=batch_size,
                offset=offset,
                with_payload=True,
            )
            records.extend(page)
            if offset is None:
                break

        ordered = sorted(
            records,
            key=lambda record: (
                record.payload.get("chunk_index", 0),
                record.payload.get("source", ""),
            ),
        )
        return [
            {
                "source": record.payload["source"],
                "chunk_text": record.payload["chunk_text"],
                "chunk_index": record.payload["chunk_index"],
                "score": 1.0,
            }
            for record in ordered
        ]

    async def has_chunks(
        self,
        user_id: str,
        source_filter: str | None = None,
    ) -> bool:
        """Return True if at least one matching chunk exists."""
        scroll_filter = self._build_filter(
            user_id=user_id,
            source_filter=source_filter,
            include_global=False,
        )
        page, _ = await self._client.scroll(
            collection_name=self._collection,
            scroll_filter=scroll_filter,
            limit=1,
            with_payload=False,
        )
        return bool(page)

    async def collection_count(self) -> int:
        """Return the number of points in the collection."""
        info = await self._client.get_collection(self._collection)
        return info.points_count
