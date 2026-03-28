import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from core.cache import EmbeddingCache
from core.memory import ConversationMemory
from core.rag_engine import RAGEngine


class _UnusedEmbeddingService:
    def __init__(self) -> None:
        self.embed_single = AsyncMock(side_effect=AssertionError("should not embed"))


class _FakeQdrantService:
    def __init__(self) -> None:
        self.search = AsyncMock(side_effect=AssertionError("should not search"))
        self.fetch_source_chunks = AsyncMock(
            return_value=[
                {
                    "source": "tested_queries.md",
                    "chunk_text": "First chunk.",
                    "chunk_index": 1,
                    "score": 1.0,
                },
                {
                    "source": "tested_queries.md",
                    "chunk_text": "Second chunk.",
                    "chunk_index": 2,
                    "score": 1.0,
                },
            ]
        )


class RAGEngineTests(unittest.IsolatedAsyncioTestCase):
    async def test_summary_of_last_uploaded_file_uses_direct_source_fetch(self) -> None:
        memory = ConversationMemory(max_history=3)
        memory.set_last_upload("123", "tested_queries.md")

        qdrant = _FakeQdrantService()
        llm = SimpleNamespace(generate=AsyncMock(return_value="Summary"))

        engine = RAGEngine(
            embedding_service=_UnusedEmbeddingService(),
            qdrant_service=qdrant,
            llm_gateway=llm,
            memory=memory,
            cache=EmbeddingCache(),
        )

        result = await engine.ask("can you summarize this file", "123")

        qdrant.fetch_source_chunks.assert_awaited_once_with(
            user_id="123",
            source="tested_queries.md",
            include_global=False,
        )
        qdrant.search.assert_not_awaited()
        llm.generate.assert_awaited_once()
        self.assertEqual(result.answer, "Summary")
        self.assertEqual(
            result.sources,
            [{"source": "tested_queries.md", "snippet": "First chunk."}],
        )

    async def test_about_query_for_named_uploaded_file_uses_direct_source_fetch(
        self,
    ) -> None:
        memory = ConversationMemory(max_history=3)
        memory.set_last_upload("123", "tested_queries.md")

        qdrant = _FakeQdrantService()
        llm = SimpleNamespace(generate=AsyncMock(return_value="Summary"))

        engine = RAGEngine(
            embedding_service=_UnusedEmbeddingService(),
            qdrant_service=qdrant,
            llm_gateway=llm,
            memory=memory,
            cache=EmbeddingCache(),
        )

        result = await engine.ask("whats the tested_queries.md file about", "123")

        qdrant.fetch_source_chunks.assert_awaited_once_with(
            user_id="123",
            source="tested_queries.md",
            include_global=False,
        )
        qdrant.search.assert_not_awaited()
        llm.generate.assert_awaited_once()
        self.assertEqual(result.answer, "Summary")
