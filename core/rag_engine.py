"""Central RAG orchestrator — the brain of the bot."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from core.cache import EmbeddingCache
from core.memory import ConversationMemory
from services.embedding import EmbeddingService
from services.intent import Intent, IntentClassifier, IntentType
from services.llm_gateway import LLMGateway
from services.qdrant_service import QdrantService

logger = logging.getLogger(__name__)

_MAX_SUMMARY_CONTEXT_CHARS = 16_000

_GREETING_RESPONSE = (
    "Hello! I'm your knowledge-base assistant. "
    "Ask me anything about your uploaded documents using "
    "/ask <your question>."
)


@dataclass
class RAGResponse:
    answer: str
    sources: list[dict] = field(default_factory=list)


class RAGEngine:

    def __init__(
        self,
        embedding_service: EmbeddingService,
        qdrant_service: QdrantService,
        llm_gateway: LLMGateway,
        intent_classifier: IntentClassifier,
        memory: ConversationMemory,
        cache: EmbeddingCache,
        top_k: int = 3,
        score_threshold: float = 0.4,
    ) -> None:
        self._embed = embedding_service
        self._qdrant = qdrant_service
        self._llm = llm_gateway
        self._intent = intent_classifier
        self._memory = memory
        self.memory = memory  # public access for handlers
        self._cache = cache
        self._top_k = top_k
        self._score_threshold = score_threshold

    # ------------------------------------------------------------------
    # Retrieval strategies
    # ------------------------------------------------------------------

    async def _retrieve_rag(self, search_query: str, user_id: str) -> list[dict]:
        """Normal semantic search across all user-visible documents."""
        vector = self._cache.get(search_query)
        if vector is None:
            vector = await self._embed.embed_single(search_query)
            self._cache.put(search_query, vector)

        results = await self._qdrant.search(
            query_embedding=vector,
            user_id=user_id,
            top_k=self._top_k,
            include_global=True,
        )

        logger.info(
            "rag_query: %d results, scores=%s",
            len(results),
            [round(r["score"], 4) for r in results],
        )

        return [r for r in results if r["score"] >= self._score_threshold]

    async def _retrieve_file_query(
        self, search_query: str, target_file: str, user_id: str,
    ) -> list[dict]:
        """Semantic search scoped to a single source file."""
        vector = self._cache.get(search_query)
        if vector is None:
            vector = await self._embed.embed_single(search_query)
            self._cache.put(search_query, vector)

        results = await self._qdrant.search(
            query_embedding=vector,
            user_id=user_id,
            top_k=self._top_k,
            source_filter=target_file,
            include_global=True,
        )

        logger.info(
            "file_query: file=%s, %d results, scores=%s",
            target_file, len(results),
            [round(r["score"], 4) for r in results],
        )

        return results  # no score filter — user explicitly asked about this file

    async def _retrieve_file_summary(
        self, target_file: str, user_id: str,
    ) -> list[dict]:
        """Fetch all chunks for a file in document order (for summarisation)."""
        results = await self._qdrant.fetch_source_chunks(
            user_id=user_id,
            source=target_file,
            include_global=True,
        )

        logger.info("file_summary: file=%s, %d chunks fetched", target_file, len(results))

        return self._truncate_chunks(results)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _truncate_chunks(chunks: list[dict]) -> list[dict]:
        """Keep ordered chunks within a conservative prompt budget."""
        selected: list[dict] = []
        total_chars = 0

        for chunk in chunks:
            projected = total_chars + len(chunk["chunk_text"])
            if selected and projected > _MAX_SUMMARY_CONTEXT_CHARS:
                break
            selected.append(chunk)
            total_chars = projected

        return selected

    @staticmethod
    def _build_context_and_sources(results: list[dict]) -> tuple[str, list[dict]]:
        """Format retrieved chunks into prompt context and source metadata."""
        context_parts: list[str] = []
        sources: list[dict] = []
        seen: set[str] = set()

        for hit in results:
            context_parts.append(f"[{hit['source']}]\n{hit['chunk_text']}")

            if hit["source"] not in seen:
                sources.append(
                    {"source": hit["source"], "snippet": hit["chunk_text"][:120]}
                )
                seen.add(hit["source"])

        return "\n\n---\n\n".join(context_parts), sources

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def ask(self, query: str, user_id: str) -> RAGResponse:
        """Full RAG pipeline: classify → retrieve → generate."""

        # 1. Classify intent
        last_upload = self._memory.get_last_upload(user_id)
        intent: Intent = await self._intent.classify(query, last_upload)

        # 2. Route based on intent
        if intent.type == IntentType.GREETING:
            self._memory.add(user_id, query, _GREETING_RESPONSE)
            return RAGResponse(answer=_GREETING_RESPONSE)

        if intent.type == IntentType.FILE_SUMMARY and intent.target_file:
            results = await self._retrieve_file_summary(intent.target_file, user_id)

        elif intent.type == IntentType.FILE_QUERY and intent.target_file:
            results = await self._retrieve_file_query(
                intent.search_query, intent.target_file, user_id,
            )

        else:  # IntentType.RAG_QUERY or fallback
            results = await self._retrieve_rag(intent.search_query, user_id)

        # 3. Handle empty retrieval
        if not results:
            answer = (
                "I couldn't find any relevant information in the knowledge base "
                "to answer your question."
            )
            self._memory.add(user_id, query, answer)
            return RAGResponse(answer=answer)

        # 4. Build context and generate
        context, sources = self._build_context_and_sources(results)
        history = self._memory.get(user_id)

        answer = await self._llm.generate(
            query=query, context=context, history=history or None,
        )

        # 5. Store in memory
        self._memory.add(user_id, query, answer)

        return RAGResponse(answer=answer, sources=sources)

    async def summarize(self, user_id: str) -> str:
        """Summarize the user's recent conversation history."""
        if not self._memory.has_history(user_id):
            return "No conversation history to summarize yet. Ask me something first!"

        history = self._memory.get(user_id)
        return await self._llm.summarize(history)
