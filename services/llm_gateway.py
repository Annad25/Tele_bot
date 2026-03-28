"""Thin async wrapper around the OpenAI chat-completions API."""

import logging
from typing import Any

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a helpful assistant for a retrieval-augmented Telegram bot. "
    "Answer factual questions using ONLY the retrieved context. "
    "Conversation history is provided only to resolve follow-up references "
    "and is not a source of facts. If the retrieved context does not contain "
    "enough information to answer, say so honestly — do not make things up.\n"
    "Keep answers concise and well-structured."
)


class LLMGateway:
    def __init__(self, api_key: str, model: str = "gpt-5-mini") -> None:
        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model

    @staticmethod
    def _format_history(history: list[dict] | None) -> str:
        """Return prior user questions as non-authoritative continuity context."""
        if not history:
            return ""

        prior_queries = [
            turn["query"].strip()
            for turn in history
            if turn.get("query") and turn["query"].strip()
        ]
        if not prior_queries:
            return ""

        bullets = "\n".join(f"- {query}" for query in prior_queries)
        return (
            "### Prior user questions (continuity only, not factual context)\n"
            f"{bullets}\n\n"
        )

    async def generate(
        self,
        query: str,
        context: str,
        history: list[dict] | None = None,
    ) -> str:
        """Build a RAG prompt and return the assistant's reply."""
        messages: list[dict[str, Any]] = [
            {"role": "developer", "content": SYSTEM_PROMPT}
        ]

        # Inject retrieved context + current query
        history_context = self._format_history(history)
        user_content = (
            f"{history_context}"
            f"### Retrieved context\n{context}\n\n"
            f"### Current question\n{query}"
        )
        messages.append({"role": "user", "content": user_content})

        response = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            max_completion_tokens=1024,
        )

        msg = response.choices[0].message
        answer = msg.content or ""
        if not answer:
            logger.warning(
                "LLM returned empty content. finish_reason=%s, message=%s",
                response.choices[0].finish_reason,
                msg,
            )
        return answer

    async def summarize(self, interactions: list[dict]) -> str:
        """Summarize a list of past user–bot interactions."""
        conversation = "\n".join(
            f"User: {t['query']}\nAssistant: {t['response']}"
            for t in interactions
        )

        messages = [
            {
                "role": "developer",
                "content": (
                    "Summarize the following conversation between a user and "
                    "an assistant. Be concise, capture the key topics and "
                    "answers discussed."
                ),
            },
            {"role": "user", "content": conversation},
        ]

        response = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            max_completion_tokens=512,
        )

        return response.choices[0].message.content or ""
