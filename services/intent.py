"""Lightweight intent classifier using GPT-4o-mini.

Replaces static keyword lists with a single cheap LLM call that
determines the user's intent and resolves vague references like
"the file" or "it" into concrete filenames.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from enum import Enum

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class IntentType(str, Enum):
    GREETING = "greeting"
    RAG_QUERY = "rag_query"
    FILE_QUERY = "file_query"
    FILE_SUMMARY = "file_summary"


@dataclass(frozen=True)
class Intent:
    type: IntentType
    target_file: str | None
    search_query: str


# The default when classification fails — safe fallback to normal search.
_FALLBACK_INTENT = IntentType.RAG_QUERY

_CLASSIFY_PROMPT = """\
You are an intent classifier for a RAG Telegram bot.

Given a user query and context, return a JSON object with exactly these fields:
- "intent": one of "greeting", "rag_query", "file_query", "file_summary"
- "target_file": the filename being referenced, or null
- "search_query": the query rewritten for semantic search (strip meta-words like "summarize", "tell me about", keep only the topical content). For greetings, return the original query.

Intent definitions:
- "greeting": casual greeting or small talk with no knowledge question (e.g. "hey", "good morning", "thanks", "how are you")
- "rag_query": a question that should search across ALL documents in the knowledge base
- "file_query": a question about a SPECIFIC file (user names it or references their last upload)
- "file_summary": user wants a summary/overview of a SPECIFIC file (not a factual question, but "summarize", "what's in", "give me an overview", "tell me about the file")

Rules:
- If the user references "the file", "it", "this document", "what I uploaded", etc., resolve it to last_uploaded_file if available.
- If last_uploaded_file is null and the user references a file vaguely, treat as "rag_query" with target_file null.
- "search_query" should be optimized for embedding similarity — remove filler words, keep the topical core.

Examples:

User: "hey there!"
Context: last_uploaded_file=null
→ {"intent": "greeting", "target_file": null, "search_query": "hey there!"}

User: "what is overfitting?"
Context: last_uploaded_file="ml_basics.md"
→ {"intent": "rag_query", "target_file": null, "search_query": "overfitting"}

User: "what does the file say about regularization?"
Context: last_uploaded_file="ml_basics.md"
→ {"intent": "file_query", "target_file": "ml_basics.md", "search_query": "regularization"}

User: "summarize it"
Context: last_uploaded_file="tested_queries.md"
→ {"intent": "file_summary", "target_file": "tested_queries.md", "search_query": "tested_queries.md"}

User: "can you give me an overview of what I just uploaded"
Context: last_uploaded_file="notes.txt"
→ {"intent": "file_summary", "target_file": "notes.txt", "search_query": "notes.txt"}

User: "tell me about pasta_recipes.md"
Context: last_uploaded_file="other.md"
→ {"intent": "file_summary", "target_file": "pasta_recipes.md", "search_query": "pasta_recipes.md"}

User: "what is gradient descent in ml_basics.md?"
Context: last_uploaded_file="ml_basics.md"
→ {"intent": "file_query", "target_file": "ml_basics.md", "search_query": "gradient descent"}

User: "thanks!"
Context: last_uploaded_file="ml_basics.md"
→ {"intent": "greeting", "target_file": null, "search_query": "thanks!"}

Respond with ONLY the JSON object. No markdown, no explanation."""

_JSON_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "intent_classification",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "intent": {
                    "type": "string",
                    "enum": ["greeting", "rag_query", "file_query", "file_summary"],
                },
                "target_file": {
                    "type": ["string", "null"],
                },
                "search_query": {
                    "type": "string",
                },
            },
            "required": ["intent", "target_file", "search_query"],
            "additionalProperties": False,
        },
    },
}


class IntentClassifier:
    """Classifies user intent with a cheap, fast LLM call."""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini") -> None:
        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model

    async def classify(
        self,
        query: str,
        last_uploaded_file: str | None = None,
    ) -> Intent:
        """Return the classified intent for *query*.

        Falls back to a safe ``rag_query`` intent if the LLM call fails.
        """
        context_line = f"last_uploaded_file={last_uploaded_file or 'null'}"

        user_content = f"User: \"{query}\"\nContext: {context_line}"

        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _CLASSIFY_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                response_format=_JSON_SCHEMA,
                temperature=0,
                max_tokens=150,
            )

            raw = response.choices[0].message.content or ""
            data = json.loads(raw)

            intent = Intent(
                type=IntentType(data["intent"]),
                target_file=data.get("target_file"),
                search_query=data.get("search_query", query),
            )

            logger.info(
                "Intent classified: query=%r → type=%s, target_file=%s, search_query=%r",
                query, intent.type.value, intent.target_file, intent.search_query,
            )
            return intent

        except Exception:
            logger.exception("Intent classification failed for query=%r — falling back to rag_query", query)
            return Intent(
                type=_FALLBACK_INTENT,
                target_file=None,
                search_query=query,
            )
