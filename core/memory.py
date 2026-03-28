"""Per-user conversation memory (last N interactions)."""

from __future__ import annotations

from collections import defaultdict, deque


class ConversationMemory:
    """Stores the most recent interactions per user in a bounded deque."""

    def __init__(self, max_history: int = 3) -> None:
        self._max = max_history
        # user_id -> deque of {"query": str, "response": str}
        self._store: dict[str, deque[dict]] = defaultdict(
            lambda: deque(maxlen=self._max)
        )
        self._last_upload: dict[str, str] = {}

    def add(self, user_id: str, query: str, response: str) -> None:
        self._store[user_id].append({"query": query, "response": response})

    def get(self, user_id: str) -> list[dict]:
        """Return the conversation history as a plain list (oldest first)."""
        return list(self._store[user_id])

    def has_history(self, user_id: str) -> bool:
        return bool(self._store[user_id])

    def set_last_upload(self, user_id: str, file_name: str) -> None:
        """Record the most recently uploaded file for a user."""
        self._last_upload[user_id] = file_name

    def get_last_upload(self, user_id: str) -> str | None:
        """Return the last uploaded filename, or None."""
        return self._last_upload.get(user_id)
