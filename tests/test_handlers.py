import importlib
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch


def _stub_module(name: str, **attrs: object) -> None:
    module = ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module


_stub_module("services.embedding", EmbeddingService=object)
_stub_module("services.qdrant_service", QdrantService=object)

handlers_module = importlib.import_module("bot.handlers")
make_handlers = handlers_module.make_handlers


class _FakeMemory:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def set_last_upload(self, user_id: str, file_name: str) -> None:
        self.calls.append((user_id, file_name))


class _FakeRAGEngine:
    def __init__(self) -> None:
        self.memory = _FakeMemory()


class HandlerTests(unittest.IsolatedAsyncioTestCase):
    async def test_plain_message_nudge_avoids_markdown_parse_mode(self) -> None:
        handlers = make_handlers(
            rag_engine=_FakeRAGEngine(),
            embedding_service=object(),
            qdrant_service=object(),
            chunk_size=300,
            chunk_overlap=50,
            max_upload_bytes=1024,
        )
        reply_text = AsyncMock()
        update = SimpleNamespace(
            message=SimpleNamespace(text="notes_[v1]`", reply_text=reply_text),
            effective_user=SimpleNamespace(id=123),
        )

        await handlers["plain_message"](update, None)

        reply_text.assert_awaited_once_with(
            "Did you mean to ask a question? Use the command:\n/ask notes_[v1]`"
        )

    @patch("bot.handlers.ingest_text", new_callable=AsyncMock)
    async def test_document_upload_confirmation_uses_plain_text(
        self, mock_ingest_text
    ) -> None:
        mock_ingest_text.return_value = 2
        rag_engine = _FakeRAGEngine()

        handlers = make_handlers(
            rag_engine=rag_engine,
            embedding_service=object(),
            qdrant_service=object(),
            chunk_size=300,
            chunk_overlap=50,
            max_upload_bytes=1024,
        )

        reply_text = AsyncMock()
        send_action = AsyncMock()
        tg_file = SimpleNamespace(
            download_as_bytearray=AsyncMock(return_value=bytearray(b"hello world"))
        )
        document = SimpleNamespace(
            file_name="notes_[v1].md",
            file_size=42,
            get_file=AsyncMock(return_value=tg_file),
        )
        update = SimpleNamespace(
            message=SimpleNamespace(
                document=document,
                reply_text=reply_text,
                chat=SimpleNamespace(send_action=send_action),
            ),
            effective_user=SimpleNamespace(id=123),
        )

        await handlers["document"](update, None)

        self.assertEqual(rag_engine.memory.calls, [("123", "notes_[v1].md")])
        reply_text.assert_awaited_once_with(
            "Uploaded notes_[v1].md and indexed it into your personal knowledge base "
            "(2 chunks created).\n\n"
            "You can now ask questions about it with /ask."
        )
