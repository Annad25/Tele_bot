"""Telegram command and message handlers."""

from __future__ import annotations

import logging

from telegram import Update
from telegram.constants import ChatAction, ParseMode
from telegram.ext import ContextTypes

from core.rag_engine import RAGEngine
from services.embedding import EmbeddingService
from services.ingestion import ingest_text
from services.intent import IntentClassifier, IntentType
from services.qdrant_service import QdrantService

logger = logging.getLogger(__name__)

# Telegram message length limit
_MAX_MSG_LEN = 4096

HELP_TEXT = (
    "🤖 *RAG Knowledge Bot*\n\n"
    "*Commands:*\n"
    "/ask `<question>` — Ask a question against the knowledge base\n"
    "/summarize — Summarize our recent conversation\n"
    "/help — Show this message\n\n"
    "*File Uploads:*\n"
    "Send me a `.txt` or `.md` file and I'll add it to your personal "
    "knowledge base. Your documents are private to you.\n\n"
    "*Group Chats:*\n"
    "In groups, use commands directly — I only respond to commands, "
    "not regular messages."
)


def _split_message(text: str) -> list[str]:
    """Split a long message into chunks that fit Telegram's limit."""
    if len(text) <= _MAX_MSG_LEN:
        return [text]

    parts: list[str] = []
    while text:
        if len(text) <= _MAX_MSG_LEN:
            parts.append(text)
            break
        # Try to split at a paragraph boundary
        cut = text.rfind("\n\n", 0, _MAX_MSG_LEN)
        if cut == -1:
            cut = text.rfind("\n", 0, _MAX_MSG_LEN)
        if cut == -1:
            cut = _MAX_MSG_LEN
        parts.append(text[:cut])
        text = text[cut:].lstrip("\n")
    return parts


def _format_response(answer: str, sources: list[dict]) -> str:
    """Build the final user-facing reply with source citations."""
    parts = [answer]

    if sources:
        source_names = ", ".join(s["source"] for s in sources)
        parts.append(f"\n\n📚 Sources: {source_names}")

    return "\n".join(parts)


def _format_upload_success(file_name: str, num_chunks: int) -> str:
    """Build a plain-text upload confirmation safe for arbitrary filenames."""
    return (
        f"Uploaded {file_name} and indexed it into your personal knowledge base "
        f"({num_chunks} chunks created).\n\n"
        "You can now ask questions about it with /ask."
    )


def _get_user_id(update: Update) -> str:
    """Extract a stable user identifier."""
    return str(update.effective_user.id)


# ---------------------------------------------------------------------------
# Handler factories — each receives the shared dependencies via closure
# ---------------------------------------------------------------------------

def make_handlers(
    rag_engine: RAGEngine,
    intent_classifier: IntentClassifier,
    embedding_service: EmbeddingService,
    qdrant_service: QdrantService,
    chunk_size: int,
    chunk_overlap: int,
    max_upload_bytes: int,
):
    """Return a dict of handler callables wired to the given services."""

    async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(HELP_TEXT, parse_mode=ParseMode.MARKDOWN)

    async def help_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(HELP_TEXT, parse_mode=ParseMode.MARKDOWN)

    async def ask_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = " ".join(context.args) if context.args else ""
        if not query.strip():
            await update.message.reply_text(
                "Please provide a question. Usage: `/ask What is overfitting?`",
                parse_mode=ParseMode.MARKDOWN,
            )
            return

        user_id = _get_user_id(update)

        # Show typing indicator while processing
        await update.message.chat.send_action(ChatAction.TYPING)

        try:
            result = await rag_engine.ask(query, user_id)

            if not result.answer or not result.answer.strip():
                await update.message.reply_text(
                    "Sorry, I received an empty response. Please try rephrasing "
                    "your question."
                )
                return

            reply = _format_response(result.answer, result.sources)

            for chunk in _split_message(reply):
                await update.message.reply_text(chunk)

        except Exception:
            logger.exception("Error processing /ask for user %s", user_id)
            await update.message.reply_text(
                "Sorry, something went wrong while processing your question. "
                "Please try again."
            )

    async def summarize_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = _get_user_id(update)
        await update.message.chat.send_action(ChatAction.TYPING)

        try:
            summary = await rag_engine.summarize(user_id)

            if not summary or not summary.strip():
                await update.message.reply_text(
                    "Sorry, I couldn't generate a summary right now. "
                    "Try asking a few more questions first."
                )
                return

            for chunk in _split_message(summary):
                await update.message.reply_text(chunk)

        except Exception:
            logger.exception("Error processing /summarize for user %s", user_id)
            await update.message.reply_text(
                "Sorry, I couldn't generate a summary right now."
            )

    async def document_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle uploaded .txt / .md files."""
        document = update.message.document
        file_name = document.file_name or ""
        user_id = _get_user_id(update)

        # Validate extension
        if not file_name.lower().endswith((".txt", ".md")):
            await update.message.reply_text(
                "I only accept `.txt` and `.md` files. "
                "Please upload a supported format."
            )
            return

        # Validate size
        if document.file_size and document.file_size > max_upload_bytes:
            size_mb = max_upload_bytes / (1024 * 1024)
            await update.message.reply_text(
                f"File too large. Maximum allowed size is {size_mb:.0f} MB."
            )
            return

        await update.message.chat.send_action(ChatAction.TYPING)

        try:
            tg_file = await document.get_file()
            file_bytes = await tg_file.download_as_bytearray()
            text = file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            try:
                text = file_bytes.decode("latin-1")
            except Exception:
                await update.message.reply_text(
                    "Could not read the file — unsupported encoding."
                )
                return

        num_chunks = await ingest_text(
            text=text,
            source=file_name,
            user_id=user_id,
            embedding_service=embedding_service,
            qdrant_service=qdrant_service,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        # Track the upload so /ask can reference "the file"
        rag_engine.memory.set_last_upload(user_id, file_name)

        await update.message.reply_text(
            _format_upload_success(file_name, num_chunks)
        )

    async def plain_message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle plain text: use intent classifier to greet or nudge."""
        text = (update.message.text or "").strip()
        if not text:
            return

        user_id = _get_user_id(update)
        last_upload = rag_engine.memory.get_last_upload(user_id)

        # Cheap intent call to decide if this is a greeting or a real question
        intent = await intent_classifier.classify(text, last_upload)

        if intent.type == IntentType.GREETING:
            await update.message.reply_text(
                "Hello! I'm your knowledge-base assistant. "
                "Ask me anything about your uploaded documents using "
                "/ask <your question>."
            )
        else:
            await update.message.reply_text(
                f"Did you mean to ask a question? Use the command:\n/ask {text}"
            )

    async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
        logger.error("Unhandled exception: %s", context.error, exc_info=context.error)

    return {
        "start": start_handler,
        "help": help_handler,
        "ask": ask_handler,
        "summarize": summarize_handler,
        "document": document_handler,
        "plain_message": plain_message_handler,
        "error": error_handler,
    }
