"""Entry point — initialises all services and starts the Telegram bot."""

import asyncio
import logging
import sys

from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters

from bot.handlers import make_handlers
from core.cache import EmbeddingCache
from core.config import Config
from core.memory import ConversationMemory
from core.rag_engine import RAGEngine
from services.embedding import EmbeddingService
from services.ingestion import ingest_directory
from services.intent import IntentClassifier
from services.llm_gateway import LLMGateway
from services.qdrant_service import QdrantService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-28s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


async def startup(config: Config) -> None:
    """Initialise services, ingest default docs, and start polling."""

    # --- Services ---
    logger.info("Initialising services …")

    embedding_service = EmbeddingService(config.embedding_model)

    qdrant_service = QdrantService(
        path=config.qdrant_path,
        collection=config.qdrant_collection,
        vector_size=config.embedding_dimension,
    )
    await qdrant_service.ensure_collection()

    llm_gateway = LLMGateway(api_key=config.openai_api_key, model=config.openai_model)

    intent_classifier = IntentClassifier(
        api_key=config.openai_api_key,
        model=config.intent_model,
    )

    # --- Core ---
    memory = ConversationMemory(max_history=config.max_history)
    cache = EmbeddingCache()

    rag_engine = RAGEngine(
        embedding_service=embedding_service,
        qdrant_service=qdrant_service,
        llm_gateway=llm_gateway,
        intent_classifier=intent_classifier,
        memory=memory,
        cache=cache,
        top_k=config.top_k,
    )

    # --- Ensure the shared default knowledge base is present ---
    has_global_docs = await qdrant_service.has_chunks(user_id="global")
    if not has_global_docs:
        logger.info("Global default docs are missing — ingesting default documents …")
        await ingest_directory(
            docs_dir=config.default_docs_dir,
            embedding_service=embedding_service,
            qdrant_service=qdrant_service,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )
    else:
        logger.info("Global default docs already present — skipping ingestion.")

    # --- Telegram bot ---
    handlers = make_handlers(
        rag_engine=rag_engine,
        intent_classifier=intent_classifier,
        embedding_service=embedding_service,
        qdrant_service=qdrant_service,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        max_upload_bytes=config.max_upload_size_bytes,
    )

    app = ApplicationBuilder().token(config.telegram_bot_token).build()

    app.add_handler(CommandHandler("start", handlers["start"]))
    app.add_handler(CommandHandler("help", handlers["help"]))
    app.add_handler(CommandHandler("ask", handlers["ask"]))
    app.add_handler(CommandHandler("summarize", handlers["summarize"]))
    app.add_handler(
        MessageHandler(filters.Document.ALL, handlers["document"])
    )
    # Catch plain text messages (not commands, not documents) to remind about /ask
    app.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handlers["plain_message"])
    )
    app.add_error_handler(handlers["error"])

    logger.info("Bot is ready — starting polling …")
    await app.initialize()
    await app.start()
    await app.updater.start_polling()

    # Keep running until interrupted
    try:
        while True:
            await asyncio.sleep(3600)
    except (KeyboardInterrupt, SystemExit):
        logger.info("Shutting down …")
    finally:
        await app.updater.stop()
        await app.stop()
        await app.shutdown()


def main() -> None:
    config = Config()
    asyncio.run(startup(config))


if __name__ == "__main__":
    main()
