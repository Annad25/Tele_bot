"""Split a document's text into overlapping chunks with metadata.

Uses LangChain's RecursiveCharacterTextSplitter which splits on natural
boundaries (paragraphs → newlines → sentences → words) rather than
cutting at arbitrary character positions.
"""

from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_text(
    text: str,
    source: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[dict]:
    """Return a list of chunk dicts from *text*.

    Each dict contains:
        - text:        the chunk content
        - source:      originating filename
        - chunk_index: positional index within the document
    """
    if not text or not text.strip():
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    raw_chunks = splitter.split_text(text)

    return [
        {"text": chunk.strip(), "source": source, "chunk_index": idx}
        for idx, chunk in enumerate(raw_chunks)
        if chunk.strip()
    ]
