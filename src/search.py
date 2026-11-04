"""Query-time retrieval: BM25 scoring over a loaded index."""

from src.bm25 import BM25Index
from src.indexing import tokenize
from src.models import Chunk, IndexedChunk, Source


def build_bm25_index(chunks: list[IndexedChunk]) -> BM25Index:
    """Build a BM25Index from a loaded index's chunks.

    Args:
        chunks: The chunks loaded from an index directory.

    Returns:
        A BM25Index whose document ids match positions in `chunks`.
    """
    return BM25Index([c.tokens for c in chunks])


def search_chunks(
    query: str, chunks: list[IndexedChunk], bm25: BM25Index, k: int
) -> list[IndexedChunk]:
    """Retrieve the top-k chunks for a query.

    Args:
        query: The natural language query.
        chunks: The indexed chunks, in the same order used to build bm25.
        bm25: A BM25Index built from `chunks`.
        k: Maximum number of chunks to retrieve.

    Returns:
        The top-k chunks, highest score first. Empty for an empty/
        whitespace-only query or k <= 0.
    """
    if not query.strip() or k <= 0:
        return []
    ranked = bm25.top_k(tokenize(query), k)
    return [chunks[doc_id] for doc_id, _score in ranked]


def chunk_to_source(chunk: Chunk) -> Source:
    """Convert a Chunk into the Source shape used in search results.

    Args:
        chunk: A chunk (or IndexedChunk) retrieved from the index.

    Returns:
        A Source with the chunk's file_path and character range.
    """
    return Source(
        file_path=chunk.file_path,
        first_character_index=chunk.first_character_index,
        last_character_index=chunk.last_character_index,
    )
