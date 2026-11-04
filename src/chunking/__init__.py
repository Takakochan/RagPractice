"""Chunking strategies that split corpus files into indexable Chunks.

Dispatches by file extension: Python files are split along AST
boundaries, Markdown files along heading boundaries, and everything else
falls back to a paragraph-based generic splitter. All strategies respect
a hard MAX_CHUNK_LENGTH ceiling and a softer target_chunk_size used only
for grouping decisions.
"""

from src.chunking.common import DEFAULT_TARGET_CHUNK_LENGTH
from src.chunking.generic_chunker import chunk_generic_text
from src.chunking.markdown_chunker import chunk_markdown_source
from src.chunking.python_chunker import chunk_python_source
from src.models import MAX_CHUNK_LENGTH, Chunk

__all__ = ["chunk_text", "DEFAULT_TARGET_CHUNK_LENGTH"]


def chunk_text(
    text: str,
    file_path: str,
    target_size: int = DEFAULT_TARGET_CHUNK_LENGTH,
    max_size: int = MAX_CHUNK_LENGTH,
) -> list[Chunk]:
    """Split a file's text into Chunks, dispatching on file extension.

    Args:
        text: The full text content of the file, read with no newline
            translation so character offsets match the file on disk.
        file_path: The corpus-relative path to record on each Chunk
            (e.g. "data/raw/vllm-0.10.1/vllm/envs.py").
        target_size: Soft preferred chunk size, used only when the
            strategy has discretion over how much sibling content to
            group into a single chunk.
        max_size: Hard ceiling in characters; no returned chunk ever
            exceeds this length.

    Returns:
        A list of Chunks covering the entire input text, ordered by
        position in the file.
    """
    if not text:
        return []
    if file_path.endswith(".py"):
        return chunk_python_source(text, file_path, target_size, max_size)
    if file_path.endswith(".md"):
        return chunk_markdown_source(text, file_path, target_size, max_size)
    return chunk_generic_text(text, file_path, 0, len(text), target_size, max_size, None)
