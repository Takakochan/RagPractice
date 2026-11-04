"""Heading-based chunker for Markdown files.

Recursively descends from the shallowest heading level present in a
section down to deeper levels until each piece fits within
target_size, splitting a still-oversized leaf section by paragraph
(never inside a fenced code block, if avoidable).
"""

import re

from src.chunking.common import make_chunk
from src.chunking.generic_chunker import chunk_generic_text
from src.models import Chunk

_HEADING_RE = re.compile(r"(?m)^(#{1,6})[ \t]+(.*)$")
_FENCE_RE = re.compile(r"(?m)^(```|~~~)")


def _fenced_spans(text: str) -> list[tuple[int, int]]:
    """Find the character spans of fenced code blocks in a Markdown file.

    Args:
        text: The full Markdown source.

    Returns:
        A list of (start, end) spans, one per fenced code block. An
        unterminated trailing fence (odd number of fence markers) is
        treated as extending to the end of the file.
    """
    starts = [m.start() for m in _FENCE_RE.finditer(text)]
    spans = []
    for i in range(0, len(starts) - 1, 2):
        spans.append((starts[i], starts[i + 1]))
    if len(starts) % 2 == 1:
        spans.append((starts[-1], len(text)))
    return spans


def _headings(text: str) -> list[tuple[int, int, str]]:
    """Find all real (non-fenced) ATX headings in a Markdown file.

    Args:
        text: The full Markdown source.

    Returns:
        A list of (start_offset, level, title) tuples, sorted by
        position. Headings inside fenced code blocks are excluded.
    """
    fences = _fenced_spans(text)

    def _in_fence(pos: int) -> bool:
        return any(start <= pos < end for start, end in fences)

    headings = []
    for match in _HEADING_RE.finditer(text):
        if _in_fence(match.start()):
            continue
        headings.append((match.start(), len(match.group(1)), match.group(2).strip()))
    return headings


def _chunk_range(
    text: str,
    file_path: str,
    start: int,
    end: int,
    headings: list[tuple[int, int, str]],
    target_size: int,
    max_size: int,
    breadcrumb: list[str],
    fenced_spans: list[tuple[int, int]],
) -> list[Chunk]:
    """Recursively chunk [start, end), descending into headings as needed.

    Args:
        text: The full Markdown source.
        file_path: The corpus-relative path to record on each chunk.
        start: Start offset of the range, inclusive.
        end: End offset of the range, exclusive.
        headings: Headings strictly within [start, end), sorted by
            position.
        target_size: Soft preferred size; a range within this size is
            kept as a single chunk rather than descending further.
        max_size: Hard ceiling in characters for every chunk.
        breadcrumb: Heading titles from the document root down to this
            range, used to build the chunk's `symbol`.
        fenced_spans: All fenced code block spans in the document, used
            to avoid splitting a code block across chunks.

    Returns:
        A list of Chunks covering [start, end).
    """
    size = end - start
    symbol = " > ".join(breadcrumb) if breadcrumb else None
    if size <= 0:
        return []
    if not headings or size <= target_size:
        if size <= max_size:
            return [make_chunk(text, file_path, start, end, symbol)]
        return chunk_generic_text(
            text, file_path, start, end, target_size, max_size, symbol, fenced_spans
        )

    min_level = min(level for _, level, _ in headings)
    top_headings = [h for h in headings if h[1] == min_level]

    chunks: list[Chunk] = []
    if start < top_headings[0][0]:
        preamble_end = top_headings[0][0]
        preamble_headings = [h for h in headings if h[0] < preamble_end]
        chunks.extend(
            _chunk_range(
                text, file_path, start, preamble_end, preamble_headings,
                target_size, max_size, breadcrumb, fenced_spans,
            )
        )
    for i, (h_start, _level, title) in enumerate(top_headings):
        h_end = top_headings[i + 1][0] if i + 1 < len(top_headings) else end
        child_headings = [h for h in headings if h_start < h[0] < h_end]
        chunks.extend(
            _chunk_range(
                text, file_path, h_start, h_end, child_headings,
                target_size, max_size, breadcrumb + [title], fenced_spans,
            )
        )
    return chunks


def chunk_markdown_source(
    text: str, file_path: str, target_size: int, max_size: int
) -> list[Chunk]:
    """Split Markdown source into chunks along heading boundaries.

    Starts from the shallowest heading level present and descends only
    as needed to fit within target_size. A leaf section still over
    max_size is split by paragraph, keeping fenced code blocks intact
    where possible.

    Args:
        text: The full source text, read with no newline translation.
        file_path: The corpus-relative path to record on each chunk.
        target_size: Soft preferred size for a section before
            descending into its subsections.
        max_size: Hard ceiling in characters for every chunk.

    Returns:
        A list of Chunks covering the entire file.
    """
    if not text:
        return []
    fenced_spans = _fenced_spans(text)
    headings = _headings(text)
    if not headings:
        return chunk_generic_text(
            text, file_path, 0, len(text), target_size, max_size, None, fenced_spans
        )
    return _chunk_range(
        text, file_path, 0, len(text), headings, target_size, max_size, [], fenced_spans
    )
