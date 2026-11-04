"""Paragraph-based fallback chunker for non-Python, non-Markdown text.

Used directly for unrecognized file extensions, and reused internally
by the Python chunker (on a SyntaxError) and the Markdown chunker (to
split an oversized section's body).
"""

import re

from src.chunking.common import greedy_group_spans, hard_split_by_lines, make_chunk
from src.models import Chunk

_BLANK_LINE_RE = re.compile(r"\n[ \t]*\n")


def _paragraph_spans(
    text: str, start: int, end: int, atomic_spans: list[tuple[int, int]] | None
) -> list[tuple[int, int]]:
    """Split [start, end) into paragraph spans at blank lines.

    A blank line that falls inside one of atomic_spans (e.g. a fenced
    code block) is not treated as a paragraph boundary, so atomic spans
    are never split across paragraphs.

    Args:
        text: The full source text.
        start: Start offset of the range to split, inclusive.
        end: End offset of the range to split, exclusive.
        atomic_spans: Spans that must never be split at an internal
            blank line.

    Returns:
        A list of non-overlapping (start, end) paragraph spans covering
        [start, end).
    """
    atomic_spans = atomic_spans or []

    def _inside_atomic(pos: int) -> bool:
        return any(a_start <= pos < a_end for a_start, a_end in atomic_spans)

    boundaries = [start]
    for match in _BLANK_LINE_RE.finditer(text, start, end):
        if not _inside_atomic(match.start()):
            boundaries.append(match.end())
    boundaries.append(end)
    spans = []
    for i in range(len(boundaries) - 1):
        span_start, span_end = boundaries[i], boundaries[i + 1]
        if span_end > span_start:
            spans.append((span_start, span_end))
    return spans or [(start, end)]


def chunk_generic_text(
    text: str,
    file_path: str,
    start: int,
    end: int,
    target_size: int,
    max_size: int,
    symbol: str | None,
    atomic_spans: list[tuple[int, int]] | None = None,
) -> list[Chunk]:
    """Split [start, end) of text into chunks by paragraph.

    Paragraphs are greedily grouped up to target_size. Any single
    paragraph (or atomic span) that alone exceeds max_size falls back to
    line-boundary splitting.

    Args:
        text: The full source text.
        file_path: The corpus-relative path to record on each chunk.
        start: Start offset of the range to chunk, inclusive.
        end: End offset of the range to chunk, exclusive.
        target_size: Soft preferred size used for grouping paragraphs.
        max_size: Hard ceiling in characters for every chunk.
        symbol: Optional label to attach to every resulting chunk.
        atomic_spans: Spans (e.g. fenced code blocks) that must never be
            split at an internal blank line.

    Returns:
        A list of Chunks covering [start, end), none longer than
        max_size.
    """
    if end <= start:
        return []
    paragraphs = _paragraph_spans(text, start, end, atomic_spans)
    groups = greedy_group_spans(paragraphs, target_size)
    chunks: list[Chunk] = []
    for group_start, group_end in groups:
        if group_end - group_start <= max_size:
            chunks.append(make_chunk(text, file_path, group_start, group_end, symbol))
        else:
            chunks.extend(
                hard_split_by_lines(text, file_path, group_start, group_end, max_size, symbol)
            )
    return chunks
