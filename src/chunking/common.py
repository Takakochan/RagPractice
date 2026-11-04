"""Shared primitives used by every chunking strategy.

All functions here operate purely on in-memory text and character
offsets (no file I/O), which keeps them easy to unit test and reuse
across the Python, Markdown, and generic chunkers.
"""

from src.models import Chunk

DEFAULT_TARGET_CHUNK_LENGTH = 1500


def make_chunk(
    text: str, file_path: str, start: int, end: int, symbol: str | None
) -> Chunk:
    """Build a Chunk from an exact [start, end) slice of the source text.

    Args:
        text: The full source text the chunk is sliced from.
        file_path: The corpus-relative path to record on the chunk.
        start: Start offset of the chunk, inclusive.
        end: End offset of the chunk, exclusive.
        symbol: Optional human/embedding-friendly label for the chunk's
            location (e.g. a qualified function name or heading
            breadcrumb).

    Returns:
        A Chunk whose text is exactly text[start:end].
    """
    return Chunk(
        chunk_id=f"{file_path}#{start}-{end}",
        file_path=file_path,
        first_character_index=start,
        last_character_index=end,
        text=text[start:end],
        symbol=symbol,
    )


def hard_split_by_chars(
    text: str, file_path: str, start: int, end: int, max_size: int, symbol: str | None
) -> list[Chunk]:
    """Split a span into fixed-size pieces with no boundary awareness.

    This is the last-resort fallback used when a span cannot be split at
    any line boundary without still exceeding max_size (e.g. a single
    line longer than max_size).

    Args:
        text: The full source text.
        file_path: The corpus-relative path to record on each chunk.
        start: Start offset of the span to split, inclusive.
        end: End offset of the span to split, exclusive.
        max_size: Maximum number of characters per piece.
        symbol: Optional label to attach to every resulting chunk.

    Returns:
        A list of Chunks covering [start, end) with no chunk longer than
        max_size.
    """
    chunks = []
    pos = start
    while pos < end:
        piece_end = min(pos + max_size, end)
        chunks.append(make_chunk(text, file_path, pos, piece_end, symbol))
        pos = piece_end
    return chunks


def hard_split_by_lines(
    text: str, file_path: str, start: int, end: int, max_size: int, symbol: str | None
) -> list[Chunk]:
    """Split a span at newline boundaries, greedily filling up to max_size.

    Falls back to hard_split_by_chars for any single line that alone
    exceeds max_size.

    Args:
        text: The full source text.
        file_path: The corpus-relative path to record on each chunk.
        start: Start offset of the span to split, inclusive.
        end: End offset of the span to split, exclusive.
        max_size: Maximum number of characters per chunk.
        symbol: Optional label to attach to every resulting chunk.

    Returns:
        A list of Chunks covering [start, end) with no chunk longer than
        max_size.
    """
    chunks: list[Chunk] = []
    group_start = start
    pos = start
    while pos < end:
        newline_idx = text.find("\n", pos, end)
        line_end = newline_idx + 1 if newline_idx != -1 else end
        line_len = line_end - pos
        if line_len > max_size:
            if group_start < pos:
                chunks.append(make_chunk(text, file_path, group_start, pos, symbol))
            chunks.extend(
                hard_split_by_chars(text, file_path, pos, line_end, max_size, symbol)
            )
            group_start = line_end
        elif pos - group_start + line_len > max_size:
            chunks.append(make_chunk(text, file_path, group_start, pos, symbol))
            group_start = pos
        pos = line_end
    if group_start < end:
        chunks.append(make_chunk(text, file_path, group_start, end, symbol))
    return chunks


def close_gaps(
    spans: list[tuple[int, int]], range_start: int, range_end: int
) -> list[tuple[int, int]]:
    """Extend spans so they contiguously cover [range_start, range_end).

    Each span absorbs the gap that precedes it (e.g. blank lines or
    comments between statements, which AST does not represent as
    nodes), so no content between range_start and range_end is ever
    dropped from every chunk.

    Args:
        spans: Non-overlapping (start, end) spans, sorted by position,
            all within [range_start, range_end).
        range_start: The start of the full range the spans should cover.
        range_end: The end of the full range the spans should cover.

    Returns:
        A list of spans, same length and order as `spans`, contiguous
        and covering exactly [range_start, range_end).
    """
    if not spans:
        return []
    closed = [spans[0]]
    for _, end in spans[1:]:
        closed.append((closed[-1][1], end))
    closed[0] = (range_start, closed[0][1])
    closed[-1] = (closed[-1][0], range_end)
    return closed


def greedy_group_spans(
    spans: list[tuple[int, int]], bound: int
) -> list[tuple[int, int]]:
    """Greedily merge consecutive, non-overlapping spans into groups.

    A span already larger than bound by itself is kept as its own
    (oversized) group; the caller is responsible for splitting it
    further. No returned group other than such a singleton ever exceeds
    bound.

    Args:
        spans: Non-overlapping (start, end) spans, sorted by position.
        bound: Maximum combined size for a group.

    Returns:
        A list of merged (start, end) groups.
    """
    if not spans:
        return []
    groups = []
    group_start, group_end = spans[0]
    for start, end in spans[1:]:
        if end - group_start <= bound:
            group_end = end
        else:
            groups.append((group_start, group_end))
            group_start, group_end = start, end
    groups.append((group_start, group_end))
    return groups
