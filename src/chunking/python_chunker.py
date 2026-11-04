"""AST-based chunker for Python source files.

Splits a module into one chunk per top-level function/class (recursing
into methods for oversized classes), with runs of other top-level
statements (imports, constants, `if __name__ == "__main__":`, ...)
grouped separately. Gaps between statements (blank lines, comments, all
invisible to ast) are closed so every chunk's neighbor starts exactly
where it ends. Falls back to the generic paragraph chunker on a
SyntaxError.
"""

import ast

from src.chunking.common import close_gaps, hard_split_by_lines, make_chunk
from src.chunking.generic_chunker import chunk_generic_text
from src.models import Chunk

_DEFS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
_Unit = tuple[ast.stmt | None, list[ast.stmt]]


def _line_offsets(text: str) -> list[int]:
    """Compute the character offset at which each source line starts.

    Args:
        text: The full source text.

    Returns:
        A list where element i (0-indexed) is the character offset of
        the start of line i+1, matching ast's 1-indexed `lineno`. The
        final element is the total length of text.
    """
    offsets = [0]
    for line in text.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    return offsets


def _stmt_span(stmt: ast.stmt, line_offsets: list[int]) -> tuple[int, int]:
    """Compute the whole-line (start, end) character span of a statement.

    Decorators, if present, are included in the span. Spans are rounded
    out to full lines rather than exact columns, which is simple and
    matches the consistently-formatted (yapf/ruff) style of the corpus.

    Args:
        stmt: The statement (or FunctionDef/ClassDef) node.
        line_offsets: Precomputed line-start offsets from _line_offsets.

    Returns:
        A (start, end) tuple of character offsets, end exclusive.
    """
    start_line = stmt.lineno
    decorators = getattr(stmt, "decorator_list", None)
    if decorators:
        start_line = min(start_line, min(d.lineno for d in decorators))
    end_lineno = stmt.end_lineno
    assert end_lineno is not None
    return line_offsets[start_line - 1], line_offsets[end_lineno]


def _child_statement_blocks(stmt: ast.stmt) -> list[ast.stmt]:
    """Flatten a compound statement's nested statement blocks, in source order.

    Args:
        stmt: A statement that may contain nested blocks (If, For, While,
            Try, With, FunctionDef, ClassDef, ...).

    Returns:
        A flat, source-ordered list of the statements nested directly
        inside stmt. Empty if stmt has no nested statement blocks (e.g.
        a bare Assign or Expr).
    """
    children: list[ast.stmt] = []
    for _, value in ast.iter_fields(stmt):
        if isinstance(value, list) and value and isinstance(value[0], ast.stmt):
            children.extend(value)
        elif isinstance(value, list) and value and isinstance(value[0], ast.ExceptHandler):
            for handler in value:
                children.extend(handler.body)
    return children


def _greedy_group_statements(
    stmts: list[ast.stmt], spans: list[tuple[int, int]], bound: int
) -> list[tuple[int, int, list[ast.stmt]]]:
    """Greedily group statements (with precomputed, gap-closed spans) up to bound.

    Because `spans` is assumed gap-closed (each span starts exactly
    where the previous one ends), any run of merged spans stays
    contiguous, and so do the boundaries between groups.

    Args:
        stmts: Statements in source order.
        spans: The matching (start, end) span for each statement.
        bound: Maximum combined size for a group.

    Returns:
        A list of (group_start, group_end, members) tuples. A group
        with a single member may still exceed bound if that one
        statement's own span already does.
    """
    groups: list[tuple[int, int, list[ast.stmt]]] = []
    group_start, group_end = spans[0]
    members = [stmts[0]]
    for stmt, (start, end) in zip(stmts[1:], spans[1:]):
        if end - group_start <= bound:
            group_end = end
            members.append(stmt)
        else:
            groups.append((group_start, group_end, members))
            group_start, group_end, members = start, end, [stmt]
    groups.append((group_start, group_end, members))
    return groups


def _split_statements(
    text: str,
    file_path: str,
    start: int,
    end: int,
    symbol: str | None,
    body: list[ast.stmt],
    line_offsets: list[int],
    max_size: int,
) -> list[Chunk]:
    """Recursively split an oversized [start, end) span at statement boundaries.

    Sibling statements in `body` are greedily grouped up to max_size. A
    single statement still larger than max_size is recursed into via its
    own nested statement blocks; a statement with no nested blocks (an
    atomic, oversized leaf like a huge literal) falls back to
    line-boundary splitting.

    Args:
        text: The full source text.
        file_path: The corpus-relative path to record on each chunk.
        start: Start offset of the span, inclusive.
        end: End offset of the span, exclusive.
        symbol: Optional label to attach to every resulting chunk.
        body: The statements that make up [start, end), in source order.
        line_offsets: Precomputed line-start offsets from _line_offsets.
        max_size: Hard ceiling in characters for every chunk.

    Returns:
        A list of Chunks covering [start, end), none longer than
        max_size.
    """
    if end - start <= max_size:
        return [make_chunk(text, file_path, start, end, symbol)]
    if not body:
        return hard_split_by_lines(text, file_path, start, end, max_size, symbol)

    raw_spans = [_stmt_span(stmt, line_offsets) for stmt in body]
    spans = close_gaps(raw_spans, start, end)

    chunks: list[Chunk] = []
    for group_start, group_end, members in _greedy_group_statements(body, spans, max_size):
        if group_end - group_start <= max_size:
            chunks.append(make_chunk(text, file_path, group_start, group_end, symbol))
        elif len(members) == 1:
            sub_body = _child_statement_blocks(members[0])
            chunks.extend(
                _split_statements(
                    text, file_path, group_start, group_end, symbol,
                    sub_body, line_offsets, max_size,
                )
            )
        else:
            # Unreachable in practice: greedy grouping never merges
            # siblings past `max_size`, so a group can only be oversized
            # if it is a single, already-oversized statement.
            chunks.extend(
                hard_split_by_lines(text, file_path, group_start, group_end, max_size, symbol)
            )
    return chunks


def _split_top_level(body: list[ast.stmt]) -> list[_Unit]:
    """Group a statement list into (def_or_none, statements) units.

    Each FunctionDef/AsyncFunctionDef/ClassDef becomes its own unit.
    Consecutive other statements (imports, constants, ...) are grouped
    together into a single unit with a None node.

    Args:
        body: A list of statements, e.g. a module or class body.

    Returns:
        A list of (node, statements) pairs in source order. `node` is
        the def/class node for a single-definition unit, or None for a
        run of plain statements (in which case `statements` holds the
        full run).
    """
    units: list[_Unit] = []
    plain_run: list[ast.stmt] = []
    for stmt in body:
        if isinstance(stmt, _DEFS):
            if plain_run:
                units.append((None, plain_run))
                plain_run = []
            units.append((stmt, [stmt]))
        else:
            plain_run.append(stmt)
    if plain_run:
        units.append((None, plain_run))
    return units


def _unit_raw_span(unit: _Unit, line_offsets: list[int]) -> tuple[int, int]:
    """Compute a top-level unit's own (ungapped) span.

    Args:
        unit: A (node, statements) pair from _split_top_level.
        line_offsets: Precomputed line-start offsets from _line_offsets.

    Returns:
        A (start, end) tuple of character offsets, end exclusive.
    """
    node, stmts = unit
    if node is not None:
        return _stmt_span(node, line_offsets)
    start = _stmt_span(stmts[0], line_offsets)[0]
    end = _stmt_span(stmts[-1], line_offsets)[1]
    return start, end


def _chunk_plain_run(
    text: str,
    file_path: str,
    start: int,
    end: int,
    stmts: list[ast.stmt],
    line_offsets: list[int],
    target_size: int,
    max_size: int,
) -> list[Chunk]:
    """Chunk a run of non-def top-level/class-level statements.

    Statements are greedily grouped up to target_size; a group that
    still exceeds max_size (a single oversized statement) is recursed
    into via _split_statements.

    Args:
        text: The full source text.
        file_path: The corpus-relative path to record on each chunk.
        start: Start offset of the run, inclusive (already gap-closed
            against its neighbors).
        end: End offset of the run, exclusive (already gap-closed).
        stmts: The statements in the run, in source order.
        line_offsets: Precomputed line-start offsets from _line_offsets.
        target_size: Soft preferred size used for grouping.
        max_size: Hard ceiling in characters for every chunk.

    Returns:
        A list of Chunks covering [start, end).
    """
    if end - start <= max_size:
        return [make_chunk(text, file_path, start, end, None)]
    raw_spans = [_stmt_span(stmt, line_offsets) for stmt in stmts]
    spans = close_gaps(raw_spans, start, end)
    chunks: list[Chunk] = []
    for group_start, group_end, members in _greedy_group_statements(stmts, spans, target_size):
        if group_end - group_start <= max_size:
            chunks.append(make_chunk(text, file_path, group_start, group_end, None))
        else:
            sub_body = _child_statement_blocks(members[0]) if len(members) == 1 else []
            body_to_split = sub_body if (len(members) == 1 and sub_body) else members
            chunks.extend(
                _split_statements(
                    text, file_path, group_start, group_end, None,
                    body_to_split, line_offsets, max_size,
                )
            )
    return chunks


def _chunk_def(
    text: str,
    file_path: str,
    node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
    qualified_name: str,
    start: int,
    end: int,
    line_offsets: list[int],
    target_size: int,
    max_size: int,
) -> list[Chunk]:
    """Chunk a single function/class definition, recursing if oversized.

    Args:
        text: The full source text.
        file_path: The corpus-relative path to record on each chunk.
        node: The FunctionDef/AsyncFunctionDef/ClassDef node.
        qualified_name: Dotted name for this node (e.g. "Foo.bar"), used
            as the chunk's `symbol`.
        start: Start offset of this definition, inclusive (already
            gap-closed against its neighbors).
        end: End offset of this definition, exclusive (already
            gap-closed).
        line_offsets: Precomputed line-start offsets from _line_offsets.
        target_size: Soft preferred size used for grouping a class's
            non-method members (docstring, class attributes).
        max_size: Hard ceiling in characters for every chunk.

    Returns:
        A list of Chunks covering [start, end).
    """
    if end - start <= max_size:
        return [make_chunk(text, file_path, start, end, qualified_name)]
    if isinstance(node, ast.ClassDef):
        units = _split_top_level(node.body)
        raw_spans = [_unit_raw_span(u, line_offsets) for u in units]
        closed_spans = close_gaps(raw_spans, start, end)
        chunks: list[Chunk] = []
        for (sub_node, stmts), (sub_start, sub_end) in zip(units, closed_spans):
            if sub_node is not None and isinstance(sub_node, _DEFS):
                chunks.extend(
                    _chunk_def(
                        text, file_path, sub_node, f"{qualified_name}.{sub_node.name}",
                        sub_start, sub_end, line_offsets, target_size, max_size,
                    )
                )
            else:
                chunks.extend(
                    _chunk_plain_run(
                        text, file_path, sub_start, sub_end, stmts,
                        line_offsets, target_size, max_size,
                    )
                )
        return chunks
    return _split_statements(
        text, file_path, start, end, qualified_name, list(node.body), line_offsets, max_size
    )


def chunk_python_source(
    text: str, file_path: str, target_size: int, max_size: int
) -> list[Chunk]:
    """Split Python source into chunks along AST boundaries.

    Each top-level function/class is its own chunk (recursed into if
    larger than max_size); runs of other top-level statements are
    grouped up to target_size. Falls back to the generic paragraph
    chunker on a SyntaxError.

    Args:
        text: The full source text, read with no newline translation.
        file_path: The corpus-relative path to record on each chunk.
        target_size: Soft preferred size used for grouping plain
            top-level statements.
        max_size: Hard ceiling in characters for every chunk.

    Returns:
        A list of Chunks covering the entire file.
    """
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return chunk_generic_text(text, file_path, 0, len(text), target_size, max_size, None)
    if not tree.body:
        # No statements (e.g. a file containing only comments/blank
        # lines, which ast does not represent as nodes at all).
        return chunk_generic_text(text, file_path, 0, len(text), target_size, max_size, None)

    line_offsets = _line_offsets(text)
    units = _split_top_level(tree.body)
    raw_spans = [_unit_raw_span(u, line_offsets) for u in units]
    closed_spans = close_gaps(raw_spans, 0, len(text))

    chunks: list[Chunk] = []
    for (node, stmts), (start, end) in zip(units, closed_spans):
        if node is not None and isinstance(node, _DEFS):
            chunks.extend(
                _chunk_def(
                    text, file_path, node, node.name, start, end,
                    line_offsets, target_size, max_size,
                )
            )
        else:
            chunks.extend(
                _chunk_plain_run(
                    text, file_path, start, end, stmts, line_offsets, target_size, max_size
                )
            )
    return chunks
