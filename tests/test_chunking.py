"""Tests for the chunking strategies in src.chunking.

Covers the hard max-length invariant (the primary requirement: no chunk
is ever returned longer than MAX_CHUNK_LENGTH), slice-exactness of
first/last_character_index, gap/overlap-free coverage, and the
boundary-specific behavior of each strategy. A parametrized regression
suite also runs every strategy against the full real corpus when it is
present on disk.
"""

from pathlib import Path

import pytest

from src.chunking import chunk_text
from src.chunking.generic_chunker import chunk_generic_text
from src.chunking.markdown_chunker import chunk_markdown_source
from src.chunking.python_chunker import chunk_python_source
from src.models import MAX_CHUNK_LENGTH, Chunk

CORPUS_ROOT = Path("data/raw/vllm-0.10.1")


def assert_valid_chunks(text: str, chunks: list[Chunk], max_size: int) -> None:
    """Assert the invariants every chunking strategy must satisfy.

    Args:
        text: The full source text the chunks were built from.
        chunks: The chunks produced for `text`.
        max_size: The hard character limit chunks must not exceed.
    """
    if not text:
        assert chunks == [], "expected no chunks for empty text"
        return
    assert chunks, "expected at least one chunk for non-empty text"
    ordered = sorted(chunks, key=lambda c: c.first_character_index)
    prev_end = 0
    for chunk in ordered:
        assert len(chunk.text) <= max_size
        assert text[chunk.first_character_index:chunk.last_character_index] == chunk.text
        assert chunk.first_character_index == prev_end, "gap or overlap before this chunk"
        prev_end = chunk.last_character_index
    assert prev_end == len(text), "chunks do not cover the full text"


class TestMaxLength:
    """The hard character ceiling must never be exceeded, anywhere."""

    def test_python_oversized_function_is_split(self) -> None:
        body = "\n".join(f"    x{i} = {i}" for i in range(400))
        source = f"def big():\n{body}\n"
        chunks = chunk_python_source(source, "f.py", 1500, 2000)
        assert all(len(c.text) <= 2000 for c in chunks)
        assert len(chunks) > 1

    def test_python_oversized_single_statement_falls_back_to_child_statements(self) -> None:
        lines = "\n".join(f'    "k{i}": {i},' for i in range(400))
        source = f"DATA = {{\n{lines}\n}}\n"
        chunks = chunk_python_source(source, "f.py", 1500, 2000)
        assert all(len(c.text) <= 2000 for c in chunks)

    def test_python_single_line_longer_than_max_is_hard_split(self) -> None:
        source = "X = '" + ("a" * 5000) + "'\n"
        chunks = chunk_python_source(source, "f.py", 1500, 2000)
        assert all(len(c.text) <= 2000 for c in chunks)
        assert len(chunks) >= 3

    def test_python_oversized_class_recurses_into_methods(self) -> None:
        methods = "\n\n".join(
            f"    def method_{i}(self):\n" + "\n".join(f"        y = {j}" for j in range(20))
            for i in range(30)
        )
        source = f"class Big:\n{methods}\n"
        chunks = chunk_python_source(source, "f.py", 1500, 2000)
        assert all(len(c.text) <= 2000 for c in chunks)
        assert any(c.symbol == "Big.method_0" for c in chunks)

    def test_markdown_oversized_section_is_split(self) -> None:
        body = "\n\n".join(f"Paragraph {i} " + "word " * 40 for i in range(30))
        source = f"# Title\n\n{body}\n"
        chunks = chunk_markdown_source(source, "f.md", 1500, 2000)
        assert all(len(c.text) <= 2000 for c in chunks)
        assert len(chunks) > 1

    def test_markdown_single_line_longer_than_max_is_hard_split(self) -> None:
        source = "# Title\n\n" + ("a" * 5000) + "\n"
        chunks = chunk_markdown_source(source, "f.md", 1500, 2000)
        assert all(len(c.text) <= 2000 for c in chunks)

    def test_markdown_fenced_code_block_longer_than_max_is_hard_split(self) -> None:
        source = "# Title\n\n```\n" + ("x" * 5000) + "\n```\n"
        chunks = chunk_markdown_source(source, "f.md", 1500, 2000)
        assert all(len(c.text) <= 2000 for c in chunks)

    def test_generic_oversized_paragraph_is_hard_split(self) -> None:
        source = "word " * 1000
        chunks = chunk_generic_text(source, "f.txt", 0, len(source), 1500, 2000, None)
        assert all(len(c.text) <= 2000 for c in chunks)


class TestInvariants:
    """Every strategy must produce gap-free, overlap-free, exact-slice chunks."""

    def test_python_small_module(self) -> None:
        source = (
            "import os\n\n\n"
            "# A helpful comment right above foo\n"
            "def foo():\n    return 1\n\n\n"
            'class Bar:\n    """Doc."""\n\n    def method(self):\n        return 2\n'
        )
        chunks = chunk_python_source(source, "f.py", 1500, 2000)
        assert_valid_chunks(source, chunks, 2000)

    def test_python_comments_only_file(self) -> None:
        source = "# just a license header\n# nothing else\n"
        chunks = chunk_python_source(source, "f.py", 1500, 2000)
        assert_valid_chunks(source, chunks, 2000)

    def test_python_syntax_error_falls_back_to_generic(self) -> None:
        source = "def broken(:\n    pass\n"
        chunks = chunk_python_source(source, "f.py", 1500, 2000)
        assert_valid_chunks(source, chunks, 2000)

    def test_markdown_with_headings(self) -> None:
        source = "# Title\n\nIntro.\n\n## Sub\n\nBody text.\n\n### Deeper\n\nMore text.\n"
        chunks = chunk_markdown_source(source, "f.md", 1500, 2000)
        assert_valid_chunks(source, chunks, 2000)

    def test_markdown_without_headings(self) -> None:
        source = "Just a paragraph.\n\nAnother paragraph.\n"
        chunks = chunk_markdown_source(source, "f.md", 1500, 2000)
        assert_valid_chunks(source, chunks, 2000)

    def test_markdown_heading_like_text_inside_fence_is_not_a_heading(self) -> None:
        source = (
            "# Title\n\n```bash\n# this is a shell comment, not a heading\necho hi\n```\n\n"
            "More text.\n"
        )
        chunks = chunk_markdown_source(source, "f.md", 1500, 2000)
        assert_valid_chunks(source, chunks, 2000)
        assert not any("this is a shell comment" in (c.symbol or "") for c in chunks)

    def test_empty_file(self) -> None:
        assert chunk_text("", "f.py") == []
        assert chunk_text("", "f.md") == []


class TestSymbols:
    """`symbol` carries a qualified name / heading breadcrumb, not the range."""

    def test_python_symbol_is_qualified_name(self) -> None:
        source = "def foo():\n    pass\n\n\nclass Bar:\n    def method(self):\n        pass\n"
        chunks = chunk_python_source(source, "f.py", 1500, 2000)
        symbols = {c.symbol for c in chunks}
        assert "foo" in symbols
        assert "Bar" in symbols
        # See TestMaxLength.test_python_oversized_class_recurses_into_methods
        # for the "Foo.method" qualified-name case, which only appears once
        # a class is too large to stay in a single chunk.

    def test_markdown_symbol_is_heading_breadcrumb(self) -> None:
        body = "\n\n".join(f"Paragraph {i} " + "word " * 300 for i in range(5))
        source = f"# Title\n\n## Sub\n\n{body}\n"
        chunks = chunk_markdown_source(source, "f.md", 200, 2000)
        assert any(c.symbol == "Title > Sub" for c in chunks)


class TestDispatch:
    """chunk_text() picks a strategy from the file extension."""

    def test_chunk_text_dispatches_by_extension(self) -> None:
        assert chunk_text("x = 1\n", "a/b.py")
        assert chunk_text("# T\n\nbody\n", "a/b.md")
        assert chunk_text("just text\n", "a/b.txt")


@pytest.mark.skipif(
    not CORPUS_ROOT.exists(), reason="vLLM corpus not present at data/raw/vllm-0.10.1"
)
class TestCorpusRegression:
    """Full-corpus property check against the real vLLM 0.10.1 source.

    Exercises every indexed .py and .md file to catch edge cases
    synthetic tests might miss (unusual formatting, comment-only files,
    huge generated functions, etc.).
    """

    @pytest.mark.parametrize(
        "rel_path",
        [str(p.relative_to(CORPUS_ROOT)) for p in sorted(CORPUS_ROOT.glob("vllm/**/*.py"))]
        + [str(p.relative_to(CORPUS_ROOT)) for p in sorted(CORPUS_ROOT.glob("docs/**/*.md"))],
    )
    def test_file(self, rel_path: str) -> None:
        with open(CORPUS_ROOT / rel_path, encoding="utf-8", newline="") as f:
            text = f.read()
        file_path = f"data/raw/vllm-0.10.1/{rel_path}"
        chunks = chunk_text(text, file_path)
        assert_valid_chunks(text, chunks, MAX_CHUNK_LENGTH)
