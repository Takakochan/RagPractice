"""Tests for src.indexing: path normalization, file walking, tokenization,
and the build_index / load_index round trip (including atomicity and
clear failure on a missing/corrupted index).
"""

import json
from pathlib import Path

import pytest

from src.indexing import (
    IndexCorruptedError,
    IndexNotFoundError,
    build_index,
    iter_corpus_files,
    load_index,
    normalize_file_path,
    tokenize,
)
from src.models import IndexedChunk


def _make_corpus(root: Path) -> Path:
    """Create a small synthetic corpus under root/vllm-0.10.1 and return it."""
    corpus = root / "vllm-0.10.1"
    (corpus / "vllm" / "sub").mkdir(parents=True)
    (corpus / "docs").mkdir(parents=True)
    (corpus / "vllm" / "envs.py").write_text("X = 1\n", encoding="utf-8")
    (corpus / "vllm" / "sub" / "mod.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    (corpus / "docs" / "guide.md").write_text("# Guide\n\nHello.\n", encoding="utf-8")
    (corpus / "tests" / "test_foo.py").parent.mkdir(parents=True, exist_ok=True)
    (corpus / "tests" / "test_foo.py").write_text("x = 1\n", encoding="utf-8")
    (corpus / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.0)\n", encoding="utf-8"
    )
    (corpus / "README.md").write_text("# Readme\n", encoding="utf-8")
    return corpus


class TestNormalizeFilePath:
    """file_path must end up the same regardless of how corpus_dir was spelled."""

    def test_absolute_relative_and_dotted_forms_agree(self, tmp_path: Path) -> None:
        project_root = tmp_path
        corpus = _make_corpus(project_root / "data" / "raw")
        target = corpus / "vllm" / "envs.py"

        absolute = normalize_file_path(target, project_root)
        via_relative = normalize_file_path(target.relative_to(project_root), project_root)
        via_dot_slash = normalize_file_path(
            project_root / "." / "data" / "raw" / "vllm-0.10.1" / "vllm" / "envs.py", project_root
        )
        via_dotdot = normalize_file_path(
            corpus / "vllm" / ".." / "vllm" / "envs.py", project_root
        )

        assert absolute == "data/raw/vllm-0.10.1/vllm/envs.py"
        assert via_relative == absolute
        assert via_dot_slash == absolute
        assert via_dotdot == absolute

    def test_result_is_forward_slash_even_as_string(self, tmp_path: Path) -> None:
        corpus = _make_corpus(tmp_path)
        result = normalize_file_path(corpus / "docs" / "guide.md", tmp_path)
        assert "\\" not in result
        assert result == "vllm-0.10.1/docs/guide.md"


class TestIterCorpusFiles:
    """The directory/file allowlist controls exactly what gets indexed."""

    def test_default_allowlist_excludes_tests_includes_root_files(self, tmp_path: Path) -> None:
        corpus = _make_corpus(tmp_path)
        files = iter_corpus_files(corpus)
        rels = {str(p.relative_to(corpus)) for p in files}
        assert "vllm/envs.py" in rels
        assert "vllm/sub/mod.py" in rels
        assert "docs/guide.md" in rels
        assert "CMakeLists.txt" in rels
        assert "README.md" in rels
        assert not any(r.startswith("tests/") for r in rels)

    def test_missing_include_dir_is_skipped_not_an_error(self, tmp_path: Path) -> None:
        corpus = _make_corpus(tmp_path)
        files = iter_corpus_files(corpus, include_dirs=("does_not_exist",), include_root_files=())
        assert files == []


class TestTokenize:
    """snake_case and CamelCase are split, but the original identifier survives."""

    def test_snake_case_keeps_original_and_splits(self) -> None:
        tokens = tokenize("trust_remote_code")
        assert "trust_remote_code" in tokens
        assert "trust" in tokens
        assert "remote" in tokens
        assert "code" in tokens

    def test_camel_case_keeps_original_and_splits(self) -> None:
        tokens = tokenize("TPUModelRunner")
        assert "tpumodelrunner" in tokens
        assert "tpu" in tokens
        assert "model" in tokens
        assert "runner" in tokens

    def test_tokens_are_deduplicated_and_lowercase(self) -> None:
        tokens = tokenize("Foo foo FOO")
        assert tokens.count("foo") == 1


class TestBuildAndLoadIndex:
    """build_index writes atomically; load_index fails clearly when it can't read."""

    def test_round_trip(self, tmp_path: Path) -> None:
        corpus = _make_corpus(tmp_path / "data" / "raw")
        index_dir = tmp_path / "data" / "processed" / "idx"

        metadata = build_index(corpus_dir=corpus, index_dir=index_dir, project_root=tmp_path)
        assert metadata.n_files > 0
        assert metadata.n_chunks > 0
        assert (index_dir / "chunks.jsonl").is_file()
        assert (index_dir / "meta.json").is_file()

        loaded_metadata, chunks = load_index(index_dir)
        assert loaded_metadata.n_chunks == metadata.n_chunks
        assert all(isinstance(c, IndexedChunk) for c in chunks)
        assert all(c.file_path.startswith("data/raw/vllm-0.10.1/") for c in chunks)

    def test_rebuild_fully_replaces_old_contents(self, tmp_path: Path) -> None:
        corpus = _make_corpus(tmp_path / "data" / "raw")
        index_dir = tmp_path / "data" / "processed" / "idx"

        build_index(corpus_dir=corpus, index_dir=index_dir, project_root=tmp_path)
        _, first_chunks = load_index(index_dir)

        (corpus / "vllm" / "extra.py").write_text("y = 2\n", encoding="utf-8")
        build_index(corpus_dir=corpus, index_dir=index_dir, project_root=tmp_path)
        _, second_chunks = load_index(index_dir)

        assert len(second_chunks) > len(first_chunks)
        # no leftover temp/backup directories from the atomic swap
        leftovers = [p.name for p in index_dir.parent.iterdir() if p.name != "idx"]
        assert leftovers == []

    def test_missing_index_raises_clear_error(self, tmp_path: Path) -> None:
        with pytest.raises(IndexNotFoundError):
            load_index(tmp_path / "does_not_exist")

    def test_corrupted_chunks_file_raises_clear_error(self, tmp_path: Path) -> None:
        corpus = _make_corpus(tmp_path / "data" / "raw")
        index_dir = tmp_path / "data" / "processed" / "idx"
        build_index(corpus_dir=corpus, index_dir=index_dir, project_root=tmp_path)

        with open(index_dir / "chunks.jsonl", "a", encoding="utf-8") as f:
            f.write("{not valid json\n")

        with pytest.raises(IndexCorruptedError):
            load_index(index_dir)

    def test_corrupted_metadata_raises_clear_error(self, tmp_path: Path) -> None:
        corpus = _make_corpus(tmp_path / "data" / "raw")
        index_dir = tmp_path / "data" / "processed" / "idx"
        build_index(corpus_dir=corpus, index_dir=index_dir, project_root=tmp_path)

        (index_dir / "meta.json").write_text("not json at all", encoding="utf-8")

        with pytest.raises(IndexCorruptedError):
            load_index(index_dir)

    def test_chunks_are_valid_json_lines(self, tmp_path: Path) -> None:
        corpus = _make_corpus(tmp_path / "data" / "raw")
        index_dir = tmp_path / "data" / "processed" / "idx"
        build_index(corpus_dir=corpus, index_dir=index_dir, project_root=tmp_path)

        with open(index_dir / "chunks.jsonl", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                assert "tokens" in record
                assert "file_path" in record
