"""Corpus walking, path normalization, tokenization, and index build/load.

`build_index` walks an allowlist of directories/files under a corpus
root, chunks and tokenizes every file, and atomically writes the result
to an index directory. `load_index` reads it back, failing with a clear
message (not a traceback) if the index is missing or corrupted.
"""

import os
import re
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

from src.chunking import DEFAULT_TARGET_CHUNK_LENGTH, chunk_text
from src.models import MAX_CHUNK_LENGTH, IndexedChunk, IndexMetadata

DEFAULT_INCLUDE_DIRS = ("vllm", "docs")
DEFAULT_INCLUDE_ROOT_FILES = (
    "CMakeLists.txt",
    "RELEASE.md",
    "SECURITY.md",
    "README.md",
    "CONTRIBUTING.md",
    "CODE_OF_CONDUCT.md",
)
TIME_BUDGET_SECONDS = 300.0

_IDENT_RE = re.compile(r"[A-Za-z0-9_]+")
_CAMEL_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+(?![a-z])|[0-9]+")


class IndexNotFoundError(RuntimeError):
    """Raised when index_dir does not contain a built index."""


class IndexCorruptedError(RuntimeError):
    """Raised when index_dir's contents exist but cannot be parsed."""


def tokenize(text: str) -> list[str]:
    """Tokenize text for BM25, splitting snake_case/CamelCase but keeping originals.

    Each identifier-like run (letters, digits, underscores) is emitted
    whole, then also split on underscores and on CamelCase boundaries,
    with every resulting piece emitted too. This lets a query for either
    "trust_remote_code" or "remote code" match the same document.

    Args:
        text: Arbitrary text: a file path, a chunk's symbol, or its body.

    Returns:
        A de-duplicated, order-preserving list of lowercase tokens.
    """
    tokens: list[str] = []
    seen: set[str] = set()

    def add(tok: str) -> None:
        low = tok.lower()
        if low and low not in seen:
            seen.add(low)
            tokens.append(low)

    for raw in _IDENT_RE.findall(text):
        add(raw)
        for part in raw.split("_"):
            if not part:
                continue
            add(part)
            sub_parts = _CAMEL_RE.findall(part)
            if len(sub_parts) > 1:
                for sub in sub_parts:
                    add(sub)
    return tokens


def normalize_file_path(path: Path, project_root: Path) -> str:
    """Normalize a file path to a project-root-relative, forward-slash path.

    This makes file_path independent of how corpus_dir was spelled on
    the command line (absolute, relative, with "./" or ".." segments,
    trailing slash, ...): the result always matches the corpus path
    convention used throughout the dataset (e.g.
    "data/raw/vllm-0.10.1/vllm/envs.py").

    A relative `path` is anchored to `project_root`, not to the
    process's current working directory, so the result does not depend
    on where the command happened to be invoked from.

    Args:
        path: The file's path, absolute or relative, in any form.
        project_root: The directory file_path values should be relative
            to (normally the repository root).

    Returns:
        A POSIX-style path relative to project_root.
    """
    if not path.is_absolute():
        path = project_root / path
    return path.resolve().relative_to(project_root.resolve()).as_posix()


def iter_corpus_files(
    corpus_dir: Path,
    include_dirs: tuple[str, ...] = DEFAULT_INCLUDE_DIRS,
    include_root_files: tuple[str, ...] = DEFAULT_INCLUDE_ROOT_FILES,
) -> list[Path]:
    """List the corpus files to index, per the directory/file allowlist.

    Args:
        corpus_dir: Root directory of the corpus (e.g. the vLLM checkout).
        include_dirs: Top-level subdirectory names to walk recursively
            for .py and .md files.
        include_root_files: Specific filenames directly under corpus_dir
            to include regardless of extension.

    Returns:
        A sorted, de-duplicated list of file paths to index.
    """
    files: set[Path] = set()
    for dirname in include_dirs:
        base = corpus_dir / dirname
        if not base.is_dir():
            continue
        files.update(p for p in base.rglob("*.py") if p.is_file())
        files.update(p for p in base.rglob("*.md") if p.is_file())
    for filename in include_root_files:
        candidate = corpus_dir / filename
        if candidate.is_file():
            files.add(candidate)
    return sorted(files)


def build_search_text(file_path: str, symbol: str | None, text: str) -> str:
    """Build the raw text handed to tokenize() for a chunk's BM25 document.

    Args:
        file_path: The chunk's project-root-relative file path.
        symbol: The chunk's optional qualified name / heading breadcrumb.
        text: The chunk's exact source text.

    Returns:
        A single string combining path, symbol, and body, in that order.
    """
    path_words = file_path.replace("/", " ").replace(".", " ")
    return f"{path_words} {symbol or ''} {text}"


def build_index(
    corpus_dir: Path,
    index_dir: Path,
    project_root: Path,
    include_dirs: tuple[str, ...] = DEFAULT_INCLUDE_DIRS,
    include_root_files: tuple[str, ...] = DEFAULT_INCLUDE_ROOT_FILES,
    target_chunk_size: int = DEFAULT_TARGET_CHUNK_LENGTH,
    max_chunk_size: int = MAX_CHUNK_LENGTH,
) -> IndexMetadata:
    """Walk the corpus, chunk and tokenize every file, and persist the index.

    Builds the whole index in a temporary directory first, then swaps it
    into place with two atomic directory renames, so a crash mid-build
    (or a concurrent reader) never observes a half-written index.

    Args:
        corpus_dir: Root directory of the corpus.
        index_dir: Directory to write chunks.jsonl and meta.json into.
        project_root: Directory that stored file_path values are
            relative to (normally the repository root).
        include_dirs: Top-level subdirectories to index.
        include_root_files: Root-level filenames to index regardless of
            extension.
        target_chunk_size: Soft preferred chunk size passed to
            chunk_text.
        max_chunk_size: Hard ceiling passed to chunk_text.

    Returns:
        The IndexMetadata describing the completed build.
    """
    start = time.perf_counter()
    files = iter_corpus_files(corpus_dir, include_dirs, include_root_files)

    index_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix=".index_build_", dir=index_dir.parent))
    try:
        n_chunks = 0
        with open(tmp_dir / "chunks.jsonl", "w", encoding="utf-8") as out:
            for path in tqdm(files, desc="indexing", unit="file"):
                try:
                    with open(path, encoding="utf-8", newline="") as f:
                        text = f.read()
                except UnicodeDecodeError:
                    tqdm.write(f"warning: skipping non-UTF-8 file: {path}")
                    continue
                file_path = normalize_file_path(path, project_root)
                for chunk in chunk_text(text, file_path, target_chunk_size, max_chunk_size):
                    search_text = build_search_text(chunk.file_path, chunk.symbol, chunk.text)
                    indexed = IndexedChunk(**chunk.model_dump(), tokens=tokenize(search_text))
                    out.write(indexed.model_dump_json() + "\n")
                    n_chunks += 1

        elapsed = time.perf_counter() - start
        if elapsed > TIME_BUDGET_SECONDS:
            tqdm.write(
                f"warning: indexing took {elapsed:.1f}s, "
                f"over the {TIME_BUDGET_SECONDS:.0f}s budget"
            )

        metadata = IndexMetadata(
            corpus_dir=str(corpus_dir),
            index_dir=str(index_dir),
            target_chunk_size=target_chunk_size,
            max_chunk_size=max_chunk_size,
            n_files=len(files),
            n_chunks=n_chunks,
            elapsed_seconds=elapsed,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        (tmp_dir / "meta.json").write_text(metadata.model_dump_json(indent=2), encoding="utf-8")

        old_dir = None
        if index_dir.exists():
            old_dir = index_dir.with_name(index_dir.name + ".old")
            if old_dir.exists():
                shutil.rmtree(old_dir)
            os.replace(index_dir, old_dir)
        os.replace(tmp_dir, index_dir)
        if old_dir is not None:
            shutil.rmtree(old_dir, ignore_errors=True)
    except BaseException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise

    return metadata


def load_index(index_dir: Path) -> tuple[IndexMetadata, list[IndexedChunk]]:
    """Load a previously built index, failing clearly if missing or broken.

    Args:
        index_dir: Directory previously passed to build_index as
            index_dir.

    Returns:
        The index's metadata and its list of chunks.

    Raises:
        IndexNotFoundError: If index_dir, chunks.jsonl, or meta.json is
            missing.
        IndexCorruptedError: If meta.json or any line of chunks.jsonl
            cannot be parsed.
    """
    meta_path = index_dir / "meta.json"
    chunks_path = index_dir / "chunks.jsonl"
    if not meta_path.is_file() or not chunks_path.is_file():
        raise IndexNotFoundError(
            f"No index found at '{index_dir}'. Run the `index` command first to build one."
        )
    try:
        metadata = IndexMetadata.model_validate_json(meta_path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as e:
        raise IndexCorruptedError(f"Could not read index metadata at '{meta_path}': {e}") from e

    chunks: list[IndexedChunk] = []
    try:
        with open(chunks_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    chunks.append(IndexedChunk.model_validate_json(line))
    except (ValueError, OSError) as e:
        raise IndexCorruptedError(f"Could not parse '{chunks_path}': {e}") from e

    return metadata, chunks
