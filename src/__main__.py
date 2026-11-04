"""CLI entry point for the RAG pipeline.

Exposes six commands through Fire: index, search, search_dataset, answer,
answer_dataset, and evaluate. Run with ``uv run python -m src <command>``.
"""

import sys
from pathlib import Path

import fire
from tqdm import tqdm

from src.chunking import DEFAULT_TARGET_CHUNK_LENGTH
from src.evaluation import compute_recall
from src.indexing import (
    DEFAULT_INCLUDE_DIRS,
    DEFAULT_INCLUDE_ROOT_FILES,
    IndexCorruptedError,
    IndexNotFoundError,
    build_index,
    load_index,
)
from src.models import (
    MAX_CHUNK_LENGTH,
    AnsweredDataset,
    IndexedChunk,
    IndexMetadata,
    SearchResult,
    StudentResults,
    UnansweredDataset,
)
from src.search import build_bm25_index, chunk_to_source, search_chunks


def _as_str_tuple(value: str | tuple[str, ...]) -> tuple[str, ...]:
    """Normalize a comma-separated-string-or-tuple CLI argument to a tuple.

    Fire parses a CLI value like "vllm,docs" into an actual tuple before
    it ever reaches here, but an unchanged default argument stays a
    plain comma-joined string; this accepts either.

    Args:
        value: Either a comma-separated string or an already-parsed
            tuple of strings.

    Returns:
        A tuple of strings.
    """
    if isinstance(value, str):
        return tuple(value.split(","))
    return tuple(value)


def index(
    corpus_dir: str = "data/raw/vllm-0.10.1",
    index_dir: str = "data/processed",
    project_root: str = ".",
    include_dirs: str | tuple[str, ...] = ",".join(DEFAULT_INCLUDE_DIRS),
    include_root_files: str | tuple[str, ...] = ",".join(DEFAULT_INCLUDE_ROOT_FILES),
    target_chunk_size: int = DEFAULT_TARGET_CHUNK_LENGTH,
    max_chunk_size: int = MAX_CHUNK_LENGTH,
) -> None:
    """Build a search index from a corpus directory.

    Args:
        corpus_dir: Path to the root directory containing the source
            corpus (e.g. data/raw/vllm-0.10.1).
        index_dir: Path to the directory where the built index will be
            written.
        project_root: Directory that stored file_path values are made
            relative to, regardless of how corpus_dir was spelled.
        include_dirs: Comma-separated top-level subdirectories of
            corpus_dir to index (.py and .md files only).
        include_root_files: Comma-separated filenames directly under
            corpus_dir to index regardless of extension.
        target_chunk_size: Soft preferred chunk size for grouping
            decisions.
        max_chunk_size: Hard ceiling in characters for every chunk.
    """
    metadata = build_index(
        corpus_dir=Path(corpus_dir),
        index_dir=Path(index_dir),
        project_root=Path(project_root),
        include_dirs=_as_str_tuple(include_dirs),
        include_root_files=_as_str_tuple(include_root_files),
        target_chunk_size=target_chunk_size,
        max_chunk_size=max_chunk_size,
    )
    print(metadata.model_dump_json(indent=2))


def _load_index_or_exit(index_dir: str) -> tuple[IndexMetadata, list[IndexedChunk]]:
    """Load an index, exiting with a clear message instead of a traceback.

    Args:
        index_dir: Path to a previously built index directory.

    Returns:
        The index's metadata and its list of chunks.
    """
    try:
        return load_index(Path(index_dir))
    except (IndexNotFoundError, IndexCorruptedError) as e:
        sys.exit(str(e))


def search(query: str, index_dir: str = "data/processed", k: int = 10) -> None:
    """Search the index for a single query and print the retrieved sources.

    Args:
        query: The natural language query to search for.
        index_dir: Path to a previously built index directory.
        k: Maximum number of sources to retrieve.
    """
    _, chunks = _load_index_or_exit(index_dir)
    bm25 = build_bm25_index(chunks)
    for i, chunk in enumerate(search_chunks(query, chunks, bm25, k), start=1):
        print(
            f"[{i}] {chunk.file_path} "
            f"[{chunk.first_character_index}:{chunk.last_character_index}] "
            f"symbol={chunk.symbol}"
        )
        print(chunk.text[:200].replace("\n", " "))
        print()


def search_dataset(
    dataset_path: str, save_directory: str, index_dir: str = "data/processed", k: int = 10
) -> None:
    """Run search for every question in a dataset and write the results.

    The output file is named after dataset_path's basename, so callers
    must scope save_directory per dataset (e.g. AnsweredQuestions vs.
    UnansweredQuestions) to avoid overwriting one dataset's results with
    another's.

    Args:
        dataset_path: Path to a dataset JSON file (AnsweredQuestions or
            UnansweredQuestions format).
        save_directory: Directory the student search results JSON will
            be written into.
        index_dir: Path to a previously built index directory.
        k: Maximum number of sources to retrieve per question.
    """
    _, chunks = _load_index_or_exit(index_dir)
    bm25 = build_bm25_index(chunks)

    try:
        dataset = UnansweredDataset.model_validate_json(
            Path(dataset_path).read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as e:
        sys.exit(f"Could not read dataset at '{dataset_path}': {e}")

    results = []
    for q in tqdm(dataset.rag_questions, desc="searching", unit="question"):
        retrieved = search_chunks(q.question, chunks, bm25, k)
        sources = [chunk_to_source(c) for c in retrieved]
        results.append(
            SearchResult(question_id=q.question_id, question=q.question, retrieved_sources=sources)
        )

    student_results = StudentResults(search_results=results, k=k)
    output_path = Path(save_directory) / Path(dataset_path).name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(student_results.model_dump_json(indent=2), encoding="utf-8")
    print(f"Saved student_search_results to {output_path}")


def answer(query: str, index_dir: str = "data/processed", k: int = 10) -> None:
    """Retrieve context for a single query and generate an answer.

    Args:
        query: The natural language question to answer.
        index_dir: Path to a previously built index directory.
        k: Maximum number of sources to retrieve as context.
    """
    raise NotImplementedError


def answer_dataset(
    dataset_path: str, output_path: str, index_dir: str = "data/processed", k: int = 10
) -> None:
    """Generate answers for every question in a dataset and write them.

    Args:
        dataset_path: Path to a dataset JSON file (AnsweredQuestions or
            UnansweredQuestions format).
        output_path: Path where the generated answers JSON will be
            written.
        index_dir: Path to a previously built index directory.
        k: Maximum number of sources to retrieve as context per question.
    """
    raise NotImplementedError


def evaluate(predictions_path: str, dataset_path: str, k: int = 10) -> None:
    """Evaluate predicted search results against a ground-truth dataset.

    Does not import or call moulinette; metrics are computed locally.

    Args:
        predictions_path: Path to a student search results JSON file, as
            produced by search_dataset.
        dataset_path: Path to the ground-truth AnsweredQuestions dataset
            JSON file.
        k: Maximum number of sources per question considered during
            evaluation.
    """
    try:
        predictions = StudentResults.model_validate_json(
            Path(predictions_path).read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as e:
        sys.exit(f"Could not read predictions at '{predictions_path}': {e}")
    try:
        dataset = AnsweredDataset.model_validate_json(
            Path(dataset_path).read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as e:
        sys.exit(f"Could not read dataset at '{dataset_path}': {e}")

    metrics = compute_recall(predictions, dataset, k)
    print(metrics.model_dump_json(indent=2))


def main() -> None:
    """Dispatch to one of the six RAG pipeline subcommands via Fire."""
    fire.Fire(
        {
            "index": index,
            "search": search,
            "search_dataset": search_dataset,
            "answer": answer,
            "answer_dataset": answer_dataset,
            "evaluate": evaluate,
        }
    )


if __name__ == "__main__":
    main()
