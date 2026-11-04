"""Pydantic models shared between the pipeline stages.

These models mirror the JSON formats consumed and produced by the CLI
commands in ``src.__main__``, including the ground-truth dataset format
and the student search-results / answers formats expected by moulinette.
"""

from pydantic import BaseModel, field_validator

MAX_CHUNK_LENGTH = 2000


class Source(BaseModel):
    """A reference to a contiguous character span within a corpus file."""

    file_path: str
    first_character_index: int
    last_character_index: int


class Chunk(BaseModel):
    """A single indexed chunk of corpus text.

    `symbol` is a human/embedding-friendly label (e.g. a qualified Python
    function name or a Markdown heading breadcrumb) for the chunk's
    location within its file. It never affects `first_character_index`,
    `last_character_index`, or `text`, which always mirror the exact
    source slice. BM25 search text is expected to prepend `file_path` and
    `symbol` to `text`; that concatenation happens at index-build time,
    not here.
    """

    chunk_id: str
    file_path: str
    first_character_index: int
    last_character_index: int
    text: str
    symbol: str | None = None

    @field_validator("text")
    @classmethod
    def validate_text_length(cls, value: str) -> str:
        """Ensure a chunk never exceeds the maximum allowed length.

        Args:
            value: The chunk text to validate.

        Returns:
            The validated chunk text.

        Raises:
            ValueError: If the chunk text exceeds MAX_CHUNK_LENGTH characters.
        """
        if len(value) > MAX_CHUNK_LENGTH:
            raise ValueError(
                f"Chunk text must not exceed {MAX_CHUNK_LENGTH} characters."
            )
        return value


class IndexedChunk(Chunk):
    """A Chunk augmented with its precomputed BM25 search tokens.

    `tokens` is derived from file_path + symbol + text at index-build
    time (see src.indexing.build_search_text / tokenize) and persisted
    alongside the chunk so search-time tokenization always matches what
    was actually indexed.
    """

    tokens: list[str]


class IndexMetadata(BaseModel):
    """Summary of a single index-build run, for inspection and debugging."""

    corpus_dir: str
    index_dir: str
    target_chunk_size: int
    max_chunk_size: int
    n_files: int
    n_chunks: int
    elapsed_seconds: float
    created_at: str


class AnsweredQuestion(BaseModel):
    """A ground-truth question with its answer and supporting sources."""

    question_id: str
    question: str
    answer: str
    sources: list[Source]


class UnansweredQuestion(BaseModel):
    """A question without a known answer or sources."""

    question_id: str
    question: str


class AnsweredDataset(BaseModel):
    """A dataset of ground-truth questions, answers, and sources."""

    rag_questions: list[AnsweredQuestion]


class UnansweredDataset(BaseModel):
    """A dataset of questions without ground-truth answers or sources."""

    rag_questions: list[UnansweredQuestion]


class SearchResult(BaseModel):
    """The sources retrieved by the search pipeline for a single question."""

    question_id: str
    question: str
    retrieved_sources: list[Source]


class StudentResults(BaseModel):
    """The full set of search results submitted for evaluation."""

    search_results: list[SearchResult]
    k: int


class StudentAnswer(BaseModel):
    """A generated answer together with the sources used to produce it."""

    question_id: str
    question: str
    answer: str
    sources: list[Source]


class StudentAnswers(BaseModel):
    """The full set of generated answers submitted for evaluation."""

    answers: list[StudentAnswer]


class EvaluationMetrics(BaseModel):
    """Recall metrics computed for a set of search results."""

    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    recall_at_10: float
