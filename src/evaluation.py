"""Recall@k computation for search results against ground truth.

This is our own, independent implementation (it does not import or call
moulinette), so its exact matching rule may differ from moulinette's.
A question counts as a hit at rank k if any of its top-k retrieved
sources overlaps (same file, intersecting character range) any of its
ground-truth sources.
"""

from src.models import AnsweredDataset, EvaluationMetrics, Source, StudentResults

RECALL_KS = (1, 3, 5, 10)

MINIMAL_IOU_THRESHOLD = 0.05


def source_iou(a: Source, b: Source) -> float:
    """Compute the Intersection-over-Union of two sources' character ranges.

    Args:
        a: One source.
        b: Another source.

    Returns:
        0.0 if `a` and `b` are in different files; otherwise the IoU of
        their half-open [first_character_index, last_character_index)
        ranges.
    """
    raise NotImplementedError


def is_source_found(
    source: Source,
    retrieved: list[Source],
    iou_threshold: float = MINIMAL_IOU_THRESHOLD,
) -> bool:
    """Whether a ground-truth source is matched by any retrieved source.

    Args:
        source: A single ground-truth source to look for.
        retrieved: Candidate sources to search for a match.
        iou_threshold: Minimum source_iou required to count as a match.

    Returns:
        True if any source in `retrieved` has a source_iou with `source`
        that is at least `iou_threshold`.
    """
    raise NotImplementedError


def sources_overlap(a: Source, b: Source) -> bool:
    """Whether two sources are in the same file with intersecting ranges.

    Args:
        a: One source.
        b: Another source.

    Returns:
        True if file_path matches and the [first, last) ranges intersect.
    """
    if a.file_path != b.file_path:
        return False
    return (
        a.first_character_index < b.last_character_index
        and b.first_character_index < a.last_character_index
    )


def is_hit(retrieved: list[Source], gt_sources: list[Source], rank: int) -> bool:
    """Whether any of the top-`rank` retrieved sources overlaps a ground-truth source.

    Args:
        retrieved: Retrieved sources, best first.
        gt_sources: The question's ground-truth sources.
        rank: How many of the top retrieved sources to consider.

    Returns:
        True if there is a hit within the top `rank` retrieved sources.
    """
    top = retrieved[:rank]
    return any(sources_overlap(r, g) for r in top for g in gt_sources)


def compute_recall(
    predictions: StudentResults, dataset: AnsweredDataset, k: int = 10
) -> EvaluationMetrics:
    """Compute recall@1/3/5/10 of predictions against ground-truth sources.

    Args:
        predictions: Student search results, as produced by
            search_dataset.
        dataset: The ground-truth AnsweredQuestions dataset.
        k: Maximum number of retrieved sources per question to consider,
            mirroring moulinette's own `--k` option.

    Returns:
        Recall@1/3/5/10 averaged over all questions in `dataset`. A
        question missing from `predictions` counts as a miss at every
        rank.
    """
    pred_by_id = {r.question_id: r.retrieved_sources[:k] for r in predictions.search_results}

    n = len(dataset.rag_questions)
    hits = {rank: 0 for rank in RECALL_KS}
    for question in dataset.rag_questions:
        retrieved = pred_by_id.get(question.question_id, [])
        for rank in RECALL_KS:
            if is_hit(retrieved, question.sources, rank):
                hits[rank] += 1

    def rate(rank: int) -> float:
        return hits[rank] / n if n else 0.0

    return EvaluationMetrics(
        recall_at_1=rate(1),
        recall_at_3=rate(3),
        recall_at_5=rate(5),
        recall_at_10=rate(10),
    )
