"""Tests for src.evaluation's IoU-based source matching.

These pin down the two real discrepancies found between our own evaluate
and moulinette's recall@5 on the public code dataset (both were questions
where the top-5 retrieved chunk barely grazes the ground-truth span):
moulinette's `list_valid_questions` rejects both at its default
`minimal_iou_threshold=0.05`, while the old any-intersection rule in
sources_overlap/is_hit accepted them. source_iou/is_source_found exist to
close that gap.
"""

import pytest

from src.evaluation import MINIMAL_IOU_THRESHOLD, is_source_found, source_iou
from src.models import Source


def _source(file_path: str, first: int, last: int) -> Source:
    """Build a Source for a given file and half-open character range."""
    return Source(file_path=file_path, first_character_index=first, last_character_index=last)


class TestSourceIou:
    """source_iou(a, b): 0.0 for different files; else IoU of the ranges."""

    def test_identical_ranges_have_iou_one(self) -> None:
        a = _source("f.py", 0, 100)
        b = _source("f.py", 0, 100)
        assert source_iou(a, b) == pytest.approx(1.0)

    def test_disjoint_ranges_have_iou_zero(self) -> None:
        a = _source("f.py", 0, 10)
        b = _source("f.py", 50, 60)
        assert source_iou(a, b) == 0.0

    def test_touching_ranges_have_iou_zero(self) -> None:
        # Half-open ranges [0, 10) and [10, 20) share no character.
        a = _source("f.py", 0, 10)
        b = _source("f.py", 10, 20)
        assert source_iou(a, b) == 0.0

    def test_different_file_is_zero_even_if_ranges_match(self) -> None:
        a = _source("a.py", 0, 100)
        b = _source("b.py", 0, 100)
        assert source_iou(a, b) == 0.0

    def test_partial_overlap(self) -> None:
        # overlap=50, union=100+100-50=150 -> IoU=1/3.
        a = _source("f.py", 0, 100)
        b = _source("f.py", 50, 150)
        assert source_iou(a, b) == pytest.approx(50 / 150)

    def test_exactly_at_threshold_boundary(self) -> None:
        # b is fully inside a: overlap=5, union=100+5-5=100 -> IoU=0.05.
        a = _source("f.py", 0, 100)
        b = _source("f.py", 95, 100)
        assert source_iou(a, b) == pytest.approx(0.05)

    def test_just_under_threshold_boundary(self) -> None:
        # overlap=4, union=100+4-4=100 -> IoU=0.04.
        a = _source("f.py", 0, 100)
        b = _source("f.py", 96, 100)
        assert source_iou(a, b) == pytest.approx(0.04)

    def test_real_near_miss_nvml_case(self) -> None:
        # Question eea9e938-...: moulinette's list_valid_questions marks
        # this INVALID at k=5 even though the ranges technically overlap.
        gt = _source("data/raw/vllm-0.10.1/vllm/third_party/pynvml.py", 82787, 83673)
        retrieved = _source("data/raw/vllm-0.10.1/vllm/third_party/pynvml.py", 83628, 83672)
        assert source_iou(gt, retrieved) == pytest.approx(44 / 886, abs=1e-6)

    def test_real_near_miss_ovisconfig_case(self) -> None:
        # Question 9495e881-...: only 2 characters of overlap.
        gt = _source(
            "data/raw/vllm-0.10.1/vllm/transformers_utils/configs/ovis.py", 3435, 4523
        )
        retrieved = _source(
            "data/raw/vllm-0.10.1/vllm/transformers_utils/configs/ovis.py", 4521, 4904
        )
        assert source_iou(gt, retrieved) == pytest.approx(2 / 1469, abs=1e-6)


class TestIsSourceFound:
    """is_source_found(source, retrieved, iou_threshold=MINIMAL_IOU_THRESHOLD)."""

    def test_true_when_one_retrieved_source_matches(self) -> None:
        gt = _source("f.py", 0, 100)
        retrieved = [_source("g.py", 0, 100), _source("f.py", 10, 110)]
        assert is_source_found(gt, retrieved) is True

    def test_false_when_no_retrieved_source_matches(self) -> None:
        gt = _source("f.py", 0, 100)
        retrieved = [_source("g.py", 0, 100), _source("f.py", 300, 400)]
        assert is_source_found(gt, retrieved) is False

    def test_false_on_empty_retrieved_list(self) -> None:
        gt = _source("f.py", 0, 100)
        assert is_source_found(gt, []) is False

    def test_respects_custom_threshold(self) -> None:
        # overlap=4, union=100 -> IoU=0.04: fails the default 0.05 bar
        # but passes a looser 0.03 bar.
        gt = _source("f.py", 0, 100)
        retrieved = [_source("f.py", 96, 100)]
        assert is_source_found(gt, retrieved) is False
        assert is_source_found(gt, retrieved, iou_threshold=0.03) is True

    def test_default_threshold_constant_is_point_zero_five(self) -> None:
        assert MINIMAL_IOU_THRESHOLD == 0.05

    def test_nvml_real_case_fails_default_threshold(self) -> None:
        gt = _source("data/raw/vllm-0.10.1/vllm/third_party/pynvml.py", 82787, 83673)
        retrieved = [_source("data/raw/vllm-0.10.1/vllm/third_party/pynvml.py", 83628, 83672)]
        assert is_source_found(gt, retrieved) is False

    def test_ovisconfig_real_case_fails_default_threshold(self) -> None:
        gt = _source(
            "data/raw/vllm-0.10.1/vllm/transformers_utils/configs/ovis.py", 3435, 4523
        )
        retrieved = [
            _source(
                "data/raw/vllm-0.10.1/vllm/transformers_utils/configs/ovis.py", 4521, 4904
            )
        ]
        assert is_source_found(gt, retrieved) is False
