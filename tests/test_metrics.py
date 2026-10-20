import math

import pytest

from retrieval_gate.metrics import METRIC_NAMES, recall, score_case


def test_perfect_binary_ranking():
    scores = score_case(["a", "b"], {"a": 1, "b": 1}, k=2)
    assert scores == {name: 1.0 for name in METRIC_NAMES}


def test_graded_partial_ranking():
    # Top 3 unique ids are a, x, b. c is relevant but outside k.
    # DCG = (2^3 - 1) / log2(2) + 0 + (2^2 - 1) / log2(4) = 7 + 1.5 = 8.5
    # IDCG uses grades 3, 2, 1 = 7 + 3 / log2(3) + 0.5
    scores = score_case(["a", "x", "b", "a", "c"], {"a": 3, "b": 2, "c": 1}, k=3)
    assert scores["hit_rate"] == 1.0
    assert scores["recall"] == pytest.approx(2 / 3)
    assert scores["precision"] == pytest.approx(2 / 3)
    assert scores["mrr"] == 1.0
    idcg = 7 + 3 / math.log2(3) + 0.5
    assert scores["ndcg"] == pytest.approx(8.5 / idcg)


def test_duplicates_do_not_consume_a_slot():
    # [a, a, b] at k=2 keeps [a, b], not [a].
    scores = score_case(["a", "a", "b"], {"a": 1, "b": 1}, k=2)
    assert scores["hit_rate"] == 1.0
    assert scores["recall"] == 1.0
    assert scores["precision"] == 1.0
    assert scores["mrr"] == 1.0
    assert scores["ndcg"] == pytest.approx(1.0)


def test_duplicate_leaves_a_short_list():
    scores = score_case(["a", "a"], {"a": 1, "b": 1}, k=2)
    assert scores["hit_rate"] == 1.0
    assert scores["recall"] == 0.5
    assert scores["precision"] == 0.5
    assert scores["mrr"] == 1.0
    # DCG = 1 at rank 1. IDCG = 1 + 1 / log2(3).
    assert scores["ndcg"] == pytest.approx(1 / (1 + 1 / math.log2(3)))


def test_no_relevant_document_retrieved():
    scores = score_case(["x", "y", "x"], {"a": 1, "b": 2}, k=3)
    assert scores == {
        "hit_rate": 0.0,
        "recall": 0.0,
        "precision": 0.0,
        "mrr": 0.0,
        "ndcg": 0.0,
    }


def test_fewer_than_k_results():
    scores = score_case(["a"], {"a": 1, "b": 1}, k=3)
    assert scores["hit_rate"] == 1.0
    assert scores["recall"] == 0.5
    assert scores["precision"] == pytest.approx(1 / 3)
    assert scores["mrr"] == 1.0
    assert scores["ndcg"] == pytest.approx(1 / (1 + 1 / math.log2(3)))

    empty = score_case([], {"a": 1}, k=3)
    assert empty["hit_rate"] == 0.0
    assert empty["recall"] == 0.0
    assert empty["precision"] == 0.0
    assert empty["mrr"] == 0.0
    assert empty["ndcg"] == 0.0


def test_reciprocal_rank_uses_first_hit():
    scores = score_case(["x", "y", "a"], {"a": 1}, k=5)
    assert scores["hit_rate"] == 1.0
    assert scores["recall"] == 1.0
    assert scores["precision"] == pytest.approx(1 / 5)
    assert scores["mrr"] == pytest.approx(1 / 3)
    # Gain 1 at rank 3 is 1 / log2(4) = 0.5. Ideal DCG is 1.
    assert scores["ndcg"] == pytest.approx(0.5)


def test_hit_past_k_does_not_count():
    scores = score_case(["x", "a"], {"a": 1}, k=1)
    assert scores["hit_rate"] == 0.0
    assert scores["recall"] == 0.0
    assert scores["precision"] == 0.0
    assert scores["mrr"] == 0.0
    assert scores["ndcg"] == 0.0


def test_empty_relevant_set_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        score_case(["a"], {}, k=2)
    with pytest.raises(ValueError, match="empty"):
        recall(["a"], set())
    with pytest.raises(ValueError, match="k must be >= 1"):
        score_case(["a"], {"a": 1}, k=0)
