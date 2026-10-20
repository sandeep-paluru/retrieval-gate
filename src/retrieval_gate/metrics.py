"""Retrieval metrics at a fixed cutoff k.

The ranking is the first k unique document ids, in the order returned.
A repeated id does not take another rank slot. Documents with grade
greater than 0 count as relevant for hit rate, recall, precision, and
MRR. nDCG uses the numeric grade. Binary labels are stored as grade 1.

An empty relevant set is rejected. Recall and nDCG are undefined when
nothing is relevant, so callers should skip or reject that case before
averaging.
"""

import math

METRIC_NAMES = ("hit_rate", "recall", "precision", "mrr", "ndcg")


def top_unique(retrieved: list[str], k: int) -> list[str]:
    """Return the first ``k`` unique ids, keeping the earliest position."""

    if k < 1:
        raise ValueError("k must be >= 1")
    seen: set[str] = set()
    ranking: list[str] = []
    for doc_id in retrieved:
        if doc_id in seen:
            continue
        seen.add(doc_id)
        ranking.append(doc_id)
        if len(ranking) == k:
            break
    return ranking


def hit_rate(ranking: list[str], relevant: set[str]) -> float:
    """Return 1.0 if ``ranking`` contains a relevant id, else 0.0.

    The reported hit rate is the mean of this value over cases. ``ranking``
    is already limited to k.
    """

    return 1.0 if any(doc_id in relevant for doc_id in ranking) else 0.0


def recall(ranking: list[str], relevant: set[str]) -> float:
    """Return recall at k.

    Recall is the number of relevant ids present in ``ranking`` divided by
    the number of relevant ids. ``ranking`` is already limited to k.
    """

    if not relevant:
        raise ValueError("relevant set is empty; recall is undefined")
    hits = sum(1 for doc_id in ranking if doc_id in relevant)
    return hits / len(relevant)


def precision(ranking: list[str], relevant: set[str], k: int) -> float:
    """Return precision at k.

    Precision is the number of relevant ids in ``ranking`` divided by k.
    The denominator stays k when fewer than k unique ids were returned, so
    the missing positions count as not relevant.
    """

    if k < 1:
        raise ValueError("k must be >= 1")
    hits = sum(1 for doc_id in ranking if doc_id in relevant)
    return hits / k


def reciprocal_rank(ranking: list[str], relevant: set[str]) -> float:
    """Return the reciprocal rank of the first relevant id in ``ranking``.

    Rank is 1-based. A hit at rank 1 scores 1.0. If ``ranking`` has no
    relevant id, the score is 0.0. The reported MRR is the mean of this
    value over cases, taken on the top k only.
    """

    for rank, doc_id in enumerate(ranking, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg(ranking: list[str], grades: dict[str, int], k: int) -> float:
    """Return normalized discounted cumulative gain at k.

    Gain at 1-based rank i is ``(2 ** grade - 1) / log2(i + 1)``. Ids
    missing from ``grades`` have grade 0. Ideal DCG sorts the positive
    grades descending and keeps at most k of them. nDCG is DCG divided by
    ideal DCG. When the ranking has no positive grade, DCG is 0 and nDCG
    is 0.
    """

    if k < 1:
        raise ValueError("k must be >= 1")
    positive = [grade for grade in grades.values() if grade > 0]
    if not positive:
        raise ValueError("relevant set is empty; nDCG is undefined")
    dcg = _dcg(grades.get(doc_id, 0) for doc_id in ranking)
    ideal = sorted(positive, reverse=True)[:k]
    ideal_dcg = _dcg(ideal)
    if ideal_dcg == 0.0:
        raise ValueError("relevant set is empty; nDCG is undefined")
    return dcg / ideal_dcg


def score_case(retrieved: list[str], grades: dict[str, int], k: int) -> dict[str, float]:
    """Score one retrieved list.

    Returns hit_rate, recall, precision, mrr, and ndcg for the first k
    unique ids. Raises ``ValueError`` when ``k < 1`` or ``grades`` has no
    positive grade.
    """

    if k < 1:
        raise ValueError("k must be >= 1")
    relevant = {doc_id for doc_id, grade in grades.items() if grade > 0}
    if not relevant:
        raise ValueError("relevant set is empty; recall and nDCG are undefined")
    ranking = top_unique(retrieved, k)
    return {
        "hit_rate": hit_rate(ranking, relevant),
        "recall": recall(ranking, relevant),
        "precision": precision(ranking, relevant, k),
        "mrr": reciprocal_rank(ranking, relevant),
        "ndcg": ndcg(ranking, grades, k),
    }


def _dcg(grades) -> float:
    total = 0.0
    for rank, grade in enumerate(grades, start=1):
        total += (2**grade - 1) / math.log2(rank + 1)
    return total
