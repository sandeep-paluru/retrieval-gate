"""Retrieval latency, in milliseconds.

Each scored retriever call is timed with ``time.perf_counter``. The
clock starts when the call is allowed to run, so time spent waiting for
a free ``--concurrency`` slot is not included. Warmup calls are not
timed.

Percentiles use linear interpolation, not nearest rank. Sort the
samples and take position ``q * (n - 1)``. A position that falls
between two samples is a weighted blend of those two. With one sample
every percentile is that sample. With two samples, p50 is the midpoint.
This is the usual inclusive percentile (Hyndman-Fan type 7), the same
linear method the bootstrap interval uses.
"""

import math
import statistics
import time

LATENCY_STATS = ("mean", "p50", "p95", "p99", "max")
PERCENTILE_METHOD = "linear"

_QUANTILES = {"p50": 0.50, "p95": 0.95, "p99": 0.99}
_PLACES = 6
# Same tolerance as the metric drop check. A gap this small is a tie.
_EPSILON = 1e-9


def clock() -> float:
    """Seconds from ``time.perf_counter``.

    Scored retriever calls read this before the call and after it
    returns. Tests replace the function so a run does not depend on
    how fast the machine is.
    """

    return time.perf_counter()


def elapsed_ms(start: float, end: float) -> float:
    """Milliseconds between two ``clock`` readings, rounded to 6 places."""

    return round((end - start) * 1000.0, _PLACES)


def percentile(sorted_values: list[float], q: float) -> float:
    """Linear percentile of an ascending list.

    ``q`` is from 0 to 1. The position is ``q * (n - 1)``. Values
    between two ranks are interpolated. Nearest rank is not used. One
    sample returns that sample. Two samples at ``q`` 0.5 return the
    midpoint.
    """

    last = len(sorted_values) - 1
    if last <= 0:
        return sorted_values[0]
    position = q * last
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return sorted_values[low]
    weight = position - low
    return sorted_values[low] * (1.0 - weight) + sorted_values[high] * weight


def summarize(samples_ms: list[float]) -> dict[str, float]:
    """Mean, p50, p95, p99, and max for one latency sample per case.

    Each stat is rounded to 6 decimal places. Percentiles are the
    linear interpolation described on ``percentile``.
    """

    if not samples_ms:
        raise ValueError("latency needs at least one sample")
    ordered = sorted(samples_ms)
    raw = {
        "mean": statistics.fmean(samples_ms),
        "max": ordered[-1],
    }
    for name, q in _QUANTILES.items():
        raw[name] = percentile(ordered, q)
    return {name: round(raw[name], _PLACES) for name in LATENCY_STATS}


def exceeded(stats: dict[str, float], thresholds: dict[str, float]) -> list[str]:
    """Stats strictly above their limit, in report order.

    A stat equal to its limit passes. The comparison ignores a gap of
    1e-9 or less.
    """

    failed = []
    for name in LATENCY_STATS:
        limit = thresholds.get(name)
        if limit is None:
            continue
        if stats[name] - float(limit) > _EPSILON:
            failed.append(name)
    return failed


def increase_exceeded(baseline: float, current: float, percent: float) -> bool:
    """Whether ``current`` is more than ``percent`` percent above ``baseline``.

    An increase equal to the limit passes. A baseline of 0 fails when
    current is above 0, because that increase has no finite percent.
    """

    allowed = baseline * (1.0 + percent / 100.0)
    return current - allowed > _EPSILON


def read_stats(raw: object) -> dict[str, float] | None:
    """Return the five stats, or None when ``raw`` is not a latency object."""

    if not isinstance(raw, dict):
        return None
    stats: dict[str, float] = {}
    for name in LATENCY_STATS:
        value = raw.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        number = float(value)
        if not math.isfinite(number):
            return None
        stats[name] = number
    return stats


def read_block(report: dict) -> dict | None:
    """Stats, thresholds, failures, and warmup from a report, if usable."""

    raw = report.get("latency")
    stats = read_stats(raw)
    if stats is None or not isinstance(raw, dict):
        return None
    thresholds: dict[str, float] = {}
    raw_limits = raw.get("thresholds")
    if isinstance(raw_limits, dict):
        for name in LATENCY_STATS:
            if name not in raw_limits:
                continue
            value = raw_limits[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            number = float(value)
            if math.isfinite(number):
                thresholds[name] = number
    failed: list[str] = []
    raw_failed = raw.get("failed")
    if isinstance(raw_failed, list):
        failed = [name for name in LATENCY_STATS if name in raw_failed]
    warmup = raw.get("warmup")
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        warmup = 0
    return {
        "stats": stats,
        "thresholds": thresholds,
        "failed": failed,
        "warmup": warmup,
    }


def comparison(
    baseline: dict,
    current: dict,
    *,
    stat: str,
    max_increase: float | None,
) -> dict:
    """Baseline vs current latency, or a note when one side has none.

    Nothing is shown when neither report has latency and no increase
    check was requested, so an older compare keeps its old text.
    A requested check against a baseline with no latency is a note and
    not a failure.
    """

    before = read_stats(baseline.get("latency"))
    after = read_stats(current.get("latency"))
    checked = max_increase is not None
    empty = {
        "show": False,
        "available": False,
        "message": None,
        "baseline": None,
        "current": None,
        "delta": None,
        "stat": stat,
        "max_increase": max_increase,
        "checked": checked,
        "failed": [],
    }
    if before is None and after is None:
        if not checked:
            return empty
        empty["show"] = True
        empty["message"] = "baseline report has no latency"
        return empty
    if before is None or after is None:
        empty["show"] = True
        if before is None:
            empty["message"] = "baseline report has no latency"
        else:
            empty["message"] = "current report has no latency"
        return empty
    delta = {name: round(after[name] - before[name], _PLACES) for name in LATENCY_STATS}
    failed: list[str] = []
    if checked and increase_exceeded(before[stat], after[stat], max_increase):
        failed.append(stat)
    return {
        "show": True,
        "available": True,
        "message": None,
        "baseline": before,
        "current": after,
        "delta": delta,
        "stat": stat,
        "max_increase": max_increase,
        "checked": checked,
        "failed": failed,
    }
