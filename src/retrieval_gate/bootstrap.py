"""Paired bootstrap over cases shared by two reports.

For each metric, the observed change is the mean of current minus
baseline on cases present in both reports. Each resample draws those
case ids with replacement and takes the same mean. The interval is the
percentile interval of those means. The p-value is a two-sided test
that the mean change is zero.

``--fail-on-significant-drop`` reads the interval. It fails when the
whole interval is below ``-max_drop``. The upper bound, rounded to 6
decimal places, has to be a larger drop than the limit. A bound equal
to ``-max_drop`` passes, same as the aggregate gate.
"""

import math
import random
import statistics

from retrieval_gate.metrics import METRIC_NAMES
from retrieval_gate.report import _case_metric, _case_results, _index_cases, render_table

DEFAULT_BOOTSTRAP_CONFIDENCE = 0.95
DEFAULT_BOOTSTRAP_SEED = 0

# Same tolerance as the aggregate drop check. A difference this small
# is treated as a tie so rounded JSON values do not flap the result.
_EPSILON = 1e-9
_PLACES = 6


def paired_bootstrap(
    baseline: dict,
    current: dict,
    *,
    samples: int,
    confidence: float,
    seed: int,
    max_drop: float,
    checked: bool,
) -> dict:
    """Resample paired case deltas and summarize each metric.

    Cases are matched by id. The first row wins when an id is repeated,
    and ids are sorted before sampling so the seed does not depend on
    file order. A case missing any of the five metrics is skipped.
    Reports with no ``results`` list, and pairs with no usable cases,
    come back with ``available`` false and a ``message`` instead of
    raising.
    """

    result = {
        "available": False,
        "samples": samples,
        "confidence": confidence,
        "seed": seed,
        "max_drop": max_drop,
        "checked": checked,
    }
    paired = _paired_deltas(baseline, current)
    if not paired["available"]:
        result["message"] = paired["message"]
        return result

    columns = [paired["columns"][name] for name in METRIC_NAMES]
    draws = _resample_means(columns, samples, random.Random(seed))
    alpha = (1.0 - confidence) / 2.0
    metrics = {}
    significant = []
    for index, name in enumerate(METRIC_NAMES):
        observed = statistics.fmean(columns[index])
        ordered = sorted(draws[index])
        low = round(_percentile(ordered, alpha), _PLACES)
        high = round(_percentile(ordered, 1.0 - alpha), _PLACES)
        metrics[name] = {
            "delta": round(observed, _PLACES),
            "low": low,
            "high": high,
            "p_value": round(_two_sided_p(observed, draws[index]), _PLACES),
        }
        if _interval_regressed(high, max_drop):
            significant.append(name)
    result["available"] = True
    result["case_count"] = paired["case_count"]
    result["skipped"] = paired["skipped"]
    result["metrics"] = metrics
    result["significant"] = significant
    return result


def format_bootstrap(result: dict, *, markdown: bool) -> str:
    """Plain text or Markdown for a ``paired_bootstrap`` result."""

    lines = ["### bootstrap", ""] if markdown else ["bootstrap"]
    lines.append(_settings_line(result))
    if not result["available"]:
        lines.append(result["message"])
        return "\n".join(lines)
    if result["checked"]:
        lines.append(f"fail when the whole interval is below {-result['max_drop']:.4f}")
    headers = ["metric", "delta", "low", "high", "p-value"]
    if result["checked"]:
        headers.append("status")
    significant = set(result["significant"])
    rows = []
    for name in METRIC_NAMES:
        metric = result["metrics"][name]
        row = [
            name,
            f"{metric['delta']:+.4f}",
            f"{metric['low']:+.4f}",
            f"{metric['high']:+.4f}",
            f"{metric['p_value']:.4f}",
        ]
        if result["checked"]:
            row.append("FAIL" if name in significant else "pass")
        rows.append(row)
    if markdown:
        lines.append(_markdown_table(headers, rows, checked=result["checked"]))
    else:
        lines.append(render_table(headers, rows))
    if result["checked"] and result["significant"]:
        lines.extend(["", "significant drop: " + ", ".join(result["significant"])])
    return "\n".join(lines)


def _paired_deltas(baseline: dict, current: dict) -> dict:
    before = _case_results(baseline)
    after = _case_results(current)
    if before is None:
        return {"available": False, "message": "baseline report has no per-case results"}
    if after is None:
        return {"available": False, "message": "current report has no per-case results"}

    before_rows = _index_cases(before)
    after_rows = _index_cases(after)
    columns = {name: [] for name in METRIC_NAMES}
    skipped = 0
    for case_id in sorted(before_rows.keys() & after_rows.keys()):
        deltas = _complete_deltas(before_rows[case_id], after_rows[case_id])
        if deltas is None:
            skipped += 1
            continue
        for name in METRIC_NAMES:
            columns[name].append(deltas[name])
    if not columns["hit_rate"]:
        return {"available": False, "message": "no paired cases to resample"}
    return {
        "available": True,
        "columns": columns,
        "case_count": len(columns["hit_rate"]),
        "skipped": skipped,
    }


def _complete_deltas(before: dict, after: dict) -> dict[str, float] | None:
    deltas = {}
    for name in METRIC_NAMES:
        base = _case_metric(before, name)
        current = _case_metric(after, name)
        if base is None or current is None:
            return None
        deltas[name] = current - base
    return deltas


def _resample_means(
    columns: list[list[float]],
    samples: int,
    rng: random.Random,
) -> list[list[float]]:
    """Draw case indexes with replacement and average every metric.

    One draw of indexes is reused across metrics so the resample stays
    paired. The loop is plain Python and stays under a second for 2000
    resamples of 1000 cases.
    """

    count = len(columns[0])
    randrange = rng.randrange
    inverse = 1.0 / count
    series = tuple(columns)
    width = len(series)
    draws: list[list[float]] = [[] for _ in range(width)]
    for _ in range(samples):
        totals = [0.0] * width
        for _case in range(count):
            index = randrange(count)
            for metric in range(width):
                totals[metric] += series[metric][index]
        for metric in range(width):
            draws[metric].append(totals[metric] * inverse)
    return draws


def _percentile(sorted_values: list[float], q: float) -> float:
    """Linear percentile of an ascending list. ``q`` is from 0 to 1.

    The position is ``q * (n - 1)``. Values between two ranks are
    interpolated. This is the usual percentile interval, not a normal
    approximation.
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


def _two_sided_p(observed: float, draws: list[float]) -> float:
    """Two-sided bootstrap p-value for a mean change of zero.

    Subtracting ``observed`` from each resampled mean recenters that
    sample on zero. A recentered mean counts as extreme when it is at
    least as far from zero as the observed mean. One is added to the
    count and to the sample count so a one-sided pile of resamples
    cannot print a p-value of zero.
    """

    threshold = abs(observed)
    count = 0
    for sample in draws:
        if abs(sample - observed) + _EPSILON >= threshold:
            count += 1
    return (count + 1) / (len(draws) + 1)


def _interval_regressed(high: float, max_drop: float) -> bool:
    """Return whether the whole interval is below ``-max_drop``.

    ``high`` is the upper bound. It is the mildest value in the
    interval, so the whole interval clears the limit only when this
    bound does. Equality with ``-max_drop`` is allowed.
    """

    return (-high) - max_drop > _EPSILON


def _settings_line(result: dict) -> str:
    line = (
        f"{result['samples']} resamples, confidence {_format_confidence(result['confidence'])}, "
        f"seed {result['seed']}"
    )
    if result["available"]:
        cases = result["case_count"]
        noun = "paired case" if cases == 1 else "paired cases"
        line += f", {cases} {noun}"
        if result["skipped"]:
            line += f", {result['skipped']} skipped"
    return line


def _format_confidence(value: float) -> str:
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return text or "0"


def _markdown_table(headers: list[str], rows: list[list[str]], *, checked: bool) -> str:
    aligns = ["---", "---:", "---:", "---:", "---:"]
    if checked:
        aligns.append("---")
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(aligns) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)
