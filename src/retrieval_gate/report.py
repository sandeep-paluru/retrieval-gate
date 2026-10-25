"""Run a dataset, format the report, and compare two reports."""

import asyncio
import json
import math
from collections.abc import Callable
from pathlib import Path

from retrieval_gate.errors import ConfigError, DatasetError
from retrieval_gate.metrics import METRIC_NAMES, score_at_cutoffs, top_unique
from retrieval_gate.models import Case
from retrieval_gate.retriever import call_retriever

_MISS_LIMIT = 10
_QUERY_LIMIT = 60
_DELTA_EPSILON = 1e-9


def build_report(
    cases: list[Case],
    retriever: Callable,
    k: int,
    *,
    dataset: str,
    retriever_name: str,
    thresholds: dict[str, float],
    tag_thresholds: dict[str, dict[str, float]] | None = None,
    concurrency: int = 1,
    k_values: list[int] | None = None,
    thresholds_by_k: dict[int, dict[str, float]] | None = None,
) -> dict:
    """Score every case and average the metrics.

    Per-case metrics are rounded to 6 decimal places. The aggregate is the
    mean of the unrounded scores, then rounded the same way. A threshold
    fails when the aggregate is strictly below the limit.

    When ``k_values`` lists more than one cutoff, the retriever is called
    once per query at the largest value. Each cutoff scores the prefix of
    that list. The top-level ``k``, ``metrics``, ``results``, and tag
    table stay on the largest cutoff, so a single-k report keeps its old
    shape. ``cutoffs`` holds one aggregate per k. Pass ``thresholds_by_k``
    to gate a cutoff other than the largest. A bare ``thresholds`` mapping
    applies to the largest cutoff. Tag limits always use that same largest
    cutoff.

    When any case has tags, the report also stores one mean per tag at
    the largest cutoff. A case is counted in each of its tags. Cases with
    no tags are left out of that table. ``tag_thresholds`` maps a tag to
    metric limits and uses the same strict-below rule.

    Calls run one at a time when ``concurrency`` is 1. A larger value
    keeps that many calls in flight. Import-path runs leave it at 1.
    """

    cutoffs = _checked_cutoffs(k, k_values)
    retrieve_k = cutoffs[-1]
    multi = len(cutoffs) > 1
    if isinstance(concurrency, bool) or not isinstance(concurrency, int) or concurrency < 1:
        raise ConfigError("concurrency must be an integer >= 1")
    if tag_thresholds is None:
        tag_thresholds = {}
    limits_by_k = _limits_by_cutoff(cutoffs, thresholds, thresholds_by_k)
    _require_tagged_cases(cases, tag_thresholds)
    retrieved_lists = asyncio.run(_retrieve_all(retriever, cases, retrieve_k, concurrency))
    results = []
    raw_by_k: dict[int, list[dict[str, float]]] = {cutoff: [] for cutoff in cutoffs}
    for case, retrieved in zip(cases, retrieved_lists, strict=True):
        try:
            scored = score_at_cutoffs(retrieved, case.relevant, cutoffs)
        except ValueError as exc:
            raise DatasetError(f"case {case.id}: {exc}") from exc
        for cutoff, scores in scored.items():
            raw_by_k[cutoff].append(scores)
        ranking = top_unique(retrieved, retrieve_k)
        present = set(ranking)
        misses = [
            doc_id
            for doc_id, grade in case.relevant.items()
            if grade > 0 and doc_id not in present
        ]
        largest = scored[retrieve_k]
        results.append(
            {
                "id": case.id,
                "query": case.query,
                "tags": list(case.tags),
                "retrieved": ranking,
                "relevant": dict(case.relevant),
                "misses": misses,
                "metrics": {name: round(largest[name], 6) for name in METRIC_NAMES},
            }
        )
    metrics = _mean_metrics(raw_by_k[retrieve_k])
    headline_limits = limits_by_k[retrieve_k]
    if multi:
        failed: list[str] = []
        cutoff_rows = []
        for cutoff in cutoffs:
            means = _mean_metrics(raw_by_k[cutoff])
            limits = limits_by_k[cutoff]
            failed_here = [name for name, limit in limits.items() if means[name] < limit]
            for name in failed_here:
                failed.append(name if cutoff == retrieve_k else f"{name}@{cutoff}")
            cutoff_rows.append(
                {
                    "k": cutoff,
                    "metrics": means,
                    "thresholds": dict(limits),
                    "failed": failed_here,
                }
            )
    else:
        failed = [name for name, limit in headline_limits.items() if metrics[name] < limit]
        cutoff_rows = []
    tag_rows = _tag_rows(cases, raw_by_k[retrieve_k])
    report = {
        "k": retrieve_k,
        "dataset": dataset,
        "retriever": retriever_name,
        "case_count": len(results),
        "metrics": metrics,
        "thresholds": dict(headline_limits),
        "failed": failed,
    }
    if multi:
        report["k_values"] = list(cutoffs)
        report["cutoffs"] = cutoff_rows
    report["results"] = results
    if tag_rows:
        report["tags"] = tag_rows
    if tag_thresholds:
        report["tag_thresholds"] = _ordered_tag_thresholds(tag_thresholds)
        report["failed_tags"] = _failed_tags(tag_rows, tag_thresholds)
    return report


def _checked_cutoffs(k: int, k_values: list[int] | None) -> list[int]:
    """Return the cutoffs to score, ascending, with the largest last.

    ``k`` is that largest cutoff. ``k_values`` replaces the single cutoff
    when it is set, and ``k`` still has to be the largest entry.
    """

    if k_values is None:
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ConfigError("k must be >= 1")
        return [k]
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ConfigError("k must be >= 1")
    if not isinstance(k_values, list) or not k_values:
        raise ConfigError("k_values must list at least one integer")
    cleaned = []
    for value in k_values:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ConfigError("k_values entries must be integers >= 1")
        cleaned.append(value)
    if len(set(cleaned)) != len(cleaned):
        raise ConfigError("k_values contains a duplicate")
    ordered = sorted(cleaned)
    if k != ordered[-1]:
        raise ConfigError("k must be the largest cutoff")
    return ordered


def _limits_by_cutoff(
    cutoffs: list[int],
    thresholds: dict[str, float],
    thresholds_by_k: dict[int, dict[str, float]] | None,
) -> dict[int, dict[str, float]]:
    """Map each cutoff to the metric limits that gate it.

    Without ``thresholds_by_k``, ``thresholds`` applies to the largest
    cutoff and the smaller ones are not gated.
    """

    if thresholds_by_k is None:
        bound = {cutoff: {} for cutoff in cutoffs}
        bound[cutoffs[-1]] = dict(thresholds)
        return bound
    unknown = sorted(key for key in thresholds_by_k if key not in set(cutoffs))
    if unknown:
        names = ", ".join(str(key) for key in unknown)
        raise ConfigError(f"thresholds_by_k has a cutoff that is not being scored: {names}")
    bound = {}
    for cutoff in cutoffs:
        limits = thresholds_by_k.get(cutoff, {})
        if not isinstance(limits, dict):
            raise ConfigError("thresholds_by_k values must be metric limits")
        bound[cutoff] = dict(limits)
    return bound


def _mean_metrics(raw_scores: list[dict[str, float]]) -> dict[str, float]:
    count = len(raw_scores)
    return {
        name: round(sum(scores[name] for scores in raw_scores) / count, 6)
        for name in METRIC_NAMES
    }


def _tag_rows(cases: list[Case], raw_scores: list[dict[str, float]]) -> list[dict]:
    """Mean the unrounded case scores for each tag.

    Tags are listed alphabetically. A case with two tags counts once in
    each. The mean is rounded to 6 decimal places, same as the overall score.
    """

    grouped: dict[str, list[dict[str, float]]] = {}
    for case, scores in zip(cases, raw_scores, strict=True):
        for tag in case.tags:
            grouped.setdefault(tag, []).append(scores)
    return [
        {
            "tag": tag,
            "case_count": len(grouped[tag]),
            "metrics": _mean_metrics(grouped[tag]),
        }
        for tag in sorted(grouped)
    ]


def _ordered_tag_thresholds(
    thresholds: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    ordered: dict[str, dict[str, float]] = {}
    for tag in sorted(thresholds):
        limits = thresholds[tag]
        ordered[tag] = {name: limits[name] for name in METRIC_NAMES if name in limits}
    return ordered


def _failed_tags(rows: list[dict], thresholds: dict[str, dict[str, float]]) -> list[dict]:
    failed = []
    for row in rows:
        limits = thresholds.get(row["tag"], {})
        for name in METRIC_NAMES:
            limit = limits.get(name)
            if limit is not None and row["metrics"][name] < limit:
                failed.append({"tag": row["tag"], "metric": name})
    return failed


def _require_tagged_cases(cases: list[Case], thresholds: dict[str, dict[str, float]]) -> None:
    if not thresholds:
        return
    known = {tag for case in cases for tag in case.tags}
    missing = [tag for tag in sorted(thresholds) if tag not in known]
    if not missing:
        return
    names = ", ".join(repr(tag) for tag in missing)
    raise ConfigError(f"no cases tagged {names}")


async def _retrieve_all(
    retriever: Callable,
    cases: list[Case],
    k: int,
    concurrency: int,
) -> list[list[str]]:
    if concurrency <= 1:
        lists = []
        for case in cases:
            lists.append(await call_retriever(retriever, case.query, k, case.id))
        return lists

    semaphore = asyncio.Semaphore(concurrency)

    async def one(case: Case) -> list[str] | ConfigError:
        async with semaphore:
            try:
                return await call_retriever(retriever, case.query, k, case.id)
            except ConfigError as exc:
                return exc

    retrieved = await asyncio.gather(*(one(case) for case in cases))
    for item in retrieved:
        if isinstance(item, ConfigError):
            raise item
    return retrieved


def format_report(report: dict) -> str:
    """Plain text table plus the worst per-case misses."""

    cutoffs = _stored_cutoffs(report)
    if cutoffs is not None:
        return _format_cutoff_report(report, cutoffs)
    thresholds = report["thresholds"]
    rows = []
    for name in METRIC_NAMES:
        limit = thresholds.get(name)
        score = report["metrics"][name]
        if limit is None:
            status = "-"
            limit_text = "-"
        else:
            status = "FAIL" if score < limit else "pass"
            limit_text = f"{limit:.4f}"
        rows.append([name, f"{score:.4f}", limit_text, status])
    lines = [
        f"dataset: {report['dataset']}",
        f"retriever: {report['retriever']}",
        f"k: {report['k']}",
        f"cases: {report['case_count']}",
        "",
        render_table(["metric", "score", "threshold", "status"], rows),
    ]
    tag_table = _format_tag_scores(report)
    if tag_table:
        lines.extend(["", tag_table])
    lines.extend(["", "worst misses", _format_misses(report["results"])])
    if report["failed"]:
        lines.extend(["", "failed: " + ", ".join(report["failed"])])
    failed_tags = _failed_tag_text(report)
    if failed_tags:
        lines.extend(["", failed_tags])
    return "\n".join(lines)


def _format_cutoff_report(report: dict, cutoffs: list[dict]) -> str:
    """Text report for a run that scored more than one cutoff.

    The score table is one row per metric and one column per k. Tag
    means and worst misses are the largest cutoff. Thresholds that name
    a cutoff are listed under that table.
    """

    largest = cutoffs[-1]["k"]
    headers = ["metric", *[f"@{row['k']}" for row in cutoffs]]
    rows = [
        [name, *[f"{row['metrics'][name]:.4f}" for row in cutoffs]]
        for name in METRIC_NAMES
    ]
    lines = [
        f"dataset: {report['dataset']}",
        f"retriever: {report['retriever']}",
        "k: " + ", ".join(str(row["k"]) for row in cutoffs),
        f"cases: {report['case_count']}",
        "",
        render_table(headers, rows),
    ]
    threshold_rows = _threshold_table_rows(cutoffs)
    if threshold_rows:
        lines.extend(
            [
                "",
                "thresholds",
                render_table(
                    ["metric", "k", "threshold", "score", "status"],
                    threshold_rows,
                ),
            ]
        )
    tag_table = _format_tag_scores(report, at_k=largest)
    if tag_table:
        lines.extend(["", tag_table])
    lines.extend(["", f"worst misses at k={largest}", _format_misses(report["results"])])
    if report["failed"]:
        lines.extend(["", "failed: " + ", ".join(report["failed"])])
    failed_tags = _failed_tag_text(report)
    if failed_tags:
        lines.extend(["", failed_tags])
    return "\n".join(lines)


def _threshold_table_rows(cutoffs: list[dict]) -> list[list[str]]:
    rows = []
    for row in cutoffs:
        limits = row.get("thresholds") or {}
        if not isinstance(limits, dict):
            continue
        for name in METRIC_NAMES:
            if name not in limits:
                continue
            limit = limits[name]
            score = row["metrics"][name]
            status = "FAIL" if score < limit else "pass"
            rows.append([name, str(row["k"]), f"{float(limit):.4f}", f"{score:.4f}", status])
    return rows


def _format_tag_scores(report: dict, *, at_k: int | None = None) -> str:
    rows = _report_tags(report)
    if not rows:
        return ""
    table = [
        [row["tag"], str(row["case_count"])]
        + [f"{row['metrics'][name]:.4f}" for name in METRIC_NAMES]
        for row in rows
    ]
    rendered = render_table(["tag", "cases", *METRIC_NAMES], table)
    title = "tags" if at_k is None else f"tags at k={at_k}"
    return f"{title}\n{rendered}"


def _failed_tag_text(report: dict) -> str:
    failed = report.get("failed_tags") or []
    if not failed:
        return ""
    parts = [f"{item['tag']} {item['metric']}" for item in failed]
    return "failed tags: " + ", ".join(parts)


def _report_tags(report: dict) -> list[dict]:
    rows = report.get("tags")
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict) and row.get("tag")]


def _format_misses(results: list[dict]) -> str:
    missed = [row for row in results if row["misses"]]
    missed.sort(key=lambda row: (row["metrics"]["recall"], -len(row["misses"]), row["id"]))
    if not missed:
        return "none"
    shown = missed[:_MISS_LIMIT]
    rows = [
        [row["id"], f"{row['metrics']['recall']:.4f}", ", ".join(row["misses"])]
        for row in shown
    ]
    table = render_table(["case", "recall", "missed"], rows)
    if len(missed) > _MISS_LIMIT:
        table += f"\nshowing {_MISS_LIMIT} of {len(missed)}"
    return table


def render_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    def format_row(row: list[str]) -> str:
        padded = [cell.ljust(widths[index]) for index, cell in enumerate(row)]
        return "  ".join(padded).rstrip()

    lines = [
        format_row(headers),
        format_row(["-" * width for width in widths]),
    ]
    lines.extend(format_row(row) for row in rows)
    return "\n".join(lines)


def write_report(path: Path, report: dict) -> None:
    try:
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"could not write {path}: {exc}") from exc


def load_report(path: Path) -> dict:
    """Load a JSON report and require all five aggregate metrics."""

    if not path.is_file():
        raise ConfigError(f"report not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{path}:{exc.lineno}: invalid JSON ({exc.msg})") from exc
    except OSError as exc:
        raise ConfigError(f"could not read {path}: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("metrics"), dict):
        raise ConfigError(f"{path}: report is missing a metrics object")
    metrics = {
        name: _metric_number(path, data["metrics"], name) for name in METRIC_NAMES
    }
    loaded = dict(data)
    loaded["metrics"] = metrics
    if "cutoffs" in data and data["cutoffs"] is not None:
        loaded["cutoffs"] = _parse_cutoffs(data["cutoffs"], prefix=f"{path}: cutoffs")
    return loaded


def _metric_number(path: Path, metrics: dict, name: str, where: str | None = None) -> float:
    label = f"metrics.{name}" if where is None else f"{where}.{name}"
    if name not in metrics:
        raise ConfigError(f"{path}: {label} is missing")
    value = metrics[name]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{path}: {label} is not a number")
    number = float(value)
    if not math.isfinite(number):
        raise ConfigError(f"{path}: {label} is not a number")
    return number


def _parse_cutoffs(raw: object, *, prefix: str) -> list[dict]:
    """Require a list of at least two cutoff rows, sorted by k."""

    if not isinstance(raw, list) or len(raw) < 2:
        raise ConfigError(f"{prefix} must list at least two k values")
    rows = []
    seen: set[int] = set()
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ConfigError(f"{prefix}[{index}] must be an object")
        k = item.get("k")
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ConfigError(f"{prefix}[{index}].k must be an integer >= 1")
        if k in seen:
            raise ConfigError(f"{prefix} cutoff {k} was given twice")
        seen.add(k)
        metrics_raw = item.get("metrics")
        if not isinstance(metrics_raw, dict):
            raise ConfigError(f"{prefix}[{index}].metrics is missing")
        where = f"{prefix}[{index}].metrics"
        metrics = {}
        for name in METRIC_NAMES:
            if name not in metrics_raw:
                raise ConfigError(f"{where}.{name} is missing")
            metrics[name] = _finite_number(metrics_raw[name], f"{where}.{name}")
        row = {"k": k, "metrics": metrics}
        thresholds = item.get("thresholds")
        if isinstance(thresholds, dict):
            row["thresholds"] = dict(thresholds)
        failed = item.get("failed")
        if isinstance(failed, list):
            row["failed"] = list(failed)
        rows.append(row)
    rows.sort(key=lambda row: row["k"])
    return rows


def _finite_number(value: object, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{where} is not a number")
    number = float(value)
    if not math.isfinite(number):
        raise ConfigError(f"{where} is not a number")
    return number


def _stored_cutoffs(report: dict | None) -> list[dict] | None:
    """Return cutoff rows when a report scored several k values.

    A missing ``cutoffs`` key means a single-k report. A present list that
    is not usable raises ``ConfigError`` so compare does not fall back to
    the headline numbers.
    """

    if not isinstance(report, dict):
        return None
    raw = report.get("cutoffs", None)
    if raw is None:
        return None
    return _parse_cutoffs(raw, prefix="cutoffs")


def regressed(baseline: float, current: float, max_drop: float) -> bool:
    """Return whether ``current`` fell by more than ``max_drop``.

    A drop equal to ``max_drop`` is allowed. The comparison ignores
    differences smaller than 1e-9 so a value that was rounded into JSON
    does not flap around the limit.
    """

    return (baseline - current) - max_drop > 1e-9


def regressed_metrics(baseline: dict, current: dict, max_drop: float) -> list[str]:
    """Return metrics that fell by more than ``max_drop``, in report order.

    Two single-k reports compare the headline metrics, as before. When
    both reports have cutoff rows, a drop at any shared k is listed as
    ``name@k``. When only one report has cutoff rows, the comparison uses
    the single report's k if that cutoff was scored. No shared cutoff
    returns an empty list. The printed comparison explains that case.
    """

    plan = _cutoff_plan(baseline, current)
    if plan["mode"] == "headline":
        return [
            name
            for name in METRIC_NAMES
            if regressed(baseline["metrics"][name], current["metrics"][name], max_drop)
        ]
    if plan["mode"] == "unmatched":
        return []
    if plan["mode"] == "matched":
        before = _metrics_at(baseline, plan["k"])
        after = _metrics_at(current, plan["k"])
        return [
            name
            for name in METRIC_NAMES
            if regressed(before[name], after[name], max_drop)
        ]
    failed = []
    for cutoff in plan["shared"]:
        before = _metrics_at(baseline, cutoff)
        after = _metrics_at(current, cutoff)
        for name in METRIC_NAMES:
            if regressed(before[name], after[name], max_drop):
                failed.append(f"{name}@{cutoff}")
    return failed


def format_comparison(
    baseline_path: Path,
    current_path: Path,
    baseline: dict,
    current: dict,
    max_drop: float,
    *,
    check_tags: bool = False,
) -> tuple[str, list[str]]:
    """Return the comparison table and the metrics that regressed.

    Per-tag deltas are included when either report has a tag breakdown.
    ``check_tags`` marks a tag drop past ``max_drop`` as FAIL. The returned
    failure list stays the overall metrics, so the caller decides whether
    a tag drop changes the exit code.

    When either report scored several cutoffs, the table is one row per
    metric and shared k. A cutoff that is missing on one side is named
    and is not a failure. If the two reports share no cutoff, the text
    says so and the failure list is empty.
    """

    plan = _cutoff_plan(baseline, current)
    if plan["mode"] != "headline":
        return _format_cutoff_comparison(
            baseline_path,
            current_path,
            baseline,
            current,
            max_drop,
            plan,
            check_tags=check_tags,
        )
    failed = regressed_metrics(baseline, current, max_drop)
    failed_set = set(failed)
    rows = []
    for name in METRIC_NAMES:
        before = baseline["metrics"][name]
        after = current["metrics"][name]
        delta = after - before
        status = "FAIL" if name in failed_set else "pass"
        rows.append([name, f"{before:.4f}", f"{after:.4f}", f"{delta:+.4f}", status])
    lines = [
        f"baseline: {baseline_path}",
        f"current: {current_path}",
        f"k: {_k_text(baseline)} -> {_k_text(current)}",
        f"max drop: {max_drop:.4f}",
        "",
        render_table(["metric", "baseline", "current", "delta", "status"], rows),
    ]
    if failed:
        lines.extend(["", "regressed: " + ", ".join(failed)])
    tag_text = format_tag_diff(
        tag_diff(baseline, current, max_drop),
        markdown=False,
        check=check_tags,
    )
    if tag_text:
        lines.extend(["", tag_text])
    return "\n".join(lines), failed


def _k_text(report: dict) -> str:
    value = report.get("k")
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    return "-"


def _format_cutoff_comparison(
    baseline_path: Path,
    current_path: Path,
    baseline: dict,
    current: dict,
    max_drop: float,
    plan: dict,
    *,
    check_tags: bool,
) -> tuple[str, list[str]]:
    failed = regressed_metrics(baseline, current, max_drop)
    lines = [
        f"baseline: {baseline_path}",
        f"current: {current_path}",
        f"k: {_k_label(baseline)} -> {_k_label(current)}",
    ]
    if plan["mode"] == "matched":
        lines.append(f"compared at k: {plan['k']}")
    if plan["mode"] == "per_k":
        if plan["only_baseline"]:
            lines.append("only in baseline: " + ", ".join(str(k) for k in plan["only_baseline"]))
        if plan["only_current"]:
            lines.append("only in current: " + ", ".join(str(k) for k in plan["only_current"]))
    lines.append(f"max drop: {max_drop:.4f}")
    lines.append("")
    if plan["mode"] == "unmatched":
        lines.append(plan["message"])
    else:
        lines.append(
            render_table(
                ["metric", "k", "baseline", "current", "delta", "status"],
                _delta_rows(baseline, current, plan, max_drop),
            )
        )
    if failed:
        lines.extend(["", "regressed: " + ", ".join(failed)])
    tag_text = format_tag_diff(
        tag_diff(baseline, current, max_drop),
        markdown=False,
        check=check_tags,
    )
    if tag_text:
        lines.extend(["", tag_text])
    return "\n".join(lines), failed


def _cutoff_plan(baseline: dict, current: dict) -> dict:
    """Decide how two reports line up across cutoffs.

    ``headline`` is two single-k reports. ``per_k`` is two multi-k reports
    that share at least one cutoff. ``matched`` is one multi-k report and
    one single-k report whose k was scored. ``unmatched`` has no cutoff
    in common. The message on that last mode is what the report prints.
    """

    before = _stored_cutoffs(baseline)
    after = _stored_cutoffs(current)
    if before is None and after is None:
        return {"mode": "headline"}
    if before is not None and after is not None:
        shared = sorted({row["k"] for row in before} & {row["k"] for row in after})
        only_baseline = sorted({row["k"] for row in before} - set(shared))
        only_current = sorted({row["k"] for row in after} - set(shared))
        if not shared:
            return {
                "mode": "unmatched",
                "message": (
                    "No shared cutoff "
                    f"(baseline {_k_list(before)}, current {_k_list(after)}). "
                    "Metrics were not compared."
                ),
            }
        return {
            "mode": "per_k",
            "shared": shared,
            "only_baseline": only_baseline,
            "only_current": only_current,
        }
    if before is not None:
        return _match_one_cutoff(before, current, multi_side="baseline", single_side="current")
    return _match_one_cutoff(after, baseline, multi_side="current", single_side="baseline")


def _match_one_cutoff(
    multi_rows: list[dict],
    single: dict,
    *,
    multi_side: str,
    single_side: str,
) -> dict:
    ks = {row["k"] for row in multi_rows}
    k = _k_value(single)
    if k is not None and k in ks:
        return {"mode": "matched", "k": k}
    listed = _k_list(multi_rows)
    if k is None:
        message = (
            f"{_cap(single_side)} report has no k. "
            f"{_cap(multi_side)} cutoffs are {listed}. "
            "Metrics were not compared."
        )
    else:
        message = (
            f"{_cap(single_side)} k={k} is not one of the {multi_side} cutoffs ({listed}). "
            "Metrics were not compared."
        )
    return {"mode": "unmatched", "message": message}


def _cap(text: str) -> str:
    if not text:
        return text
    return text[0].upper() + text[1:]


def _k_list(rows: list[dict]) -> str:
    return ", ".join(str(row["k"]) for row in rows)


def _k_label(report: dict) -> str:
    rows = _stored_cutoffs(report)
    if rows is not None:
        return _k_list(rows)
    return _k_text(report)


def _metrics_at(report: dict, k: int) -> dict[str, float]:
    rows = _stored_cutoffs(report)
    if rows is not None:
        for row in rows:
            if row["k"] == k:
                return row["metrics"]
    return report["metrics"]


def _delta_rows(baseline: dict, current: dict, plan: dict, max_drop: float) -> list[list[str]]:
    if plan["mode"] == "matched":
        ks = [plan["k"]]
    else:
        ks = plan["shared"]
    rows = []
    for cutoff in ks:
        before = _metrics_at(baseline, cutoff)
        after = _metrics_at(current, cutoff)
        for name in METRIC_NAMES:
            delta = after[name] - before[name]
            status = "FAIL" if regressed(before[name], after[name], max_drop) else "pass"
            rows.append(
                [
                    name,
                    str(cutoff),
                    f"{before[name]:.4f}",
                    f"{after[name]:.4f}",
                    f"{delta:+.4f}",
                    status,
                ]
            )
    return rows


def format_markdown(
    report: dict,
    *,
    baseline: dict | None = None,
    max_drop: float = 0.02,
    baseline_path: str | None = None,
    current_path: str | None = None,
    check_thresholds: bool = True,
    check_tags: bool = False,
) -> str:
    """Return a compact Markdown table for a pull request or job summary.

    Columns are the metric, score, threshold, baseline, delta, and status.
    When ``baseline`` is omitted, the baseline and delta cells are dashes
    and no metric is marked as a regression. A threshold fails when the
    score is strictly below the limit. A regression fails when the score
    drops by more than ``max_drop``. ``check_thresholds`` can be turned
    off by ``compare``, which still prints stored limits but gates only
    on the baseline drop. When the report has tags, a per-tag table is
    added. If the baseline has tags too, that table shows deltas.
    ``check_tags`` marks a tag drop past ``max_drop`` as FAIL. It does
    not by itself decide the process exit code.

    A report with several cutoffs gets a metric-by-k table. A baseline
    that also has cutoffs adds a delta for each shared k. If only one
    side has cutoffs, the delta is at the single report's k when that
    cutoff was scored. Otherwise the report says the metrics were not
    compared.
    """

    if _stored_cutoffs(report) is not None or (
        baseline is not None and _stored_cutoffs(baseline) is not None
    ):
        return _format_markdown_cutoffs(
            report,
            baseline=baseline,
            max_drop=max_drop,
            baseline_path=baseline_path,
            current_path=current_path,
            check_thresholds=check_thresholds,
            check_tags=check_tags,
        )

    thresholds = report.get("thresholds") or {}
    dropped = regressed_metrics(baseline, report, max_drop) if baseline is not None else []
    dropped_set = set(dropped)
    failed: list[str] = []
    rows: list[list[str]] = []
    for name in METRIC_NAMES:
        score = report["metrics"][name]
        limit = thresholds.get(name)
        if baseline is None:
            base_text = "-"
            delta_text = "-"
            drop_failed = False
        else:
            before = baseline["metrics"][name]
            base_text = f"{before:.4f}"
            delta_text = f"{score - before:+.4f}"
            drop_failed = name in dropped_set
        threshold_failed = bool(check_thresholds and limit is not None and score < limit)
        if threshold_failed:
            failed.append(name)
        if threshold_failed or drop_failed:
            status = "FAIL"
        elif (check_thresholds and limit is not None) or baseline is not None:
            status = "pass"
        else:
            status = "-"
        limit_text = "-" if limit is None else f"{limit:.4f}"
        rows.append([name, f"{score:.4f}", limit_text, base_text, delta_text, status])

    lines = ["## retrieval-gate"]
    context = _markdown_context(report, baseline, max_drop, baseline_path, current_path)
    if context:
        lines.append("")
        lines.extend(context)
    lines.extend(["", _markdown_table(rows)])
    notes = []
    if failed:
        notes.append("failed: " + ", ".join(failed))
    if dropped:
        notes.append("regressed: " + ", ".join(dropped))
    if notes:
        lines.append("")
        lines.extend(notes)
    tag_text = _markdown_tag_block(
        report,
        baseline,
        max_drop,
        check_thresholds=check_thresholds,
        check_tags=check_tags,
    )
    if tag_text:
        lines.extend(["", tag_text])
    return "\n".join(lines)


def _format_markdown_cutoffs(
    report: dict,
    *,
    baseline: dict | None,
    max_drop: float,
    baseline_path: str | None,
    current_path: str | None,
    check_thresholds: bool,
    check_tags: bool,
) -> str:
    plan = _cutoff_plan(baseline, report) if baseline is not None else None
    lines = ["## retrieval-gate"]
    context = _markdown_context(report, baseline, max_drop, baseline_path, current_path)
    if plan is not None and plan["mode"] == "matched":
        context.append(f"compared at k: {plan['k']}")
    if plan is not None and plan["mode"] == "per_k":
        if plan["only_baseline"]:
            context.append(
                "only in baseline: " + ", ".join(str(k) for k in plan["only_baseline"])
            )
        if plan["only_current"]:
            context.append("only in current: " + ", ".join(str(k) for k in plan["only_current"]))
    if context:
        lines.append("")
        lines.extend(context)

    blocks: list[str] = []
    current_rows = _stored_cutoffs(report)
    if current_rows is not None:
        blocks.append(_markdown_score_table(current_rows))
        if check_thresholds:
            threshold = _markdown_threshold_table(current_rows)
            if threshold:
                blocks.append(threshold)
    elif plan is not None and plan["mode"] == "unmatched":
        blocks.append(_markdown_single_scores(report, check_thresholds=check_thresholds))
    if plan is not None and plan["mode"] in {"per_k", "matched"}:
        blocks.append(_markdown_delta_table(baseline, report, plan, max_drop))
    if plan is not None and plan["mode"] == "unmatched":
        blocks.append(plan["message"])

    notes = []
    if check_thresholds and report.get("failed"):
        notes.append("failed: " + ", ".join(report["failed"]))
    if baseline is not None:
        dropped = regressed_metrics(baseline, report, max_drop)
        if dropped:
            notes.append("regressed: " + ", ".join(dropped))
    if notes:
        blocks.append("\n".join(notes))
    if blocks:
        lines.append("")
        lines.append("\n\n".join(blocks))

    at_k = current_rows[-1]["k"] if current_rows is not None else None
    tag_text = _markdown_tag_block(
        report,
        baseline,
        max_drop,
        check_thresholds=check_thresholds,
        check_tags=check_tags,
        at_k=at_k,
    )
    if tag_text:
        lines.extend(["", tag_text])
    return "\n".join(lines)


def _markdown_score_table(cutoffs: list[dict]) -> str:
    headers = ["metric", *[f"@{row['k']}" for row in cutoffs]]
    rows = [
        [name, *[f"{row['metrics'][name]:.4f}" for row in cutoffs]]
        for name in METRIC_NAMES
    ]
    return _markdown_grid(headers, rows, numeric_from=1)


def _markdown_threshold_table(cutoffs: list[dict]) -> str:
    rows = _threshold_table_rows(cutoffs)
    if not rows:
        return ""
    return _markdown_grid(
        ["metric", "k", "threshold", "score", "status"],
        rows,
        numeric_from=1,
        status_at=4,
    )


def _markdown_delta_table(
    baseline: dict,
    current: dict,
    plan: dict,
    max_drop: float,
) -> str:
    return _markdown_grid(
        ["metric", "k", "baseline", "current", "delta", "status"],
        _delta_rows(baseline, current, plan, max_drop),
        numeric_from=1,
        status_at=5,
    )


def _markdown_single_scores(report: dict, *, check_thresholds: bool) -> str:
    thresholds = report.get("thresholds") or {}
    rows = []
    for name in METRIC_NAMES:
        score = report["metrics"][name]
        limit = thresholds.get(name)
        if check_thresholds and limit is not None:
            status = "FAIL" if score < limit else "pass"
        elif check_thresholds and limit is None:
            status = "-"
        else:
            status = "-"
        limit_text = "-" if limit is None else f"{float(limit):.4f}"
        rows.append([name, f"{score:.4f}", limit_text, status])
    return _markdown_grid(
        ["metric", "score", "threshold", "status"],
        rows,
        numeric_from=1,
        status_at=3,
    )


def _markdown_grid(
    headers: list[str],
    rows: list[list[str]],
    *,
    numeric_from: int,
    status_at: int | None = None,
) -> str:
    aligns = []
    for index in range(len(headers)):
        if status_at is not None and index == status_at:
            aligns.append("---")
        elif index >= numeric_from:
            aligns.append("---:")
        else:
            aligns.append("---")
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(aligns) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _markdown_tag_block(
    report: dict,
    baseline: dict | None,
    max_drop: float,
    *,
    check_thresholds: bool,
    check_tags: bool,
    at_k: int | None = None,
) -> str:
    if baseline is None:
        block = _format_tag_scores_markdown(report, at_k=at_k)
    else:
        diff = tag_diff(baseline, report, max_drop)
        # A run still shows the scores it just computed when the baseline
        # predates per-tag results. compare only prints the note.
        if (
            check_thresholds
            and diff.get("message") == "baseline report has no per-tag results"
            and _report_tags(report)
        ):
            block = _format_tag_scores_markdown(report, at_k=at_k)
            if block:
                block = f"{block}\n\n{diff['message']}"
        else:
            block = format_tag_diff(diff, markdown=True, check=check_tags)
    if check_thresholds:
        failed = _failed_tag_text(report)
        if failed:
            block = f"{block}\n\n{failed}" if block else failed
    return block


def _format_tag_scores_markdown(report: dict, *, at_k: int | None = None) -> str:
    rows = _report_tags(report)
    if not rows:
        return ""
    headers = ["tag", "cases", *METRIC_NAMES]
    aligns = ["---", "---:", *["---:"] * len(METRIC_NAMES)]
    heading = "### tags" if at_k is None else f"### tags at k={at_k}"
    lines = [
        heading,
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(aligns) + " |",
    ]
    for row in rows:
        cells = [_markdown_cell(str(row["tag"])), str(row["case_count"])]
        cells.extend(f"{row['metrics'][name]:.4f}" for name in METRIC_NAMES)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _markdown_context(
    report: dict,
    baseline: dict | None,
    max_drop: float,
    baseline_path: str | None,
    current_path: str | None,
) -> list[str]:
    lines = []
    dataset = report.get("dataset")
    if isinstance(dataset, str) and dataset:
        lines.append(f"dataset: {dataset}")
    retriever = report.get("retriever")
    if isinstance(retriever, str) and retriever:
        lines.append(f"retriever: {retriever}")
    rows = _stored_cutoffs(report)
    if rows is not None:
        lines.append("k: " + _k_list(rows))
    else:
        k = report.get("k")
        if isinstance(k, int) and not isinstance(k, bool):
            lines.append(f"k: {k}")
    count = report.get("case_count")
    if isinstance(count, int) and not isinstance(count, bool):
        lines.append(f"cases: {count}")
    if baseline_path:
        lines.append(f"baseline: {baseline_path}")
    if current_path:
        lines.append(f"current: {current_path}")
    if baseline is not None:
        lines.append(f"max drop: {max_drop:.4f}")
    return lines


def _markdown_table(rows: list[list[str]]) -> str:
    headers = ["metric", "score", "threshold", "baseline", "delta", "status"]
    aligns = ["---", "---:", "---:", "---:", "---:", "---"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(aligns) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def case_diff(baseline: dict, current: dict) -> dict:
    """Match cases by id and list the ones that got worse.

    A case is worse when recall dropped, nDCG dropped, or it went from a
    hit to a miss. The worse list is sorted largest drop first: the bigger
    of the recall drop and the nDCG drop, then recall, then nDCG, then id.
    A drop of 1e-9 or less is ignored.

    Cases that exist on only one side come back as ``added`` and
    ``removed`` ids. They are not regressions. The first row wins when an
    id is repeated.

    Older reports have no ``results`` list. Then ``available`` is false
    and ``message`` says which side is missing, so callers can print that
    instead of failing.
    """

    before = _case_results(baseline)
    after = _case_results(current)
    empty = {
        "available": False,
        "message": None,
        "worse": [],
        "added": [],
        "removed": [],
    }
    if before is None:
        empty["message"] = "baseline report has no per-case results"
        return empty
    if after is None:
        empty["message"] = "current report has no per-case results"
        return empty

    before_rows = _index_cases(before)
    after_rows = _index_cases(after)
    worse = []
    for case_id in before_rows.keys() & after_rows.keys():
        row = _worse_case(before_rows[case_id], after_rows[case_id], case_id)
        if row is not None:
            worse.append(row)
    worse.sort(key=_worse_sort_key)
    return {
        "available": True,
        "message": None,
        "worse": worse,
        "added": sorted(after_rows.keys() - before_rows.keys()),
        "removed": sorted(before_rows.keys() - after_rows.keys()),
    }


def format_case_diff(diff: dict, limit: int) -> str:
    """Plain text section for the cases that got worse."""

    return _render_case_diff(diff, limit, markdown=False)


def format_case_diff_markdown(diff: dict, limit: int) -> str:
    """Markdown section for the cases that got worse."""

    return _render_case_diff(diff, limit, markdown=True)


def comparison_payload(
    baseline_path: Path,
    current_path: Path,
    baseline: dict,
    current: dict,
    max_drop: float,
    diff: dict,
    limit: int,
    *,
    check_tags: bool = False,
    bootstrap: dict | None = None,
) -> dict:
    """JSON body for ``compare --json``.

    ``cases.worse`` follows ``limit`` (0 stores none). ``worse_count`` and
    ``recall_dropped`` cover every matched case, so the cap does not hide
    a recall drop from the gate.

    ``tags.regressed`` lists tag metrics that fell by more than ``max_drop``.
    ``tags.checked`` is true when that list is allowed to fail the command.

    ``bootstrap`` is included only when a resample was requested.
    ``bootstrap.checked`` is true when a significant drop can fail the command.
    """

    failed = regressed_metrics(baseline, current, max_drop)
    plan = _cutoff_plan(baseline, current)
    metrics = _payload_metrics(baseline, current, max_drop, plan)
    shown = [] if limit <= 0 else diff["worse"][:limit]
    cases: dict = {
        "available": diff["available"],
        "worse_count": len(diff["worse"]),
        "worse": [_public_case(row) for row in shown],
        "added": {"count": len(diff["added"]), "ids": list(diff["added"])},
        "removed": {"count": len(diff["removed"]), "ids": list(diff["removed"])},
        "recall_dropped": [row["id"] for row in diff["worse"] if row["recall_dropped"]],
    }
    if diff["message"]:
        cases["message"] = diff["message"]
    tags = tag_diff(baseline, current, max_drop)
    payload = {
        "baseline": str(baseline_path),
        "current": str(current_path),
        "k": {"baseline": _k_value(baseline), "current": _k_value(current)},
        "max_drop": max_drop,
        "metrics": metrics,
        "regressed": failed,
        "cases": cases,
        "tags": _public_tag_diff(tags, check_tags),
    }
    if plan["mode"] == "matched":
        payload["compared_at"] = plan["k"]
    elif plan["mode"] == "per_k":
        payload["cutoffs"] = [
            {
                "k": cutoff,
                "metrics": _pair_metrics(
                    _metrics_at(baseline, cutoff),
                    _metrics_at(current, cutoff),
                    max_drop,
                ),
            }
            for cutoff in plan["shared"]
        ]
        if plan["only_baseline"] or plan["only_current"]:
            payload["cutoff_only"] = {
                "baseline": plan["only_baseline"],
                "current": plan["only_current"],
            }
    elif plan["mode"] == "unmatched":
        payload["message"] = plan["message"]
    if bootstrap is not None:
        payload["bootstrap"] = bootstrap
    return payload


def _payload_metrics(baseline: dict, current: dict, max_drop: float, plan: dict) -> dict:
    """Headline metric object for a comparison payload.

    For two multi-k reports this is the largest shared cutoff. The full
    per-k deltas live under ``cutoffs``. When the reports share no cutoff
    the numbers are left empty and ``status`` is a dash.
    """

    if plan["mode"] == "unmatched":
        return {
            name: {"baseline": None, "current": None, "delta": None, "status": "-"}
            for name in METRIC_NAMES
        }
    if plan["mode"] == "matched":
        cutoff = plan["k"]
    elif plan["mode"] == "per_k":
        cutoff = plan["shared"][-1]
    else:
        return _pair_metrics(baseline["metrics"], current["metrics"], max_drop)
    return _pair_metrics(_metrics_at(baseline, cutoff), _metrics_at(current, cutoff), max_drop)


def _pair_metrics(before: dict, after: dict, max_drop: float) -> dict:
    paired = {}
    for name in METRIC_NAMES:
        base = before[name]
        cur = after[name]
        paired[name] = {
            "baseline": base,
            "current": cur,
            "delta": round(cur - base, 6),
            "status": "FAIL" if regressed(base, cur, max_drop) else "pass",
        }
    return paired


def tag_diff(baseline: dict, current: dict, max_drop: float) -> dict:
    """Compare per-tag means for tags that appear in both reports.

    Tags found on only one side are listed as added or removed. They are
    not regressions. When a report has no usable ``tags`` list,
    ``available`` is false and ``message`` names that side so callers can
    print a note instead of failing. ``regressed`` is every shared tag
    metric that fell by more than ``max_drop``. A drop equal to the limit
    is allowed.
    """

    before = _index_tags(baseline)
    after = _index_tags(current)
    empty = {
        "available": False,
        "message": None,
        "rows": [],
        "added": [],
        "removed": [],
        "regressed": [],
    }
    if before is None and after is None:
        return empty
    if before is None:
        empty["message"] = "baseline report has no per-tag results"
        return empty
    if after is None:
        empty["message"] = "current report has no per-tag results"
        return empty

    rows = []
    regressed_rows = []
    for tag in sorted(before.keys() & after.keys()):
        metrics = {}
        for name in METRIC_NAMES:
            base = before[tag]["metrics"][name]
            cur = after[tag]["metrics"][name]
            dropped = regressed(base, cur, max_drop)
            metrics[name] = {
                "baseline": base,
                "current": cur,
                "delta": round(cur - base, 6),
                "regressed": dropped,
            }
            if dropped:
                regressed_rows.append({"tag": tag, "metric": name})
        rows.append(
            {
                "tag": tag,
                "baseline_cases": before[tag]["case_count"],
                "current_cases": after[tag]["case_count"],
                "metrics": metrics,
            }
        )
    return {
        "available": True,
        "message": None,
        "rows": rows,
        "added": sorted(after.keys() - before.keys()),
        "removed": sorted(before.keys() - after.keys()),
        "regressed": regressed_rows,
    }


def format_tag_diff(diff: dict, *, markdown: bool, check: bool) -> str:
    """Plain text or Markdown for per-tag deltas.

    Returns an empty string when neither report has per-tag results.
    ``check`` adds a status column and a regressed line. Without it the
    deltas are shown and nothing is marked FAIL.
    """

    if not diff["available"] and not diff["message"]:
        return ""
    lines = ["### tags", ""] if markdown else ["tags"]
    if not diff["available"]:
        lines.append(diff["message"])
        return "\n".join(lines)
    if diff["rows"]:
        if markdown:
            table = _markdown_tag_delta_table(diff, check)
        else:
            table = _text_tag_delta_table(diff, check)
        lines.append(table)
    membership = []
    if diff["added"]:
        membership.append(_id_line("added tags", diff["added"]))
    if diff["removed"]:
        membership.append(_id_line("removed tags", diff["removed"]))
    if membership:
        if diff["rows"]:
            lines.append("")
        lines.extend(membership)
    if check and diff["regressed"]:
        parts = [f"{item['tag']} {item['metric']}" for item in diff["regressed"]]
        lines.extend(["", "regressed tags: " + ", ".join(parts)])
    return "\n".join(lines)


def _public_tag_diff(diff: dict, check: bool) -> dict:
    payload: dict = {"available": diff["available"]}
    if diff["message"]:
        payload["message"] = diff["message"]
    if not diff["available"]:
        return payload
    payload["checked"] = check
    payload["rows"] = [
        {
            "tag": row["tag"],
            "baseline_cases": row["baseline_cases"],
            "current_cases": row["current_cases"],
            "metrics": {
                name: {
                    "baseline": row["metrics"][name]["baseline"],
                    "current": row["metrics"][name]["current"],
                    "delta": row["metrics"][name]["delta"],
                    "status": _tag_status(row["metrics"][name]["regressed"], check),
                }
                for name in METRIC_NAMES
            },
        }
        for row in diff["rows"]
    ]
    payload["regressed"] = list(diff["regressed"])
    payload["added"] = list(diff["added"])
    payload["removed"] = list(diff["removed"])
    return payload


def _tag_status(dropped: bool, check: bool) -> str:
    if not check:
        return "-"
    return "FAIL" if dropped else "pass"


def _index_tags(report: dict) -> dict[str, dict] | None:
    raw = report.get("tags", None)
    if not isinstance(raw, list):
        return None
    indexed: dict[str, dict] = {}
    for item in raw:
        parsed = _parse_tag_row(item)
        if parsed is None or parsed["tag"] in indexed:
            continue
        indexed[parsed["tag"]] = parsed
    if not indexed:
        return None
    return indexed


def _parse_tag_row(item: object) -> dict | None:
    if not isinstance(item, dict):
        return None
    tag = item.get("tag")
    if isinstance(tag, str):
        tag = tag.strip()
    if not isinstance(tag, str) or not tag:
        return None
    count = item.get("case_count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        return None
    metrics_raw = item.get("metrics")
    if not isinstance(metrics_raw, dict):
        return None
    metrics: dict[str, float] = {}
    for name in METRIC_NAMES:
        if name not in metrics_raw:
            return None
        value = metrics_raw[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        number = float(value)
        if not math.isfinite(number):
            return None
        metrics[name] = number
    return {"tag": tag, "case_count": count, "metrics": metrics}


def _text_tag_delta_table(diff: dict, check: bool) -> str:
    headers = ["tag", "cases", "metric", "baseline", "current", "delta"]
    if check:
        headers.append("status")
    rows = []
    for row in diff["rows"]:
        cases = f"{row['baseline_cases']} -> {row['current_cases']}"
        for name in METRIC_NAMES:
            metric = row["metrics"][name]
            cells = [
                row["tag"],
                cases,
                name,
                f"{metric['baseline']:.4f}",
                f"{metric['current']:.4f}",
                f"{metric['delta']:+.4f}",
            ]
            if check:
                cells.append(_tag_status(metric["regressed"], True))
            rows.append(cells)
    return render_table(headers, rows)


def _markdown_tag_delta_table(diff: dict, check: bool) -> str:
    headers = ["tag", "cases", "metric", "baseline", "current", "delta"]
    aligns = ["---", "---", "---", "---:", "---:", "---:"]
    if check:
        headers.append("status")
        aligns.append("---")
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(aligns) + " |",
    ]
    for row in diff["rows"]:
        cases = f"{row['baseline_cases']} -> {row['current_cases']}"
        for name in METRIC_NAMES:
            metric = row["metrics"][name]
            cells = [
                _markdown_cell(row["tag"]),
                cases,
                name,
                f"{metric['baseline']:.4f}",
                f"{metric['current']:.4f}",
                f"{metric['delta']:+.4f}",
            ]
            if check:
                cells.append(_tag_status(metric["regressed"], True))
            lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _case_results(report: dict) -> list | None:
    results = report.get("results", None)
    if isinstance(results, list):
        return results
    return None


def _index_cases(results: list) -> dict[str, dict]:
    indexed: dict[str, dict] = {}
    for row in results:
        if not isinstance(row, dict):
            continue
        case_id = row.get("id")
        if isinstance(case_id, str):
            case_id = case_id.strip()
        if not isinstance(case_id, str) or not case_id or case_id in indexed:
            continue
        indexed[case_id] = row
    return indexed


def _worse_case(before: dict, after: dict, case_id: str) -> dict | None:
    base_recall = _case_metric(before, "recall")
    cur_recall = _case_metric(after, "recall")
    base_ndcg = _case_metric(before, "ndcg")
    cur_ndcg = _case_metric(after, "ndcg")
    base_hit = _case_metric(before, "hit_rate")
    cur_hit = _case_metric(after, "hit_rate")
    recall_dropped = _metric_dropped(base_recall, cur_recall)
    ndcg_dropped = _metric_dropped(base_ndcg, cur_ndcg)
    hit_lost = _hit_lost(base_hit, cur_hit)
    if not (recall_dropped or ndcg_dropped or hit_lost):
        return None
    query = after.get("query")
    if not isinstance(query, str) or not query.strip():
        query = before.get("query")
    if not isinstance(query, str):
        query = ""
    return {
        "id": case_id,
        "query": query.strip(),
        "recall": {"baseline": base_recall, "current": cur_recall},
        "ndcg": {"baseline": base_ndcg, "current": cur_ndcg},
        "hit_rate": {"baseline": base_hit, "current": cur_hit},
        "hit_lost": hit_lost,
        "recall_dropped": recall_dropped,
        "newly_missed": _newly_missed(before, after),
    }


def _worse_sort_key(row: dict) -> tuple:
    recall_drop = _positive_drop(row["recall"]["baseline"], row["recall"]["current"])
    ndcg_drop = _positive_drop(row["ndcg"]["baseline"], row["ndcg"]["current"])
    hit = 0 if row["hit_lost"] else 1
    return (-max(recall_drop, ndcg_drop), -recall_drop, -ndcg_drop, hit, row["id"])


def _metric_dropped(before: float | None, after: float | None) -> bool:
    if before is None or after is None:
        return False
    return before - after > _DELTA_EPSILON


def _positive_drop(before: float | None, after: float | None) -> float:
    if before is None or after is None:
        return 0.0
    drop = before - after
    if drop <= _DELTA_EPSILON:
        return 0.0
    return drop


def _hit_lost(before: float | None, after: float | None) -> bool:
    if before is None or after is None:
        return False
    return before >= 1.0 - _DELTA_EPSILON and after <= _DELTA_EPSILON


def _case_metric(row: dict, name: str) -> float | None:
    metrics = row.get("metrics")
    if not isinstance(metrics, dict) or name not in metrics:
        return None
    value = metrics[name]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return number


def _newly_missed(before: dict, after: dict) -> list[str]:
    previous = set(_misses(before))
    return [doc_id for doc_id in _misses(after) if doc_id not in previous]


def _misses(row: dict) -> list[str]:
    misses = row.get("misses")
    if isinstance(misses, list):
        return [doc_id for doc_id in misses if isinstance(doc_id, str) and doc_id]
    retrieved = row.get("retrieved")
    relevant = row.get("relevant")
    if not isinstance(retrieved, list) or not isinstance(relevant, dict):
        return []
    present = {doc_id for doc_id in retrieved if isinstance(doc_id, str)}
    found = []
    for doc_id, grade in relevant.items():
        if not isinstance(doc_id, str) or not doc_id:
            continue
        if isinstance(grade, bool) or not isinstance(grade, (int, float)):
            continue
        if grade > 0 and doc_id not in present:
            found.append(doc_id)
    return found


def _public_case(row: dict) -> dict:
    return {
        "id": row["id"],
        "query": row["query"],
        "recall": dict(row["recall"]),
        "ndcg": dict(row["ndcg"]),
        "hit_lost": row["hit_lost"],
        "newly_missed": list(row["newly_missed"]),
    }


def _k_value(report: dict) -> int | None:
    value = report.get("k")
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def _render_case_diff(diff: dict, limit: int, *, markdown: bool) -> str:
    if markdown:
        lines = ["### worse cases", ""]
    else:
        lines = ["worse cases"]
    if not diff["available"]:
        lines.append(diff["message"])
    else:
        worse = diff["worse"]
        if limit <= 0:
            lines.append("none" if not worse else f"{len(worse)} hidden")
        elif not worse:
            lines.append("none")
        else:
            shown = worse[:limit]
            if markdown:
                lines.append(_markdown_case_table(shown))
            else:
                lines.append(_text_case_table(shown))
            if len(worse) > limit:
                lines.append(f"showing {limit} of {len(worse)}")
    membership = _membership_lines(diff)
    if membership:
        lines.append("")
        lines.extend(membership)
    return "\n".join(lines)


def _text_case_table(rows: list[dict]) -> str:
    table = [
        [
            row["id"],
            _truncate_query(row["query"]),
            _score_text(row["recall"]["baseline"]),
            _score_text(row["recall"]["current"]),
            _score_text(row["ndcg"]["baseline"]),
            _score_text(row["ndcg"]["current"]),
            _missed_text(row["newly_missed"]),
        ]
        for row in rows
    ]
    return render_table(
        [
            "case",
            "query",
            "recall baseline",
            "recall current",
            "ndcg baseline",
            "ndcg current",
            "newly missed",
        ],
        table,
    )


def _markdown_case_table(rows: list[dict]) -> str:
    headers = [
        "case",
        "query",
        "recall baseline",
        "recall current",
        "ndcg baseline",
        "ndcg current",
        "newly missed",
    ]
    aligns = ["---", "---", "---:", "---:", "---:", "---:", "---"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(aligns) + " |",
    ]
    for row in rows:
        cells = [
            _markdown_cell(row["id"]),
            _markdown_cell(_truncate_query(row["query"])),
            _score_text(row["recall"]["baseline"]),
            _score_text(row["recall"]["current"]),
            _score_text(row["ndcg"]["baseline"]),
            _score_text(row["ndcg"]["current"]),
            _markdown_cell(_missed_text(row["newly_missed"])),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _membership_lines(diff: dict) -> list[str]:
    if not diff["available"]:
        return []
    lines = []
    if diff["added"]:
        lines.append(_id_line("added", diff["added"]))
    if diff["removed"]:
        lines.append(_id_line("removed", diff["removed"]))
    return lines


def _id_line(label: str, ids: list[str]) -> str:
    return f"{label}: {len(ids)} ({', '.join(ids)})"


def _score_text(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.4f}"


def _missed_text(doc_ids: list[str]) -> str:
    if not doc_ids:
        return "-"
    return ", ".join(doc_ids)


def _truncate_query(query: str, limit: int = _QUERY_LIMIT) -> str:
    text = " ".join(query.split())
    if len(text) <= limit:
        return text
    if limit <= 3:
        return text[:limit]
    return text[: limit - 3] + "..."


def _markdown_cell(value: str) -> str:
    return value.replace("\\", "\\\\").replace("|", "\\|")


def write_output(path: Path, text: str) -> None:
    """Write a formatted report, adding a trailing newline when needed."""

    payload = text if text.endswith("\n") else text + "\n"
    try:
        path.write_text(payload, encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"could not write {path}: {exc}") from exc
