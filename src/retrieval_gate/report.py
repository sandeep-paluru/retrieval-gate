"""Run a dataset, format the report, and compare two reports."""

import asyncio
import json
import math
from collections.abc import Callable
from pathlib import Path

from retrieval_gate.errors import ConfigError, DatasetError
from retrieval_gate.metrics import METRIC_NAMES, score_case, top_unique
from retrieval_gate.models import Case
from retrieval_gate.retriever import call_retriever

_MISS_LIMIT = 10


def build_report(
    cases: list[Case],
    retriever: Callable,
    k: int,
    *,
    dataset: str,
    retriever_name: str,
    thresholds: dict[str, float],
) -> dict:
    """Score every case and average the metrics.

    Per-case metrics are rounded to 6 decimal places. The aggregate is the
    mean of the unrounded scores, then rounded the same way. A threshold
    fails when the aggregate is strictly below the limit.
    """

    if k < 1:
        raise ConfigError("k must be >= 1")
    retrieved_lists = asyncio.run(_retrieve_all(retriever, cases, k))
    results = []
    raw_scores = []
    for case, retrieved in zip(cases, retrieved_lists, strict=True):
        try:
            scores = score_case(retrieved, case.relevant, k)
        except ValueError as exc:
            raise DatasetError(f"case {case.id}: {exc}") from exc
        ranking = top_unique(retrieved, k)
        present = set(ranking)
        misses = [
            doc_id
            for doc_id, grade in case.relevant.items()
            if grade > 0 and doc_id not in present
        ]
        raw_scores.append(scores)
        results.append(
            {
                "id": case.id,
                "query": case.query,
                "retrieved": ranking,
                "relevant": dict(case.relevant),
                "misses": misses,
                "metrics": {name: round(scores[name], 6) for name in METRIC_NAMES},
            }
        )
    count = len(raw_scores)
    metrics = {
        name: round(sum(scores[name] for scores in raw_scores) / count, 6)
        for name in METRIC_NAMES
    }
    failed = [name for name, limit in thresholds.items() if metrics[name] < limit]
    return {
        "k": k,
        "dataset": dataset,
        "retriever": retriever_name,
        "case_count": count,
        "metrics": metrics,
        "thresholds": dict(thresholds),
        "failed": failed,
        "results": results,
    }


async def _retrieve_all(retriever: Callable, cases: list[Case], k: int) -> list[list[str]]:
    lists = []
    for case in cases:
        lists.append(await call_retriever(retriever, case.query, k, case.id))
    return lists


def format_report(report: dict) -> str:
    """Plain text table plus the worst per-case misses."""

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
        "",
        "worst misses",
        _format_misses(report["results"]),
    ]
    if report["failed"]:
        lines.extend(["", "failed: " + ", ".join(report["failed"])])
    return "\n".join(lines)


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
    return loaded


def _metric_number(path: Path, metrics: dict, name: str) -> float:
    if name not in metrics:
        raise ConfigError(f"{path}: metrics.{name} is missing")
    value = metrics[name]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{path}: metrics.{name} is not a number")
    number = float(value)
    if not math.isfinite(number):
        raise ConfigError(f"{path}: metrics.{name} is not a number")
    return number


def regressed(baseline: float, current: float, max_drop: float) -> bool:
    """Return whether ``current`` fell by more than ``max_drop``.

    A drop equal to ``max_drop`` is allowed. The comparison ignores
    differences smaller than 1e-9 so a value that was rounded into JSON
    does not flap around the limit.
    """

    return (baseline - current) - max_drop > 1e-9


def regressed_metrics(baseline: dict, current: dict, max_drop: float) -> list[str]:
    """Return metrics that fell by more than ``max_drop``, in report order."""

    return [
        name
        for name in METRIC_NAMES
        if regressed(baseline["metrics"][name], current["metrics"][name], max_drop)
    ]


def format_comparison(
    baseline_path: Path,
    current_path: Path,
    baseline: dict,
    current: dict,
    max_drop: float,
) -> tuple[str, list[str]]:
    """Return the comparison table and the metrics that regressed."""

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
    return "\n".join(lines), failed


def _k_text(report: dict) -> str:
    value = report.get("k")
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    return "-"


def format_markdown(
    report: dict,
    *,
    baseline: dict | None = None,
    max_drop: float = 0.02,
    baseline_path: str | None = None,
    current_path: str | None = None,
    check_thresholds: bool = True,
) -> str:
    """Return a compact Markdown table for a pull request or job summary.

    Columns are the metric, score, threshold, baseline, delta, and status.
    When ``baseline`` is omitted, the baseline and delta cells are dashes
    and no metric is marked as a regression. A threshold fails when the
    score is strictly below the limit. A regression fails when the score
    drops by more than ``max_drop``. ``check_thresholds`` can be turned
    off by ``compare``, which still prints stored limits but gates only
    on the baseline drop.
    """

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


def write_output(path: Path, text: str) -> None:
    """Write a formatted report, adding a trailing newline when needed."""

    payload = text if text.endswith("\n") else text + "\n"
    try:
        path.write_text(payload, encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"could not write {path}: {exc}") from exc
