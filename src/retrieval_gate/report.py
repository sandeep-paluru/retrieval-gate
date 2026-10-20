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


def format_comparison(
    baseline_path: Path,
    current_path: Path,
    baseline: dict,
    current: dict,
    max_drop: float,
) -> tuple[str, list[str]]:
    """Return the comparison table and the metrics that regressed."""

    rows = []
    failed = []
    for name in METRIC_NAMES:
        before = baseline["metrics"][name]
        after = current["metrics"][name]
        delta = after - before
        status = "FAIL" if regressed(before, after, max_drop) else "pass"
        if status == "FAIL":
            failed.append(name)
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
