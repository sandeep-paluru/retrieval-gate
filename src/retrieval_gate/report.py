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
) -> dict:
    """JSON body for ``compare --json``.

    ``cases.worse`` follows ``limit`` (0 stores none). ``worse_count`` and
    ``recall_dropped`` cover every matched case, so the cap does not hide
    a recall drop from the gate.
    """

    failed = regressed_metrics(baseline, current, max_drop)
    failed_set = set(failed)
    metrics = {}
    for name in METRIC_NAMES:
        before = baseline["metrics"][name]
        after = current["metrics"][name]
        metrics[name] = {
            "baseline": before,
            "current": after,
            "delta": round(after - before, 6),
            "status": "FAIL" if name in failed_set else "pass",
        }
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
    return {
        "baseline": str(baseline_path),
        "current": str(current_path),
        "k": {"baseline": _k_value(baseline), "current": _k_value(current)},
        "max_drop": max_drop,
        "metrics": metrics,
        "regressed": failed,
        "cases": cases,
    }


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
