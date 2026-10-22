"""Command line interface for retrieval-gate."""

import math
from pathlib import Path

import typer
import yaml

from retrieval_gate.dataset import load_dataset
from retrieval_gate.errors import ConfigError, DatasetError
from retrieval_gate.metrics import METRIC_NAMES
from retrieval_gate.report import (
    build_report,
    case_diff,
    comparison_payload,
    format_case_diff,
    format_case_diff_markdown,
    format_comparison,
    format_markdown,
    format_report,
    load_report,
    regressed_metrics,
    tag_diff,
    write_output,
    write_report,
)
from retrieval_gate.retriever import load_retriever

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    pretty_exceptions_enable=False,
    help="Score a retriever against labeled queries and fail when metrics drop.",
)

_CONFIG_KEYS = {
    "dataset",
    "retriever",
    "k",
    "fail_under",
    "fail_under_tag",
    "baseline",
    "max_drop",
}


def main() -> None:
    app()


@app.command()
def run(
    dataset: Path | None = typer.Option(
        None,
        "--dataset",
        "-d",
        help="JSONL or YAML file of labeled queries.",
    ),
    retriever: str | None = typer.Option(
        None,
        "--retriever",
        "-r",
        help="Import path of the retriever, as module.submodule:function.",
    ),
    k: int | None = typer.Option(
        None,
        "--k",
        help="Number of unique results to score per query. Defaults to 5.",
    ),
    fail_under: list[str] | None = typer.Option(
        None,
        "--fail-under",
        help="Gate a metric, as name=value. Repeat for more than one metric.",
    ),
    fail_under_tag: list[str] | None = typer.Option(
        None,
        "--fail-under-tag",
        help="Gate one tag, as tag:metric=value. Repeat for more than one.",
    ),
    json_output: Path | None = typer.Option(
        None,
        "--json",
        help="Write the JSON report to this path.",
    ),
    report_format: str = typer.Option(
        "text",
        "--format",
        help="Report format: text or markdown.",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        help="Write the formatted report to this path.",
    ),
    config: Path | None = typer.Option(
        None,
        "--config",
        help=(
            "YAML file with dataset, retriever, k, fail_under, fail_under_tag, "
            "baseline, and max_drop."
        ),
    ),
    baseline: Path | None = typer.Option(
        None,
        "--baseline",
        help="JSON report to compare against. A drop past --max-drop fails the run.",
    ),
    max_drop: float | None = typer.Option(
        None,
        "--max-drop",
        help="Maximum allowed drop versus the baseline. Defaults to 0.02.",
    ),
) -> None:
    """Score a retriever on a labeled dataset."""

    try:
        fmt = _checked_format(report_format)
        cfg = load_config(config) if config is not None else _empty_config()
        chosen_dataset = dataset if dataset is not None else cfg["dataset"]
        chosen_retriever = retriever if retriever is not None else cfg["retriever"]
        chosen_k = _choose_k(k, cfg["k"])
        thresholds = parse_fail_under(fail_under) if fail_under else cfg["thresholds"]
        tag_thresholds = (
            parse_fail_under_tag(fail_under_tag) if fail_under_tag else cfg["tag_thresholds"]
        )
        chosen_baseline = baseline if baseline is not None else cfg["baseline"]
        chosen_drop = _choose_max_drop(max_drop, cfg["max_drop"], chosen_baseline)
        if chosen_dataset is None:
            raise ConfigError("pass --dataset or set dataset in --config")
        if not chosen_retriever:
            raise ConfigError("pass --retriever or set retriever in --config")
        if chosen_k < 1:
            raise ConfigError("k must be >= 1")
        cases = load_dataset(chosen_dataset)
        fn = load_retriever(chosen_retriever)
        report = build_report(
            cases,
            fn,
            chosen_k,
            dataset=str(chosen_dataset),
            retriever_name=chosen_retriever,
            thresholds=thresholds,
            tag_thresholds=tag_thresholds,
        )
        baseline_report = (
            load_report(chosen_baseline) if chosen_baseline is not None else None
        )
        rendered = _render_run(
            fmt,
            report,
            baseline_report,
            chosen_baseline,
            chosen_drop,
            json_output,
        )
        _emit(rendered, output)
        if json_output is not None:
            write_report(json_output, report)
    except (DatasetError, ConfigError) as exc:
        _fail(str(exc))
    dropped = (
        regressed_metrics(baseline_report, report, chosen_drop)
        if baseline_report is not None
        else []
    )
    if report["failed"] or report.get("failed_tags") or dropped:
        raise typer.Exit(code=1)


@app.command()
def compare(
    baseline: Path = typer.Argument(..., help="JSON report to treat as the baseline."),
    current: Path = typer.Argument(..., help="JSON report from the current run."),
    max_drop: float = typer.Option(
        0.02,
        "--max-drop",
        help="Fail when any metric drops by more than this. An equal drop passes.",
    ),
    report_format: str = typer.Option(
        "text",
        "--format",
        help="Report format: text or markdown.",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        help="Write the formatted report to this path.",
    ),
    json_output: Path | None = typer.Option(
        None,
        "--json",
        help="Write the JSON comparison to this path.",
    ),
    show_cases: int = typer.Option(
        10,
        "--show-cases",
        help="How many worse cases to list. The default is 10. 0 hides the list.",
    ),
    fail_on_case_regression: bool = typer.Option(
        False,
        "--fail-on-case-regression",
        help="Exit 1 when any matched case has a lower recall. Off by default.",
    ),
    fail_on_tag_regression: bool = typer.Option(
        False,
        "--fail-on-tag-regression",
        help="Exit 1 when any tag metric drops by more than --max-drop. Off by default.",
    ),
) -> None:
    """Compare two JSON reports and fail on a regression."""

    try:
        fmt = _checked_format(report_format)
        _checked_max_drop(max_drop)
        _checked_show_cases(show_cases)
        before = load_report(baseline)
        after = load_report(current)
        diff = case_diff(before, after)
        tags = tag_diff(before, after, max_drop)
        if fmt == "markdown":
            text = format_markdown(
                after,
                baseline=before,
                max_drop=max_drop,
                baseline_path=str(baseline),
                current_path=str(current),
                check_thresholds=False,
                check_tags=fail_on_tag_regression,
            )
            failed = regressed_metrics(before, after, max_drop)
            section = format_case_diff_markdown(diff, show_cases)
        else:
            text, failed = format_comparison(
                baseline,
                current,
                before,
                after,
                max_drop,
                check_tags=fail_on_tag_regression,
            )
            section = format_case_diff(diff, show_cases)
        recall_ids = [row["id"] for row in diff["worse"] if row["recall_dropped"]]
        if fail_on_case_regression and recall_ids:
            section = f"{section}\n\ncase recall dropped: {', '.join(recall_ids)}"
            case_failed = True
        else:
            case_failed = False
        tag_failed = bool(fail_on_tag_regression and tags["regressed"])
        _emit(f"{text}\n\n{section}", output)
        if json_output is not None:
            write_report(
                json_output,
                comparison_payload(
                    baseline,
                    current,
                    before,
                    after,
                    max_drop,
                    diff,
                    show_cases,
                    check_tags=fail_on_tag_regression,
                ),
            )
    except (DatasetError, ConfigError) as exc:
        _fail(str(exc))
    if failed or case_failed or tag_failed:
        raise typer.Exit(code=1)


def parse_fail_under(items: list[str]) -> dict[str, float]:
    """Parse repeated ``name=value`` thresholds.

    Names are hit_rate, recall, precision, mrr, and ndcg. The value is
    the minimum allowed aggregate, from 0 to 1 inclusive.
    """

    allowed = set(METRIC_NAMES)
    thresholds: dict[str, float] = {}
    for item in items:
        if "=" not in item:
            raise ConfigError(f"--fail-under {item!r} must look like name=value")
        name, raw = item.split("=", 1)
        name = name.strip()
        if name not in allowed:
            names = ", ".join(METRIC_NAMES)
            raise ConfigError(f"unknown metric {name!r} in --fail-under (expected {names})")
        if name in thresholds:
            raise ConfigError(f"--fail-under metric {name!r} was given twice")
        thresholds[name] = _threshold_value("--fail-under", item, name, raw)
    return thresholds


def parse_fail_under_tag(items: list[str]) -> dict[str, dict[str, float]]:
    """Parse repeated ``tag:metric=value`` thresholds.

    Metric names match ``--fail-under``. The value is the minimum allowed
    mean for cases that carry that tag, from 0 to 1 inclusive.
    """

    allowed = set(METRIC_NAMES)
    thresholds: dict[str, dict[str, float]] = {}
    for item in items:
        left, sep, raw = item.partition("=")
        if not sep or ":" not in left:
            raise ConfigError(f"--fail-under-tag {item!r} must look like tag:metric=value")
        tag, name = left.rsplit(":", 1)
        tag = tag.strip()
        name = name.strip()
        if not tag:
            raise ConfigError(f"--fail-under-tag {item!r} is missing a tag")
        if name not in allowed:
            names = ", ".join(METRIC_NAMES)
            raise ConfigError(
                f"unknown metric {name!r} in --fail-under-tag (expected {names})"
            )
        limits = thresholds.setdefault(tag, {})
        if name in limits:
            raise ConfigError(f"--fail-under-tag {tag} {name} was given twice")
        limits[name] = _threshold_value("--fail-under-tag", item, name, raw)
    return thresholds


def _threshold_value(flag: str, item: str, name: str, raw: str) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigError(f"{flag} {item!r} value is not a number") from exc
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ConfigError(f"{flag} {name} must be between 0 and 1")
    return value


def load_config(path: Path) -> dict:
    """Load run settings from a YAML file.

    Paths are resolved relative to the config file. ``fail_under`` may be
    a mapping of metric name to limit, or a list of ``name=value`` strings.
    ``fail_under_tag`` may be a mapping of tag to metric limits, or a list
    of ``tag:metric=value`` strings.
    """

    if not path.is_file():
        raise ConfigError(f"config not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: invalid YAML ({exc})") from exc
    except OSError as exc:
        raise ConfigError(f"could not read {path}: {exc}") from exc
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: config must be a mapping")
    unknown = sorted(key for key in raw if key not in _CONFIG_KEYS)
    if unknown:
        names = ", ".join(unknown)
        raise ConfigError(f"{path}: unknown config keys: {names}")
    return {
        "dataset": _config_path(path, raw.get("dataset"), "dataset"),
        "retriever": _config_text(path, raw.get("retriever"), "retriever"),
        "k": _config_k(path, raw.get("k")),
        "baseline": _config_path(path, raw.get("baseline"), "baseline"),
        "max_drop": _config_max_drop(path, raw.get("max_drop")),
        "thresholds": _config_thresholds(path, raw.get("fail_under")),
        "tag_thresholds": _config_tag_thresholds(path, raw.get("fail_under_tag")),
    }


def _empty_config() -> dict:
    return {
        "dataset": None,
        "retriever": None,
        "k": None,
        "baseline": None,
        "max_drop": None,
        "thresholds": {},
        "tag_thresholds": {},
    }


def _choose_k(flag: int | None, configured: int | None) -> int:
    if flag is not None:
        return flag
    if configured is not None:
        return configured
    return 5


def _choose_max_drop(
    flag: float | None,
    configured: float | None,
    baseline: Path | None,
) -> float:
    if flag is not None:
        chosen = _checked_max_drop(flag)
    elif configured is not None:
        chosen = configured
    else:
        chosen = 0.02
    if baseline is None and (flag is not None or configured is not None):
        raise ConfigError("max_drop requires a baseline")
    return chosen


def _config_path(path: Path, value: object, key: str) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{path}: {key} must be a path")
    candidate = Path(value.strip())
    if not candidate.is_absolute():
        candidate = path.parent / candidate
    return candidate


def _config_text(path: Path, value: object, key: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{path}: {key} must be a string")
    return value.strip()


def _config_k(path: Path, value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ConfigError(f"{path}: k must be an integer >= 1")
    return value


def _config_max_drop(path: Path, value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{path}: max_drop must be a number >= 0")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ConfigError(f"{path}: max_drop must be a number >= 0")
    return number


def _config_thresholds(path: Path, value: object) -> dict[str, float]:
    if value is None:
        return {}
    if isinstance(value, dict):
        items = []
        for name, limit in value.items():
            if isinstance(limit, bool) or not isinstance(limit, (int, float)):
                raise ConfigError(f"{path}: fail_under.{name} must be a number")
            items.append(f"{name}={limit}")
        return parse_fail_under(items)
    if isinstance(value, list):
        items = []
        for item in value:
            if not isinstance(item, str):
                raise ConfigError(f"{path}: fail_under entries must be name=value strings")
            items.append(item)
        return parse_fail_under(items)
    raise ConfigError(f"{path}: fail_under must be a mapping or a list")


def _config_tag_thresholds(path: Path, value: object) -> dict[str, dict[str, float]]:
    if value is None:
        return {}
    if isinstance(value, dict):
        items = []
        for tag, limits in value.items():
            if not isinstance(tag, str) or not tag.strip():
                raise ConfigError(f"{path}: fail_under_tag keys must be tag names")
            if not isinstance(limits, dict):
                raise ConfigError(
                    f"{path}: fail_under_tag.{tag} must be a mapping of metric to limit"
                )
            for name, limit in limits.items():
                if isinstance(limit, bool) or not isinstance(limit, (int, float)):
                    raise ConfigError(f"{path}: fail_under_tag.{tag}.{name} must be a number")
                items.append(f"{tag.strip()}:{name}={limit}")
        return parse_fail_under_tag(items)
    if isinstance(value, list):
        items = []
        for item in value:
            if not isinstance(item, str):
                raise ConfigError(
                    f"{path}: fail_under_tag entries must be tag:metric=value strings"
                )
            items.append(item)
        return parse_fail_under_tag(items)
    raise ConfigError(f"{path}: fail_under_tag must be a mapping or a list")


def _checked_format(value: str) -> str:
    fmt = value.strip().lower()
    if fmt not in {"text", "markdown"}:
        raise ConfigError("--format must be text or markdown")
    return fmt


def _checked_max_drop(value: float) -> float:
    if not math.isfinite(value) or value < 0:
        raise ConfigError("--max-drop must be a number greater than or equal to 0")
    return value


def _checked_show_cases(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ConfigError("--show-cases must be an integer >= 0")
    return value


def _render_run(
    fmt: str,
    report: dict,
    baseline_report: dict | None,
    baseline_path: Path | None,
    max_drop: float,
    json_output: Path | None,
) -> str:
    if fmt == "markdown":
        current = None
        if json_output is not None and baseline_report is not None:
            current = str(json_output)
        return format_markdown(
            report,
            baseline=baseline_report,
            max_drop=max_drop,
            baseline_path=None if baseline_path is None else str(baseline_path),
            current_path=current,
        )
    text = format_report(report)
    if baseline_report is None or baseline_path is None:
        return text
    current_path = json_output if json_output is not None else Path("this run")
    comparison, _ = format_comparison(
        baseline_path,
        current_path,
        baseline_report,
        report,
        max_drop,
    )
    return f"{text}\n\n{comparison}"


def _emit(text: str, output: Path | None) -> None:
    typer.echo(text)
    if output is not None:
        write_output(output, text)


def _fail(message: str) -> None:
    typer.echo(message, err=True)
    raise typer.Exit(code=2)
