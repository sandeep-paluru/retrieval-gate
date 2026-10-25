"""Command line interface for retrieval-gate."""

import math
import re
from pathlib import Path

import typer
import yaml

from retrieval_gate.adapters import normalize_adapter
from retrieval_gate.bootstrap import (
    DEFAULT_BOOTSTRAP_CONFIDENCE,
    DEFAULT_BOOTSTRAP_SEED,
    format_bootstrap,
    paired_bootstrap,
)
from retrieval_gate.dataset import load_dataset
from retrieval_gate.errors import ConfigError, DatasetError
from retrieval_gate.http_retriever import (
    DEFAULT_CONCURRENCY,
    DEFAULT_ID_FIELD,
    DEFAULT_TIMEOUT,
    build_http_retriever,
    checked_concurrency,
    checked_timeout,
    normalize_method,
    public_url,
)
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
    "retriever_url",
    "method",
    "headers",
    "timeout",
    "concurrency",
    "results_key",
    "id_field",
    "adapter",
    "k",
    "k_values",
    "fail_under",
    "fail_under_tag",
    "baseline",
    "max_drop",
}

_HTTP_FLAGS = {
    "method": "--method",
    "header": "--header",
    "timeout": "--timeout",
    "concurrency": "--concurrency",
    "results_key": "--results-key",
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
        help=(
            "Import path, as module.submodule:attribute. A (query, k) function, "
            "or a LangChain or LlamaIndex retriever."
        ),
    ),
    adapter: str | None = typer.Option(
        None,
        "--adapter",
        help="How to call --retriever: auto, langchain, or llamaindex. Defaults to auto.",
    ),
    retriever_url: str | None = typer.Option(
        None,
        "--retriever-url",
        help="Score this HTTP service. POST JSON with query and k by default.",
    ),
    method: str | None = typer.Option(
        None,
        "--method",
        help="HTTP method for --retriever-url: get or post. Defaults to post.",
    ),
    header: list[str] | None = typer.Option(
        None,
        "--header",
        help=(
            "Header for --retriever-url, as 'Name: Value'. Repeatable. "
            "${ENV_VAR} in the value is expanded. The value is not printed."
        ),
    ),
    timeout: float | None = typer.Option(
        None,
        "--timeout",
        help=(
            "Per-request timeout in seconds for --retriever-url. "
            f"Defaults to {DEFAULT_TIMEOUT:g}."
        ),
    ),
    concurrency: int | None = typer.Option(
        None,
        "--concurrency",
        help=f"Max in-flight HTTP requests. Defaults to {DEFAULT_CONCURRENCY}.",
    ),
    results_key: str | None = typer.Option(
        None,
        "--results-key",
        help="JSON key that holds the result list. Defaults to results, then documents, then hits.",
    ),
    id_field: str | None = typer.Option(
        None,
        "--id-field",
        help=(
            "Where to read a document id. HTTP bodies use this object field "
            "(default id). LangChain reads Document.id unless this names a metadata key. "
            "LlamaIndex reads node_id unless this is ref_doc_id or a metadata key."
        ),
    ),
    k: int | None = typer.Option(
        None,
        "--k",
        help="Number of unique results to score per query. Defaults to 5.",
    ),
    k_values: str | None = typer.Option(
        None,
        "--k-values",
        help=(
            "Comma-separated cutoffs, like 1,3,5,10. The retriever is called "
            "once at the largest. Do not combine with --k."
        ),
    ),
    fail_under: list[str] | None = typer.Option(
        None,
        "--fail-under",
        help=(
            "Gate a metric, as name=value or name@k=value. A bare name applies "
            "to the largest cutoff. Repeat for more than one metric."
        ),
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
            "YAML file with dataset, retriever or retriever_url, k or k_values, "
            "fail_under, fail_under_tag, baseline, and max_drop. Import-path runs also read "
            "adapter and id_field. HTTP runs also read method, headers, timeout, "
            "concurrency, results_key, and id_field."
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
        cutoffs = _choose_cutoffs(k, k_values, cfg["k"], cfg["k_values"])
        parsed_thresholds = parse_fail_under(fail_under) if fail_under else cfg["thresholds"]
        bound_thresholds = bind_thresholds(parsed_thresholds, cutoffs)
        tag_thresholds = (
            parse_fail_under_tag(fail_under_tag) if fail_under_tag else cfg["tag_thresholds"]
        )
        chosen_baseline = baseline if baseline is not None else cfg["baseline"]
        chosen_drop = _choose_max_drop(max_drop, cfg["max_drop"], chosen_baseline)
        if chosen_dataset is None:
            raise ConfigError("pass --dataset or set dataset in --config")
        chosen_k = cutoffs[-1]
        multi = len(cutoffs) > 1
        fn, retriever_name, request_concurrency = _prepare_retriever(
            retriever,
            retriever_url,
            adapter,
            method,
            header,
            timeout,
            concurrency,
            results_key,
            id_field,
            cfg,
        )
        cases = load_dataset(chosen_dataset)
        report = build_report(
            cases,
            fn,
            chosen_k,
            dataset=str(chosen_dataset),
            retriever_name=retriever_name,
            thresholds=bound_thresholds[chosen_k],
            tag_thresholds=tag_thresholds,
            concurrency=request_concurrency,
            k_values=cutoffs if multi else None,
            thresholds_by_k=bound_thresholds if multi else None,
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
    bootstrap_samples: int = typer.Option(
        0,
        "--bootstrap",
        help="Paired resamples of shared cases. 0 skips the check. 2000 is the suggested count.",
    ),
    confidence: float = typer.Option(
        DEFAULT_BOOTSTRAP_CONFIDENCE,
        "--confidence",
        help="Confidence level for the bootstrap interval, above 0 and below 1. Default 0.95.",
    ),
    seed: int = typer.Option(
        DEFAULT_BOOTSTRAP_SEED,
        "--seed",
        help="Random seed for the bootstrap. Default 0 so repeated runs match.",
    ),
    fail_on_significant_drop: bool = typer.Option(
        False,
        "--fail-on-significant-drop",
        help=(
            "Exit 1 when the whole bootstrap interval is below -max-drop. "
            "Off by default. Applies only when --bootstrap is set."
        ),
    ),
) -> None:
    """Compare two JSON reports and fail on a regression.

    ``--bootstrap`` resamples cases shared by both reports. It is off by
    default. ``--fail-on-significant-drop`` is what lets that interval
    change the exit code.
    """

    try:
        fmt = _checked_format(report_format)
        _checked_max_drop(max_drop)
        _checked_show_cases(show_cases)
        _checked_bootstrap(bootstrap_samples)
        _checked_confidence(confidence)
        _checked_seed(seed)
        before = load_report(baseline)
        after = load_report(current)
        diff = case_diff(before, after)
        tags = tag_diff(before, after, max_drop)
        bootstrap = None
        if bootstrap_samples > 0:
            bootstrap = paired_bootstrap(
                before,
                after,
                samples=bootstrap_samples,
                confidence=confidence,
                seed=seed,
                max_drop=max_drop,
                checked=fail_on_significant_drop,
            )
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
        parts = [text]
        if bootstrap is not None:
            parts.append(format_bootstrap(bootstrap, markdown=(fmt == "markdown")))
        parts.append(section)
        _emit("\n\n".join(parts), output)
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
                    bootstrap=bootstrap,
                ),
            )
    except (DatasetError, ConfigError) as exc:
        _fail(str(exc))
    bootstrap_failed = bool(
        fail_on_significant_drop
        and bootstrap is not None
        and bootstrap["available"]
        and bootstrap["significant"]
    )
    if failed or case_failed or tag_failed or bootstrap_failed:
        raise typer.Exit(code=1)


def parse_fail_under(items: list[str]) -> dict[str, dict[int | None, float]]:
    """Parse repeated ``name=value`` or ``name@k=value`` thresholds.

    Names are hit_rate, recall, precision, mrr, and ndcg. A bare name is
    stored under cutoff ``None`` and later bound to the largest k. The
    value is the minimum allowed aggregate, from 0 to 1 inclusive.
    """

    allowed = set(METRIC_NAMES)
    thresholds: dict[str, dict[int | None, float]] = {}
    for item in items:
        if "=" not in item:
            raise ConfigError(f"--fail-under {item!r} must look like name=value")
        name, raw = item.split("=", 1)
        name = name.strip()
        metric, cutoff = _split_metric_cutoff(name)
        if metric not in allowed:
            names = ", ".join(METRIC_NAMES)
            raise ConfigError(f"unknown metric {metric!r} in --fail-under (expected {names})")
        limits = thresholds.setdefault(metric, {})
        if cutoff in limits:
            label = metric if cutoff is None else f"{metric}@{cutoff}"
            raise ConfigError(f"--fail-under metric {label!r} was given twice")
        limits[cutoff] = _threshold_value("--fail-under", item, metric, raw)
    return thresholds


def bind_thresholds(
    parsed: dict[str, dict[int | None, float]],
    cutoffs: list[int],
) -> dict[int, dict[str, float]]:
    """Attach each threshold to a cutoff that this run scores.

    A bare metric name uses the largest cutoff. ``name@k`` has to name one
    of ``cutoffs``. The same metric at the same cutoff is an error, including
    a bare name together with ``name@k`` when that k is the largest.
    """

    largest = cutoffs[-1]
    allowed = set(cutoffs)
    bound: dict[int, dict[str, float]] = {cutoff: {} for cutoff in cutoffs}
    for metric, limits in parsed.items():
        for cutoff, value in limits.items():
            target = largest if cutoff is None else cutoff
            if target not in allowed:
                shown = ", ".join(str(item) for item in cutoffs)
                label = metric if cutoff is None else f"{metric}@{cutoff}"
                raise ConfigError(f"--fail-under {label} does not match a cutoff ({shown})")
            if metric in bound[target]:
                raise ConfigError(
                    f"--fail-under metric {metric!r} was given twice for k={target}"
                )
            bound[target][metric] = value
    return bound


def parse_k_values(text: str, *, label: str = "--k-values") -> list[int]:
    """Parse ``1,3,5,10`` into ascending cutoffs.

    Duplicates are an error. Values are sorted so the last one is what the
    retriever is called with.
    """

    if not isinstance(text, str) or not text.strip():
        raise ConfigError(
            f"{label} must be a comma-separated list of integers >= 1, like 1,3,5,10"
        )
    parts = [part.strip() for part in text.split(",")]
    if any(not part for part in parts):
        raise ConfigError(
            f"{label} must be a comma-separated list of integers >= 1, like 1,3,5,10"
        )
    values = []
    for part in parts:
        if not re.fullmatch(r"[0-9]+", part):
            raise ConfigError(f"{label} entry {part!r} is not an integer >= 1")
        value = int(part)
        if value < 1:
            raise ConfigError(f"{label} entry {part!r} is not an integer >= 1")
        values.append(value)
    if len(values) != len(set(values)):
        raise ConfigError(f"{label} contains a duplicate")
    return sorted(values)


def _split_metric_cutoff(name: str) -> tuple[str, int | None]:
    metric, sep, raw_k = name.partition("@")
    if not sep:
        return metric, None
    if not re.fullmatch(r"[0-9]+", raw_k):
        return name, None
    cutoff = int(raw_k)
    if cutoff < 1:
        raise ConfigError(f"--fail-under cutoff in {name!r} must be >= 1")
    return metric, cutoff


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


def _prepare_retriever(
    retriever: str | None,
    retriever_url: str | None,
    adapter: str | None,
    method: str | None,
    header: list[str] | None,
    timeout: float | None,
    concurrency: int | None,
    results_key: str | None,
    id_field: str | None,
    cfg: dict,
) -> tuple:
    """Pick the import path or the HTTP service and return ``(fn, name, concurrency)``."""

    if retriever and retriever_url:
        raise ConfigError("pass --retriever or --retriever-url, not both")

    from_config = retriever is None and retriever_url is None
    if retriever:
        spec, url = retriever, None
    elif retriever_url:
        spec, url = None, retriever_url
    else:
        spec, url = cfg["retriever"], cfg["retriever_url"]
        if spec and url:
            raise ConfigError("config sets both retriever and retriever_url")

    if spec is None and url is None:
        raise ConfigError("pass --retriever or --retriever-url, or set one in --config")

    if url is not None and (adapter is not None or (from_config and cfg["adapter"] is not None)):
        raise ConfigError("adapter requires a retriever import path")

    cli_http = _first_set(
        ("method", method),
        ("header", header),
        ("timeout", timeout),
        ("concurrency", concurrency),
        ("results_key", results_key),
    )
    if spec is not None:
        if cli_http:
            raise ConfigError(f"{_HTTP_FLAGS[cli_http]} requires --retriever-url")
        if from_config:
            config_http = _first_set(
                ("method", cfg["method"]),
                ("headers", cfg["headers"]),
                ("timeout", cfg["timeout"]),
                ("concurrency", cfg["concurrency"]),
                ("results_key", cfg["results_key"]),
            )
            if config_http:
                raise ConfigError(f"config key {config_http} requires retriever_url")
        if adapter is not None:
            chosen_adapter = normalize_adapter(adapter)
        elif from_config:
            chosen_adapter = cfg["adapter"]
        else:
            chosen_adapter = None
        if id_field is not None:
            chosen_id_field = id_field
        elif from_config:
            chosen_id_field = cfg["id_field"]
        else:
            chosen_id_field = None
        return (
            load_retriever(
                spec,
                adapter=chosen_adapter or "auto",
                id_field=chosen_id_field,
            ),
            spec,
            1,
        )

    chosen_method = method if method is not None else cfg["method"]
    chosen_headers = list(header) if header else list(cfg["headers"])
    chosen_timeout = timeout if timeout is not None else cfg["timeout"]
    chosen_concurrency = concurrency if concurrency is not None else cfg["concurrency"]
    chosen_results_key = results_key if results_key is not None else cfg["results_key"]
    chosen_id_field = id_field if id_field is not None else cfg["id_field"]
    fn = build_http_retriever(
        url,
        method=chosen_method or "POST",
        headers=chosen_headers,
        timeout=DEFAULT_TIMEOUT if chosen_timeout is None else chosen_timeout,
        results_key=chosen_results_key,
        id_field=chosen_id_field or DEFAULT_ID_FIELD,
    )
    limit = DEFAULT_CONCURRENCY if chosen_concurrency is None else chosen_concurrency
    limit = checked_concurrency(limit)
    return fn, public_url(url), limit


def _first_set(*pairs: tuple[str, object]) -> str | None:
    for name, value in pairs:
        if _is_set(value):
            return name
    return None


def _is_set(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, (list, tuple, str)) and len(value) == 0:
        return False
    return True


def load_config(path: Path) -> dict:
    """Load run settings from a YAML file.

    Paths are resolved relative to the config file. ``fail_under`` may be
    a mapping of metric name to limit, or a list of ``name=value`` strings.
    ``fail_under_tag`` may be a mapping of tag to metric limits, or a list
    of ``tag:metric=value`` strings. A ``fail_under`` metric may be
    ``recall`` or ``recall@5``. ``k_values`` is a list of cutoffs or the
    same comma-separated text as ``--k-values``. Set ``k`` or ``k_values``,
    not both. ``retriever_url`` calls an HTTP
    service. ``method``, ``headers``, ``timeout``, ``concurrency``,
    and ``results_key`` apply to that service. ``id_field`` applies to
    the service and to a LangChain or LlamaIndex retriever. ``adapter``
    is auto, langchain, or llamaindex.
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
        "retriever_url": _config_text(path, raw.get("retriever_url"), "retriever_url"),
        "method": _config_method(path, raw.get("method")),
        "headers": _config_headers(path, raw.get("headers")),
        "timeout": _config_timeout(path, raw.get("timeout")),
        "concurrency": _config_concurrency(path, raw.get("concurrency")),
        "results_key": _config_text(path, raw.get("results_key"), "results_key"),
        "id_field": _config_text(path, raw.get("id_field"), "id_field"),
        "adapter": _config_adapter(path, raw.get("adapter")),
        "k": _config_k(path, raw.get("k")),
        "k_values": _config_k_values(path, raw.get("k_values")),
        "baseline": _config_path(path, raw.get("baseline"), "baseline"),
        "max_drop": _config_max_drop(path, raw.get("max_drop")),
        "thresholds": _config_thresholds(path, raw.get("fail_under")),
        "tag_thresholds": _config_tag_thresholds(path, raw.get("fail_under_tag")),
    }


def _empty_config() -> dict:
    return {
        "dataset": None,
        "retriever": None,
        "retriever_url": None,
        "method": None,
        "headers": [],
        "timeout": None,
        "concurrency": None,
        "results_key": None,
        "id_field": None,
        "adapter": None,
        "k": None,
        "k_values": None,
        "baseline": None,
        "max_drop": None,
        "thresholds": {},
        "tag_thresholds": {},
    }


def _choose_cutoffs(
    k_flag: int | None,
    k_values_flag: str | None,
    configured_k: int | None,
    configured_k_values: list[int] | None,
) -> list[int]:
    """Pick the cutoffs for this run.

    ``--k`` and ``--k-values`` together are an error. A flag replaces both
    config keys. A config file that sets both ``k`` and ``k_values`` is an
    error when neither flag is set. The default is a single cutoff of 5.
    """

    if k_flag is not None and k_values_flag is not None:
        raise ConfigError("pass --k or --k-values, not both")
    if k_values_flag is not None:
        return parse_k_values(k_values_flag)
    if k_flag is not None:
        if isinstance(k_flag, bool) or not isinstance(k_flag, int) or k_flag < 1:
            raise ConfigError("k must be >= 1")
        return [k_flag]
    if configured_k is not None and configured_k_values is not None:
        raise ConfigError("config sets both k and k_values")
    if configured_k_values is not None:
        return list(configured_k_values)
    if configured_k is not None:
        if configured_k < 1:
            raise ConfigError("k must be >= 1")
        return [configured_k]
    return [5]


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


def _config_adapter(path: Path, value: object) -> str | None:
    text = _config_text(path, value, "adapter")
    if text is None:
        return None
    try:
        return normalize_adapter(text)
    except ConfigError as exc:
        raise ConfigError(f"{path}: {exc}") from exc


def _config_text(path: Path, value: object, key: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{path}: {key} must be a string")
    return value.strip()


def _config_method(path: Path, value: object) -> str | None:
    text = _config_text(path, value, "method")
    if text is None:
        return None
    try:
        return normalize_method(text)
    except ConfigError as exc:
        raise ConfigError(f"{path}: {exc}") from exc


def _config_headers(path: Path, value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        items = []
        for item in value:
            if not isinstance(item, str):
                raise ConfigError(f"{path}: headers entries must be 'Name: Value' strings")
            items.append(item)
        return items
    if isinstance(value, dict):
        items = []
        for name, raw in value.items():
            if not isinstance(name, str) or not isinstance(raw, str):
                raise ConfigError(f"{path}: headers values must be strings")
            items.append(f"{name}: {raw}")
        return items
    raise ConfigError(f"{path}: headers must be a list or a mapping")


def _config_timeout(path: Path, value: object) -> float | None:
    if value is None:
        return None
    try:
        return checked_timeout(value)
    except ConfigError as exc:
        raise ConfigError(f"{path}: {exc}") from exc


def _config_concurrency(path: Path, value: object) -> int | None:
    if value is None:
        return None
    try:
        return checked_concurrency(value)
    except ConfigError as exc:
        raise ConfigError(f"{path}: {exc}") from exc


def _config_k(path: Path, value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ConfigError(f"{path}: k must be an integer >= 1")
    return value


def _config_k_values(path: Path, value: object) -> list[int] | None:
    if value is None:
        return None
    try:
        if isinstance(value, str):
            return parse_k_values(value, label="k_values")
        if isinstance(value, list):
            if not value:
                raise ConfigError("k_values must list at least one integer")
            rendered = []
            for item in value:
                if isinstance(item, bool) or not isinstance(item, int):
                    raise ConfigError("k_values entries must be integers >= 1")
                rendered.append(str(item))
            return parse_k_values(",".join(rendered), label="k_values")
        raise ConfigError("k_values must be a list of integers or a comma-separated string")
    except ConfigError as exc:
        raise ConfigError(f"{path}: {exc}") from exc


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


def _checked_bootstrap(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ConfigError("--bootstrap must be an integer >= 0")
    return value


def _checked_confidence(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ConfigError("--confidence must be greater than 0 and less than 1")
    if not 0.0 < float(value) < 1.0:
        raise ConfigError("--confidence must be greater than 0 and less than 1")
    return float(value)


def _checked_seed(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError("--seed must be an integer")
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
