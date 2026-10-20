"""Command line interface for retrieval-gate."""

import math
from pathlib import Path

import typer

from retrieval_gate.dataset import load_dataset
from retrieval_gate.errors import ConfigError, DatasetError
from retrieval_gate.metrics import METRIC_NAMES
from retrieval_gate.report import (
    build_report,
    format_comparison,
    format_report,
    load_report,
    write_report,
)
from retrieval_gate.retriever import load_retriever

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    pretty_exceptions_enable=False,
    help="Score a retriever against labeled queries and fail when metrics drop.",
)


def main() -> None:
    app()


@app.command()
def run(
    dataset: Path = typer.Option(
        ...,
        "--dataset",
        "-d",
        help="JSONL or YAML file of labeled queries.",
    ),
    retriever: str = typer.Option(
        ...,
        "--retriever",
        "-r",
        help="Import path of the retriever, as module.submodule:function.",
    ),
    k: int = typer.Option(5, "--k", help="Number of unique results to score per query."),
    fail_under: list[str] | None = typer.Option(
        None,
        "--fail-under",
        help="Gate a metric, as name=value. Repeat for more than one metric.",
    ),
    json_output: Path | None = typer.Option(
        None,
        "--json",
        help="Write the JSON report to this path.",
    ),
) -> None:
    """Score a retriever on a labeled dataset."""

    try:
        if k < 1:
            raise ConfigError("k must be >= 1")
        thresholds = parse_fail_under(fail_under or [])
        cases = load_dataset(dataset)
        fn = load_retriever(retriever)
        report = build_report(
            cases,
            fn,
            k,
            dataset=str(dataset),
            retriever_name=retriever,
            thresholds=thresholds,
        )
        typer.echo(format_report(report))
        if json_output is not None:
            write_report(json_output, report)
    except (DatasetError, ConfigError) as exc:
        _fail(str(exc))
    if report["failed"]:
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
) -> None:
    """Compare two JSON reports and fail on a regression."""

    try:
        if not math.isfinite(max_drop) or max_drop < 0:
            raise ConfigError("--max-drop must be a number greater than or equal to 0")
        before = load_report(baseline)
        after = load_report(current)
        text, failed = format_comparison(baseline, current, before, after, max_drop)
    except (DatasetError, ConfigError) as exc:
        _fail(str(exc))
    typer.echo(text)
    if failed:
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
        try:
            value = float(raw)
        except ValueError as exc:
            raise ConfigError(f"--fail-under {item!r} value is not a number") from exc
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ConfigError(f"--fail-under {name} must be between 0 and 1")
        thresholds[name] = value
    return thresholds


def _fail(message: str) -> None:
    typer.echo(message, err=True)
    raise typer.Exit(code=2)
