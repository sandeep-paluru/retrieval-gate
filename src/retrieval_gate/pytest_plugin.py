"""pytest plugin for retrieval-gate.

Pytest loads this module through the ``pytest11`` entry point when the
package is installed. Importing ``retrieval_gate`` does not import it.
A test that does not use the ``retrieval_gate`` fixture does not score
anything and does not read the config file.
"""

import math
from pathlib import Path

import pytest

from retrieval_gate.adapters import adapt_retriever
from retrieval_gate.cli import (
    _choose_cutoffs,
    _empty_config,
    bind_thresholds,
    load_config,
    parse_fail_under,
    parse_fail_under_tag,
    parse_max_latency,
)
from retrieval_gate.dataset import load_dataset
from retrieval_gate.errors import ConfigError
from retrieval_gate.latency import LATENCY_STATS
from retrieval_gate.metrics import METRIC_NAMES
from retrieval_gate.report import _format_misses, build_report, write_report
from retrieval_gate.retriever import load_retriever

_CONFIG_OPTION = "retrieval_gate_config"
_JSON_OPTION = "retrieval_gate_json"


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("retrieval-gate", "Score a retriever against labeled queries")
    group.addoption(
        "--retrieval-gate-config",
        action="store",
        default=None,
        dest=_CONFIG_OPTION,
        help="gate.yml file. Its thresholds apply to assert_scores().",
    )
    group.addoption(
        "--retrieval-gate-json",
        action="store",
        default=None,
        dest=_JSON_OPTION,
        help="Write the JSON report (same schema as retrieval-gate run --json).",
    )
    parser.addini(_CONFIG_OPTION, "Path to a retrieval-gate YAML config.", default=None)
    parser.addini(_JSON_OPTION, "Path to write the JSON report.", default=None)


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "retrieval_gate(config): gate.yml for the retrieval_gate fixture in this test. "
        "Overrides --retrieval-gate-config.",
    )


@pytest.fixture
def retrieval_gate(request: pytest.FixtureRequest) -> "RetrievalGate":
    """Score a labeled dataset.

    Pass a dataset path and a retriever. The retriever is a
    ``(query, k)`` callable, an import path (``module:attribute``), or a
    LangChain or LlamaIndex retriever. ``k`` defaults to 5, or to the
    value in the config file when the call omits it. The result's
    ``assert_scores`` fails the test when a score is below its threshold.
    """

    return RetrievalGate(request)


class RetrievalGate:
    """Callable fixture. One call scores one dataset."""

    def __init__(self, request: pytest.FixtureRequest) -> None:
        self._request = request

    def __call__(
        self,
        dataset: str | Path | None = None,
        retriever: object | None = None,
        *,
        k: int | None = None,
        k_values: list[int] | str | None = None,
        adapter: str | None = None,
        id_field: str | None = None,
        fail_under: dict | list | None = None,
        fail_under_tag: dict | list | None = None,
        warmup: int | None = None,
        max_latency: dict | list | None = None,
    ) -> "GateResult":
        """Score ``dataset`` with ``retriever`` and return the result.

        Either argument can be omitted when the config file sets it.
        ``fail_under`` uses the same names as the CLI: ``recall`` or
        ``recall@5``. A value passed here replaces the config file's
        list for that setting. ``adapter`` is ``auto``, ``langchain``,
        or ``llamaindex``.
        """

        cfg = _load_cfg(self._request)
        dataset_path, dataset_label = _dataset(dataset, cfg)
        fn, retriever_name = _open_retriever(retriever, cfg, adapter, id_field)
        cutoffs = _cutoffs(k, k_values, cfg)
        thresholds = bind_thresholds(_fail_under(fail_under, cfg), cutoffs)
        tag_thresholds = _fail_under_tag(fail_under_tag, cfg)
        chosen_warmup = cfg["warmup"] if warmup is None else warmup
        if chosen_warmup is None:
            chosen_warmup = 0
        latency_limits = _max_latency(max_latency, cfg)
        cases = load_dataset(dataset_path)
        largest = cutoffs[-1]
        multi = len(cutoffs) > 1
        report = build_report(
            cases,
            fn,
            largest,
            dataset=dataset_label,
            retriever_name=retriever_name,
            thresholds=thresholds[largest],
            tag_thresholds=tag_thresholds,
            k_values=cutoffs if multi else None,
            thresholds_by_k=thresholds if multi else None,
            warmup=chosen_warmup,
            latency_thresholds=latency_limits,
        )
        json_path = _json_path(self._request)
        if json_path is not None:
            write_report(json_path, report)
        return GateResult(report)


class GateResult:
    """Scores from one ``retrieval_gate`` call.

    ``report`` is the mapping ``retrieval-gate run --json`` writes.
    ``metrics`` is the aggregate at the largest cutoff.
    """

    def __init__(self, report: dict) -> None:
        self.report = report

    @property
    def metrics(self) -> dict[str, float]:
        return self.report["metrics"]

    def assert_scores(self, **thresholds: float) -> None:
        """Fail if a metric is below its threshold.

        Keyword names are ``hit_rate``, ``recall``, ``precision``,
        ``mrr``, and ``ndcg``. They override the same metric from the
        config file or from ``fail_under``. Other thresholds still
        apply, including ``name@k``, per-tag gates, and ``max_latency``.
        With no keywords, only those configured thresholds are checked.
        A score equal to its threshold passes.

        The failure lists each miss, the score, and the threshold, then
        the worst cases.
        """

        message = failure_message(self.report, _checked_overrides(thresholds))
        if message is not None:
            pytest.fail(message, pytrace=False)


def failure_message(report: dict, overrides: dict[str, float]) -> str | None:
    """Return the assertion text, or None when every threshold passes."""

    lines = [
        f"{label}: {score:.4f} < {limit:.4f}"
        for label, score, limit in _metric_misses(report, overrides)
    ]
    for label, score, limit in _latency_misses(report):
        lines.append(f"{label}: {score:.4f} ms > {limit:.4f} ms")
    if not lines:
        return None
    title = "worst misses" if not report.get("cutoffs") else f"worst misses at k={report['k']}"
    body = [
        "retrieval gate failed",
        *lines,
        "",
        title,
        _format_misses(report["results"]),
    ]
    return "\n".join(body)


def _metric_misses(report: dict, overrides: dict[str, float]) -> list[tuple[str, float, float]]:
    misses: list[tuple[str, float, float]] = []
    limits = dict(report.get("thresholds") or {})
    limits.update(overrides)
    metrics = report["metrics"]
    for name in METRIC_NAMES:
        limit = limits.get(name)
        if limit is not None and metrics[name] < limit:
            misses.append((name, metrics[name], limit))
    largest = report["k"]
    for row in report.get("cutoffs") or []:
        if row["k"] == largest:
            continue
        row_limits = row.get("thresholds") or {}
        for name in METRIC_NAMES:
            limit = row_limits.get(name)
            if limit is not None and row["metrics"][name] < limit:
                misses.append((f"{name}@{row['k']}", row["metrics"][name], limit))
    scores = {
        row["tag"]: row["metrics"]
        for row in report.get("tags") or []
        if isinstance(row, dict) and row.get("tag")
    }
    tag_limits = report.get("tag_thresholds") or {}
    for item in report.get("failed_tags") or []:
        tag = item["tag"]
        name = item["metric"]
        misses.append((f"{tag}:{name}", scores[tag][name], tag_limits[tag][name]))
    return misses


def _latency_misses(report: dict) -> list[tuple[str, float, float]]:
    block = report.get("latency") or {}
    failed = set(block.get("failed") or [])
    limits = block.get("thresholds") or {}
    return [
        (name, float(block[name]), float(limits[name]))
        for name in LATENCY_STATS
        if name in failed and name in limits
    ]


def _checked_overrides(thresholds: dict[str, float]) -> dict[str, float]:
    allowed = set(METRIC_NAMES)
    checked: dict[str, float] = {}
    for name, value in thresholds.items():
        if name not in allowed:
            names = ", ".join(METRIC_NAMES)
            raise ConfigError(f"unknown metric {name!r} (expected {names})")
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
        ):
            raise ConfigError(f"{name} threshold must be a number between 0 and 1")
        checked[name] = float(value)
    return checked


def _load_cfg(request: pytest.FixtureRequest) -> dict:
    path = _config_path(request)
    if path is None:
        return _empty_config()
    return load_config(path)


def _config_path(request: pytest.FixtureRequest) -> Path | None:
    marker = request.node.get_closest_marker("retrieval_gate")
    if marker is not None:
        unknown = set(marker.kwargs) - {"config"}
        if marker.args or unknown:
            raise ConfigError("retrieval_gate mark accepts only config='path/to/gate.yml'")
        raw = marker.kwargs.get("config")
        if raw:
            return _resolve(raw, request.config)
    return _option_path(request.config, _CONFIG_OPTION)


def _json_path(request: pytest.FixtureRequest) -> Path | None:
    return _option_path(request.config, _JSON_OPTION)


def _option_path(config: pytest.Config, name: str) -> Path | None:
    value = config.getoption(name)
    if _blank(value):
        value = config.getini(name)
    if _blank(value):
        return None
    return _resolve(value, config)


def _resolve(value: str | Path, config: pytest.Config) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(config.invocation_params.dir) / path


def _blank(value: object) -> bool:
    return value is None or value == ""


def _dataset(dataset: str | Path | None, cfg: dict) -> tuple[Path, str]:
    if dataset is None:
        path = cfg["dataset"]
        if path is None:
            raise ConfigError("pass a dataset, or set dataset in the config")
        return path, str(path)
    path = Path(dataset)
    return path, str(path)


def _open_retriever(
    retriever: object | None,
    cfg: dict,
    adapter: str | None,
    id_field: str | None,
) -> tuple:
    if retriever is None:
        if cfg["retriever"]:
            retriever = cfg["retriever"]
        elif cfg["retriever_url"]:
            raise ConfigError(
                "pass a retriever callable or import path. "
                "The pytest plugin does not call retriever_url."
            )
        else:
            raise ConfigError("pass a retriever, or set retriever in the config")
    chosen_adapter = adapter if adapter is not None else cfg["adapter"]
    if chosen_adapter is None:
        chosen_adapter = "auto"
    chosen_id = id_field if id_field is not None else cfg["id_field"]
    if isinstance(retriever, str):
        spec = retriever.strip()
        if not spec:
            raise ConfigError("retriever import path is empty")
        return (
            load_retriever(spec, adapter=chosen_adapter, id_field=chosen_id),
            spec,
        )
    label = _retriever_label(retriever)
    return (
        adapt_retriever(retriever, adapter=chosen_adapter, id_field=chosen_id, label=label),
        label,
    )


def _retriever_label(retriever: object) -> str:
    qualname = getattr(retriever, "__qualname__", None) or getattr(retriever, "__name__", None)
    module = getattr(retriever, "__module__", None)
    if isinstance(qualname, str) and qualname:
        if isinstance(module, str) and module not in {"__main__", "builtins"}:
            return f"{module}:{qualname}"
        return qualname
    return type(retriever).__name__


def _cutoffs(k: int | None, k_values: list[int] | str | None, cfg: dict) -> list[int]:
    if k_values is not None:
        if k is not None:
            raise ConfigError("pass k or k_values, not both")
        return _choose_cutoffs(None, _k_values_text(k_values), None, None)
    return _choose_cutoffs(k, None, cfg["k"], cfg["k_values"])


def _k_values_text(value: list[int] | str) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, bool) or not isinstance(item, int):
                raise ConfigError("k_values entries must be integers >= 1")
            parts.append(str(item))
        return ",".join(parts)
    raise ConfigError("k_values must be a list of integers or a comma-separated string")


def _fail_under(value: dict | list | None, cfg: dict) -> dict:
    if value is None:
        return cfg["thresholds"]
    try:
        return parse_fail_under(_named_items(value, "fail_under", "name=value"))
    except ConfigError as exc:
        raise ConfigError(str(exc).replace("--fail-under", "fail_under")) from exc


def _fail_under_tag(value: dict | list | None, cfg: dict) -> dict:
    if value is None:
        return cfg["tag_thresholds"]
    try:
        if isinstance(value, dict):
            items: list[str] = []
            for tag, limits in value.items():
                if not isinstance(limits, dict):
                    raise ConfigError(f"fail_under_tag.{tag} must be a mapping of metric to limit")
                for name, limit in limits.items():
                    if isinstance(limit, bool) or not isinstance(limit, (int, float)):
                        raise ConfigError(f"fail_under_tag.{tag}.{name} must be a number")
                    items.append(f"{tag}:{name}={limit}")
            return parse_fail_under_tag(items)
        return parse_fail_under_tag(_string_items(value, "fail_under_tag", "tag:metric=value"))
    except ConfigError as exc:
        raise ConfigError(str(exc).replace("--fail-under-tag", "fail_under_tag")) from exc


def _max_latency(value: dict | list | None, cfg: dict) -> dict:
    if value is None:
        return cfg["latency_thresholds"]
    try:
        return parse_max_latency(
            _named_items(value, "max_latency", "stat=milliseconds"),
            label="max_latency",
        )
    except ConfigError as exc:
        text = str(exc).replace("--max-latency", "max_latency")
        raise ConfigError(text) from exc


def _named_items(value: dict | list, label: str, shape: str) -> list[str]:
    if isinstance(value, dict):
        items: list[str] = []
        for name, limit in value.items():
            if isinstance(limit, bool) or not isinstance(limit, (int, float)):
                raise ConfigError(f"{label}.{name} must be a number")
            items.append(f"{name}={limit}")
        return items
    return _string_items(value, label, shape)


def _string_items(value: object, label: str, shape: str) -> list[str]:
    if not isinstance(value, list):
        raise ConfigError(f"{label} must be a mapping or a list of {shape} strings")
    items: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ConfigError(f"{label} entries must be {shape} strings")
        items.append(item)
    return items
