"""Latency stats, gates, and compare."""

import asyncio
import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from typer.testing import CliRunner

from retrieval_gate import latency
from retrieval_gate.adapters import adapt_retriever
from retrieval_gate.cli import app
from retrieval_gate.http_retriever import build_http_retriever
from retrieval_gate.models import Case
from retrieval_gate.report import _retrieve_all, build_report, format_markdown, format_report
from tests import fakes

runner = CliRunner()
ROOT = Path(__file__).resolve().parents[1]


def _output(result) -> str:
    return (result.stdout or "") + (result.stderr or "")


def _case(case_id: str, query: str, relevant: list[str] | None = None) -> Case:
    return Case.model_validate(
        {"id": case_id, "query": query, "relevant": relevant or ["d1"]}
    )


def _write_dataset(path: Path) -> None:
    path.write_text(
        '{"id": "good", "query": "cats", "relevant": ["d1", "d2"]}\n'
        '{"id": "bad", "query": "none", "relevant": ["d1"]}\n',
        encoding="utf-8",
    )


class ManualClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_clock_reads_perf_counter(monkeypatch):
    seen = {}

    def fake() -> float:
        seen["called"] = True
        return 1.25

    monkeypatch.setattr(latency.time, "perf_counter", fake)
    assert latency.clock() == 1.25
    assert seen["called"] is True


def test_percentile_linear_for_one_and_two_samples():
    assert latency.percentile([10.0], 0.95) == 10.0
    single = latency.summarize([10.0])
    assert single == {"mean": 10.0, "p50": 10.0, "p95": 10.0, "p99": 10.0, "max": 10.0}

    # Nearest rank would pick the slower sample for p95. Linear does not.
    assert latency.percentile([10.0, 20.0], 0.5) == 15.0
    assert latency.percentile([10.0, 20.0], 0.95) == 19.5
    assert latency.percentile([10.0, 20.0], 0.99) == pytest.approx(19.9)
    pair = latency.summarize([20.0, 10.0])
    assert pair["mean"] == 15.0
    assert pair["p50"] == 15.0
    assert pair["p95"] == 19.5
    assert pair["p99"] == 19.9
    assert pair["max"] == 20.0


def test_percentile_interpolates_a_longer_list():
    values = [0.0, 1.0, 2.0, 3.0]
    assert latency.percentile(values, 0.0) == 0.0
    assert latency.percentile(values, 1.0) == 3.0
    assert latency.percentile(values, 0.5) == 1.5
    assert latency.percentile(values, 0.95) == pytest.approx(2.85)
    stats = latency.summarize(values)
    assert stats["mean"] == 1.5
    assert stats["p50"] == 1.5
    assert stats["max"] == 3.0


def test_increase_boundary_and_zero_baseline():
    assert latency.increase_exceeded(100.0, 120.0, 20) is False
    assert latency.increase_exceeded(100.0, 120.0001, 20) is True
    assert latency.increase_exceeded(0.0, 0.0, 20) is False
    assert latency.increase_exceeded(0.0, 0.1, 20) is True
    stats = {"mean": 10.0, "p50": 10.0, "p95": 19.5, "p99": 19.9, "max": 20.0}
    assert latency.exceeded(stats, {"p95": 19.5}) == []
    assert latency.exceeded(stats, {"p95": 19.4}) == ["p95"]


def _score(
    monkeypatch,
    retriever,
    durations,
    *,
    concurrency=1,
    k=2,
    cases=None,
    warmup=0,
    limits=None,
):
    clock = fakes.scripted_clock(durations)
    monkeypatch.setattr(latency, "clock", clock)
    chosen = cases or [_case("q", "cats")]
    report = build_report(
        chosen,
        retriever,
        k,
        dataset="cases.jsonl",
        retriever_name="local",
        thresholds={},
        concurrency=concurrency,
        warmup=warmup,
        latency_thresholds=limits,
    )
    assert clock.remaining() == 0
    return report


def test_sync_and_async_import_paths_are_timed(monkeypatch):
    def sync(query: str, k: int) -> list[str]:
        return ["d1", "d2"]

    async def async_retrieve(query: str, k: int) -> list[str]:
        return ["d1", "d2"]

    sync_report = _score(
        monkeypatch,
        sync,
        [0.010, 0.020],
        cases=[_case("a", "cats"), _case("b", "none")],
    )
    assert [row["latency_ms"] for row in sync_report["results"]] == [10.0, 20.0]
    assert sync_report["latency"]["p95"] == 19.5
    assert sync_report["metrics"]["hit_rate"] == 1.0

    async_report = _score(monkeypatch, async_retrieve, [0.004])
    assert async_report["results"][0]["latency_ms"] == 4.0
    assert async_report["results"][0]["retrieved"] == ["d1", "d2"]


def test_langchain_and_llamaindex_are_timed(monkeypatch):
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever
    from llama_index.core.base.base_retriever import BaseRetriever as LlamaRetriever
    from llama_index.core.schema import NodeWithScore, TextNode

    class OneDoc(BaseRetriever):
        def _get_relevant_documents(self, query: str, *, run_manager) -> list[Document]:
            return [Document(id="d1", page_content=query)]

    class OneNode(LlamaRetriever):
        def _retrieve(self, query_bundle) -> list[NodeWithScore]:
            return [NodeWithScore(node=TextNode(text="a", id_="d1"), score=1.0)]

    langchain = adapt_retriever(OneDoc(), adapter="langchain", label="app:docs")
    report = _score(monkeypatch, langchain, [0.007])
    assert report["results"][0]["retrieved"] == ["d1"]
    assert report["results"][0]["latency_ms"] == 7.0
    assert report["latency"]["method"] == "linear"

    llama = adapt_retriever(OneNode(), adapter="llamaindex", label="app:index")
    report = _score(monkeypatch, llama, [0.009])
    assert report["results"][0]["retrieved"] == ["d1"]
    assert report["results"][0]["latency_ms"] == 9.0


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: object) -> None:
        return

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0") or 0)
        if length:
            self.rfile.read(length)
        body = json.dumps({"results": [{"id": "d1"}, {"id": "d2"}]}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@contextmanager
def _server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(
        target=server.serve_forever,
        kwargs={"poll_interval": 0.05},
        daemon=True,
    )
    thread.start()
    host, port = server.server_address
    try:
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_http_times_each_request_including_under_concurrency(monkeypatch):
    with _server() as base:
        fn = build_http_retriever(f"{base}/search")
        one = _score(monkeypatch, fn, [0.012], concurrency=1, k=1)
        assert one["results"][0]["latency_ms"] == 12.0
        assert one["results"][0]["retrieved"] == ["d1"]

        cases = [_case(f"q{index}", "cats") for index in range(3)]
        many = _score(
            monkeypatch,
            fn,
            [0.004, 0.008, 0.012],
            concurrency=2,
            k=1,
            cases=cases,
        )
    assert sorted(row["latency_ms"] for row in many["results"]) == [4.0, 8.0, 12.0]
    assert many["latency"]["max"] == 12.0
    assert len(many["results"]) == 3


def test_warmup_is_not_scored_or_timed(monkeypatch):
    seen: list[tuple[str, str]] = []

    def retrieve(query: str, k: int) -> list[str]:
        seen.append((query, "warmup" if len(seen) == 0 else "score"))
        if len(seen) == 1:
            return ["not-scored"]
        return ["d1", "d2"]

    cases = [_case("a", "first", ["d1"]), _case("b", "second", ["d1"])]
    report = _score(monkeypatch, retrieve, [0.010, 0.030], cases=cases, warmup=1)
    assert seen == [("first", "warmup"), ("first", "score"), ("second", "score")]
    assert [row["latency_ms"] for row in report["results"]] == [10.0, 30.0]
    assert report["results"][0]["retrieved"] == ["d1", "d2"]
    assert report["latency"]["warmup"] == 1
    assert "warmup: 1" in format_report(report)
    assert report["metrics"]["hit_rate"] == 1.0


def test_k_values_records_one_latency_per_case(monkeypatch, tmp_path):
    dataset = tmp_path / "cases.jsonl"
    _write_dataset(dataset)
    clock = fakes.scripted_clock([0.010, 0.020])
    monkeypatch.setattr(latency, "clock", clock)
    fakes.calls.clear()
    result = runner.invoke(
        app,
        [
            "run",
            "--dataset",
            str(dataset),
            "--retriever",
            "tests.fakes:retrieve",
            "--k-values",
            "1,3",
            "--json",
            str(tmp_path / "report.json"),
        ],
    )
    assert result.exit_code == 0, _output(result)
    assert fakes.calls == [("cats", 3), ("none", 3)]
    assert clock.remaining() == 0
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert [row["latency_ms"] for row in report["results"]] == [10.0, 20.0]
    assert "latency" in report
    assert "p95" not in report["cutoffs"][0]
    assert report["metrics"]["recall"] == 0.5


def test_concurrency_wait_is_not_timed(monkeypatch):
    clock = ManualClock()
    monkeypatch.setattr(latency, "clock", clock)
    second_in = asyncio.Event()
    release = asyncio.Event()

    async def retrieve(query: str, k: int) -> list[str]:
        if query == "c":
            clock.advance(0.010)
            return ["d1"]
        if query == "b":
            second_in.set()
            await release.wait()
            clock.advance(0.005)
            return ["d1"]
        await second_in.wait()
        clock.advance(0.030)
        release.set()
        return ["d1"]

    cases = [_case(name, name) for name in ("a", "b", "c")]

    async def run():
        return await asyncio.wait_for(
            _retrieve_all(retrieve, cases, 1, 2, timed=True),
            timeout=2,
        )

    _ids, samples = asyncio.run(run())
    assert dict(zip(("a", "b", "c"), samples, strict=True))["c"] == 10.0


def test_latency_gate_pass_fail_and_json_shape(monkeypatch, tmp_path):
    dataset = tmp_path / "cases.jsonl"
    _write_dataset(dataset)
    json_path = tmp_path / "report.json"
    monkeypatch.setattr(latency, "clock", fakes.scripted_clock([0.010, 0.020]))
    passed = runner.invoke(
        app,
        [
            "run",
            "--dataset",
            str(dataset),
            "--retriever",
            "tests.fakes:retrieve",
            "--k",
            "2",
            "--fail-under",
            "recall=0.5",
            "--max-latency",
            "p95=19.5",
            "--json",
            str(json_path),
            "--format",
            "markdown",
        ],
    )
    assert passed.exit_code == 0, _output(passed)
    text = passed.stdout or ""
    assert "| p95 | 19.5000 | 19.5000 | pass |" in text
    assert "failed latency" not in text
    report = json.loads(json_path.read_text(encoding="utf-8"))
    assert report["metrics"] == {
        "hit_rate": 0.5,
        "recall": 0.5,
        "precision": 0.5,
        "mrr": 0.5,
        "ndcg": 0.5,
    }
    assert report["failed"] == []
    assert report["latency"] == {
        "unit": "ms",
        "method": "linear",
        "warmup": 0,
        "mean": 15.0,
        "p50": 15.0,
        "p95": 19.5,
        "p99": 19.9,
        "max": 20.0,
        "thresholds": {"p95": 19.5},
        "failed": [],
    }
    assert [row["latency_ms"] for row in report["results"]] == [10.0, 20.0]
    assert "latency_ms" not in report["latency"]

    monkeypatch.setattr(latency, "clock", fakes.scripted_clock([0.010, 0.020]))
    failed = runner.invoke(
        app,
        [
            "run",
            "--dataset",
            str(dataset),
            "--retriever",
            "tests.fakes:retrieve",
            "--k",
            "2",
            "--max-latency",
            "p95=19.4",
            "--max-latency",
            "max=20",
        ],
    )
    failed_text = failed.stdout or ""
    assert failed.exit_code == 1, _output(failed)
    assert "failed latency: p95" in failed_text
    assert "failed: recall" not in failed_text
    assert "FAIL" in failed_text
    plain = format_report(
        {
            "dataset": "cases.jsonl",
            "retriever": "local",
            "k": 2,
            "case_count": 2,
            "metrics": report["metrics"],
            "thresholds": {},
            "failed": [],
            "results": report["results"],
            "latency": report["latency"],
        }
    )
    assert "latency (ms)" in plain
    assert plain.index("ndcg") < plain.index("latency (ms)")


def test_config_yaml_json_and_flag_override(monkeypatch, tmp_path):
    dataset = tmp_path / "cases.jsonl"
    _write_dataset(dataset)
    yaml_config = tmp_path / "gate.yml"
    yaml_config.write_text(
        "dataset: cases.jsonl\n"
        "retriever: tests.fakes:retrieve\n"
        "k: 2\n"
        "warmup: 1\n"
        "max_latency:\n"
        "  p95: 19.4\n",
        encoding="utf-8",
    )
    fakes.calls.clear()
    monkeypatch.setattr(latency, "clock", fakes.scripted_clock([0.010, 0.020]))
    from_yaml = runner.invoke(app, ["run", "--config", str(yaml_config)])
    assert from_yaml.exit_code == 1, _output(from_yaml)
    assert fakes.calls == [("cats", 2), ("cats", 2), ("none", 2)]
    assert "failed latency: p95" in (from_yaml.stdout or "")

    fakes.calls.clear()
    monkeypatch.setattr(latency, "clock", fakes.scripted_clock([0.010, 0.020]))
    overridden = runner.invoke(
        app,
        ["run", "--config", str(yaml_config), "--warmup", "0", "--max-latency", "p95=19.5"],
    )
    assert overridden.exit_code == 0, _output(overridden)
    assert fakes.calls == [("cats", 2), ("none", 2)]

    json_config = tmp_path / "gate.json"
    json_config.write_text(
        json.dumps(
            {
                "dataset": "cases.jsonl",
                "retriever": "tests.fakes:retrieve",
                "k": 2,
                "warmup": 0,
                "max_latency": ["mean=100", "max=100"],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(latency, "clock", fakes.scripted_clock([0.010, 0.020]))
    from_json = runner.invoke(app, ["run", "--config", str(json_config)])
    assert from_json.exit_code == 0, _output(from_json)

    listed = tmp_path / "listed.yml"
    listed.write_text(
        "dataset: cases.jsonl\n"
        "retriever: tests.fakes:retrieve\n"
        "max_latency:\n"
        "  - p95=1\n"
        "  - p95=2\n",
        encoding="utf-8",
    )
    duplicate = runner.invoke(app, ["run", "--config", str(listed)])
    assert duplicate.exit_code == 2
    assert "twice" in _output(duplicate)


def test_bad_latency_flags_exit_2(tmp_path):
    dataset = tmp_path / "cases.jsonl"
    _write_dataset(dataset)
    base = ["run", "--dataset", str(dataset), "--retriever", "tests.fakes:retrieve", "--k", "2"]

    unknown = runner.invoke(app, [*base, "--max-latency", "p90=10"])
    assert unknown.exit_code == 2
    assert "unknown latency stat" in _output(unknown)

    negative = runner.invoke(app, [*base, "--max-latency", "p95=-1"])
    assert negative.exit_code == 2
    assert ">= 0" in _output(negative)

    missing = runner.invoke(app, [*base, "--max-latency", "p95"])
    assert missing.exit_code == 2

    too_many = runner.invoke(app, [*base, "--warmup", "3"])
    assert too_many.exit_code == 2
    assert "larger than the dataset" in _output(too_many)

    bad_warmup = runner.invoke(app, [*base, "--warmup", "-1"])
    assert bad_warmup.exit_code == 2

    bad_stat = runner.invoke(
        app,
        ["compare", str(dataset), str(dataset), "--latency-stat", "average"],
    )
    assert bad_stat.exit_code == 2
    assert "--latency-stat" in _output(bad_stat)

    bad_increase = runner.invoke(
        app,
        ["compare", str(dataset), str(dataset), "--max-latency-increase", "-5"],
    )
    assert bad_increase.exit_code == 2
    assert "--max-latency-increase" in _output(bad_increase)

    config = tmp_path / "bad.yml"
    config.write_text("dataset: cases.jsonl\nwarmup: 1.5\n", encoding="utf-8")
    bad_config = runner.invoke(app, ["run", "--config", str(config)])
    assert bad_config.exit_code == 2
    assert "warmup" in _output(bad_config)


def _metrics(score: float) -> dict[str, float]:
    return {name: score for name in ("hit_rate", "recall", "precision", "mrr", "ndcg")}


def _report(path: Path, score: float, latency_stats: dict | None) -> None:
    payload: dict = {"k": 5, "metrics": _metrics(score)}
    if latency_stats is not None:
        payload["latency"] = {
            "unit": "ms",
            "method": "linear",
            "warmup": 0,
            **latency_stats,
            "thresholds": {},
            "failed": [],
        }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_compare_latency_delta_and_increase(tmp_path):
    baseline = tmp_path / "baseline.json"
    current = tmp_path / "current.json"
    stats = {"mean": 10.0, "p50": 10.0, "p95": 100.0, "p99": 100.0, "max": 100.0}
    _report(baseline, 1.0, stats)
    grown = dict(stats)
    grown["p95"] = 120.0
    _report(current, 1.0, grown)

    equal = runner.invoke(
        app,
        [
            "compare",
            str(baseline),
            str(current),
            "--max-latency-increase",
            "20",
            "--json",
            str(tmp_path / "equal.json"),
        ],
    )
    equal_text = equal.stdout or ""
    assert equal.exit_code == 0, _output(equal)
    assert "max latency increase: 20% on p95" in equal_text
    assert "100.0000" in equal_text and "120.0000" in equal_text
    assert "+20.0000" in equal_text
    assert "failed latency" not in equal_text
    saved = json.loads((tmp_path / "equal.json").read_text(encoding="utf-8"))
    assert saved["latency"]["available"] is True
    assert saved["latency"]["checked"] is True
    assert saved["latency"]["status"] == "pass"
    assert saved["latency"]["stat"] == "p95"
    assert saved["latency"]["delta"]["p95"] == 20.0
    assert saved["regressed"] == []

    grown["p95"] = 121.0
    _report(current, 1.0, grown)
    over = runner.invoke(
        app,
        [
            "compare",
            str(baseline),
            str(current),
            "--max-latency-increase",
            "20",
            "--format",
            "markdown",
        ],
    )
    assert over.exit_code == 1, _output(over)
    assert "failed latency: p95" in (over.stdout or "")
    assert "| p95 | 100.0000 | 121.0000 | +21.0000 | FAIL |" in (over.stdout or "")

    # The gate follows --latency-stat. A p95 jump does not fail a mean check.
    mean_ok = runner.invoke(
        app,
        [
            "compare",
            str(baseline),
            str(current),
            "--max-latency-increase",
            "20",
            "--latency-stat",
            "mean",
        ],
    )
    assert mean_ok.exit_code == 0, _output(mean_ok)
    assert "on mean" in (mean_ok.stdout or "")


def test_compare_without_baseline_latency(tmp_path):
    baseline = tmp_path / "baseline.json"
    current = tmp_path / "current.json"
    stats = {"mean": 10.0, "p50": 10.0, "p95": 10.0, "p99": 10.0, "max": 10.0}
    _report(baseline, 1.0, None)
    _report(current, 1.0, stats)
    missing = runner.invoke(
        app,
        [
            "compare",
            str(baseline),
            str(current),
            "--max-latency-increase",
            "1",
            "--json",
            str(tmp_path / "missing.json"),
            "--format",
            "markdown",
        ],
    )
    assert missing.exit_code == 0, _output(missing)
    assert "baseline report has no latency" in (missing.stdout or "")
    assert "failed latency" not in (missing.stdout or "")
    payload = json.loads((tmp_path / "missing.json").read_text(encoding="utf-8"))
    assert payload["latency"]["available"] is False
    assert payload["latency"]["message"] == "baseline report has no latency"
    assert payload["latency"]["checked"] is True

    broken = tmp_path / "broken.json"
    broken.write_text(
        json.dumps({"k": 5, "metrics": _metrics(1.0), "latency": {"mean": 1}}),
        encoding="utf-8",
    )
    partial = runner.invoke(
        app,
        ["compare", str(broken), str(current), "--max-latency-increase", "1"],
    )
    assert partial.exit_code == 0, _output(partial)
    assert "baseline report has no latency" in (partial.stdout or "")

    _report(baseline, 1.0, stats)
    _report(current, 1.0, None)
    other = runner.invoke(
        app,
        ["compare", str(baseline), str(current), "--max-latency-increase", "1"],
    )
    assert other.exit_code == 0, _output(other)
    assert "current report has no latency" in (other.stdout or "")

    # Older reports, and no latency flag, keep the comparison they had.
    _report(baseline, 1.0, None)
    _report(current, 0.99, None)
    plain = runner.invoke(
        app,
        ["compare", str(baseline), str(current), "--json", str(tmp_path / "old.json")],
    )
    assert plain.exit_code == 0, _output(plain)
    assert "latency" not in (plain.stdout or "")
    old = json.loads((tmp_path / "old.json").read_text(encoding="utf-8"))
    assert "latency" not in old
    markdown = format_markdown(
        {"metrics": _metrics(0.99), "thresholds": {}},
        baseline={"metrics": _metrics(1.0)},
        max_drop=0.02,
    )
    assert "latency" not in markdown


def test_run_markdown_shows_baseline_latency_or_a_note():
    current = {
        "metrics": _metrics(1.0),
        "thresholds": {"recall": 0.8},
        "latency": {
            "unit": "ms",
            "method": "linear",
            "warmup": 0,
            "mean": 12.0,
            "p50": 10.0,
            "p95": 20.0,
            "p99": 22.0,
            "max": 25.0,
            "thresholds": {"p95": 250.0},
            "failed": [],
        },
    }
    baseline = {
        "metrics": _metrics(1.0),
        "latency": {
            "mean": 10.0,
            "p50": 10.0,
            "p95": 16.0,
            "p99": 18.0,
            "max": 20.0,
        },
    }
    text = format_markdown(current, baseline=baseline, max_drop=0.02)
    assert "| p95 | 20.0000 | 250.0000 | 16.0000 | +4.0000 | pass |" in text
    assert text.count("### latency") == 1

    missing = format_markdown(current, baseline={"metrics": _metrics(1.0)}, max_drop=0.02)
    assert "| p95 | 20.0000 | 250.0000 | pass |" in missing
    assert "baseline report has no latency" in missing


def test_action_yml_accepts_latency_inputs():
    text = (ROOT / "action.yml").read_text(encoding="utf-8")
    assert "warmup:" in text
    assert "max-latency:" in text
    assert "max_latency" in text
    assert "RG_WARMUP" in text
    assert "RG_MAX_LATENCY" in text
    assert '--warmup "$RG_WARMUP"' in text
    assert '--max-latency "$item"' in text
