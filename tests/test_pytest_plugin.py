"""The pytest plugin, exercised with pytester."""

import asyncio
import json
import subprocess
import sys
from importlib.metadata import entry_points

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore
from typer.testing import CliRunner

from retrieval_gate.cli import _empty_config, app
from retrieval_gate.errors import ConfigError
from retrieval_gate.metrics import METRIC_NAMES
from retrieval_gate.pytest_plugin import GateResult, _open_retriever, failure_message
from retrieval_gate.report import load_report

pytest_plugins = ["pytester"]

_CASES = (
    '{"id": "good", "query": "cats", "relevant": ["d1", "d2"]}\n'
    '{"id": "bad", "query": "none", "relevant": ["d1"]}\n'
)

_RETRIEVER = '''
LIBRARY = {"cats": ["d1", "d2", "d3"], "none": ["d8"]}


def retrieve(query, k):
    return list(LIBRARY[query])
'''

_GATE = """\
dataset: cases.jsonl
retriever: retriever:retrieve
k: 2
fail_under:
  recall: 0.8
  mrr: 0.4
"""

_CONFIG_TEST = """\
def test_scores(retrieval_gate):
    result = retrieval_gate()
    assert result.report["k"] == 2
    assert result.report["thresholds"]["recall"] == 0.8
    result.assert_scores()
"""

_MARKED_CONFIG_TEST = """\
import pytest


@pytest.mark.retrieval_gate(config="gate.yml")
def test_scores(retrieval_gate):
    result = retrieval_gate()
    assert result.report["k"] == 2
    assert result.report["thresholds"]["recall"] == 0.8
    result.assert_scores()
"""


class _FlatEmbedding(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]


def _isolate(pytester, monkeypatch, ini: str = "[pytest]\n") -> None:
    # A local pytest.ini keeps the nested run from picking up this repo's
    # pyproject testpaths.
    monkeypatch.chdir(pytester.path)
    (pytester.path / "pytest.ini").write_text(ini, encoding="utf-8")
    (pytester.path / "cases.jsonl").write_text(_CASES, encoding="utf-8")


def _run(pytester, *args: str):
    return pytester.runpytest("--color=no", *args)


def test_assert_scores_passes_when_metrics_meet_the_threshold(pytester, monkeypatch):
    _isolate(pytester, monkeypatch)
    pytester.makepyfile(
        """
        def test_scores(retrieval_gate):
            def retrieve(query, k):
                if query == "cats":
                    return ["d1", "d2", "d3"]
                return ["d8"]

            result = retrieval_gate("cases.jsonl", retrieve, k=2)
            assert result.metrics["recall"] == 0.5
            result.assert_scores(recall=0.5, mrr=0.5)
        """
    )
    result = _run(pytester)
    result.assert_outcomes(passed=1)


def test_assert_scores_lists_failing_metrics_and_worst_cases(pytester, monkeypatch):
    _isolate(pytester, monkeypatch)
    pytester.makepyfile(retriever=_RETRIEVER)
    pytester.makepyfile(
        """
        def test_scores(retrieval_gate):
            result = retrieval_gate("cases.jsonl", "retriever:retrieve", k=2)
            result.assert_scores(recall=0.8, mrr=0.6)
        """
    )
    result = _run(pytester)
    result.assert_outcomes(failed=1)
    text = result.stdout.str()
    assert "retrieval gate failed" in text
    assert "recall: 0.5000 < 0.8000" in text
    assert "mrr: 0.5000 < 0.6000" in text
    assert "worst misses" in text
    assert "bad   0.0000  d1" in text
    misses = text.split("worst misses", 1)[1]
    assert "good" not in misses


@pytest.mark.parametrize("mode", ["cli", "ini", "mark"])
def test_config_file_thresholds(pytester, monkeypatch, mode):
    ini = "[pytest]\n"
    if mode == "ini":
        ini = "[pytest]\nretrieval_gate_config = gate.yml\n"
    _isolate(pytester, monkeypatch, ini)
    pytester.makepyfile(retriever=_RETRIEVER)
    (pytester.path / "gate.yml").write_text(_GATE, encoding="utf-8")
    pytester.makepyfile(_MARKED_CONFIG_TEST if mode == "mark" else _CONFIG_TEST)
    args = ["--retrieval-gate-config", "gate.yml"] if mode == "cli" else []
    result = _run(pytester, *args)
    result.assert_outcomes(failed=1)
    text = result.stdout.str()
    assert "recall: 0.5000 < 0.8000" in text
    assert "mrr:" not in text
    assert "bad   0.0000  d1" in text


def test_json_report_matches_run_json(pytester, monkeypatch):
    _isolate(pytester, monkeypatch)
    pytester.makepyfile(retriever=_RETRIEVER)
    pytester.makepyfile(
        """
        def test_scores(retrieval_gate):
            result = retrieval_gate("cases.jsonl", "retriever:retrieve", k=2)
            result.assert_scores(recall=0.5)
        """
    )
    result = _run(pytester, "--retrieval-gate-json", "report.json")
    result.assert_outcomes(passed=1)
    report_path = pytester.path / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    loaded = load_report(report_path)
    assert loaded["metrics"] == report["metrics"]
    assert report["retriever"] == "retriever:retrieve"
    assert report["k"] == 2
    assert report["case_count"] == 2
    assert report["failed"] == []
    assert [row["id"] for row in report["results"]] == ["good", "bad"]
    assert report["results"][1]["misses"] == ["d1"]

    cli_path = pytester.path / "cli.json"
    cli = CliRunner().invoke(
        app,
        [
            "run",
            "--dataset",
            "cases.jsonl",
            "--retriever",
            "retriever:retrieve",
            "--k",
            "2",
            "--json",
            str(cli_path),
        ],
    )
    assert cli.exit_code == 0, (cli.stdout or "") + (cli.stderr or "")
    cli_report = json.loads(cli_path.read_text(encoding="utf-8"))
    assert _without_timings(report) == _without_timings(cli_report)


def test_plugin_is_inert_when_unused(pytester, monkeypatch):
    _isolate(pytester, monkeypatch)
    pytester.makepyfile(
        """
        def test_plain():
            assert 1 + 1 == 2
        """
    )
    result = _run(
        pytester,
        "--retrieval-gate-config",
        "missing.yml",
        "--retrieval-gate-json",
        "report.json",
    )
    result.assert_outcomes(passed=1)
    assert not (pytester.path / "report.json").exists()
    assert "retrieval gate failed" not in result.stdout.str()
    assert "missing.yml" not in result.stdout.str()
    listed = _run(pytester, "--fixtures")
    assert listed.ret == 0
    assert "retrieval_gate" in listed.stdout.str()
    assert "Score a labeled dataset" in listed.stdout.str()


def test_importing_the_package_does_not_load_the_plugin():
    script = (
        "import sys\n"
        "import retrieval_gate\n"
        "assert 'pytest' not in sys.modules\n"
        "assert 'retrieval_gate.pytest_plugin' not in sys.modules\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_pytest11_entry_point_is_registered():
    found = [item for item in entry_points(group="pytest11") if item.name == "retrieval_gate"]
    assert [item.value for item in found] == ["retrieval_gate.pytest_plugin"]


def test_langchain_store_is_scored_through_the_adapter():
    store = InMemoryVectorStore(embedding=_FlatEmbedding())
    store.add_documents([Document(id="d1", page_content="cats")])
    fn, _name = _open_retriever(store, _empty_config(), adapter=None, id_field=None)
    assert asyncio.run(fn("cats", 1)) == ["d1"]


def test_failure_message_includes_cutoff_tag_and_latency():
    metrics = {name: 1.0 for name in METRIC_NAMES}
    metrics["recall"] = 0.4
    smaller = dict(metrics)
    smaller["precision"] = 0.0
    report = {
        "k": 5,
        "metrics": metrics,
        "thresholds": {"recall": 0.8},
        "cutoffs": [
            {
                "k": 1,
                "metrics": smaller,
                "thresholds": {"precision": 0.5},
                "failed": ["precision"],
            },
            {
                "k": 5,
                "metrics": metrics,
                "thresholds": {"recall": 0.8},
                "failed": ["recall"],
            },
        ],
        "tags": [{"tag": "billing", "case_count": 1, "metrics": {**metrics, "recall": 0.2}}],
        "tag_thresholds": {"billing": {"recall": 0.5}},
        "failed_tags": [{"tag": "billing", "metric": "recall"}],
        "results": [{"id": "q1", "misses": ["d9"], "metrics": {"recall": 0.0}}],
        "latency": {
            "failed": ["p95"],
            "thresholds": {"p95": 10.0},
            "p95": 20.0,
        },
    }
    text = failure_message(report, {})
    assert text is not None
    assert "recall: 0.4000 < 0.8000" in text
    assert "precision@1: 0.0000 < 0.5000" in text
    assert "billing:recall: 0.2000 < 0.5000" in text
    assert "p95: 20.0000 ms > 10.0000 ms" in text
    assert "worst misses at k=5" in text
    assert "q1" in text and "d9" in text
    loosened = failure_message(report, {"recall": 0.4})
    assert loosened is not None
    assert "\nrecall:" not in loosened
    assert "billing:recall: 0.2000 < 0.5000" in loosened


def test_assert_scores_rejects_an_unknown_metric():
    report = {
        "k": 1,
        "metrics": {name: 1.0 for name in METRIC_NAMES},
        "thresholds": {},
        "results": [],
    }
    with pytest.raises(ConfigError, match="unknown metric 'recalll'"):
        GateResult(report).assert_scores(recalll=0.5)


def _without_timings(report: dict) -> dict:
    body = {key: value for key, value in report.items() if key != "latency"}
    body["results"] = [
        {key: value for key, value in row.items() if key != "latency_ms"}
        for row in body["results"]
    ]
    return body
