import subprocess
import sys
from pathlib import Path

from retrieval_gate.dataset import load_dataset
from retrieval_gate.report import build_report
from retrieval_gate.retriever import load_retriever

ROOT = Path(__file__).resolve().parents[1]


def test_example_files_describe_the_same_cases():
    jsonl = [case.model_dump() for case in load_dataset(ROOT / "examples/cases.jsonl")]
    yaml_cases = [case.model_dump() for case in load_dataset(ROOT / "examples/cases.yaml")]
    assert jsonl == yaml_cases
    assert len(jsonl) == 10


def test_example_retriever_ranks_the_primary_doc_first(monkeypatch):
    _primary_doc_is_first(monkeypatch, "examples.retriever:retrieve")


def _primary_doc_is_first(monkeypatch, spec: str):
    monkeypatch.chdir(ROOT)
    cases = load_dataset(ROOT / "examples/cases.jsonl")
    fn = load_retriever(spec)
    report = build_report(
        cases,
        fn,
        5,
        dataset="examples/cases.jsonl",
        retriever_name=spec,
        thresholds={"recall": 0.8, "mrr": 0.6},
    )
    assert report["failed"] == []
    assert report["metrics"]["hit_rate"] == 1.0
    for case, row in zip(cases, report["results"], strict=True):
        primary = max(case.relevant.items(), key=lambda item: item[1])[0]
        assert row["retrieved"][0] == primary, (case.id, row["retrieved"])


def test_langchain_example_ranks_the_primary_doc_first(monkeypatch):
    _primary_doc_is_first(monkeypatch, "examples.langchain_retriever:store")


def test_llamaindex_example_ranks_the_primary_doc_first(monkeypatch):
    _primary_doc_is_first(monkeypatch, "examples.llamaindex_retriever:index")


def test_console_script_quickstart():
    script = Path(sys.executable).parent / "retrieval-gate"
    result = subprocess.run(
        [
            str(script),
            "run",
            "--dataset",
            "examples/cases.jsonl",
            "--retriever",
            "examples.retriever:retrieve",
            "--k",
            "5",
            "--fail-under",
            "recall=0.8",
            "--fail-under",
            "mrr=0.6",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "hit_rate" in result.stdout
    assert "worst misses" in result.stdout


def _run_script(spec: str) -> subprocess.CompletedProcess[str]:
    script = Path(sys.executable).parent / "retrieval-gate"
    return subprocess.run(
        [
            str(script),
            "run",
            "--dataset",
            "examples/cases.jsonl",
            "--retriever",
            spec,
            "--k",
            "5",
            "--fail-under",
            "recall=0.8",
            "--fail-under",
            "mrr=0.6",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_langchain_example_command():
    result = _run_script("examples.langchain_retriever:store")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "hit_rate   1.0000" in result.stdout


def test_llamaindex_example_command():
    result = _run_script("examples.llamaindex_retriever:index")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "hit_rate   1.0000" in result.stdout
