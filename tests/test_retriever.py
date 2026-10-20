import pytest

from retrieval_gate.errors import ConfigError
from retrieval_gate.models import Case
from retrieval_gate.report import build_report
from retrieval_gate.retriever import coerce_ids, load_retriever
from tests import fakes


def test_coerce_ids_accepts_string_dict_and_object():
    class Doc:
        def __init__(self, id: str) -> None:
            self.id = id

    assert coerce_ids([" d1 ", {"id": "d2"}, Doc("d3")], case_id="q") == ["d1", "d2", "d3"]


def test_coerce_ids_rejects_bad_values():
    with pytest.raises(ConfigError, match=r"result\[0\]"):
        coerce_ids([{"name": "d1"}], case_id="q1")
    with pytest.raises(ConfigError, match="must return a list"):
        coerce_ids(("d1",), case_id="q1")


def test_load_retriever_checks_the_import_path():
    with pytest.raises(ConfigError, match="package.module:function"):
        load_retriever("notapath")
    with pytest.raises(ConfigError, match="has no attribute"):
        load_retriever("tests.fakes:missing")
    with pytest.raises(ConfigError, match="could not import"):
        load_retriever("tests.does_not_exist:retrieve")


def test_scoring_keeps_only_k_unique_ids():
    fakes.calls.clear()
    case = Case.model_validate(
        {"id": "q", "query": "cats", "relevant": ["d1", "d2", "d3"]}
    )
    report = build_report(
        [case],
        fakes.retrieve,
        2,
        dataset="cases.jsonl",
        retriever_name="tests.fakes:retrieve",
        thresholds={},
    )
    row = report["results"][0]
    assert fakes.calls == [("cats", 2)]
    assert row["retrieved"] == ["d1", "d2"]
    assert row["misses"] == ["d3"]
    assert row["metrics"]["recall"] == pytest.approx(2 / 3)


def test_async_results_are_scored():
    case = Case.model_validate({"id": "q", "query": "x", "relevant": ["d1"]})
    report = build_report(
        [case],
        fakes.retrieve_async,
        2,
        dataset="cases.jsonl",
        retriever_name="tests.fakes:retrieve_async",
        thresholds={},
    )
    assert report["results"][0]["retrieved"] == ["d1", "d2"]
    assert report["metrics"]["hit_rate"] == 1.0
