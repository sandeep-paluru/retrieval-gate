"""Several cutoffs in one run."""

import json
import math
from pathlib import Path

from typer.testing import CliRunner

from retrieval_gate.cli import app, bind_thresholds, load_config, parse_fail_under, parse_k_values
from retrieval_gate.errors import ConfigError
from retrieval_gate.metrics import METRIC_NAMES, score_at_cutoffs
from retrieval_gate.models import Case
from retrieval_gate.report import (
    build_report,
    format_markdown,
    format_report,
    load_report,
    regressed_metrics,
    write_report,
)
from tests import fakes

runner = CliRunner()
ROOT = Path(__file__).resolve().parents[1]


def _output(result) -> str:
    return (result.stdout or "") + (result.stderr or "")


def _metrics(score: float, **overrides: float) -> dict[str, float]:
    values = {name: score for name in METRIC_NAMES}
    values.update(overrides)
    return values


def _write_dataset(path: Path) -> None:
    path.write_text(
        '{"id": "good", "query": "cats", "relevant": ["d1", "d2"]}\n'
        '{"id": "bad", "query": "none", "relevant": ["d1"]}\n',
        encoding="utf-8",
    )


def test_parse_k_values_sorts_and_rejects_bad_lists():
    assert parse_k_values("10, 1, 3") == [1, 3, 10]
    assert parse_k_values("5") == [5]
    for text in ("", "1,", "1,1", "0", "-1", "1.5", "a"):
        try:
            parse_k_values(text)
        except ConfigError:
            continue
        raise AssertionError(text)


def test_parse_fail_under_accepts_a_cutoff_and_binds_a_bare_name_to_the_largest():
    parsed = parse_fail_under(["recall@5=0.8", "mrr=0.6", "hit_rate@1=0.4"])
    assert parsed["recall"][5] == 0.8
    assert parsed["mrr"][None] == 0.6
    bound = bind_thresholds(parsed, [1, 5, 10])
    assert bound[1] == {"hit_rate": 0.4}
    assert bound[5] == {"recall": 0.8}
    assert bound[10] == {"mrr": 0.6}
    assert "recall" not in bound[1]

    try:
        bind_thresholds(parsed, [1, 10])
    except ConfigError as exc:
        assert "recall@5" in str(exc)
    else:
        raise AssertionError("missing cutoff")

    twice = parse_fail_under(["recall=0.8", "recall@10=0.9"])
    try:
        bind_thresholds(twice, [1, 10])
    except ConfigError as exc:
        assert "twice" in str(exc)
    else:
        raise AssertionError("duplicate")


def test_score_at_cutoffs_truncates_duplicates_and_short_lists():
    grades = {"a": 1, "b": 1, "c": 1}
    scored = score_at_cutoffs(["a", "a", "b"], grades, [1, 3, 5])
    assert scored[1]["recall"] == 1 / 3
    assert scored[1]["precision"] == 1.0
    assert scored[1]["hit_rate"] == 1.0
    assert scored[3]["recall"] == 2 / 3
    assert scored[3]["precision"] == 2 / 3
    assert scored[5]["recall"] == 2 / 3
    assert scored[5]["precision"] == 2 / 5
    assert scored[5]["mrr"] == 1.0
    # Two of three relevant ids, so ideal DCG uses three grades at both k=3 and k=5.
    ideal = 1 + 1 / math.log2(3) + 1 / math.log2(4)
    gained = 1 + 1 / math.log2(3)
    assert scored[3]["ndcg"] == scored[5]["ndcg"]
    assert scored[5]["ndcg"] == gained / ideal

    short = score_at_cutoffs(["a"], {"a": 1, "b": 1}, [1, 5])
    assert short[1]["recall"] == 0.5
    assert short[1]["precision"] == 1.0
    assert short[5]["recall"] == 0.5
    assert short[5]["precision"] == 1 / 5
    assert short[5]["hit_rate"] == 1.0


def test_build_report_calls_the_retriever_once_at_the_largest_k():
    seen: list[int] = []

    def retrieve(query: str, k: int) -> list[str]:
        seen.append(k)
        return ["a", "a", "b"]

    case = Case(id="q", query="q", relevant={"a": 1, "b": 1, "c": 1}, tags=["billing"])
    report = build_report(
        [case],
        retrieve,
        5,
        dataset="cases.jsonl",
        retriever_name="local:retrieve",
        thresholds={"mrr": 0.5},
        k_values=[5, 1, 3],
        thresholds_by_k={1: {"recall": 0.5}, 3: {}, 5: {"mrr": 0.5}},
    )
    assert seen == [5]
    assert report["k"] == 5
    assert report["k_values"] == [1, 3, 5]
    assert [row["k"] for row in report["cutoffs"]] == [1, 3, 5]
    assert report["results"][0]["retrieved"] == ["a", "b"]
    assert report["results"][0]["misses"] == ["c"]
    assert report["metrics"]["recall"] == report["cutoffs"][-1]["metrics"]["recall"]
    assert report["cutoffs"][0]["metrics"]["recall"] == round(1 / 3, 6)
    assert report["cutoffs"][1]["metrics"]["precision"] == round(2 / 3, 6)
    assert report["cutoffs"][2]["metrics"]["precision"] == round(2 / 5, 6)
    assert report["cutoffs"][0]["failed"] == ["recall"]
    assert report["failed"] == ["recall@1"]
    assert report["tags"][0]["metrics"]["recall"] == report["metrics"]["recall"]
    text = format_report(report)
    assert "k: 1, 3, 5" in text
    assert "@1" in text and "@5" in text
    assert "worst misses at k=5" in text
    assert "tags at k=5" in text
    assert "failed: recall@1" in text
    assert "cutoffs" not in build_report(
        [case],
        retrieve,
        2,
        dataset="cases.jsonl",
        retriever_name="local:retrieve",
        thresholds={},
    )


def test_run_k_values_scores_prefixes_and_keeps_single_k_json(tmp_path):
    dataset = tmp_path / "cases.jsonl"
    _write_dataset(dataset)
    json_path = tmp_path / "report.json"
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
            "--fail-under",
            "recall@1=0.9",
            "--fail-under",
            "mrr=0.4",
            "--json",
            str(json_path),
        ],
    )
    text = result.stdout or ""
    assert result.exit_code == 1, _output(result)
    assert fakes.calls == [("cats", 3), ("none", 3)]
    assert "k: 1, 3" in text
    assert "metric" in text and "@1" in text and "@3" in text
    assert "failed: recall@1" in text
    assert "worst misses at k=3" in text
    report = json.loads(json_path.read_text(encoding="utf-8"))
    assert report["k"] == 3
    assert report["k_values"] == [1, 3]
    assert report["thresholds"] == {"mrr": 0.4}
    assert report["cutoffs"][0]["thresholds"] == {"recall": 0.9}
    assert report["cutoffs"][0]["failed"] == ["recall"]
    assert report["cutoffs"][1]["failed"] == []
    assert report["failed"] == ["recall@1"]
    # cats returns d1, d2, d3. none returns d8. Mean recall at k=1 is 0.5.
    assert report["cutoffs"][0]["metrics"]["recall"] == 0.25
    assert report["metrics"]["mrr"] == 0.5
    loaded = load_report(json_path)
    assert loaded["cutoffs"][0]["metrics"]["recall"] == report["cutoffs"][0]["metrics"]["recall"]
    assert loaded["metrics"] == report["metrics"]

    single_path = tmp_path / "single.json"
    fakes.calls.clear()
    single = runner.invoke(
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
            "recall@2=0.5",
            "--json",
            str(single_path),
        ],
    )
    assert single.exit_code == 0, _output(single)
    assert fakes.calls == [("cats", 2), ("none", 2)]
    body = json.loads(single_path.read_text(encoding="utf-8"))
    assert body["k"] == 2
    assert body["thresholds"] == {"recall": 0.5}
    assert "cutoffs" not in body
    assert "k_values" not in body
    assert "k: 2\n" in (single.stdout or "")
    assert "@1" not in (single.stdout or "")


def test_k_and_k_values_together_are_a_usage_error(tmp_path):
    dataset = tmp_path / "cases.jsonl"
    _write_dataset(dataset)
    both = runner.invoke(
        app,
        [
            "run",
            "--dataset",
            str(dataset),
            "--retriever",
            "tests.fakes:retrieve",
            "--k",
            "5",
            "--k-values",
            "1,5",
        ],
    )
    assert both.exit_code == 2
    assert "not both" in _output(both)

    missing = runner.invoke(
        app,
        [
            "run",
            "--dataset",
            str(dataset),
            "--retriever",
            "tests.fakes:retrieve",
            "--k",
            "5",
            "--fail-under",
            "recall@3=0.8",
        ],
    )
    assert missing.exit_code == 2
    assert "does not match a cutoff" in _output(missing)

    bad = runner.invoke(
        app,
        [
            "run",
            "--dataset",
            str(dataset),
            "--retriever",
            "tests.fakes:retrieve",
            "--k-values",
            "1,1,3",
        ],
    )
    assert bad.exit_code == 2
    assert "duplicate" in _output(bad)


def test_config_k_values_and_flag_override(tmp_path):
    dataset = tmp_path / "cases.jsonl"
    _write_dataset(dataset)
    config = tmp_path / "gate.yml"
    config.write_text(
        "dataset: cases.jsonl\n"
        "retriever: tests.fakes:retrieve\n"
        "k_values: [3, 1]\n"
        "fail_under:\n"
        "  recall@1: 0.9\n"
        "  mrr: 0\n",
        encoding="utf-8",
    )
    loaded = load_config(config)
    assert loaded["k_values"] == [1, 3]
    assert loaded["thresholds"]["recall"][1] == 0.9
    assert loaded["thresholds"]["mrr"][None] == 0.0
    failed = runner.invoke(app, ["run", "--config", str(config)])
    assert failed.exit_code == 1, _output(failed)
    assert "failed: recall@1" in (failed.stdout or "")

    listed = tmp_path / "list.yml"
    listed.write_text(
        "dataset: cases.jsonl\n"
        "retriever: tests.fakes:retrieve\n"
        'k_values: "1, 3"\n'
        "fail_under:\n"
        "  - recall@1=0\n",
        encoding="utf-8",
    )
    passed = runner.invoke(app, ["run", "--config", str(listed)])
    assert passed.exit_code == 0, _output(passed)

    fakes.calls.clear()
    overridden = runner.invoke(
        app,
        ["run", "--config", str(config), "--k", "2", "--fail-under", "recall=0"],
    )
    assert overridden.exit_code == 0, _output(overridden)
    assert fakes.calls == [("cats", 2), ("none", 2)]
    assert "k: 2\n" in (overridden.stdout or "")
    assert "@1" not in (overridden.stdout or "")

    both = tmp_path / "both.yml"
    both.write_text(
        "dataset: cases.jsonl\n"
        "retriever: tests.fakes:retrieve\n"
        "k: 2\n"
        "k_values: [1, 2]\n",
        encoding="utf-8",
    )
    conflict = runner.invoke(app, ["run", "--config", str(both)])
    assert conflict.exit_code == 2
    assert "both k and k_values" in _output(conflict)


def test_markdown_run_has_a_metric_by_k_table(tmp_path):
    dataset = tmp_path / "cases.jsonl"
    _write_dataset(dataset)
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
            "--fail-under",
            "recall@1=0.9",
            "--format",
            "markdown",
        ],
    )
    text = result.stdout or ""
    assert result.exit_code == 1, _output(result)
    assert "| metric | @1 | @3 |" in text
    assert "| --- | ---: | ---: |" in text
    assert "| metric | k | threshold | score | status |" in text
    assert "| recall | 1 | 0.9000 |" in text
    assert "failed: recall@1" in text
    assert "worst misses" not in text


def _multi(metrics_by_k: dict[int, dict[str, float]]) -> dict:
    ks = sorted(metrics_by_k)
    return {
        "k": ks[-1],
        "k_values": ks,
        "metrics": metrics_by_k[ks[-1]],
        "cutoffs": [
            {"k": k, "metrics": metrics_by_k[k], "thresholds": {}, "failed": []} for k in ks
        ],
    }


def test_compare_deltas_per_k_and_falls_back_to_the_matching_k(tmp_path):
    baseline = tmp_path / "baseline.json"
    current = tmp_path / "current.json"
    before = _multi({1: _metrics(1.0), 5: _metrics(1.0), 10: _metrics(1.0)})
    after = _multi({1: _metrics(0.9), 5: _metrics(1.0), 10: _metrics(1.0)})
    baseline.write_text(json.dumps(before), encoding="utf-8")
    current.write_text(json.dumps(after), encoding="utf-8")
    output = tmp_path / "compare.json"
    result = runner.invoke(
        app,
        ["compare", str(baseline), str(current), "--max-drop", "0.02", "--json", str(output)],
    )
    text = result.stdout or ""
    assert result.exit_code == 1, _output(result)
    assert "recall" in text and "1" in text and "-0.1000" in text
    assert "regressed: hit_rate@1, recall@1, precision@1, mrr@1, ndcg@1" in text
    assert "FAIL" in text
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["regressed"] == ["hit_rate@1", "recall@1", "precision@1", "mrr@1", "ndcg@1"]
    assert payload["metrics"]["recall"]["status"] == "pass"
    assert payload["metrics"]["recall"]["delta"] == 0.0
    by_k = {row["k"]: row for row in payload["cutoffs"]}
    assert by_k[1]["metrics"]["recall"]["status"] == "FAIL"
    assert by_k[5]["metrics"]["recall"]["status"] == "pass"
    assert by_k[10]["metrics"]["recall"]["status"] == "pass"
    loaded_before = load_report(baseline)
    loaded_after = load_report(current)
    assert regressed_metrics(loaded_before, loaded_after, 0.02)[0] == "hit_rate@1"

    # Headline at k=10 is perfect. The matching cutoff k=5 is the one that dropped.
    single = {"k": 5, "metrics": _metrics(1.0)}
    swept = _multi({5: _metrics(0.5), 10: _metrics(1.0)})
    baseline.write_text(json.dumps(single), encoding="utf-8")
    current.write_text(json.dumps(swept), encoding="utf-8")
    matched = runner.invoke(
        app,
        ["compare", str(baseline), str(current), "--max-drop", "0.02", "--format", "markdown"],
    )
    matched_text = matched.stdout or ""
    assert matched.exit_code == 1, _output(matched)
    assert "compared at k: 5" in matched_text
    assert "| recall | 5 | 1.0000 | 0.5000 | -0.5000 | FAIL |" in matched_text
    assert "regressed: hit_rate, recall, precision, mrr, ndcg" in matched_text

    missed = {"k": 2, "metrics": _metrics(1.0)}
    baseline.write_text(json.dumps(missed), encoding="utf-8")
    none = runner.invoke(app, ["compare", str(baseline), str(current)])
    assert none.exit_code == 0, _output(none)
    assert "not one of the current cutoffs" in (none.stdout or "")
    assert "Metrics were not compared." in (none.stdout or "")
    assert "regressed:" not in (none.stdout or "")

    other = _multi({1: _metrics(1.0), 2: _metrics(1.0)})
    baseline.write_text(json.dumps(other), encoding="utf-8")
    apart = runner.invoke(app, ["compare", str(baseline), str(current)])
    assert apart.exit_code == 0, _output(apart)
    assert "No shared cutoff" in (apart.stdout or "")

    partial = _multi({1: _metrics(1.0), 5: _metrics(1.0)})
    baseline.write_text(json.dumps(partial), encoding="utf-8")
    current.write_text(json.dumps(_multi({5: _metrics(1.0), 10: _metrics(0.9)})), encoding="utf-8")
    overlap = runner.invoke(app, ["compare", str(baseline), str(current), "--max-drop", "0.02"])
    overlap_text = overlap.stdout or ""
    assert overlap.exit_code == 0, _output(overlap)
    assert "only in baseline: 1" in overlap_text
    assert "only in current: 10" in overlap_text
    assert "regressed:" not in overlap_text


def test_compare_json_round_trip_of_a_sweep(tmp_path):
    dataset = tmp_path / "cases.jsonl"
    _write_dataset(dataset)
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    runner.invoke(
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
            str(first),
        ],
    )
    report = load_report(first)
    write_report(second, report)
    again = load_report(second)
    assert again["k_values"] == [1, 3]
    assert again["cutoffs"][1]["metrics"] == report["cutoffs"][1]["metrics"]
    assert regressed_metrics(report, again, 0.02) == []
    markdown = format_markdown(again, baseline=report, max_drop=0.02)
    assert "| metric | @1 | @3 |" in markdown
    assert "regressed:" not in markdown
    assert "| recall | 1 |" in markdown


def test_action_yml_accepts_k_values():
    text = (ROOT / "action.yml").read_text(encoding="utf-8")
    assert "k-values:" in text
    assert "k_values" in text
    assert "RG_K_VALUES" in text
    assert '--k-values "$RG_K_VALUES"' in text
