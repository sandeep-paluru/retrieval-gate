import json

from typer.testing import CliRunner

from retrieval_gate.cli import app
from tests import fakes

runner = CliRunner()


def _write_dataset(path):
    path.write_text(
        '{"id": "good", "query": "cats", "relevant": ["d1", "d2"]}\n'
        '{"id": "bad", "query": "none", "relevant": ["d1"]}\n',
        encoding="utf-8",
    )


def _output(result) -> str:
    return (result.stdout or "") + (result.stderr or "")


def test_run_report_and_threshold_failure(tmp_path):
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
            "--k",
            "2",
            "--fail-under",
            "recall=0.8",
            "--fail-under",
            "mrr=0.4",
            "--json",
            str(json_path),
        ],
    )
    text = result.stdout or ""
    assert result.exit_code == 1, _output(result)
    assert fakes.calls == [("cats", 2), ("none", 2)]
    assert text == (
        f"dataset: {dataset}\n"
        "retriever: tests.fakes:retrieve\n"
        "k: 2\n"
        "cases: 2\n"
        "\n"
        "metric     score   threshold  status\n"
        "---------  ------  ---------  ------\n"
        "hit_rate   0.5000  -          -\n"
        "recall     0.5000  0.8000     FAIL\n"
        "precision  0.5000  -          -\n"
        "mrr        0.5000  0.4000     pass\n"
        "ndcg       0.5000  -          -\n"
        "\n"
        "worst misses\n"
        "case  recall  missed\n"
        "----  ------  ------\n"
        "bad   0.0000  d1\n"
        "\n"
        "failed: recall\n"
    )
    report = json.loads(json_path.read_text(encoding="utf-8"))
    assert report["failed"] == ["recall"]
    assert report["metrics"]["recall"] == 0.5
    assert report["results"][1]["misses"] == ["d1"]
    assert report["results"][1]["id"] == "bad"


def test_run_passes_when_thresholds_are_met(tmp_path):
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
            "--k",
            "2",
            "--fail-under",
            "recall=0.5",
        ],
    )
    assert result.exit_code == 0, _output(result)
    assert "failed:" not in (result.stdout or "")


def test_async_retriever_from_the_cli(tmp_path):
    dataset = tmp_path / "cases.jsonl"
    dataset.write_text('{"id": "q1", "query": "x", "relevant": ["d1"]}\n', encoding="utf-8")
    result = runner.invoke(
        app,
        ["run", "--dataset", str(dataset), "--retriever", "tests.fakes:retrieve_async", "--k", "2"],
    )
    assert result.exit_code == 0, _output(result)
    assert "hit_rate   1.0000" in (result.stdout or "")


def test_run_markdown_format_and_output(tmp_path):
    dataset = tmp_path / "cases.jsonl"
    _write_dataset(dataset)
    output = tmp_path / "report.md"
    result = runner.invoke(
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
            "recall=0.8",
            "--fail-under",
            "mrr=0.4",
            "--format",
            "markdown",
            "--output",
            str(output),
        ],
    )
    text = result.stdout or ""
    assert result.exit_code == 1, _output(result)
    assert text == (
        "## retrieval-gate\n"
        "\n"
        f"dataset: {dataset}\n"
        "retriever: tests.fakes:retrieve\n"
        "k: 2\n"
        "cases: 2\n"
        "\n"
        "| metric | score | threshold | baseline | delta | status |\n"
        "| --- | ---: | ---: | ---: | ---: | --- |\n"
        "| hit_rate | 0.5000 | - | - | - | - |\n"
        "| recall | 0.5000 | 0.8000 | - | - | FAIL |\n"
        "| precision | 0.5000 | - | - | - | - |\n"
        "| mrr | 0.5000 | 0.4000 | - | - | pass |\n"
        "| ndcg | 0.5000 | - | - | - | - |\n"
        "\n"
        "failed: recall\n"
    )
    assert output.read_text(encoding="utf-8") == text
    assert "worst misses" not in text


def test_run_baseline_markdown_marks_regression(tmp_path):
    dataset = tmp_path / "cases.jsonl"
    _write_dataset(dataset)
    baseline = tmp_path / "baseline.json"
    metrics = {name: 1.0 for name in ("hit_rate", "recall", "precision", "mrr", "ndcg")}
    baseline.write_text(json.dumps({"k": 2, "metrics": metrics}), encoding="utf-8")
    result = runner.invoke(
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
            "recall=0.4",
            "--baseline",
            str(baseline),
            "--format",
            "markdown",
        ],
    )
    text = result.stdout or ""
    assert result.exit_code == 1, _output(result)
    assert "| recall | 0.5000 | 0.4000 | 1.0000 | -0.5000 | FAIL |" in text
    assert "failed:" not in text
    assert "regressed: hit_rate, recall, precision, mrr, ndcg" in text


def test_config_supplies_paths_and_flags_override(tmp_path):
    dataset = tmp_path / "cases.jsonl"
    _write_dataset(dataset)
    baseline = tmp_path / "baseline.json"
    metrics = {name: 0.5 for name in ("hit_rate", "recall", "precision", "mrr", "ndcg")}
    baseline.write_text(json.dumps({"metrics": metrics}), encoding="utf-8")
    config = tmp_path / "gate.yml"
    config.write_text(
        "dataset: cases.jsonl\n"
        "retriever: tests.fakes:retrieve\n"
        "k: 2\n"
        "fail_under:\n"
        "  recall: 0.4\n"
        "baseline: baseline.json\n",
        encoding="utf-8",
    )
    matched = runner.invoke(app, ["run", "--config", str(config), "--format", "markdown"])
    matched_text = matched.stdout or ""
    assert matched.exit_code == 0, _output(matched)
    assert "regressed" not in matched_text
    assert f"dataset: {dataset}" in matched_text
    assert "| recall | 0.5000 | 0.4000 | 0.5000 | +0.0000 | pass |" in matched_text

    other = tmp_path / "other.jsonl"
    other.write_text('{"id": "q1", "query": "cats", "relevant": ["d1"]}\n', encoding="utf-8")
    overridden = runner.invoke(
        app,
        [
            "run",
            "--config",
            str(config),
            "--dataset",
            str(other),
            "--fail-under",
            "recall=0.5",
            "--format",
            "markdown",
        ],
    )
    text = overridden.stdout or ""
    assert overridden.exit_code == 0, _output(overridden)
    assert f"dataset: {other}" in text
    assert "| recall | 1.0000 | 0.5000 | 0.5000 | +0.5000 | pass |" in text


def test_config_errors_exit_2(tmp_path):
    dataset = tmp_path / "cases.jsonl"
    dataset.write_text('{"id": "q1", "query": "cats", "relevant": ["d1"]}\n', encoding="utf-8")
    missing = runner.invoke(
        app,
        ["run", "--dataset", str(tmp_path / "nope.jsonl"), "--retriever", "tests.fakes:retrieve"],
    )
    assert missing.exit_code == 2
    assert "dataset not found" in _output(missing)

    bad_k = runner.invoke(
        app,
        ["run", "--dataset", str(dataset), "--retriever", "tests.fakes:retrieve", "--k", "0"],
    )
    assert bad_k.exit_code == 2
    assert "k must be >= 1" in _output(bad_k)

    bad_metric = runner.invoke(
        app,
        [
            "run",
            "--dataset",
            str(dataset),
            "--retriever",
            "tests.fakes:retrieve",
            "--fail-under",
            "nope=0.1",
        ],
    )
    assert bad_metric.exit_code == 2
    assert "unknown metric" in _output(bad_metric)

    bad_value = runner.invoke(
        app,
        [
            "run",
            "--dataset",
            str(dataset),
            "--retriever",
            "tests.fakes:retrieve",
            "--fail-under",
            "recall=high",
        ],
    )
    assert bad_value.exit_code == 2

    boom = runner.invoke(
        app,
        ["run", "--dataset", str(dataset), "--retriever", "tests.fakes:retrieve_boom"],
    )
    assert boom.exit_code == 2
    assert "case q1" in _output(boom)
    assert "backend down" in _output(boom)

    bad_return = runner.invoke(
        app,
        ["run", "--dataset", str(dataset), "--retriever", "tests.fakes:retrieve_bad"],
    )
    assert bad_return.exit_code == 2
    assert "must return a list" in _output(bad_return)

    usage = runner.invoke(app, ["run"])
    assert usage.exit_code == 2

    bad_format = runner.invoke(
        app,
        [
            "run",
            "--dataset",
            str(dataset),
            "--retriever",
            "tests.fakes:retrieve",
            "--format",
            "html",
        ],
    )
    assert bad_format.exit_code == 2
    assert "--format must be text or markdown" in _output(bad_format)

    no_baseline = runner.invoke(
        app,
        [
            "run",
            "--dataset",
            str(dataset),
            "--retriever",
            "tests.fakes:retrieve",
            "--max-drop",
            "0.01",
        ],
    )
    assert no_baseline.exit_code == 2
    assert "baseline" in _output(no_baseline)

    config = tmp_path / "gate.yml"
    config.write_text("dataset: cases.jsonl\nextra: 1\n", encoding="utf-8")
    unknown = runner.invoke(app, ["run", "--config", str(config)])
    assert unknown.exit_code == 2
    assert "unknown config keys: extra" in _output(unknown)
