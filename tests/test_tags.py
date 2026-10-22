import json

from typer.testing import CliRunner

from retrieval_gate.cli import app
from retrieval_gate.models import Case
from retrieval_gate.report import build_report, format_markdown, format_report

runner = CliRunner()


def _retrieve(query: str, k: int) -> list[str]:
    return {"partial": ["d1", "d2"], "miss": ["d9"], "other": ["d1"]}[query]


def _cases() -> list[Case]:
    return [
        Case(
            id="partial",
            query="partial",
            relevant={"d1": 1, "d2": 1, "d3": 1},
            tags=["billing"],
        ),
        Case(id="miss", query="miss", relevant={"d1": 1}, tags=["billing", "long-tail"]),
        Case(id="other", query="other", relevant={"d1": 1}),
    ]


def test_tag_mean_skips_untagged_cases_and_counts_a_case_in_each_tag():
    report = build_report(
        _cases(),
        _retrieve,
        3,
        dataset="cases.jsonl",
        retriever_name="local:retrieve",
        thresholds={},
    )
    # partial recall is 2/3, miss is 0, so billing is 1/3. other is not in the slice.
    by_tag = {row["tag"]: row for row in report["tags"]}
    assert list(by_tag) == ["billing", "long-tail"]
    assert by_tag["billing"]["case_count"] == 2
    assert by_tag["billing"]["metrics"]["recall"] == 0.333333
    assert by_tag["billing"]["metrics"]["precision"] == 0.333333
    assert by_tag["billing"]["metrics"]["hit_rate"] == 0.5
    assert by_tag["long-tail"]["case_count"] == 1
    assert by_tag["long-tail"]["metrics"]["recall"] == 0.0
    assert report["metrics"]["recall"] == 0.555556
    assert report["results"][0]["tags"] == ["billing"]
    assert report["results"][1]["tags"] == ["billing", "long-tail"]
    assert report["results"][2]["tags"] == []
    assert "failed_tags" not in report
    assert "untagged" not in by_tag


def test_report_omits_the_tag_section_when_no_case_is_tagged():
    report = build_report(
        [Case(id="other", query="other", relevant={"d1": 1})],
        _retrieve,
        3,
        dataset="cases.jsonl",
        retriever_name="local:retrieve",
        thresholds={},
    )
    assert "tags" not in report
    assert report["results"][0]["tags"] == []
    assert "\ntags\n" not in format_report(report)
    assert "### tags" not in format_markdown(report)


def _write_cli_dataset(path):
    path.write_text(
        '{"id": "good", "query": "cats", "relevant": ["d1", "d2"], "tags": ["billing"]}\n'
        '{"id": "bad", "query": "none", "relevant": ["d1"], "tags": ["billing", "api"]}\n'
        '{"id": "plain", "query": "cats", "relevant": ["d1"]}\n',
        encoding="utf-8",
    )


def _output(result) -> str:
    return (result.stdout or "") + (result.stderr or "")


def test_run_text_markdown_and_json_include_tags(tmp_path):
    dataset = tmp_path / "cases.jsonl"
    _write_cli_dataset(dataset)
    json_path = tmp_path / "report.json"
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
            "--fail-under-tag",
            "billing:recall=0.8",
            "--fail-under-tag",
            "api:mrr=0",
            "--format",
            "markdown",
            "--output",
            str(output),
            "--json",
            str(json_path),
        ],
    )
    text = result.stdout or ""
    assert result.exit_code == 1, _output(result)
    assert "### tags" in text
    assert "| api | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |" in text
    assert "| billing | 2 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 |" in text
    assert "failed tags: billing recall" in text
    assert "failed: recall" not in text
    assert output.read_text(encoding="utf-8") == text
    report = json.loads(json_path.read_text(encoding="utf-8"))
    assert report["failed"] == []
    assert report["tag_thresholds"] == {"api": {"mrr": 0.0}, "billing": {"recall": 0.8}}
    assert report["failed_tags"] == [{"tag": "billing", "metric": "recall"}]
    assert [row["tag"] for row in report["tags"]] == ["api", "billing"]
    assert report["tags"][1]["case_count"] == 2
    assert report["tags"][1]["metrics"]["recall"] == 0.5
    assert report["results"][2]["tags"] == []
    assert "tags" in format_report(report)
    assert "failed tags: billing recall" in format_report(report)


def test_fail_under_tag_passes_when_the_score_equals_the_limit(tmp_path):
    dataset = tmp_path / "cases.jsonl"
    _write_cli_dataset(dataset)
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
            "--fail-under-tag",
            "billing:recall=0.5",
        ],
    )
    text = result.stdout or ""
    assert result.exit_code == 0, _output(result)
    assert "failed tags:" not in text
    assert "api      1      0.0000    0.0000  0.0000     0.0000  0.0000" in text
    assert "billing  2      0.5000    0.5000  0.5000     0.5000  0.5000" in text


def test_fail_under_tag_rejects_bad_arguments(tmp_path):
    dataset = tmp_path / "cases.jsonl"
    _write_cli_dataset(dataset)
    base = ["run", "--dataset", str(dataset), "--retriever", "tests.fakes:retrieve", "--k", "2"]

    unknown_metric = runner.invoke(app, [*base, "--fail-under-tag", "billing:nope=0.1"])
    assert unknown_metric.exit_code == 2
    assert "unknown metric" in _output(unknown_metric)

    missing_shape = runner.invoke(app, [*base, "--fail-under-tag", "recall=0.1"])
    assert missing_shape.exit_code == 2
    assert "tag:metric=value" in _output(missing_shape)

    missing_tag = runner.invoke(app, [*base, "--fail-under-tag", ":recall=0.1"])
    assert missing_tag.exit_code == 2
    assert "missing a tag" in _output(missing_tag)

    bad_value = runner.invoke(app, [*base, "--fail-under-tag", "billing:recall=high"])
    assert bad_value.exit_code == 2
    assert "not a number" in _output(bad_value)

    too_high = runner.invoke(app, [*base, "--fail-under-tag", "billing:recall=1.1"])
    assert too_high.exit_code == 2
    assert "between 0 and 1" in _output(too_high)

    twice = runner.invoke(
        app,
        [*base, "--fail-under-tag", "billing:recall=0.8", "--fail-under-tag", "billing:recall=0.4"],
    )
    assert twice.exit_code == 2
    assert "was given twice" in _output(twice)

    unknown_tag = runner.invoke(app, [*base, "--fail-under-tag", "missing:recall=0.1"])
    assert unknown_tag.exit_code == 2
    assert "no cases tagged 'missing'" in _output(unknown_tag)

    colon_tag = runner.invoke(app, [*base, "--fail-under-tag", "team:billing:recall=0.5"])
    assert colon_tag.exit_code == 2
    assert "no cases tagged 'team:billing'" in _output(colon_tag)


def test_config_fail_under_tag_and_flag_override(tmp_path):
    dataset = tmp_path / "cases.jsonl"
    _write_cli_dataset(dataset)
    config = tmp_path / "gate.yml"
    config.write_text(
        "dataset: cases.jsonl\n"
        "retriever: tests.fakes:retrieve\n"
        "k: 2\n"
        "fail_under_tag:\n"
        "  billing:\n"
        "    recall: 0.8\n",
        encoding="utf-8",
    )
    failed = runner.invoke(app, ["run", "--config", str(config)])
    assert failed.exit_code == 1, _output(failed)
    assert "failed tags: billing recall" in (failed.stdout or "")

    passed = runner.invoke(
        app,
        ["run", "--config", str(config), "--fail-under-tag", "billing:recall=0.5"],
    )
    assert passed.exit_code == 0, _output(passed)

    listed = tmp_path / "list.yml"
    listed.write_text(
        "dataset: cases.jsonl\n"
        "retriever: tests.fakes:retrieve\n"
        "k: 2\n"
        "fail_under_tag:\n"
        "  - billing:recall=0.5\n",
        encoding="utf-8",
    )
    from_list = runner.invoke(app, ["run", "--config", str(listed)])
    assert from_list.exit_code == 0, _output(from_list)
