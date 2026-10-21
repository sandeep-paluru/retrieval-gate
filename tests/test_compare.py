import json

from typer.testing import CliRunner

from retrieval_gate.cli import app
from retrieval_gate.report import regressed

runner = CliRunner()


def _report(score: float) -> dict:
    metrics = {
        "hit_rate": score,
        "recall": score,
        "precision": score,
        "mrr": score,
        "ndcg": score,
    }
    return {"k": 5, "metrics": metrics}


def _write(path, score: float) -> None:
    path.write_text(json.dumps(_report(score)), encoding="utf-8")


def _output(result) -> str:
    return (result.stdout or "") + (result.stderr or "")


def test_drop_equal_to_the_limit_passes():
    assert regressed(1.0, 0.98, 0.02) is False
    assert regressed(1.0, 0.97, 0.02) is True
    assert regressed(0.5, 0.6, 0.02) is False


def test_compare_command_pass_and_fail(tmp_path):
    baseline = tmp_path / "baseline.json"
    current = tmp_path / "current.json"
    _write(baseline, 1.0)
    _write(current, 0.98)
    allowed = runner.invoke(
        app,
        ["compare", str(baseline), str(current), "--max-drop", "0.02"],
    )
    assert allowed.exit_code == 0, _output(allowed)
    assert "FAIL" not in (allowed.stdout or "")
    assert "-0.0200" in (allowed.stdout or "")

    _write(current, 0.97)
    dropped = runner.invoke(
        app,
        ["compare", str(baseline), str(current), "--max-drop", "0.02"],
    )
    assert dropped.exit_code == 1, _output(dropped)
    assert "regressed:" in (dropped.stdout or "")
    assert "-0.0300" in (dropped.stdout or "")


def test_compare_markdown_marks_regression(tmp_path):
    baseline = tmp_path / "baseline.json"
    current = tmp_path / "current.json"
    _write(baseline, 1.0)
    _write(current, 0.97)
    output = tmp_path / "compare.md"
    result = runner.invoke(
        app,
        [
            "compare",
            str(baseline),
            str(current),
            "--max-drop",
            "0.02",
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
        "k: 5\n"
        f"baseline: {baseline}\n"
        f"current: {current}\n"
        "max drop: 0.0200\n"
        "\n"
        "| metric | score | threshold | baseline | delta | status |\n"
        "| --- | ---: | ---: | ---: | ---: | --- |\n"
        "| hit_rate | 0.9700 | - | 1.0000 | -0.0300 | FAIL |\n"
        "| recall | 0.9700 | - | 1.0000 | -0.0300 | FAIL |\n"
        "| precision | 0.9700 | - | 1.0000 | -0.0300 | FAIL |\n"
        "| mrr | 0.9700 | - | 1.0000 | -0.0300 | FAIL |\n"
        "| ndcg | 0.9700 | - | 1.0000 | -0.0300 | FAIL |\n"
        "\n"
        "regressed: hit_rate, recall, precision, mrr, ndcg\n"
    )
    assert output.read_text(encoding="utf-8") == text


def test_compare_markdown_shows_thresholds_without_gating_them(tmp_path):
    baseline = tmp_path / "baseline.json"
    current = tmp_path / "current.json"
    score = 0.5
    payload = _report(score)
    payload["thresholds"] = {"recall": 0.8}
    baseline.write_text(json.dumps(_report(score)), encoding="utf-8")
    current.write_text(json.dumps(payload), encoding="utf-8")
    result = runner.invoke(
        app,
        ["compare", str(baseline), str(current), "--format", "markdown"],
    )
    text = result.stdout or ""
    assert result.exit_code == 0, _output(result)
    assert "| recall | 0.5000 | 0.8000 | 0.5000 | +0.0000 | pass |" in text
    assert "failed:" not in text
    assert "regressed" not in text


def test_compare_rejects_bad_input(tmp_path):
    baseline = tmp_path / "baseline.json"
    current = tmp_path / "current.json"
    _write(baseline, 1.0)
    _write(current, 1.0)
    missing = runner.invoke(app, ["compare", str(baseline), str(tmp_path / "nope.json")])
    assert missing.exit_code == 2
    assert "report not found" in _output(missing)

    current.write_text("{", encoding="utf-8")
    broken = runner.invoke(app, ["compare", str(baseline), str(current)])
    assert broken.exit_code == 2
    assert "invalid JSON" in _output(broken)

    current.write_text(json.dumps({"metrics": {"hit_rate": 1}}), encoding="utf-8")
    partial = runner.invoke(app, ["compare", str(baseline), str(current)])
    assert partial.exit_code == 2
    assert "metrics.recall is missing" in _output(partial)

    negative = runner.invoke(
        app,
        ["compare", str(baseline), str(current), "--max-drop", "-0.01"],
    )
    assert negative.exit_code == 2
    assert "--max-drop" in _output(negative)
