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
