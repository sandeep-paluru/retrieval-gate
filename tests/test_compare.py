import json

from typer.testing import CliRunner

from retrieval_gate.cli import app
from retrieval_gate.report import case_diff, load_report, regressed

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
        "\n"
        "### worse cases\n"
        "\n"
        "baseline report has no per-case results\n"
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

    hidden = runner.invoke(
        app,
        ["compare", str(baseline), str(current), "--show-cases", "-1"],
    )
    assert hidden.exit_code == 2
    assert "--show-cases" in _output(hidden)


def _case(
    case_id,
    query="query",
    *,
    recall=1.0,
    ndcg=1.0,
    hit_rate=1.0,
    misses=None,
    retrieved=None,
    relevant=None,
):
    row = {
        "id": case_id,
        "query": query,
        "metrics": {
            "hit_rate": hit_rate,
            "recall": recall,
            "precision": 1.0,
            "mrr": 1.0,
            "ndcg": ndcg,
        },
    }
    if misses is not None:
        row["misses"] = misses
    if retrieved is not None:
        row["retrieved"] = retrieved
    if relevant is not None:
        row["relevant"] = relevant
    return row


def _with_cases(path, cases, score=1.0) -> None:
    payload = _report(score)
    payload["results"] = cases
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_case_diff_matches_sorts_and_splits_membership(tmp_path):
    baseline = tmp_path / "baseline.json"
    current = tmp_path / "current.json"
    _with_cases(
        baseline,
        [
            _case("q_same", "unchanged", misses=["d0"]),
            _case("q_better", "improved", recall=0.4, ndcg=0.4, hit_rate=1.0),
            _case("q_b", "small drop", recall=1.0, ndcg=1.0, misses=[]),
            _case("q_a", "big recall", recall=1.0, ndcg=1.0, misses=["d9"]),
            _case("q_d", "tied recall", recall=1.0, ndcg=0.8, misses=[]),
            _case("q_c", "ndcg only", recall=1.0, ndcg=1.0, hit_rate=1.0),
            _case("q_hit", "lost the hit", recall=0.5, ndcg=0.5, hit_rate=1.0, misses=["d4"]),
            _case("q_old", "removed query"),
            _case("dup", "first baseline", recall=1.0),
            _case("dup", "second baseline", recall=0.1),
            "not a case",
            {"query": "missing id"},
        ],
    )
    _with_cases(
        current,
        [
            _case("q_c", "ndcg only", recall=1.0, ndcg=0.1, hit_rate=1.0, misses=[]),
            _case("q_same", "unchanged", misses=["d0"]),
            _case("q_better", "improved", recall=0.9, ndcg=0.9, hit_rate=1.0),
            _case("q_b", "small drop", recall=0.8, ndcg=0.8, misses=["d7"]),
            _case("q_a", "big recall", recall=0.5, ndcg=1.0, misses=["d9", "d8"]),
            _case("q_d", "tied recall", recall=0.5, ndcg=0.2, misses=[]),
            _case(
                "q_hit",
                "lost the hit",
                recall=0.5,
                ndcg=0.5,
                hit_rate=0.0,
                retrieved=["other"],
                relevant={"d4": 1, "d5": 1},
            ),
            _case("q_new", "added query"),
            _case("dup", "current", recall=0.2),
            {"id": " "},
        ],
    )
    diff = case_diff(load_report(baseline), load_report(current))
    assert diff["available"] is True
    assert [row["id"] for row in diff["worse"]] == ["q_c", "dup", "q_d", "q_a", "q_b", "q_hit"]
    by_id = {row["id"]: row for row in diff["worse"]}
    assert by_id["q_a"]["newly_missed"] == ["d8"]
    assert by_id["q_b"]["newly_missed"] == ["d7"]
    assert by_id["q_c"]["newly_missed"] == []
    assert by_id["q_hit"]["hit_lost"] is True
    assert by_id["q_hit"]["newly_missed"] == ["d5"]
    assert by_id["q_hit"]["recall_dropped"] is False
    assert by_id["dup"]["query"] == "current"
    assert by_id["dup"]["recall"] == {"baseline": 1.0, "current": 0.2}
    assert diff["added"] == ["q_new"]
    assert diff["removed"] == ["q_old"]
    assert "q_same" not in by_id
    assert "q_better" not in by_id


def test_compare_caps_sorts_and_lists_added_cases(tmp_path):
    baseline = tmp_path / "baseline.json"
    current = tmp_path / "current.json"
    _with_cases(
        baseline,
        [
            _case("q_c", "ndcg only"),
            _case("q_d", "tied recall"),
            _case("q_a", "big recall", misses=["d9"]),
            _case("q_b", "small drop"),
            _case("q_old", "gone"),
            _case("q_same", "stays"),
        ],
    )
    _with_cases(
        current,
        [
            _case("q_b", "small drop", recall=0.8, ndcg=0.8, misses=["d7"]),
            _case("q_a", "big recall", recall=0.5, misses=["d9", "d8"]),
            _case("q_d", "tied recall", recall=0.5, ndcg=0.2),
            _case("q_c", "ndcg only", ndcg=0.1),
            _case("q_same", "stays"),
            _case("q_new", "fresh"),
        ],
    )
    capped = runner.invoke(
        app,
        ["compare", str(baseline), str(current), "--show-cases", "2"],
    )
    text = capped.stdout or ""
    assert capped.exit_code == 0, _output(capped)
    assert text.index("q_c") < text.index("q_d")
    assert "q_a" not in text
    assert "q_b" not in text
    assert "showing 2 of 4" in text
    assert "added: 1 (q_new)" in text
    assert "removed: 1 (q_old)" in text
    assert "d8" not in text
    assert "case recall dropped" not in text

    hidden = runner.invoke(
        app,
        ["compare", str(baseline), str(current), "--show-cases", "0"],
    )
    hidden_text = hidden.stdout or ""
    assert hidden.exit_code == 0, _output(hidden)
    assert "4 hidden" in hidden_text
    assert "ndcg only" not in hidden_text
    assert "added: 1 (q_new)" in hidden_text
    assert "removed: 1 (q_old)" in hidden_text

    shown = runner.invoke(app, ["compare", str(baseline), str(current)])
    shown_text = shown.stdout or ""
    assert shown.exit_code == 0, _output(shown)
    assert "showing" not in shown_text
    assert shown_text.index("q_c") < shown_text.index("q_d") < shown_text.index("q_a")
    assert shown_text.index("q_a") < shown_text.index("q_b")
    assert "1.0000" in shown_text
    assert "d8" in shown_text


def test_fail_on_case_regression_exit_codes(tmp_path):
    baseline = tmp_path / "baseline.json"
    current = tmp_path / "current.json"
    _with_cases(baseline, [_case("q1", "cats", misses=[])])
    _with_cases(current, [_case("q1", "cats", recall=0.5, misses=["d1"])])

    quiet = runner.invoke(app, ["compare", str(baseline), str(current), "--max-drop", "0.02"])
    assert quiet.exit_code == 0, _output(quiet)
    assert "case recall dropped" not in (quiet.stdout or "")
    assert "d1" in (quiet.stdout or "")

    failed = runner.invoke(
        app,
        ["compare", str(baseline), str(current), "--fail-on-case-regression"],
    )
    assert failed.exit_code == 1, _output(failed)
    assert "case recall dropped: q1" in (failed.stdout or "")

    _with_cases(current, [_case("q1", "cats", ndcg=0.4)])
    ndcg_only = runner.invoke(
        app,
        ["compare", str(baseline), str(current), "--fail-on-case-regression"],
    )
    assert ndcg_only.exit_code == 0, _output(ndcg_only)
    assert "q1" in (ndcg_only.stdout or "")
    assert "case recall dropped" not in (ndcg_only.stdout or "")

    _with_cases(baseline, [_case("q1", "cats"), _case("q_old", "gone")])
    _with_cases(current, [_case("q1", "cats"), _case("q_new", "fresh")])
    membership = runner.invoke(
        app,
        ["compare", str(baseline), str(current), "--fail-on-case-regression"],
    )
    assert membership.exit_code == 0, _output(membership)
    assert "added: 1 (q_new)" in (membership.stdout or "")
    assert "removed: 1 (q_old)" in (membership.stdout or "")
    assert "\nnone\n" in (membership.stdout or "")


def test_compare_markdown_renders_worse_cases(tmp_path):
    baseline = tmp_path / "baseline.json"
    current = tmp_path / "current.json"
    query = "reset|password " + ("x" * 50) + "\ntail"
    _with_cases(
        baseline,
        [_case("q1", query, misses=["d0"]), _case("q_old", "gone")],
    )
    _with_cases(
        current,
        [_case("q1", query, recall=0.25, ndcg=0.5, misses=["d0", "d9"]), _case("q_new", "fresh")],
    )
    result = runner.invoke(
        app,
        ["compare", str(baseline), str(current), "--format", "markdown", "--show-cases", "10"],
    )
    text = result.stdout or ""
    assert result.exit_code == 0, _output(result)
    shortened = "reset\\|password " + ("x" * 42) + "..."
    assert shortened in text
    assert "tail" not in text
    header = (
        "| case | query | recall baseline | recall current | "
        "ndcg baseline | ndcg current | newly missed |"
    )
    assert text.endswith(
        "### worse cases\n"
        "\n"
        f"{header}\n"
        "| --- | --- | ---: | ---: | ---: | ---: | --- |\n"
        f"| q1 | {shortened} | 1.0000 | 0.2500 | 1.0000 | 0.5000 | d9 |\n"
        "\n"
        "added: 1 (q_new)\n"
        "removed: 1 (q_old)\n"
    )


def test_compare_json_includes_capped_cases(tmp_path):
    baseline = tmp_path / "baseline.json"
    current = tmp_path / "current.json"
    long_query = "y" * 80
    _with_cases(
        baseline,
        [
            _case("q_big", long_query),
            _case("q_small", "beta"),
            _case("q_old", "gone"),
        ],
    )
    _with_cases(
        current,
        [
            _case("q_small", "beta", recall=0.9, misses=["d2"]),
            _case("q_big", long_query, recall=0.0, ndcg=0.0, hit_rate=0.0, misses=["d1"]),
            _case("q_new", "fresh"),
        ],
    )
    output = tmp_path / "compare.json"
    result = runner.invoke(
        app,
        [
            "compare",
            str(baseline),
            str(current),
            "--show-cases",
            "1",
            "--json",
            str(output),
            "--fail-on-case-regression",
        ],
    )
    assert result.exit_code == 1, _output(result)
    assert long_query not in (result.stdout or "")
    assert "yyy..." in (result.stdout or "")
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["regressed"] == []
    assert payload["cases"]["worse_count"] == 2
    assert payload["cases"]["recall_dropped"] == ["q_big", "q_small"]
    assert [row["id"] for row in payload["cases"]["worse"]] == ["q_big"]
    assert payload["cases"]["worse"][0]["query"] == long_query
    assert payload["cases"]["worse"][0]["newly_missed"] == ["d1"]
    assert payload["cases"]["worse"][0]["hit_lost"] is True
    assert payload["cases"]["added"] == {"count": 1, "ids": ["q_new"]}
    assert payload["cases"]["removed"] == {"count": 1, "ids": ["q_old"]}
    assert payload["metrics"]["recall"]["status"] == "pass"


def test_compare_old_baseline_without_cases(tmp_path):
    baseline = tmp_path / "baseline.json"
    current = tmp_path / "current.json"
    _write(baseline, 1.0)
    _with_cases(current, [_case("q1", "cats", recall=0.0)])

    noted = runner.invoke(
        app,
        ["compare", str(baseline), str(current), "--fail-on-case-regression"],
    )
    assert noted.exit_code == 0, _output(noted)
    assert "baseline report has no per-case results" in (noted.stdout or "")
    assert "q1" not in (noted.stdout or "")

    _write(current, 0.5)
    current_payload = json.loads(current.read_text(encoding="utf-8"))
    current_payload["results"] = [_case("q1", "cats", recall=0.0)]
    current.write_text(json.dumps(current_payload), encoding="utf-8")
    dropped = runner.invoke(
        app,
        ["compare", str(baseline), str(current), "--max-drop", "0.02", "--format", "markdown"],
    )
    assert dropped.exit_code == 1, _output(dropped)
    assert "regressed: hit_rate, recall, precision, mrr, ndcg" in (dropped.stdout or "")
    assert "baseline report has no per-case results" in (dropped.stdout or "")

    _with_cases(baseline, [_case("q1", "cats")])
    bare = _report(1.0)
    current.write_text(json.dumps(bare), encoding="utf-8")
    missing_current = runner.invoke(app, ["compare", str(baseline), str(current)])
    assert missing_current.exit_code == 0, _output(missing_current)
    assert "current report has no per-case results" in (missing_current.stdout or "")
