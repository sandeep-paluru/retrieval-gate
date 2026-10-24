"""Paired bootstrap on retrieval-gate compare."""

import json
import random
import statistics
import time

import pytest
from typer.testing import CliRunner

from retrieval_gate.bootstrap import _percentile, format_bootstrap, paired_bootstrap
from retrieval_gate.cli import app
from retrieval_gate.metrics import METRIC_NAMES

runner = CliRunner()

_HEADLINE = {name: 1.0 for name in METRIC_NAMES}


def _case(case_id, **overrides):
    metrics = {name: 0.5 for name in METRIC_NAMES}
    metrics.update(overrides)
    return {"id": case_id, "query": case_id, "metrics": metrics}


def _report(cases, headline=None):
    payload = {"k": 5, "metrics": dict(headline or _HEADLINE)}
    if cases is not None:
        payload["results"] = cases
    return payload


def _write(path, cases, headline=None) -> None:
    path.write_text(json.dumps(_report(cases, headline)), encoding="utf-8")


def _output(result) -> str:
    return (result.stdout or "") + (result.stderr or "")


def _constant_reports(count, **deltas):
    before = [_case(f"q{index:04d}") for index in range(count)]
    after = []
    for index in range(count):
        shifted = {}
        for name in METRIC_NAMES:
            shifted[name] = before[index]["metrics"][name] + deltas.get(name, 0.0)
        after.append(_case(f"q{index:04d}", **shifted))
    return _report(before), _report(after)


def _run_constant(count, samples, max_drop, **deltas):
    before, after = _constant_reports(count, **deltas)
    return paired_bootstrap(
        before,
        after,
        samples=samples,
        confidence=0.95,
        seed=0,
        max_drop=max_drop,
        checked=True,
    )


def test_percentile_interpolates_between_ranks():
    values = [0.0, 1.0, 2.0, 3.0]
    assert _percentile(values, 0.0) == 0.0
    assert _percentile(values, 1.0) == 3.0
    assert _percentile(values, 0.25) == pytest.approx(0.75)
    assert _percentile(values, 0.5) == pytest.approx(1.5)
    assert _percentile([5.0], 0.3) == 5.0


def test_constant_shift_interval_and_p_value():
    result = _run_constant(12, samples=99, max_drop=0.02, recall=-0.4)
    recall = result["metrics"]["recall"]
    assert result["case_count"] == 12
    assert result["skipped"] == 0
    assert recall["delta"] == -0.4
    assert recall["low"] == -0.4
    assert recall["high"] == -0.4
    assert recall["low"] <= recall["high"]
    assert recall["p_value"] == round(1 / 100, 6)
    assert result["significant"] == ["recall"]
    for name in METRIC_NAMES:
        if name == "recall":
            continue
        metric = result["metrics"][name]
        assert metric["delta"] == 0.0
        assert metric["low"] == 0.0
        assert metric["high"] == 0.0
        assert metric["p_value"] == 1.0


def test_drop_inside_the_limit_is_not_a_significant_fail():
    # The whole interval is below zero, and it is still inside max-drop.
    inside = _run_constant(8, samples=20, max_drop=0.02, recall=-0.01)
    assert inside["metrics"]["recall"]["high"] == -0.01
    assert inside["significant"] == []

    # A bound equal to -max-drop passes, same as the aggregate gate.
    equal = _run_constant(8, samples=20, max_drop=0.02, recall=-0.02)
    assert equal["metrics"]["recall"]["high"] == -0.02
    assert equal["significant"] == []

    past = _run_constant(8, samples=20, max_drop=0.02, ndcg=-0.05)
    assert past["significant"] == ["ndcg"]


def test_noise_interval_crosses_zero_and_stays_ordered():
    before_cases = []
    after_cases = []
    for index in range(40):
        before_cases.append(_case(f"q{index:02d}", recall=0.5))
        recall = 0.9 if index < 20 else 0.1
        after_cases.append(_case(f"q{index:02d}", recall=recall))
    before = _report(before_cases)
    after = _report(after_cases)

    def run(seed, confidence):
        return paired_bootstrap(
            before,
            after,
            samples=80,
            confidence=confidence,
            seed=seed,
            max_drop=0.02,
            checked=True,
        )

    first = run(0, 0.95)
    second = run(0, 0.95)
    other = run(1, 0.95)
    wider = run(0, 0.99)
    assert first == second
    assert first["metrics"] != other["metrics"]
    recall = first["metrics"]["recall"]
    assert recall["delta"] == 0.0
    assert recall["low"] < 0 < recall["high"]
    assert recall["low"] <= recall["high"]
    assert first["significant"] == []
    for name in METRIC_NAMES:
        metric = first["metrics"][name]
        assert metric["low"] <= metric["high"]
        wide = wider["metrics"][name]
        assert wide["low"] <= metric["low"]
        assert wide["high"] >= metric["high"]


def test_same_seed_matches_a_hand_rolled_resample():
    deltas = {
        "hit_rate": [-0.2, 0.0, 0.1, 0.4],
        "recall": [0.2, -0.2, 0.0, 0.0],
        "precision": [0.0, 0.0, 0.0, 0.0],
        "mrr": [0.5, -0.25, 0.0, -0.25],
        "ndcg": [-0.1, -0.1, 0.3, -0.05],
    }
    before = _report([_case(f"q{i}") for i in range(4)])
    after_cases = []
    for index in range(4):
        shifted = {
            name: before["results"][index]["metrics"][name] + deltas[name][index]
            for name in METRIC_NAMES
        }
        after_cases.append(_case(f"q{index}", **shifted))
    after = _report(after_cases)
    samples = 40
    confidence = 0.9
    seed = 7
    result = paired_bootstrap(
        before,
        after,
        samples=samples,
        confidence=confidence,
        seed=seed,
        max_drop=0.02,
        checked=False,
    )
    columns = {
        name: [
            after["results"][index]["metrics"][name] - before["results"][index]["metrics"][name]
            for index in range(4)
        ]
        for name in METRIC_NAMES
    }
    expected = _reference(columns, samples, confidence, seed, max_drop=0.02)
    assert result["case_count"] == 4
    assert result["significant"] == expected["significant"]
    for name in METRIC_NAMES:
        assert result["metrics"][name] == expected["metrics"][name]


def test_case_order_and_duplicate_ids_do_not_change_the_draw():
    before_cases = [_case("b", recall=1.0), _case("a", recall=1.0)]
    after_cases = [_case("a", recall=0.0), _case("b", recall=1.0)]
    flipped_before = list(reversed(before_cases))
    flipped_after = list(reversed(after_cases))
    kwargs = {"samples": 25, "confidence": 0.95, "seed": 3, "max_drop": 0.02, "checked": False}
    forward = paired_bootstrap(_report(before_cases), _report(after_cases), **kwargs)
    backward = paired_bootstrap(_report(flipped_before), _report(flipped_after), **kwargs)
    assert forward == backward
    assert forward["metrics"]["recall"]["delta"] == -0.5

    duplicated = _report(
        [_case("q1", recall=1.0), _case("q1", recall=0.0), _case("q2", recall=1.0)]
    )
    current = _report([_case("q1", recall=0.0), _case("q2", recall=1.0)])
    used_first = paired_bootstrap(duplicated, current, **kwargs)
    assert used_first["case_count"] == 2
    assert used_first["metrics"]["recall"]["delta"] == -0.5


def test_incomplete_cases_are_skipped():
    before = _report([_case("kept", recall=1.0), _case("dropme", recall=1.0)])
    broken = _case("dropme", recall=0.0)
    del broken["metrics"]["precision"]
    after = _report([_case("kept", recall=0.25), broken])
    result = paired_bootstrap(
        before,
        after,
        samples=10,
        confidence=0.95,
        seed=0,
        max_drop=0.02,
        checked=False,
    )
    assert result["available"] is True
    assert result["case_count"] == 1
    assert result["skipped"] == 1
    assert result["metrics"]["recall"]["delta"] == -0.75
    text = format_bootstrap(result, markdown=False)
    assert "1 paired case, 1 skipped" in text


def test_missing_per_case_results_are_a_note():
    bare = _report(None)
    current = _report([_case("q1")])
    kwargs = {"samples": 10, "confidence": 0.8, "seed": 1, "max_drop": 0.02, "checked": True}
    missing_base = paired_bootstrap(bare, current, **kwargs)
    assert missing_base["available"] is False
    assert missing_base["message"] == "baseline report has no per-case results"
    assert "metrics" not in missing_base

    missing_current = paired_bootstrap(current, bare, **kwargs)
    assert missing_current["message"] == "current report has no per-case results"

    disjoint = paired_bootstrap(_report([_case("a")]), _report([_case("b")]), **kwargs)
    assert disjoint["message"] == "no paired cases to resample"
    note = format_bootstrap(missing_base, markdown=True)
    assert "### bootstrap" in note
    assert "baseline report has no per-case results" in note
    assert "FAIL" not in note


def test_thousand_cases_stay_fast_enough():
    count = 1000
    before = _report([_case(f"q{index:04d}", recall=0.5) for index in range(count)])
    after = _report(
        [
            _case(f"q{index:04d}", recall=0.4 if index % 2 == 0 else 0.7)
            for index in range(count)
        ]
    )
    started = time.perf_counter()
    result = paired_bootstrap(
        before,
        after,
        samples=2000,
        confidence=0.95,
        seed=0,
        max_drop=0.02,
        checked=False,
    )
    assert time.perf_counter() - started < 5
    assert result["case_count"] == count
    for name in METRIC_NAMES:
        metric = result["metrics"][name]
        assert metric["low"] <= metric["high"]
        assert 0.0 < metric["p_value"] <= 1.0


def test_compare_bootstrap_text_markdown_and_json(tmp_path):
    baseline = tmp_path / "baseline.json"
    current = tmp_path / "current.json"
    before, after = _constant_reports(6, recall=-0.4)
    baseline.write_text(json.dumps(before), encoding="utf-8")
    current.write_text(json.dumps(after), encoding="utf-8")
    output = tmp_path / "compare.json"

    quiet = runner.invoke(
        app,
        ["compare", str(baseline), str(current), "--bootstrap", "99", "--seed", "0"],
    )
    assert quiet.exit_code == 0, _output(quiet)
    text = quiet.stdout or ""
    assert "99 resamples, confidence 0.95, seed 0, 6 paired cases" in text
    assert "significant drop:" not in text
    assert "fail when the whole interval" not in text
    bootstrap_text = text.split("bootstrap\n", 1)[1].split("worse cases", 1)[0]
    assert "recall" in bootstrap_text
    assert "-0.4000" in bootstrap_text
    assert "0.0100" in bootstrap_text

    markdown = runner.invoke(
        app,
        [
            "compare",
            str(baseline),
            str(current),
            "--bootstrap",
            "99",
            "--format",
            "markdown",
            "--json",
            str(output),
            "--fail-on-significant-drop",
        ],
    )
    assert markdown.exit_code == 1, _output(markdown)
    rendered = markdown.stdout or ""
    assert (
        "| metric | delta | low | high | p-value | status |\n"
        "| --- | ---: | ---: | ---: | ---: | --- |\n"
        "| hit_rate | +0.0000 | +0.0000 | +0.0000 | 1.0000 | pass |"
    ) in rendered
    assert "| recall | -0.4000 | -0.4000 | -0.4000 | 0.0100 | FAIL |" in rendered
    assert "significant drop: recall" in rendered
    assert "fail when the whole interval is below -0.0200" in rendered
    assert rendered.index("### bootstrap") < rendered.index("### worse cases")
    payload = json.loads(output.read_text(encoding="utf-8"))
    block = payload["bootstrap"]
    assert block["available"] is True
    assert block["checked"] is True
    assert block["samples"] == 99
    assert block["confidence"] == 0.95
    assert block["seed"] == 0
    assert block["case_count"] == 6
    assert block["significant"] == ["recall"]
    assert block["metrics"]["recall"]["delta"] == -0.4
    assert block["metrics"]["recall"]["low"] == -0.4
    assert block["metrics"]["recall"]["high"] == -0.4
    assert block["metrics"]["recall"]["p_value"] == 0.01
    assert block["metrics"]["hit_rate"]["p_value"] == 1.0
    assert payload["regressed"] == []


def test_compare_exit_codes_for_noise_and_a_real_drop(tmp_path):
    baseline = tmp_path / "baseline.json"
    current = tmp_path / "current.json"
    before_cases = []
    after_cases = []
    for index in range(30):
        before_cases.append(_case(f"q{index:02d}", recall=0.5))
        recall = 0.8 if index % 2 == 0 else 0.2
        after_cases.append(_case(f"q{index:02d}", recall=recall))
    _write(baseline, before_cases)
    _write(current, after_cases)

    omitted = runner.invoke(app, ["compare", str(baseline), str(current), "--max-drop", "0.02"])
    assert omitted.exit_code == 0, _output(omitted)
    assert "bootstrap" not in (omitted.stdout or "")

    flagged = runner.invoke(
        app,
        [
            "compare",
            str(baseline),
            str(current),
            "--bootstrap",
            "300",
            "--fail-on-significant-drop",
        ],
    )
    assert flagged.exit_code == 0, _output(flagged)
    assert "bootstrap" in (flagged.stdout or "")
    assert "significant drop:" not in (flagged.stdout or "")

    before, after = _constant_reports(10, recall=-0.5)
    baseline.write_text(json.dumps(before), encoding="utf-8")
    current.write_text(json.dumps(after), encoding="utf-8")
    shown = runner.invoke(
        app,
        ["compare", str(baseline), str(current), "--bootstrap", "50"],
    )
    assert shown.exit_code == 0, _output(shown)
    assert "significant drop:" not in (shown.stdout or "")

    failed = runner.invoke(
        app,
        [
            "compare",
            str(baseline),
            str(current),
            "--bootstrap",
            "50",
            "--fail-on-significant-drop",
        ],
    )
    assert failed.exit_code == 1, _output(failed)
    assert "significant drop: recall" in (failed.stdout or "")

    # The headline gate still fails on its own when the interval is only noise.
    _write(baseline, before_cases, headline={name: 1.0 for name in METRIC_NAMES})
    _write(current, after_cases, headline={name: 0.9 for name in METRIC_NAMES})
    headline = runner.invoke(
        app,
        [
            "compare",
            str(baseline),
            str(current),
            "--max-drop",
            "0.02",
            "--bootstrap",
            "200",
            "--fail-on-significant-drop",
        ],
    )
    assert headline.exit_code == 1, _output(headline)
    assert "regressed: hit_rate, recall, precision, mrr, ndcg" in (headline.stdout or "")
    assert "significant drop:" not in (headline.stdout or "")


def test_compare_old_baseline_skips_bootstrap(tmp_path):
    baseline = tmp_path / "baseline.json"
    current = tmp_path / "current.json"
    baseline.write_text(json.dumps(_report(None)), encoding="utf-8")
    _write(current, [_case("q1", recall=0.0)])
    output = tmp_path / "compare.json"
    noted = runner.invoke(
        app,
        [
            "compare",
            str(baseline),
            str(current),
            "--bootstrap",
            "20",
            "--fail-on-significant-drop",
            "--format",
            "markdown",
            "--json",
            str(output),
        ],
    )
    assert noted.exit_code == 0, _output(noted)
    text = noted.stdout or ""
    section = text.split("### bootstrap", 1)[1].split("### worse cases", 1)[0]
    assert "baseline report has no per-case results" in section
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["bootstrap"]["available"] is False
    assert payload["bootstrap"]["message"] == "baseline report has no per-case results"
    assert payload["bootstrap"]["checked"] is True
    assert "metrics" not in payload["bootstrap"]

    _write(baseline, [_case("q1")])
    current.write_text(json.dumps(_report(None)), encoding="utf-8")
    missing_current = runner.invoke(
        app,
        [
            "compare",
            str(baseline),
            str(current),
            "--bootstrap",
            "20",
            "--fail-on-significant-drop",
        ],
    )
    assert missing_current.exit_code == 0, _output(missing_current)
    assert "current report has no per-case results" in (missing_current.stdout or "")

    _write(baseline, [_case("only-base")])
    _write(current, [_case("only-new")])
    disjoint = runner.invoke(
        app,
        ["compare", str(baseline), str(current), "--bootstrap", "20", "--json", str(output)],
    )
    assert disjoint.exit_code == 0, _output(disjoint)
    assert "no paired cases to resample" in (disjoint.stdout or "")
    skipped = json.loads(output.read_text(encoding="utf-8"))
    assert skipped["bootstrap"]["available"] is False
    assert skipped["bootstrap"]["checked"] is False


def test_compare_rejects_bad_bootstrap_flags(tmp_path):
    baseline = tmp_path / "baseline.json"
    current = tmp_path / "current.json"
    _write(baseline, [_case("q1")])
    _write(current, [_case("q1")])
    negative = runner.invoke(
        app,
        ["compare", str(baseline), str(current), "--bootstrap", "-1"],
    )
    assert negative.exit_code == 2
    assert "--bootstrap" in _output(negative)

    for value in ("0", "1", "95"):
        bad = runner.invoke(
            app,
            ["compare", str(baseline), str(current), "--bootstrap", "10", "--confidence", value],
        )
        assert bad.exit_code == 2, value
        assert "--confidence" in _output(bad)


def test_bootstrap_ignores_tag_rows(tmp_path):
    baseline = tmp_path / "baseline.json"
    current = tmp_path / "current.json"
    before, after = _constant_reports(4, mrr=-0.3)
    before["tags"] = [
        {"tag": "billing", "case_count": 4, "metrics": {name: 1.0 for name in METRIC_NAMES}}
    ]
    after["tags"] = [
        {"tag": "billing", "case_count": 4, "metrics": {name: 0.5 for name in METRIC_NAMES}}
    ]
    baseline.write_text(json.dumps(before), encoding="utf-8")
    current.write_text(json.dumps(after), encoding="utf-8")
    result = runner.invoke(
        app,
        ["compare", str(baseline), str(current), "--bootstrap", "30", "--seed", "2"],
    )
    assert result.exit_code == 0, _output(result)
    text = result.stdout or ""
    assert "billing" in text
    section = text.split("bootstrap\n", 1)[1].split("worse cases", 1)[0]
    assert "billing" not in section
    assert section.count("mrr") == 1


def test_default_seed_matches_seed_zero(tmp_path):
    baseline = tmp_path / "baseline.json"
    current = tmp_path / "current.json"
    before_cases = [_case(f"q{i}", recall=0.2 if i % 3 else 0.9) for i in range(15)]
    after_cases = [_case(f"q{i}", recall=0.55) for i in range(15)]
    _write(baseline, before_cases)
    _write(current, after_cases)
    default = runner.invoke(
        app,
        ["compare", str(baseline), str(current), "--bootstrap", "80"],
    )
    explicit = runner.invoke(
        app,
        ["compare", str(baseline), str(current), "--bootstrap", "80", "--seed", "0"],
    )
    assert default.exit_code == 0, _output(default)
    assert default.stdout == explicit.stdout


def _reference(deltas, samples, confidence, seed, max_drop):
    names = list(METRIC_NAMES)
    columns = [deltas[name] for name in names]
    count = len(columns[0])
    rng = random.Random(seed)
    draws = [[] for _ in names]
    for _ in range(samples):
        totals = [0.0] * len(names)
        for _case in range(count):
            index = rng.randrange(count)
            for metric in range(len(names)):
                totals[metric] += columns[metric][index]
        for metric in range(len(names)):
            draws[metric].append(totals[metric] / count)
    alpha = (1.0 - confidence) / 2.0
    metrics = {}
    significant = []
    for metric, name in enumerate(names):
        observed = statistics.fmean(columns[metric])
        ordered = sorted(draws[metric])
        high = round(_percentile(ordered, 1.0 - alpha), 6)
        metrics[name] = {
            "delta": round(observed, 6),
            "low": round(_percentile(ordered, alpha), 6),
            "high": high,
            "p_value": round(_reference_p(observed, draws[metric]), 6),
        }
        if (-high) - max_drop > 1e-9:
            significant.append(name)
    return {"metrics": metrics, "significant": significant}


def _reference_p(observed, draws):
    threshold = abs(observed)
    count = sum(1 for sample in draws if abs(sample - observed) + 1e-9 >= threshold)
    return (count + 1) / (len(draws) + 1)
