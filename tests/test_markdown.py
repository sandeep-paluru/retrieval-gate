from retrieval_gate.metrics import METRIC_NAMES
from retrieval_gate.report import format_markdown


def _metrics(**overrides: float) -> dict[str, float]:
    values = {name: 1.0 for name in METRIC_NAMES}
    values.update(overrides)
    return values


def test_markdown_snapshot():
    report = {
        "dataset": "examples/cases.jsonl",
        "retriever": "examples.retriever:retrieve",
        "k": 5,
        "case_count": 10,
        "metrics": {
            "hit_rate": 1.0,
            "recall": 1.0,
            "precision": 0.22,
            "mrr": 1.0,
            "ndcg": 0.9964,
        },
        "thresholds": {"recall": 0.8, "mrr": 0.6},
    }
    assert format_markdown(report) == (
        "## retrieval-gate\n"
        "\n"
        "dataset: examples/cases.jsonl\n"
        "retriever: examples.retriever:retrieve\n"
        "k: 5\n"
        "cases: 10\n"
        "\n"
        "| metric | score | threshold | baseline | delta | status |\n"
        "| --- | ---: | ---: | ---: | ---: | --- |\n"
        "| hit_rate | 1.0000 | - | - | - | - |\n"
        "| recall | 1.0000 | 0.8000 | - | - | pass |\n"
        "| precision | 0.2200 | - | - | - | - |\n"
        "| mrr | 1.0000 | 0.6000 | - | - | pass |\n"
        "| ndcg | 0.9964 | - | - | - | - |"
    )


def test_markdown_marks_a_regression():
    report = {
        "metrics": _metrics(recall=0.9),
        "thresholds": {"recall": 0.8, "mrr": 0.6},
    }
    text = format_markdown(
        report,
        baseline={"metrics": _metrics()},
        max_drop=0.02,
        baseline_path="baseline.json",
        current_path="report.json",
    )
    assert text == (
        "## retrieval-gate\n"
        "\n"
        "baseline: baseline.json\n"
        "current: report.json\n"
        "max drop: 0.0200\n"
        "\n"
        "| metric | score | threshold | baseline | delta | status |\n"
        "| --- | ---: | ---: | ---: | ---: | --- |\n"
        "| hit_rate | 1.0000 | - | 1.0000 | +0.0000 | pass |\n"
        "| recall | 0.9000 | 0.8000 | 1.0000 | -0.1000 | FAIL |\n"
        "| precision | 1.0000 | - | 1.0000 | +0.0000 | pass |\n"
        "| mrr | 1.0000 | 0.6000 | 1.0000 | +0.0000 | pass |\n"
        "| ndcg | 1.0000 | - | 1.0000 | +0.0000 | pass |\n"
        "\n"
        "regressed: recall"
    )

    # A drop equal to the limit is not a regression.
    equal = format_markdown(
        {"metrics": _metrics(recall=0.98)},
        baseline={"metrics": _metrics()},
        max_drop=0.02,
    )
    assert "regressed" not in equal
    assert "FAIL" not in equal
    assert "| recall | 0.9800 | - | 1.0000 | -0.0200 | pass |" in equal


def test_markdown_empty_baseline():
    report = {
        "metrics": {name: 0.5 for name in METRIC_NAMES},
        "thresholds": {"recall": 0.8},
    }
    assert format_markdown(report, baseline=None) == (
        "## retrieval-gate\n"
        "\n"
        "| metric | score | threshold | baseline | delta | status |\n"
        "| --- | ---: | ---: | ---: | ---: | --- |\n"
        "| hit_rate | 0.5000 | - | - | - | - |\n"
        "| recall | 0.5000 | 0.8000 | - | - | FAIL |\n"
        "| precision | 0.5000 | - | - | - | - |\n"
        "| mrr | 0.5000 | - | - | - | - |\n"
        "| ndcg | 0.5000 | - | - | - | - |\n"
        "\n"
        "failed: recall"
    )
