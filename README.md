# retrieval-gate

retrieval-gate scores the retrieval step of a RAG pipeline against a labeled query set. You provide queries and the document ids that should come back. It calls your retriever, reports hit rate, recall, precision, MRR, and nDCG, and exits non-zero when a score falls under a threshold you set. The core makes no model call and no network call, so a CI job can run it the way it runs a coverage check.

## Install

Python 3.10 or newer. From a checkout of this repo:

```bash
pip install -e ".[dev]"
```

That installs the `retrieval-gate` command plus pytest and ruff. Drop `.[dev]` if you only want the command.

## Quickstart

`examples/` is a self-contained demo: 10 short documents, 10 labeled queries, and a keyword retriever (BM25 in the standard library). From the repository root:

```bash
retrieval-gate run \
  --dataset examples/cases.jsonl \
  --retriever examples.retriever:retrieve \
  --k 5
```

The same run with thresholds:

```bash
retrieval-gate run \
  --dataset examples/cases.jsonl \
  --retriever examples.retriever:retrieve \
  --k 5 \
  --fail-under recall=0.8 \
  --fail-under mrr=0.6
```

That prints:

```
dataset: examples/cases.jsonl
retriever: examples.retriever:retrieve
k: 5
cases: 10

metric     score   threshold  status
---------  ------  ---------  ------
hit_rate   1.0000  -          -
recall     1.0000  0.8000     pass
precision  0.2200  -          -
mrr        1.0000  0.6000     pass
ndcg       0.9964  -          -

tags
tag      cases  hit_rate  recall  precision  mrr     ndcg
-------  -----  --------  ------  ---------  ------  ------
api      2      1.0000    1.0000  0.3000     1.0000  0.9820
billing  3      1.0000    1.0000  0.2667     1.0000  0.9880

worst misses
none
```

Exit code 0 means every threshold passed. Exit code 1 means a threshold failed. Exit code 2 means the dataset, retriever, or arguments could not be used. The text report still prints when a threshold fails.

Write a JSON report:

```bash
retrieval-gate run \
  --dataset examples/cases.jsonl \
  --retriever examples.retriever:retrieve \
  --k 5 \
  --json report.json
```

`--retriever` is an import path, `package.module:function`. The current working directory is put on `sys.path`, so run the command from the directory that contains that module (the repo root for the example).

The callable is `(query: str, k: int)`. It can be sync or async. Return a list of document id strings, or a list of dicts or objects with an `id` field.

## Dataset format

JSONL is one case per line. YAML is a list of the same objects. Each case has an `id`, a `query`, and `relevant`. `relevant` is either a list of document ids (each stored as grade 1) or a mapping of document id to a positive integer grade.

An empty relevant set is rejected at load time. Recall and nDCG are undefined when nothing is relevant. Validation errors name the file line (JSONL) or the list index (YAML), and the case id when the line has one.

`tags` is optional. It is a list of non-empty strings, such as `billing` or `long-tail`. Leave it out when a case has no tags. A case can carry more than one.

```json
{"id": "q01", "query": "reset password", "relevant": ["d01"]}
{"id": "q10", "query": "webhook when an invoice is paid", "relevant": {"d10": 2, "d03": 1}, "tags": ["billing", "api"]}
```

```yaml
- id: q01
  query: reset password
  relevant:
    - d01
- id: q10
  query: webhook when an invoice is paid
  relevant:
    d10: 2
    d03: 1
  tags:
    - billing
    - api
```

The checked-in files are `examples/cases.jsonl` and `examples/cases.yaml`.

## Metrics

Scores use the cutoff `--k`. The ranking is the first k unique document ids, in the order returned. A repeated id does not take another rank slot. If that leaves fewer than k ids, the missing positions count as not relevant.

Hit rate, recall, precision, and MRR treat a document as relevant when its grade is greater than 0. nDCG uses the numeric grade. Binary labels are grade 1. The number in the report is the unweighted mean across cases. If any case has tags, a second table lists each tag, how many cases carry it, and the same five scores. A tag score is the unweighted mean of the cases with that tag. A case with two tags counts in both. Cases with no tags stay out of that table. Tags are listed alphabetically.

Threshold names are `hit_rate`, `recall`, `precision`, `mrr`, and `ndcg`. A score below the limit fails. A score equal to the limit passes. A dash in the threshold column means that metric is not gated.

`--fail-under-tag billing:recall=0.8` gates one tag the same way. Repeat the flag for more than one gate. The tag has to be on at least one case, or the run exits 2. In a config file the key is `fail_under_tag`, a mapping of tag to metric limits.

- **Hit rate.** 1 when any relevant id is in the top k, otherwise 0.
- **Recall.** Relevant ids in the top k, divided by the number of relevant ids.
- **Precision.** Relevant ids in the top k, divided by k (not by the number of ids actually returned).
- **MRR.** Reciprocal rank of the first relevant id in the top k. Rank 1 scores 1. No hit scores 0.
- **nDCG.** Normalized discounted cumulative gain. Gain at 1-based rank i is `(2 ** grade - 1) / log2(i + 1)`. The ideal ranking sorts the case grades descending and keeps at most k of them. nDCG is 0 when no relevant document was retrieved.

The report then lists up to 10 cases that missed at least one relevant id, lowest recall first. The JSON report has every case, including the retrieved ids and the misses. When cases have tags, it also has a `tags` list with each tag's case count and scores.

## Compare and CI

Save a JSON report from a known-good retriever and check later runs against it. `compare` prints the per-metric delta (current minus baseline) and exits 1 when any metric drops by more than `--max-drop` (default 0.02). A drop equal to the limit passes.

It also lists cases that got worse, matched by id. A case is listed when recall dropped, nDCG dropped, or a hit became a miss. The worst 10 are shown. `--show-cases N` changes that cap, and `--show-cases 0` hides the list. Each row has the case id, the query shortened to 60 characters, baseline and current recall and nDCG, and doc ids that are missed now but were not missed in the baseline. Cases found in only one report are printed as a count and their ids. That does not fail the command. `--fail-on-case-regression` exits 1 when any single case recall drops. It is off by default, so the exit code stays the aggregate check. The same section is in the text report, in `--format markdown`, and in the file from `--json`. A baseline written before per-case results existed prints a note instead of the list.

When both reports have per-tag scores, compare also prints each tag's delta. A baseline written before tags existed prints a note and does not fail. `--fail-on-tag-regression` applies `--max-drop` to every tag metric and exits 1 on a drop. It is off by default, so a drop in one tag does not change the exit code. Tags found in only one report are listed and do not fail the command.

```bash
retrieval-gate compare baseline.json report.json --max-drop 0.02
retrieval-gate compare baseline.json report.json --show-cases 10 --json compare.json
retrieval-gate compare baseline.json report.json --fail-on-tag-regression
```

A GitHub Actions job that regenerates the report and compares it:

```yaml
name: retrieval
on: [pull_request]
jobs:
  retrieval:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - name: Install
        run: pip install retrieval-gate
      - name: Score the retriever
        run: >
          retrieval-gate run
          --dataset tests/retrieval/cases.jsonl
          --retriever myapp.search:retrieve
          --k 10
          --json report.json
      - name: Compare to the baseline
        run: retrieval-gate compare tests/retrieval/baseline.json report.json --max-drop 0.02
```

You can also gate a single run without a stored baseline:

```bash
retrieval-gate run \
  --dataset tests/retrieval/cases.jsonl \
  --retriever myapp.search:retrieve \
  --k 10 \
  --fail-under recall=0.8
```

## GitHub Actions

`--format markdown` prints a table of metrics, thresholds, baseline deltas, and pass/fail. When cases have tags, it adds the per-tag table. Use it in a pull request comment or `$GITHUB_STEP_SUMMARY`. `--output` writes that same text to a file. `run` and `compare` both accept the two flags.

```bash
retrieval-gate run \
  --dataset examples/cases.jsonl \
  --retriever examples.retriever:retrieve \
  --k 5 \
  --fail-under recall=0.8 \
  --fail-under mrr=0.6 \
  --baseline examples/baseline.json \
  --format markdown \
  --output report.md
```

With no baseline, the baseline and delta columns are dashes. A score below its threshold fails. With a baseline, a drop larger than `--max-drop` (default 0.02) also fails. A drop equal to the limit passes. The text report is unchanged, and a comparison table is appended when a baseline is set.

`compare` prints the same markdown table, then the worse-case section. The status column is the baseline check. Thresholds stored in the current report are shown and do not change the compare exit code.

`action.yml` installs the package, runs the gate, and appends the report to the job summary. Copy `examples/retrieval.yml` to `.github/workflows/retrieval.yml`:

```yaml
name: retrieval
on: [pull_request]
jobs:
  retrieval:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: ./
        with:
          config: examples/gate.yml
          dataset: examples/cases.jsonl
          baseline: examples/baseline.json
```

`examples/gate.yml` holds the retriever, cutoff, and thresholds for the checked-in example. Paths in that file are relative to the file. The `dataset` and `baseline` inputs override it when you set them. In another repository, replace `./` with `sandeep-paluru/retrieval-gate@<ref>`. The step fails on a missed threshold or a baseline drop, after the summary is written.

## Roadmap

- Adapters for LangChain and LlamaIndex retrievers
- An HTTP retriever for FastAPI services
