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

```json
{"id": "q01", "query": "reset password", "relevant": ["d01"]}
{"id": "q10", "query": "webhook when an invoice is paid", "relevant": {"d10": 2, "d03": 1}}
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
```

The checked-in files are `examples/cases.jsonl` and `examples/cases.yaml`.

## Metrics

Scores use the cutoff `--k`. The ranking is the first k unique document ids, in the order returned. A repeated id does not take another rank slot. If that leaves fewer than k ids, the missing positions count as not relevant.

Hit rate, recall, precision, and MRR treat a document as relevant when its grade is greater than 0. nDCG uses the numeric grade. Binary labels are grade 1. The number in the report is the unweighted mean across cases.

Threshold names are `hit_rate`, `recall`, `precision`, `mrr`, and `ndcg`. A score below the limit fails. A score equal to the limit passes. A dash in the threshold column means that metric is not gated.

- **Hit rate.** 1 when any relevant id is in the top k, otherwise 0.
- **Recall.** Relevant ids in the top k, divided by the number of relevant ids.
- **Precision.** Relevant ids in the top k, divided by k (not by the number of ids actually returned).
- **MRR.** Reciprocal rank of the first relevant id in the top k. Rank 1 scores 1. No hit scores 0.
- **nDCG.** Normalized discounted cumulative gain. Gain at 1-based rank i is `(2 ** grade - 1) / log2(i + 1)`. The ideal ranking sorts the case grades descending and keeps at most k of them. nDCG is 0 when no relevant document was retrieved.

The report then lists up to 10 cases that missed at least one relevant id, lowest recall first. The JSON report has every case, including the retrieved ids and the misses.

## Compare and CI

Save a JSON report from a known-good retriever and check later runs against it. `compare` prints the per-metric delta (current minus baseline) and exits 1 when any metric drops by more than `--max-drop` (default 0.02). A drop equal to the limit passes.

```bash
retrieval-gate compare baseline.json report.json --max-drop 0.02
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

## Roadmap

- Adapters for LangChain and LlamaIndex retrievers
- An HTTP retriever for FastAPI services
- Per-tag breakdowns when cases carry a tag
