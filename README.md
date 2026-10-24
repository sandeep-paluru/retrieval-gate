# retrieval-gate

retrieval-gate scores the retrieval step of a RAG pipeline against a labeled query set. You provide queries and the document ids that should come back. It calls your retriever, reports hit rate, recall, precision, MRR, and nDCG, and exits non-zero when a score falls under a threshold you set. Passed an import path, it makes no model call and no network call, so a CI job can run it the way it runs a coverage check. `--retriever-url` opts in to calling an HTTP service.

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

`--retriever` is an import path, `package.module:attribute`. The current working directory is put on `sys.path`, so run the command from the directory that contains that module (the repo root for the example).

The callable is `(query: str, k: int)`. It can be sync or async. Return a list of document id strings, or a list of dicts or objects with an `id` field. A LangChain or LlamaIndex retriever is accepted too. See LangChain and LlamaIndex.

## HTTP retriever

`--retriever-url` scores a search service. The import path stays local and still makes no network call. Pass one of the two, not both.

The default request is a POST with this JSON body:

```json
{"query": "reset password", "k": 5}
```

`--method get` sends `query` and `k` as query parameters. `--header` can be repeated, as `Name: Value`. Write `${ENV_VAR}` in the value and the variable is read from the environment when the run starts. The header value is sent to the service. It is left out of the report and out of error text.

A 2xx body can be a JSON list of id strings or objects. It can also be an object with that list under `results`, `documents`, or `hits`, in that order. `--results-key` picks a different key. `--id-field` picks the id field on each object and defaults to `id`. A status that is not 2xx, or a body that is not one of those shapes, exits 2 and names the case id.

`--timeout` is the per-request limit in seconds and defaults to 10. `--concurrency` is how many requests run at once and defaults to 4.

The report line that names the retriever prints the URL with userinfo and the query string removed, so a token there is not copied into the report or the JSON file.

The YAML config takes the same settings: `retriever_url`, `method`, `headers`, `timeout`, `concurrency`, `results_key`, and `id_field`. `headers` is a list of `Name: Value` strings, or a mapping of name to value:

```yaml
retriever_url: http://localhost:8000/search
method: post
timeout: 10
concurrency: 4
headers:
  Authorization: Bearer ${SEARCH_TOKEN}
```

`examples/fastapi_service.py` serves the example BM25 retriever. From the repository root:

```bash
pip install -e ".[examples]"
uvicorn examples.fastapi_service:app --port 8000
```

```bash
retrieval-gate run \
  --dataset examples/cases.jsonl \
  --retriever-url http://localhost:8000/search \
  --k 5 \
  --fail-under recall=0.8
```

POST `/search` reads the JSON body above. GET `/search` reads `query` and `k`. Both return `{"results": [{"id": "d01"}, ...]}`.

A run that keeps the token in the environment:

```bash
retrieval-gate run \
  --dataset examples/cases.jsonl \
  --retriever-url http://localhost:8000/search \
  --header "Authorization: Bearer ${SEARCH_TOKEN}" \
  --k 5
```

## LangChain and LlamaIndex

`--retriever` can point at a LangChain retriever or vector store, or a LlamaIndex retriever or index. The type is detected. `--adapter langchain` or `--adapter llamaindex` forces one. Both libraries are optional:

```bash
pip install "retrieval-gate[langchain]"
pip install "retrieval-gate[llamaindex]"
```

From the repository root, after `pip install -e ".[langchain]"`:

```bash
retrieval-gate run \
  --dataset examples/cases.jsonl \
  --retriever examples.langchain_retriever:store \
  --k 5
```

`examples/langchain_retriever.py` builds an in-memory store with a deterministic embedding, so the run needs no API key and no network call.

LlamaIndex, after `pip install -e ".[llamaindex]"`:

```bash
retrieval-gate run \
  --dataset examples/cases.jsonl \
  --retriever examples.llamaindex_retriever:index \
  --k 5
```

Document ids come from `Document.id` for LangChain and from `node.node_id` for LlamaIndex. `--id-field source` reads a metadata key instead. For LlamaIndex, `--id-field ref_doc_id` reads the source document id. A result with no id exits 2.

`k` is passed through as LangChain `search_kwargs` or vector store `k`, and as LlamaIndex `similarity_top_k`. Results are truncated to `k`. Async `ainvoke` and `aretrieve` are used when the object has them.

The YAML config takes `adapter` and `id_field` next to `retriever`:

```yaml
retriever: examples.langchain_retriever:store
adapter: langchain
k: 5
```

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

On a small labeled set, a 0.02 drop in recall or nDCG is often just which queries are in the file. `--bootstrap N` resamples the cases shared by both reports. For hit rate, recall, precision, MRR, and nDCG it prints the mean of current minus baseline, a confidence interval, and a two-sided p-value for a mean change of zero. Omit the flag, or pass 0, and the check stays off. Use 2000 resamples when you want an interval. `--confidence` defaults to 0.95. `--seed` defaults to 0, so the same two files print the same interval on every run.

`--fail-on-significant-drop` exits 1 when the whole interval is below `-max-drop`. The upper bound is the mildest value in the interval, and it has to be a larger drop than the limit. An upper bound equal to `-max-drop` passes. That flag reads the interval, and it does nothing unless `--bootstrap` is set. Leaving both off keeps the existing exit behavior. The aggregate `--max-drop` gate still runs either way.

The bootstrap mean uses paired cases, so it can differ from the headline delta when the reports do not list the same cases. A case missing any of the five metrics is skipped. Per-tag scores are not resampled. A report written before per-case results existed prints a note and skips the check. The block is in the text report, in `--format markdown`, and in the `--json` file.

```bash
retrieval-gate compare baseline.json report.json --max-drop 0.02
retrieval-gate compare baseline.json report.json --show-cases 10 --json compare.json
retrieval-gate compare baseline.json report.json --fail-on-tag-regression
retrieval-gate compare baseline.json report.json --bootstrap 2000 --fail-on-significant-drop
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
