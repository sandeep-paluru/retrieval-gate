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

Scores use the cutoff `--k`. Pass `--k-values` to score several cutoffs from one retrieval. See Several cutoffs. The ranking is the first k unique document ids, in the order returned. A repeated id does not take another rank slot. If that leaves fewer than k ids, the missing positions count as not relevant.

Hit rate, recall, precision, and MRR treat a document as relevant when its grade is greater than 0. nDCG uses the numeric grade. Binary labels are grade 1. The number in the report is the unweighted mean across cases. If any case has tags, a second table lists each tag, how many cases carry it, and the same five scores. A tag score is the unweighted mean of the cases with that tag. A case with two tags counts in both. Cases with no tags stay out of that table. Tags are listed alphabetically.

Threshold names are `hit_rate`, `recall`, `precision`, `mrr`, and `ndcg`. A score below the limit fails. A score equal to the limit passes. A dash in the threshold column means that metric is not gated.

`--fail-under-tag billing:recall=0.8` gates one tag the same way. Repeat the flag for more than one gate. The tag has to be on at least one case, or the run exits 2. In a config file the key is `fail_under_tag`, a mapping of tag to metric limits.

- **Hit rate.** 1 when any relevant id is in the top k, otherwise 0.
- **Recall.** Relevant ids in the top k, divided by the number of relevant ids.
- **Precision.** Relevant ids in the top k, divided by k (not by the number of ids actually returned).
- **MRR.** Reciprocal rank of the first relevant id in the top k. Rank 1 scores 1. No hit scores 0.
- **nDCG.** Normalized discounted cumulative gain. Gain at 1-based rank i is `(2 ** grade - 1) / log2(i + 1)`. The ideal ranking sorts the case grades descending and keeps at most k of them. nDCG is 0 when no relevant document was retrieved.

The report then lists up to 10 cases that missed at least one relevant id, lowest recall first. The JSON report has every case, including the retrieved ids and the misses. When cases have tags, it also has a `tags` list with each tag's case count and scores.

## Several cutoffs

`--k-values 1,3,5,10` scores every cutoff in one run. The retriever is called once per query, at the largest value, and each smaller cutoff is the prefix of that list. Duplicate ids still take one slot, and a short list still divides precision by k. The values are sorted. A duplicate, or a value below 1, is an error. `--k` and `--k-values` together exit 2. A config file that sets both `k` and `k_values` does the same when neither flag is passed. Either flag replaces both config keys.

`recall@5=0.8` gates recall at that cutoff, which has to be one of the values being scored. A bare `recall=0.8` applies to the largest. On a single-k run, `recall@5` with `--k 5` is the same gate as a bare `recall`. Tag gates do not take `@k`. The tag table, the worst misses, the per-case rows, and a bootstrap all use the largest k. The text report says so in those headings.

JSON keeps the old top-level shape. `k`, `metrics`, `thresholds`, `failed`, `results`, and `tags` are the largest cutoff, so a single-k file still loads as a baseline. `k_values` and `cutoffs` are added only when there is more than one cutoff. A failure at a smaller cutoff is `recall@5` in `failed`. A failure at the largest stays `recall`. In a config file the key is `k_values`, a list or the same comma-separated text. The action input is `k-values`.

When both reports have `cutoffs`, compare prints a delta at every shared k and fails if any of those drops by more than `--max-drop`. A cutoff present on only one side is listed and does not fail. When only one report has `cutoffs`, compare uses the other report's `k` if that cutoff was scored. If it was not, or that report has no `k`, the command says the metrics were not compared and does not fail the metric check. Tag deltas and worse cases still use the stored scores, which are each report's largest k.

```bash
retrieval-gate run \
  --dataset examples/cases.jsonl \
  --retriever examples.retriever:retrieve \
  --k-values 1,3,5,10 \
  --fail-under recall@5=0.8 \
  --fail-under mrr=0.6
```

That prints:

```
dataset: examples/cases.jsonl
retriever: examples.retriever:retrieve
k: 1, 3, 5, 10
cases: 10

metric     @1      @3      @5      @10
---------  ------  ------  ------  ------
hit_rate   1.0000  1.0000  1.0000  1.0000
recall     0.9500  1.0000  1.0000  1.0000
precision  1.0000  0.3667  0.2200  0.1100
mrr        1.0000  1.0000  1.0000  1.0000
ndcg       1.0000  0.9964  0.9964  0.9964

thresholds
metric  k   threshold  score   status
------  --  ---------  ------  ------
recall  5   0.8000     1.0000  pass
mrr     10  0.6000     1.0000  pass

tags at k=10
tag      cases  hit_rate  recall  precision  mrr     ndcg
-------  -----  --------  ------  ---------  ------  ------
api      2      1.0000    1.0000  0.1500     1.0000  0.9820
billing  3      1.0000    1.0000  0.1333     1.0000  0.9880

worst misses at k=10
none
```

## Latency

Every scored retriever call is timed with `time.perf_counter`. The report adds the mean, p50, p95, p99, and max, in milliseconds. There is one sample per case. `--k-values` still calls the retriever once, at the largest cutoff, so it still records one latency per case. With `--concurrency`, each HTTP request is timed on its own. Time spent waiting for a free slot is not included.

Percentiles use linear interpolation, not nearest rank. The samples are sorted and the position of quantile `q` is `q * (n - 1)`. A position between two samples is a weighted blend of those two. With one sample, every stat is that sample. With two samples, p50 is the midpoint and p95 sits close to the slower one. The JSON report names this method `"linear"`.

`--warmup N` calls the first N queries once before scoring. Those calls are not timed and their ids are not scored. The default is 0. If N is larger than the dataset, the run exits 2. Warmup is there to drop cold-start cost, such as the first import or the first connection.

`--max-latency p95=250` fails the run when that stat is above 250 milliseconds. Repeat the flag for more than one stat. The names are `mean`, `p50`, `p95`, `p99`, and `max`. A stat equal to its limit passes. The table has the same threshold and status columns as `--fail-under`. A miss is `failed latency:` and the exit code is 1. Quality metrics are unchanged, and a latency miss does not change the `failed` list of metric names.

```bash
retrieval-gate run \
  --dataset examples/cases.jsonl \
  --retriever examples.retriever:retrieve \
  --k 5 \
  --fail-under recall=0.8 \
  --fail-under mrr=0.6 \
  --warmup 2 \
  --max-latency p95=250
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

latency (ms)
warmup: 2
stat  ms      threshold  status
----  ------  ---------  ------
mean  0.0214  -          -
p50   0.0202  -          -
p95   0.0263  250.0000   pass
p99   0.0266  -          -
max   0.0266  -          -
```

The millisecond numbers above are from one run of the example retriever. The next run will not match them. See the note on noise below.

The JSON report adds a `latency` object and a `latency_ms` field on each case:

```json
{
  "unit": "ms",
  "method": "linear",
  "warmup": 2,
  "mean": 0.021412,
  "p50": 0.020192,
  "p95": 0.026307,
  "p99": 0.026556,
  "max": 0.026618,
  "thresholds": {"p95": 250.0},
  "failed": []
}
```

`compare` prints baseline, current, and the delta (current minus baseline) when both reports have latency. `--max-latency-increase 20` fails when the chosen stat is more than 20 percent above the baseline. `--latency-stat` picks that stat and defaults to `p95`. An increase equal to the limit passes. A baseline of 0 fails the check when the current stat is above 0.

A baseline written before latency existed prints `baseline report has no latency` and skips the check. It does not fail and it does not crash. The same note is printed when the current report has no latency. If neither report has latency and the flag is left off, compare is unchanged.

```bash
retrieval-gate compare baseline.json report.json --max-latency-increase 50
```

Two runs of the example, after warmup, appended this:

```
latency (ms)
max latency increase: 50% on p95
stat  baseline  current  delta    status
----  --------  -------  -------  ------
mean  0.0214    0.0222   +0.0008  -
p50   0.0202    0.0216   +0.0014  -
p95   0.0263    0.0267   +0.0004  pass
p99   0.0266    0.0281   +0.0015  -
max   0.0266    0.0284   +0.0018  -
```

`examples/baseline.json` has no latency object. Comparing a new report to it prints `baseline report has no latency` and still checks the quality metrics.

In a config file the keys are `warmup` and `max_latency`. `max_latency` is a mapping of stat to limit, or a list of `stat=milliseconds` strings, same shape as `fail_under`:

```yaml
warmup: 2
max_latency:
  p95: 250
  max: 500
```

Latency moves around. The same retriever on the same cases will not hit the same millisecond twice, and a CI runner is slower than a laptop. Use `--warmup` so the first calls do not set p95, and set the gate well above the number you usually see. A gate at the exact p95 from one run will flap.

## Compare and CI

Save a JSON report from a known-good retriever and check later runs against it. `compare` prints the per-metric delta (current minus baseline) and exits 1 when any metric drops by more than `--max-drop` (default 0.02). A drop equal to the limit passes.

It also lists cases that got worse, matched by id. A case is listed when recall dropped, nDCG dropped, or a hit became a miss. The worst 10 are shown. `--show-cases N` changes that cap, and `--show-cases 0` hides the list. Each row has the case id, the query shortened to 60 characters, baseline and current recall and nDCG, and doc ids that are missed now but were not missed in the baseline. Cases found in only one report are printed as a count and their ids. That does not fail the command. `--fail-on-case-regression` exits 1 when any single case recall drops. It is off by default, so the exit code stays the aggregate check. The same section is in the text report, in `--format markdown`, and in the file from `--json`. A baseline written before per-case results existed prints a note instead of the list.

When both reports have per-tag scores, compare also prints each tag's delta. A baseline written before tags existed prints a note and does not fail. `--fail-on-tag-regression` applies `--max-drop` to every tag metric and exits 1 on a drop. It is off by default, so a drop in one tag does not change the exit code. Tags found in only one report are listed and do not fail the command.

When both reports include latency, compare also prints those stats and the delta. `--max-latency-increase` can fail the command when one stat grew by more than a percent limit. A baseline with no latency prints a note and skips that check. See Latency.

On a small labeled set, a 0.02 drop in recall or nDCG is often just which queries are in the file. `--bootstrap N` resamples the cases shared by both reports. For hit rate, recall, precision, MRR, and nDCG it prints the mean of current minus baseline, a confidence interval, and a two-sided p-value for a mean change of zero. Omit the flag, or pass 0, and the check stays off. Use 2000 resamples when you want an interval. `--confidence` defaults to 0.95. `--seed` defaults to 0, so the same two files print the same interval on every run.

`--fail-on-significant-drop` exits 1 when the whole interval is below `-max-drop`. The upper bound is the mildest value in the interval, and it has to be a larger drop than the limit. An upper bound equal to `-max-drop` passes. That flag reads the interval, and it does nothing unless `--bootstrap` is set. Leaving both off keeps the existing exit behavior. The aggregate `--max-drop` gate still runs either way.

The bootstrap mean uses paired cases, so it can differ from the headline delta when the reports do not list the same cases. A case missing any of the five metrics is skipped. Per-tag scores are not resampled. A report written before per-case results existed prints a note and skips the check. The block is in the text report, in `--format markdown`, and in the `--json` file.

```bash
retrieval-gate compare baseline.json report.json --max-drop 0.02
retrieval-gate compare baseline.json report.json --show-cases 10 --json compare.json
retrieval-gate compare baseline.json report.json --fail-on-tag-regression
retrieval-gate compare baseline.json report.json --bootstrap 2000 --fail-on-significant-drop
retrieval-gate compare baseline.json report.json --max-latency-increase 20 --latency-stat p95
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

`--format markdown` prints a table of metrics, thresholds, baseline deltas, and pass/fail. When cases have tags, it adds the per-tag table. When the run timed calls, it adds a latency table. Use it in a pull request comment or `$GITHUB_STEP_SUMMARY`. `--output` writes that same text to a file. `run` and `compare` both accept the two flags.

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

`examples/gate.yml` holds the retriever, cutoff, and thresholds for the checked-in example. Paths in that file are relative to the file. The `dataset` and `baseline` inputs override it when you set them. The action also takes `warmup` and `max-latency` (`p95=250`, space-separated if you set more than one stat). Both override the config file when set. In another repository, replace `./` with `sandeep-paluru/retrieval-gate@<ref>`. The step fails on a missed threshold, a latency gate, or a baseline drop, after the summary is written.
