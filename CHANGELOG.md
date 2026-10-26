# Changelog

## 0.7.0

- pytest can run the gate. The `retrieval_gate` fixture scores a dataset with the same code as `retrieval-gate run`. `result.assert_scores(recall=0.8, mrr=0.6)` fails the test and lists each metric that is below its threshold, the score, the threshold, and the worst misses.
- `--retrieval-gate-config` reads `fail_under`, `fail_under_tag`, and `max_latency` from a gate.yml file. The `retrieval_gate_config` ini option and `@pytest.mark.retrieval_gate(config=...)` point at the same kind of file.
- `--retrieval-gate-json` writes the JSON report `compare` already accepts. A baseline in the config is not compared from pytest.
