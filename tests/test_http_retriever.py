"""HTTP retriever tests. The server listens on localhost only."""

import json
import socket
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

import pytest
from typer.testing import CliRunner

from retrieval_gate.cli import app, load_config
from retrieval_gate.http_retriever import (
    DEFAULT_CONCURRENCY,
    DEFAULT_TIMEOUT,
    build_http_retriever,
    public_url,
)
from retrieval_gate.models import Case
from retrieval_gate.report import build_report

runner = CliRunner()

_SECRETS = (
    "sekret-header-value",
    "sekret-query-value",
    "sekret-body-value",
    "sekret-not-json",
    "literal-secret",
)


class _State:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.requests: list[dict] = []
        self.inflight = 0
        self.max_inflight = 0
        self.entered = 0
        self.release = threading.Event()


_state = _State()


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: object) -> None:
        return

    def do_GET(self) -> None:
        self._handle()

    def do_POST(self) -> None:
        self._handle()

    def _handle(self) -> None:
        length = int(self.headers.get("Content-Length", "0") or 0)
        body = self.rfile.read(length) if length else b""
        parsed = urlsplit(self.path)
        header_map = {key.lower(): value for key, value in self.headers.items()}
        with _state.lock:
            _state.requests.append(
                {
                    "method": self.command,
                    "path": parsed.path,
                    "query": parse_qs(parsed.query),
                    "headers": header_map,
                    "body": body,
                }
            )
        self._route(parsed.path, header_map)

    def _route(self, path: str, headers: dict[str, str]) -> None:
        if path == "/search":
            self._json(200, {"results": [{"id": "d1"}, {"id": "d2"}]})
        elif path == "/list":
            self._json(200, [" d1 ", {"id": " d2 "}])
        elif path == "/documents":
            self._json(200, {"documents": [{"id": "d1"}]})
        elif path == "/hits":
            self._json(200, {"hits": [{"id": "d1"}]})
        elif path == "/prefer":
            self._json(
                200,
                {
                    "hits": [{"id": "d9"}],
                    "documents": [{"id": "d8"}],
                    "results": [{"id": "d1"}],
                },
            )
        elif path == "/custom":
            self._json(200, {"items": [{"doc_id": "d1"}]})
        elif path == "/bad-id":
            self._json(200, {"results": [{"name": "d1"}]})
        elif path == "/bad-results":
            self._json(200, {"results": {"nested": True}, "hits": [{"id": "d1"}]})
        elif path == "/object":
            self._json(200, {"error": "nope"})
        elif path == "/malformed":
            self._send(200, b"sekret-not-json")
        elif path == "/error":
            self._json(500, {"detail": "nope"})
        elif path == "/reflect":
            reflected = headers.get("authorization", "")
            self._json(500, {"detail": reflected, "body": "sekret-body-value"})
        elif path == "/redirect":
            self._redirect()
        elif path == "/slow":
            time.sleep(2)
            self._json(200, ["d1"])
        elif path == "/overlap":
            self._overlap()
        else:
            self._json(404, {"detail": "missing"})

    def _overlap(self) -> None:
        with _state.lock:
            _state.inflight += 1
            _state.max_inflight = max(_state.max_inflight, _state.inflight)
            _state.entered += 1
            entered = _state.entered
        try:
            if entered >= 2:
                _state.release.set()
            else:
                _state.release.wait(3)
            self._json(200, ["d1"])
        finally:
            with _state.lock:
                _state.inflight -= 1

    def _redirect(self) -> None:
        try:
            self.send_response(302)
            self.send_header("Location", "/search")
            self.send_header("Content-Length", "0")
            self.end_headers()
        except OSError:
            return

    def _json(self, status: int, payload: object) -> None:
        self._send(status, json.dumps(payload).encode("utf-8"))

    def _send(self, status: int, body: bytes) -> None:
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except OSError:
            return


class _Server(ThreadingHTTPServer):
    daemon_threads = True


@pytest.fixture(scope="module")
def base_url() -> Iterator[str]:
    server = _Server(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(
        target=server.serve_forever,
        kwargs={"poll_interval": 0.05},
        daemon=True,
    )
    thread.start()
    host, port = server.server_address
    try:
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.fixture(autouse=True)
def _reset_http_state() -> None:
    with _state.lock:
        _state.requests.clear()
        _state.inflight = 0
        _state.max_inflight = 0
        _state.entered = 0
    _state.release.clear()


def _output(result) -> str:
    return (result.stdout or "") + (result.stderr or "")


def _assert_no_secrets(text: str) -> None:
    for secret in _SECRETS:
        assert secret not in text


def _write_cases(path, text: str | None = None) -> None:
    path.write_text(
        text
        or '{"id": "q1", "query": "cats", "relevant": ["d1"]}\n',
        encoding="utf-8",
    )


def _run(args: list[str]):
    return runner.invoke(app, args)


def test_public_url_drops_userinfo_query_and_fragment():
    raw = "http://user:sekret-header-value@localhost:8000/search?token=sekret-query-value#part"
    shown = public_url(raw)
    assert shown == "http://localhost:8000/search"
    assert public_url("http://[::1]:8000/search?x=1") == "http://[::1]:8000/search"
    assert DEFAULT_TIMEOUT == 10
    assert DEFAULT_CONCURRENCY == 4


def test_post_scores_results_and_hides_secrets(tmp_path, base_url, monkeypatch):
    monkeypatch.setenv("GATE_TOKEN", "sekret-header-value")
    dataset = tmp_path / "cases.jsonl"
    _write_cases(
        dataset,
        '{"id": "q1", "query": "cats", "relevant": ["d1", "d2"]}\n'
        '{"id": "q2", "query": "dogs", "relevant": ["d1"]}\n',
    )
    report_path = tmp_path / "report.json"
    result = _run(
        [
            "run",
            "--dataset",
            str(dataset),
            "--retriever-url",
            f"{base_url}/search?token=sekret-query-value",
            "--k",
            "2",
            "--concurrency",
            "2",
            "--header",
            "Authorization: Bearer ${GATE_TOKEN}",
            "--header",
            "X-Index: docs",
            "--json",
            str(report_path),
        ]
    )
    output = _output(result)
    assert result.exit_code == 0, output
    _assert_no_secrets(output)
    assert f"retriever: {base_url}/search\n" in (result.stdout or "")
    assert "hit_rate   1.0000" in (result.stdout or "")
    saved = report_path.read_text(encoding="utf-8")
    _assert_no_secrets(saved)
    report = json.loads(saved)
    assert report["retriever"] == f"{base_url}/search"
    assert report["results"][0]["retrieved"] == ["d1", "d2"]
    assert len(_state.requests) == 2
    queries = set()
    for recorded in _state.requests:
        assert recorded["method"] == "POST"
        assert recorded["path"] == "/search"
        assert recorded["headers"]["authorization"] == "Bearer sekret-header-value"
        assert recorded["headers"]["x-index"] == "docs"
        assert recorded["headers"]["content-type"].startswith("application/json")
        assert recorded["query"]["token"] == ["sekret-query-value"]
        body = json.loads(recorded["body"])
        assert body["k"] == 2
        queries.add(body["query"])
    assert queries == {"cats", "dogs"}


def test_get_sends_query_and_k(tmp_path, base_url):
    dataset = tmp_path / "cases.jsonl"
    _write_cases(dataset, '{"id": "q1", "query": "reset password", "relevant": ["d1"]}\n')
    result = _run(
        [
            "run",
            "--dataset",
            str(dataset),
            "--retriever-url",
            f"{base_url}/search?token=sekret-query-value&query=old",
            "--method",
            "GET",
            "--k",
            "2",
        ]
    )
    output = _output(result)
    assert result.exit_code == 0, output
    _assert_no_secrets(output)
    assert f"retriever: {base_url}/search\n" in (result.stdout or "")
    recorded = _state.requests[0]
    assert recorded["method"] == "GET"
    assert recorded["query"]["query"] == ["reset password"]
    assert recorded["query"]["k"] == ["2"]
    assert recorded["query"]["token"] == ["sekret-query-value"]
    assert "old" not in recorded["query"].get("query", [])
    assert recorded["body"] == b""


@pytest.mark.parametrize(
    ("path", "extra", "first_id"),
    [
        ("/list", [], "d1"),
        ("/documents", [], "d1"),
        ("/hits", [], "d1"),
        ("/prefer", [], "d1"),
        ("/custom", ["--results-key", "items", "--id-field", "doc_id"], "d1"),
        ("/list", ["--results-key", "items"], "d1"),
    ],
)
def test_response_shapes(tmp_path, base_url, path, extra, first_id):
    dataset = tmp_path / "cases.jsonl"
    _write_cases(dataset)
    report_path = tmp_path / "report.json"
    result = _run(
        [
            "run",
            "--dataset",
            str(dataset),
            "--retriever-url",
            f"{base_url}{path}",
            "--k",
            "2",
            "--json",
            str(report_path),
            *extra,
        ]
    )
    output = _output(result)
    assert result.exit_code == 0, output
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["results"][0]["retrieved"][0] == first_id
    if path == "/list":
        assert report["results"][0]["retrieved"] == ["d1", "d2"]


def test_config_file_get_headers_and_cli_override(tmp_path, base_url, monkeypatch):
    monkeypatch.setenv("GATE_TOKEN", "sekret-header-value")
    dataset = tmp_path / "cases.jsonl"
    _write_cases(dataset, '{"id": "q1", "query": "cats", "relevant": ["d1"]}\n')
    config = tmp_path / "gate.yml"
    config.write_text(
        "dataset: cases.jsonl\n"
        f"retriever_url: {base_url}/search\n"
        "method: get\n"
        "headers:\n"
        "  X-Token: ${GATE_TOKEN}\n"
        "  X-Index: docs\n"
        "timeout: 5\n"
        "concurrency: 2\n"
        "results_key: results\n"
        "id_field: id\n"
        "k: 2\n",
        encoding="utf-8",
    )
    result = _run(["run", "--config", str(config)])
    output = _output(result)
    assert result.exit_code == 0, output
    _assert_no_secrets(output)
    assert f"retriever: {base_url}/search\n" in (result.stdout or "")
    recorded = _state.requests[0]
    assert recorded["method"] == "GET"
    assert recorded["headers"]["x-token"] == "sekret-header-value"
    assert recorded["headers"]["x-index"] == "docs"
    assert recorded["query"]["k"] == ["2"]

    _state.requests.clear()
    overridden = _run(
        [
            "run",
            "--config",
            str(config),
            "--method",
            "post",
            "--header",
            "X-Other: plain",
        ]
    )
    output = _output(overridden)
    assert overridden.exit_code == 0, output
    _assert_no_secrets(output)
    posted = _state.requests[0]
    assert posted["method"] == "POST"
    assert posted["headers"]["x-other"] == "plain"
    assert "x-token" not in posted["headers"]
    assert json.loads(posted["body"]) == {"query": "cats", "k": 2}


def test_config_header_list_and_invalid_values(tmp_path):
    config = tmp_path / "gate.yml"
    config.write_text(
        "retriever_url: http://127.0.0.1/search\n"
        "headers:\n"
        "  - \"Authorization: Bearer ${GATE_TOKEN}\"\n"
        "  - \"X-Index: docs\"\n",
        encoding="utf-8",
    )
    loaded = load_config(config)
    assert loaded["headers"] == [
        "Authorization: Bearer ${GATE_TOKEN}",
        "X-Index: docs",
    ]

    config.write_text(
        "dataset: cases.jsonl\n"
        "retriever: tests.fakes:retrieve\n"
        "retriever_url: http://127.0.0.1/search\n",
        encoding="utf-8",
    )
    dataset = tmp_path / "cases.jsonl"
    _write_cases(dataset)
    both = _run(["run", "--config", str(config)])
    assert both.exit_code == 2
    assert "both retriever and retriever_url" in _output(both)

    config.write_text("retriever: tests.fakes:retrieve\ntimeout: 1\n", encoding="utf-8")
    orphan = _run(["run", "--config", str(config), "--dataset", str(dataset)])
    assert orphan.exit_code == 2
    assert "config key timeout requires retriever_url" in _output(orphan)

    config.write_text("retriever_url: http://127.0.0.1/search\nmethod: put\n", encoding="utf-8")
    bad_method = _run(["run", "--config", str(config), "--dataset", str(dataset)])
    assert bad_method.exit_code == 2
    assert "method must be get or post" in _output(bad_method)

    config.write_text(
        "retriever_url: http://127.0.0.1/search\nconcurrency: true\n",
        encoding="utf-8",
    )
    bad_concurrency = _run(["run", "--config", str(config), "--dataset", str(dataset)])
    assert bad_concurrency.exit_code == 2
    assert "concurrency must be an integer >= 1" in _output(bad_concurrency)


def test_import_path_ignores_config_url_and_does_not_connect(tmp_path, monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("import path opened a connection")

    monkeypatch.setattr("urllib.request.OpenerDirector.open", boom)
    dataset = tmp_path / "cases.jsonl"
    _write_cases(dataset, '{"id": "q1", "query": "cats", "relevant": ["d1"]}\n')
    config = tmp_path / "gate.yml"
    config.write_text(
        "dataset: cases.jsonl\n"
        "retriever_url: http://user:sekret-header-value@127.0.0.1:1/search"
        "?token=sekret-query-value\n"
        "timeout: 1\n"
        "headers:\n"
        "  Authorization: Bearer sekret-header-value\n",
        encoding="utf-8",
    )
    result = _run(
        [
            "run",
            "--config",
            str(config),
            "--retriever",
            "tests.fakes:retrieve",
            "--k",
            "2",
        ]
    )
    output = _output(result)
    assert result.exit_code == 0, output
    assert "retriever: tests.fakes:retrieve\n" in (result.stdout or "")
    _assert_no_secrets(output)


def test_non_2xx_names_the_case_and_hides_header_values(tmp_path, base_url, monkeypatch):
    monkeypatch.setenv("GATE_TOKEN", "sekret-header-value")
    dataset = tmp_path / "cases.jsonl"
    _write_cases(dataset)
    result = _run(
        [
            "run",
            "--dataset",
            str(dataset),
            "--retriever-url",
            f"{base_url}/reflect?token=sekret-query-value",
            "--header",
            "Authorization: Bearer ${GATE_TOKEN}",
        ]
    )
    output = _output(result)
    assert result.exit_code == 2
    assert "case q1:" in output
    assert f"HTTP 500 from {base_url}/reflect" in output
    _assert_no_secrets(output)


def test_malformed_and_unexpected_shapes_name_the_case(tmp_path, base_url):
    dataset = tmp_path / "cases.jsonl"
    _write_cases(dataset)

    malformed = _run(
        [
            "run",
            "--dataset",
            str(dataset),
            "--retriever-url",
            f"{base_url}/malformed?token=sekret-query-value",
        ]
    )
    malformed_output = _output(malformed)
    assert malformed.exit_code == 2
    assert "case q1:" in malformed_output
    assert "is not JSON" in malformed_output
    _assert_no_secrets(malformed_output)

    missing = _run(
        [
            "run",
            "--dataset",
            str(dataset),
            "--retriever-url",
            f"{base_url}/object",
        ]
    )
    missing_output = _output(missing)
    assert missing.exit_code == 2
    assert "case q1:" in missing_output
    assert "no results, documents, or hits list" in missing_output

    bad_field = _run(
        [
            "run",
            "--dataset",
            str(dataset),
            "--retriever-url",
            f"{base_url}/bad-results",
        ]
    )
    bad_output = _output(bad_field)
    assert bad_field.exit_code == 2
    assert "case q1:" in bad_output
    assert "field 'results' is not a list" in bad_output

    bad_id = _run(
        [
            "run",
            "--dataset",
            str(dataset),
            "--retriever-url",
            f"{base_url}/bad-id",
        ]
    )
    id_output = _output(bad_id)
    assert bad_id.exit_code == 2
    assert "case q1:" in id_output
    assert "result[0]" in id_output
    assert "no string field 'id'" in id_output

    custom_missing = _run(
        [
            "run",
            "--dataset",
            str(dataset),
            "--retriever-url",
            f"{base_url}/custom",
            "--results-key",
            "missing",
        ]
    )
    custom_output = _output(custom_missing)
    assert custom_missing.exit_code == 2
    assert "case q1:" in custom_output
    assert "missing 'missing'" in custom_output


def test_timeout_names_the_case(tmp_path, base_url):
    dataset = tmp_path / "cases.jsonl"
    _write_cases(dataset)
    result = _run(
        [
            "run",
            "--dataset",
            str(dataset),
            "--retriever-url",
            f"{base_url}/slow",
            "--timeout",
            "0.4",
        ]
    )
    output = _output(result)
    assert result.exit_code == 2, output
    assert f"case q1: request to {base_url}/slow timed out after 0.4s" in output


def test_redirect_is_not_followed(tmp_path, base_url):
    dataset = tmp_path / "cases.jsonl"
    _write_cases(dataset)
    result = _run(
        [
            "run",
            "--dataset",
            str(dataset),
            "--retriever-url",
            f"{base_url}/redirect",
        ]
    )
    output = _output(result)
    assert result.exit_code == 2, output
    assert f"case q1: HTTP 302 from {base_url}/redirect" in output


def test_unreachable_host_names_the_case(tmp_path):
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    dataset = tmp_path / "cases.jsonl"
    _write_cases(dataset)
    result = _run(
        [
            "run",
            "--dataset",
            str(dataset),
            "--retriever-url",
            f"http://127.0.0.1:{port}/search",
            "--timeout",
            "1",
        ]
    )
    output = _output(result)
    assert result.exit_code == 2, output
    assert f"case q1: could not reach http://127.0.0.1:{port}/search" in output


def test_header_errors_do_not_print_values(tmp_path, monkeypatch):
    monkeypatch.delenv("GATE_TOKEN_MISSING", raising=False)
    dataset = tmp_path / "cases.jsonl"
    _write_cases(dataset)
    missing = _run(
        [
            "run",
            "--dataset",
            str(dataset),
            "--retriever-url",
            "http://127.0.0.1/search",
            "--header",
            "Authorization: Bearer literal-secret ${GATE_TOKEN_MISSING}",
        ]
    )
    missing_output = _output(missing)
    assert missing.exit_code == 2
    assert (
        "header Authorization references unset environment variable GATE_TOKEN_MISSING"
        in missing_output
    )
    _assert_no_secrets(missing_output)

    broken = _run(
        [
            "run",
            "--dataset",
            str(dataset),
            "--retriever-url",
            "http://127.0.0.1/search",
            "--header",
            "Bearer sekret-header-value",
        ]
    )
    broken_output = _output(broken)
    assert broken.exit_code == 2
    assert "Name: Value" in broken_output
    _assert_no_secrets(broken_output)

    newline = _run(
        [
            "run",
            "--dataset",
            str(dataset),
            "--retriever-url",
            "http://127.0.0.1/search",
            "--header",
            "X-Token: one\ntwo",
        ]
    )
    newline_output = _output(newline)
    assert newline.exit_code == 2
    assert "header X-Token contains a newline" in newline_output
    assert "one\ntwo" not in newline_output


def test_flags_reject_a_bad_url_method_timeout_and_both_sources(tmp_path):
    dataset = tmp_path / "cases.jsonl"
    _write_cases(dataset)
    secret_url = "ftp://user:sekret-header-value@example.com/search?token=sekret-query-value"
    bad_url = _run(
        ["run", "--dataset", str(dataset), "--retriever-url", secret_url]
    )
    bad_url_output = _output(bad_url)
    assert bad_url.exit_code == 2
    assert "retriever URL must be http:// or https://" in bad_url_output
    _assert_no_secrets(bad_url_output)

    bad_method = _run(
        [
            "run",
            "--dataset",
            str(dataset),
            "--retriever-url",
            "http://127.0.0.1/search",
            "--method",
            "put",
        ]
    )
    assert bad_method.exit_code == 2
    assert "method must be get or post" in _output(bad_method)

    bad_timeout = _run(
        [
            "run",
            "--dataset",
            str(dataset),
            "--retriever-url",
            "http://127.0.0.1/search",
            "--timeout",
            "0",
        ]
    )
    assert bad_timeout.exit_code == 2
    assert "timeout must be a number greater than 0" in _output(bad_timeout)

    bad_concurrency = _run(
        [
            "run",
            "--dataset",
            str(dataset),
            "--retriever-url",
            "http://127.0.0.1/search",
            "--concurrency",
            "0",
        ]
    )
    assert bad_concurrency.exit_code == 2
    assert "concurrency must be an integer >= 1" in _output(bad_concurrency)

    both = _run(
        [
            "run",
            "--dataset",
            str(dataset),
            "--retriever",
            "tests.fakes:retrieve",
            "--retriever-url",
            "http://127.0.0.1/search",
        ]
    )
    assert both.exit_code == 2
    assert "not both" in _output(both)

    orphan = _run(
        [
            "run",
            "--dataset",
            str(dataset),
            "--retriever",
            "tests.fakes:retrieve",
            "--timeout",
            "1",
        ]
    )
    assert orphan.exit_code == 2
    assert "--timeout requires --retriever-url" in _output(orphan)


def test_concurrency_stays_at_the_limit(base_url):
    fn = build_http_retriever(f"{base_url}/overlap", timeout=5)
    cases = [
        Case.model_validate({"id": f"q{index}", "query": "cats", "relevant": ["d1"]})
        for index in range(3)
    ]
    report = build_report(
        cases,
        fn,
        1,
        dataset="cases.jsonl",
        retriever_name=f"{base_url}/overlap",
        thresholds={},
        concurrency=2,
    )
    assert _state.max_inflight == 2
    assert report["case_count"] == 3
    assert report["metrics"]["hit_rate"] == 1


def test_example_fastapi_service_wraps_bm25():
    pytest.importorskip("fastapi")
    from examples.fastapi_service import SearchRequest, search_get, search_post

    payload = search_get("reset password", 5)
    assert payload["results"][0]["id"] == "d01"
    posted = search_post(SearchRequest(query="reset password", k=5))
    assert posted == payload
