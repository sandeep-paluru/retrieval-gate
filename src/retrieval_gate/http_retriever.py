"""Call a retriever that is an HTTP service.

This module is the only place that opens a connection. An import-path
retriever does not use it.
"""

import asyncio
import json
import math
import os
import re
from collections.abc import Callable
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, OpenerDirector, Request, build_opener

from retrieval_gate.errors import ConfigError, RetrieverResponseError

DEFAULT_TIMEOUT = 10.0
DEFAULT_CONCURRENCY = 4
DEFAULT_ID_FIELD = "id"

_RESULT_KEYS = ("results", "documents", "hits")
_HEADER_NAME = re.compile(r"^[!#$%&'*+\-.0-9A-Za-z^_`|~]+$")
_ENV_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def public_url(url: str) -> str:
    """Return ``url`` with userinfo, query, and fragment removed.

    Those parts are where a token usually sits. The request still sends
    them. Reports and errors use this form.
    """

    parts = urlsplit(url.strip())
    host = parts.hostname or ""
    if ":" in host:
        host = f"[{host}]"
    netloc = host
    if parts.port is not None:
        netloc = f"{netloc}:{parts.port}"
    return urlunsplit((parts.scheme, netloc, parts.path, "", ""))


def check_url(url: str) -> str:
    """Require an http or https URL. The message does not echo the URL."""

    text = url.strip()
    parts = urlsplit(text)
    if parts.scheme not in {"http", "https"} or not parts.netloc:
        raise ConfigError("retriever URL must be http:// or https://")
    return text


def checked_timeout(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError("timeout must be a number greater than 0")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ConfigError("timeout must be a number greater than 0")
    return number


def checked_concurrency(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ConfigError("concurrency must be an integer >= 1")
    return value


def normalize_method(value: str) -> str:
    method = value.strip().upper()
    if method not in {"GET", "POST"}:
        raise ConfigError("method must be get or post")
    return method


def parse_headers(items: list[str]) -> list[tuple[str, str]]:
    """Parse ``Name: Value`` headers and expand ``${ENV_VAR}`` in values.

    The returned values are what is sent. Callers must not print them.
    A missing variable names the header and the variable, not the value.
    """

    parsed: list[tuple[str, str]] = []
    for item in items:
        parsed.append(_one_header(item))
    return parsed


def build_http_retriever(
    url: str,
    *,
    method: str = "POST",
    headers: list[str] | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    results_key: str | None = None,
    id_field: str = DEFAULT_ID_FIELD,
) -> Callable:
    """Return an async ``(query, k)`` callable that POSTs or GETs ``url``.

    POST sends JSON ``{"query": ..., "k": ...}``. GET sends those two as
    query parameters and keeps any other parameters already on the URL.
    ``headers`` are raw ``Name: Value`` strings.
    """

    target = check_url(url)
    verb = normalize_method(method)
    timeout_s = checked_timeout(timeout)
    field = _checked_field(id_field, "id_field")
    key = None if results_key is None else _checked_field(results_key, "results_key")
    prepared = _with_content_type(parse_headers(headers or []), verb)

    async def retrieve(query: str, k: int) -> list[str]:
        return await asyncio.to_thread(
            _perform,
            target,
            verb,
            tuple(prepared),
            timeout_s,
            key,
            field,
            query,
            k,
        )

    return retrieve


def parse_response(
    body: bytes,
    *,
    source: str,
    results_key: str | None,
    id_field: str,
) -> list[str]:
    """Turn a response body into document ids.

    A JSON list is the result list. An object is read from ``results_key``
    when that is set, otherwise from the first of ``results``,
    ``documents``, and ``hits``. List items may be id strings or objects.
    """

    try:
        payload = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RetrieverResponseError(f"response from {source} is not JSON") from exc
    items = _result_items(payload, source, results_key)
    return _ids(items, id_field)


class _RejectRedirects(HTTPRedirectHandler):
    """Refuse redirects so a header is not forwarded to another URL."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise HTTPError(req.full_url, code, msg, headers, fp)


def _perform(
    url: str,
    method: str,
    headers: tuple[tuple[str, str], ...],
    timeout: float,
    results_key: str | None,
    id_field: str,
    query: str,
    k: int,
) -> list[str]:
    shown = public_url(url)
    request = _build_request(url, method, headers, query, k)
    opener = _opener()
    try:
        with opener.open(request, timeout=timeout) as response:
            status = getattr(response, "status", None)
            if status is None:
                status = response.getcode()
            body = response.read()
    except HTTPError as exc:
        exc.close()
        raise RetrieverResponseError(f"HTTP {exc.code} from {shown}") from exc
    except URLError as exc:
        if isinstance(exc.reason, TimeoutError):
            raise RetrieverResponseError(_timeout_message(shown, timeout)) from exc
        raise RetrieverResponseError(f"could not reach {shown}") from exc
    except TimeoutError as exc:
        raise RetrieverResponseError(_timeout_message(shown, timeout)) from exc
    except OSError as exc:
        raise RetrieverResponseError(f"could not reach {shown}") from exc
    if status < 200 or status >= 300:
        raise RetrieverResponseError(f"HTTP {status} from {shown}")
    return parse_response(body, source=shown, results_key=results_key, id_field=id_field)


def _opener() -> OpenerDirector:
    return build_opener(_RejectRedirects)


def _build_request(
    url: str,
    method: str,
    headers: tuple[tuple[str, str], ...],
    query: str,
    k: int,
) -> Request:
    data = None
    target = _request_url(url)
    if method == "POST":
        data = json.dumps({"query": query, "k": k}).encode("utf-8")
    else:
        target = _get_url(target, query, k)
    request = Request(target, data=data, method=method)
    for name, value in headers:
        request.add_header(name, value)
    return request


def _request_url(url: str) -> str:
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, parts.query, ""))


def _get_url(url: str, query: str, k: int) -> str:
    parts = urlsplit(url)
    pairs = [
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if key not in {"query", "k"}
    ]
    pairs.append(("query", query))
    pairs.append(("k", str(k)))
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(pairs), ""))


def _with_content_type(headers: list[tuple[str, str]], method: str) -> list[tuple[str, str]]:
    if method != "POST":
        return headers
    if any(name.lower() == "content-type" for name, _ in headers):
        return headers
    return [*headers, ("Content-Type", "application/json")]


def _result_items(payload: object, source: str, results_key: str | None) -> list:
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        raise RetrieverResponseError(f"response from {source} must be a list or an object")
    if results_key is not None:
        if results_key not in payload:
            raise RetrieverResponseError(f"response from {source} is missing {results_key!r}")
        return _as_list(payload[results_key], source, results_key)
    for key in _RESULT_KEYS:
        if key in payload:
            return _as_list(payload[key], source, key)
    raise RetrieverResponseError(
        f"response from {source} has no results, documents, or hits list"
    )


def _as_list(value: object, source: str, key: str) -> list:
    if isinstance(value, list):
        return value
    raise RetrieverResponseError(f"response from {source} field {key!r} is not a list")


def _ids(items: list, id_field: str) -> list[str]:
    ids: list[str] = []
    for index, item in enumerate(items):
        doc_id = _item_id(item, id_field)
        if doc_id is None:
            if isinstance(item, dict):
                raise RetrieverResponseError(
                    f"result[{index}] has no string field {id_field!r}"
                )
            raise RetrieverResponseError(f"result[{index}] has no string id")
        ids.append(doc_id)
    return ids


def _item_id(item: object, id_field: str) -> str | None:
    if isinstance(item, str):
        text = item.strip()
        return text or None
    if isinstance(item, dict):
        raw = item.get(id_field)
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    return None


def _one_header(item: str) -> tuple[str, str]:
    if not isinstance(item, str) or ":" not in item:
        raise ConfigError("header must look like 'Name: Value'")
    name, value = item.split(":", 1)
    name = name.strip()
    if not _HEADER_NAME.fullmatch(name):
        raise ConfigError("header name must be an HTTP token")
    if value[:1] == " ":
        value = value[1:]
    if "\n" in value or "\r" in value:
        raise ConfigError(f"header {name} contains a newline")
    value = _expand_env(value, name)
    if "\n" in value or "\r" in value:
        raise ConfigError(f"header {name} contains a newline")
    try:
        value.encode("latin-1")
    except UnicodeEncodeError as exc:
        raise ConfigError(f"header {name} must be latin-1") from exc
    return name, value


def _expand_env(value: str, header_name: str) -> str:
    def replace(match: re.Match[str]) -> str:
        var = match.group(1)
        if var not in os.environ:
            raise ConfigError(
                f"header {header_name} references unset environment variable {var}"
            )
        return os.environ[var]

    return _ENV_VAR.sub(replace, value)


def _checked_field(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{label} must be a non-empty string")
    return value.strip()


def _timeout_message(source: str, timeout: float) -> str:
    return f"request to {source} timed out after {_format_timeout(timeout)}s"


def _format_timeout(timeout: float) -> str:
    return f"{timeout:.3f}".rstrip("0").rstrip(".")
