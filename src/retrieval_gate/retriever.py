"""Import a retriever and normalize its results."""

import importlib
import inspect
import sys
from collections.abc import Callable
from pathlib import Path

from retrieval_gate.errors import ConfigError


def load_retriever(spec: str) -> Callable:
    """Import ``module.submodule:function``.

    The current working directory is inserted on ``sys.path`` so a local
    module resolves the same way it would under pytest.
    """

    if spec.count(":") != 1:
        raise ConfigError(f"retriever {spec!r} must look like 'package.module:function'")
    module_name, func_name = spec.split(":")
    if not module_name or not func_name:
        raise ConfigError(f"retriever {spec!r} must look like 'package.module:function'")

    cwd = str(Path.cwd())
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        raise ConfigError(
            f"could not import {module_name!r} ({type(exc).__name__}: {exc})"
        ) from exc
    try:
        fn = getattr(module, func_name)
    except AttributeError as exc:
        raise ConfigError(f"{module_name!r} has no attribute {func_name!r}") from exc
    if not callable(fn):
        raise ConfigError(f"{spec} is not callable")
    return fn


async def call_retriever(fn: Callable, query: str, k: int, case_id: str) -> list[str]:
    """Call ``fn(query, k)`` and return document ids.

    Sync and async callables are both accepted. Each item may be a string
    id, a dict with an ``id`` field, or an object with an ``id`` attribute.
    """

    try:
        value = fn(query, k)
        if inspect.isawaitable(value):
            value = await value
    except ConfigError:
        raise
    except Exception as exc:
        raise ConfigError(
            f"case {case_id}: retriever raised {type(exc).__name__}: {exc}"
        ) from exc
    return coerce_ids(value, case_id=case_id)


def coerce_ids(value: object, case_id: str) -> list[str]:
    if not isinstance(value, list):
        raise ConfigError(
            f"case {case_id}: retriever must return a list, got {type(value).__name__}"
        )
    ids: list[str] = []
    for index, item in enumerate(value):
        doc_id = _doc_id(item)
        if doc_id is None:
            raise ConfigError(f"case {case_id}: result[{index}] has no string id")
        ids.append(doc_id)
    return ids


def _doc_id(item: object) -> str | None:
    if isinstance(item, str):
        text = item.strip()
        return text or None
    raw = item.get("id") if isinstance(item, dict) else getattr(item, "id", None)
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return None
