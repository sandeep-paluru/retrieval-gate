"""Wrap LangChain and LlamaIndex retrievers so they return document ids.

Neither library is imported unless a retriever needs it. Install the extra
named in the error when the import fails:

    pip install "retrieval-gate[langchain]"
    pip install "retrieval-gate[llamaindex]"
"""

import asyncio
import importlib
import inspect
from collections.abc import Callable

from retrieval_gate.errors import ConfigError, RetrieverResponseError

ADAPTERS = ("auto", "langchain", "llamaindex")

_LANGCHAIN_INSTALL = 'pip install "retrieval-gate[langchain]"'
_LLAMA_INSTALL = 'pip install "retrieval-gate[llamaindex]"'
_LLAMA_NODE_FIELDS = {
    "node_id": "node_id",
    "id_": "id_",
    "ref_doc_id": "ref_doc_id",
    "id": "node_id",
}


def normalize_adapter(value: str) -> str:
    """Return ``auto``, ``langchain``, or ``llamaindex``."""

    name = value.strip().lower()
    if name not in ADAPTERS:
        raise ConfigError("adapter must be auto, langchain, or llamaindex")
    return name


def checked_id_field(value: str | None) -> str | None:
    """Return a stripped id field, or None when the caller did not set one."""

    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ConfigError("id_field must be a non-empty string")
    return value.strip()


def adapt_retriever(
    obj: object,
    *,
    adapter: str = "auto",
    id_field: str | None = None,
    label: str,
) -> Callable:
    """Return a ``(query, k)`` callable for ``obj``.

    ``auto`` leaves a plain callable alone and wraps a LangChain retriever
    or vector store, or a LlamaIndex retriever or index. ``id_field`` selects
    a metadata key. LangChain reads ``Document.id`` when it is omitted.
    LlamaIndex reads ``node_id`` when it is omitted, and also accepts
    ``ref_doc_id``.
    """

    kind = normalize_adapter(adapter)
    field = checked_id_field(id_field)
    if kind == "auto":
        detected = _library(obj)
        if detected is None:
            return _as_callable(obj, field, label)
        kind = detected
    if kind == "langchain":
        return _adapt_langchain(obj, field, label)
    return _adapt_llamaindex(obj, field, label)


def _as_callable(obj: object, id_field: str | None, label: str) -> Callable:
    if id_field is not None:
        raise ConfigError(
            "id_field applies to an HTTP retriever or a LangChain or LlamaIndex retriever"
        )
    if not callable(obj):
        raise ConfigError(f"{label} is not callable")
    return obj


def _library(obj: object) -> str | None:
    for cls in type(obj).mro():
        root = (cls.__module__ or "").split(".", 1)[0]
        if root == "langchain" or root.startswith("langchain_"):
            return "langchain"
        if root == "llama_index":
            return "llamaindex"
    return None


def _import_attr(module: str, attr: str, extra: str, install: str):
    try:
        imported = importlib.import_module(module)
    except ImportError as exc:
        raise ConfigError(f"{extra} support needs the extra: {install}") from exc
    try:
        return getattr(imported, attr)
    except AttributeError as exc:
        raise ConfigError(f"{extra} support needs the extra: {install}") from exc


def _adapt_langchain(obj: object, id_field: str | None, label: str) -> Callable:
    base_retriever = _import_attr(
        "langchain_core.retrievers",
        "BaseRetriever",
        "LangChain",
        _LANGCHAIN_INSTALL,
    )
    vector_store = _import_attr(
        "langchain_core.vectorstores",
        "VectorStore",
        "LangChain",
        _LANGCHAIN_INSTALL,
    )
    if isinstance(obj, base_retriever):
        return _langchain_retriever(obj, id_field)
    if isinstance(obj, vector_store):
        return _langchain_store(obj, id_field)
    raise ConfigError(f"{label} is not a LangChain retriever or vector store")


def _adapt_llamaindex(obj: object, id_field: str | None, label: str) -> Callable:
    base_retriever = _import_attr(
        "llama_index.core.base.base_retriever",
        "BaseRetriever",
        "LlamaIndex",
        _LLAMA_INSTALL,
    )
    base_index = _import_attr(
        "llama_index.core.indices.base",
        "BaseIndex",
        "LlamaIndex",
        _LLAMA_INSTALL,
    )
    if isinstance(obj, base_retriever):
        return _llamaindex_retriever(obj, id_field)
    if isinstance(obj, base_index):
        return _llamaindex_index(obj, id_field)
    raise ConfigError(f"{label} is not a LlamaIndex retriever or index")


def _langchain_store(store: object, id_field: str | None) -> Callable:
    async def retrieve(query: str, k: int) -> list[str]:
        docs = await _search_store(store, query, k)
        return _ids(docs, k, id_field, _langchain_id, _missing_langchain)

    return retrieve


def _langchain_retriever(retriever: object, id_field: str | None) -> Callable:
    lock = asyncio.Lock()

    async def retrieve(query: str, k: int) -> list[str]:
        async with lock:
            docs = await _invoke_langchain(retriever, query, k)
        return _ids(docs, k, id_field, _langchain_id, _missing_langchain)

    return retrieve


async def _search_store(store: object, query: str, k: int) -> list:
    search = getattr(store, "asimilarity_search", None)
    if inspect.iscoroutinefunction(search):
        found = await search(query, k=k)
    else:
        fallback = getattr(store, "similarity_search", None)
        if not callable(fallback):
            raise RetrieverResponseError("LangChain vector store has no similarity_search method")
        found = await asyncio.to_thread(fallback, query, k=k)
    return _as_list(found)


async def _invoke_langchain(retriever: object, query: str, k: int) -> list:
    bound = _with_search_k(retriever, k)
    mutated = bound is retriever and isinstance(getattr(retriever, "search_kwargs", None), dict)
    previous: dict | None = None
    if mutated:
        search_kwargs = retriever.search_kwargs
        previous = dict(search_kwargs)
        search_kwargs["k"] = k
    try:
        ainvoke = getattr(bound, "ainvoke", None)
        if inspect.iscoroutinefunction(ainvoke):
            found = await ainvoke(query)
        else:
            invoke = getattr(bound, "invoke", None)
            if not callable(invoke):
                raise RetrieverResponseError("LangChain retriever has no invoke method")
            found = await asyncio.to_thread(invoke, query)
    finally:
        if mutated and previous is not None:
            search_kwargs = retriever.search_kwargs
            search_kwargs.clear()
            search_kwargs.update(previous)
    return _as_list(found)


def _with_search_k(retriever: object, k: int) -> object:
    search_kwargs = getattr(retriever, "search_kwargs", None)
    if not isinstance(search_kwargs, dict):
        return retriever
    model_copy = getattr(retriever, "model_copy", None)
    if not callable(model_copy):
        return retriever
    try:
        return model_copy(update={"search_kwargs": {**search_kwargs, "k": k}})
    except (TypeError, ValueError):
        return retriever


def _llamaindex_retriever(retriever: object, id_field: str | None) -> Callable:
    lock = asyncio.Lock()

    async def retrieve(query: str, k: int) -> list[str]:
        async with lock:
            nodes = await _retrieve_nodes(retriever, query, k)
        return _ids(nodes, k, id_field, _llamaindex_id, _missing_llamaindex)

    return retrieve


def _llamaindex_index(index: object, id_field: str | None) -> Callable:
    async def retrieve(query: str, k: int) -> list[str]:
        retriever = _index_retriever(index, k)
        nodes = await _retrieve_nodes(retriever, query, k)
        return _ids(nodes, k, id_field, _llamaindex_id, _missing_llamaindex)

    return retrieve


def _index_retriever(index: object, k: int) -> object:
    as_retriever = getattr(index, "as_retriever", None)
    if not callable(as_retriever):
        raise RetrieverResponseError("LlamaIndex index has no as_retriever method")
    try:
        return as_retriever(similarity_top_k=k)
    except TypeError:
        return as_retriever()


async def _retrieve_nodes(retriever: object, query: str, k: int) -> list:
    previous = _set_top_k(retriever, k)
    try:
        aretrieve = getattr(retriever, "aretrieve", None)
        if inspect.iscoroutinefunction(aretrieve):
            found = await aretrieve(query)
        else:
            retrieve = getattr(retriever, "retrieve", None)
            if not callable(retrieve):
                raise RetrieverResponseError("LlamaIndex retriever has no retrieve method")
            found = await asyncio.to_thread(retrieve, query)
    finally:
        if previous is not None:
            retriever.similarity_top_k = previous
    return _as_list(found)


def _set_top_k(retriever: object, k: int) -> int | None:
    current = getattr(retriever, "similarity_top_k", None)
    if isinstance(current, bool) or not isinstance(current, int):
        return None
    try:
        retriever.similarity_top_k = k
    except (AttributeError, TypeError):
        return None
    return current


def _as_list(value: object) -> list:
    if isinstance(value, list):
        return value
    raise RetrieverResponseError(
        f"retriever must return a list, got {type(value).__name__}"
    )


def _ids(
    items: list,
    k: int,
    id_field: str | None,
    read: Callable[[object, str | None], str | None],
    missing: Callable[[int, str | None], str],
) -> list[str]:
    chosen = items[:k] if k >= 0 else []
    ids: list[str] = []
    for index, item in enumerate(chosen):
        doc_id = read(item, id_field)
        if doc_id is None:
            raise RetrieverResponseError(missing(index, id_field))
        ids.append(doc_id)
    return ids


def _text(value: object) -> str | None:
    if isinstance(value, str):
        text = value.strip()
        return text or None
    return None


def _metadata(obj: object) -> dict:
    raw = getattr(obj, "metadata", None)
    if isinstance(raw, dict):
        return raw
    return {}


def _langchain_id(doc: object, id_field: str | None) -> str | None:
    if id_field is None:
        return _text(getattr(doc, "id", None))
    return _text(_metadata(doc).get(id_field))


def _missing_langchain(index: int, id_field: str | None) -> str:
    if id_field is None:
        return f"result[{index}] has no string id"
    return f"result[{index}] has no string metadata field {id_field!r}"


def _llamaindex_node(item: object) -> object:
    node = getattr(item, "node", None)
    if node is None:
        return item
    return node


def _llamaindex_id(item: object, id_field: str | None) -> str | None:
    node = _llamaindex_node(item)
    field = "node_id" if id_field is None else id_field
    attr = _LLAMA_NODE_FIELDS.get(field)
    if attr is not None:
        return _text(getattr(node, attr, None))
    return _text(_metadata(node).get(field))


def _missing_llamaindex(index: int, id_field: str | None) -> str:
    field = "node_id" if id_field is None else id_field
    if field in _LLAMA_NODE_FIELDS:
        shown = "node_id" if field == "id" else field
        return f"result[{index}] has no string {shown}"
    return f"result[{index}] has no string metadata field {field!r}"
