"""Framework adapters. LangChain and LlamaIndex are optional extras."""

import asyncio
import importlib

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import InMemoryVectorStore
from llama_index.core.base.base_retriever import BaseRetriever as LlamaRetriever
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.indices import VectorStoreIndex
from llama_index.core.schema import NodeRelationship, NodeWithScore, RelatedNodeInfo, TextNode
from typer.testing import CliRunner

from retrieval_gate.adapters import adapt_retriever
from retrieval_gate.cli import app
from retrieval_gate.errors import ConfigError, RetrieverResponseError
from retrieval_gate.models import Case
from retrieval_gate.report import build_report
from retrieval_gate.retriever import load_retriever
from tests import fakes


class ConstantEmbedding(Embeddings):
    """Same vector for every string, so rank is not the thing under test."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]


class _ListedDocs(BaseRetriever):
    def _get_relevant_documents(self, query: str, *, run_manager) -> list[Document]:
        return [
            Document(id="internal", page_content=query, metadata={"source": "d1"}),
        ]


class _NoId(BaseRetriever):
    def _get_relevant_documents(self, query: str, *, run_manager) -> list[Document]:
        return [Document(page_content=query, metadata={})]


class _NotAList(BaseRetriever):
    def _get_relevant_documents(self, query: str, *, run_manager):
        return "nope"


class _LlamaFixed(LlamaRetriever):
    def _retrieve(self, query_bundle) -> list[NodeWithScore]:
        return [
            NodeWithScore(node=TextNode(text="a", id_="d1"), score=1.0),
            NodeWithScore(node=TextNode(text="b", id_="d2"), score=1.0),
            NodeWithScore(node=TextNode(text="c", id_="d3"), score=1.0),
        ]


def _run(fn, query: str, k: int):
    return asyncio.run(fn(query, k))


def _output(result) -> str:
    return (result.stdout or "") + (result.stderr or "")


runner = CliRunner()


def _store(*docs: Document) -> InMemoryVectorStore:
    store = InMemoryVectorStore(embedding=ConstantEmbedding())
    store.add_documents(list(docs))
    return store


def _index(*nodes: TextNode) -> VectorStoreIndex:
    return VectorStoreIndex(list(nodes), embed_model=MockEmbedding(embed_dim=8))


def test_langchain_store_reads_document_id_and_passes_k():
    store = _store(
        Document(id="d1", page_content="cats", metadata={"source": "other"}),
        Document(id="d2", page_content="cats", metadata={"source": "also"}),
        Document(id="d3", page_content="cats", metadata={"source": "third"}),
    )
    seen: list[int] = []
    real = store.asimilarity_search

    async def spy(query: str, k: int = 4, **kwargs):
        seen.append(k)
        return await real(query, k=k, **kwargs)

    store.asimilarity_search = spy
    fn = adapt_retriever(store, label="app:store")
    ids = _run(fn, "cats", 2)
    assert seen == [2]
    assert len(ids) == 2
    assert set(ids) <= {"d1", "d2", "d3"}

    by_source = adapt_retriever(store, id_field="source", label="app:store")
    assert set(_run(by_source, "cats", 3)) == {"other", "also", "third"}


def test_langchain_retriever_passes_k_without_mutating_search_kwargs():
    store = _store(
        Document(id="d1", page_content="cats"),
        Document(id="d2", page_content="cats"),
        Document(id="d3", page_content="cats"),
    )
    async_calls: list[int] = []
    sync_calls: list[int] = []
    real_async = store.asimilarity_search
    real_sync = store.similarity_search

    async def async_spy(query: str, k: int = 4, **kwargs):
        async_calls.append(k)
        return await real_async(query, k=k, **kwargs)

    def sync_spy(query: str, k: int = 4, **kwargs):
        sync_calls.append(k)
        return real_sync(query, k=k, **kwargs)

    store.asimilarity_search = async_spy
    store.similarity_search = sync_spy
    retriever = store.as_retriever(search_kwargs={"k": 1})
    fn = adapt_retriever(retriever, adapter="langchain", label="app:retriever")
    ids = _run(fn, "cats", 3)
    assert len(ids) == 3
    assert set(ids) == {"d1", "d2", "d3"}
    assert async_calls == [3]
    assert sync_calls == []
    assert retriever.search_kwargs == {"k": 1}


def test_langchain_store_falls_back_to_sync_search_and_truncates():
    store = _store(Document(id="d1", page_content="cats"))
    seen: list[int] = []

    def sync_search(query: str, k: int = 4, **kwargs):
        seen.append(k)
        return [
            Document(id="d1", page_content="cats"),
            Document(id="d2", page_content="cats"),
            Document(id="d3", page_content="cats"),
        ]

    def not_async(*args, **kwargs):
        raise AssertionError("asimilarity_search was used")

    store.asimilarity_search = not_async
    store.similarity_search = sync_search
    fn = adapt_retriever(store, label="app:store")
    assert _run(fn, "cats", 1) == ["d1"]
    assert seen == [1]


def test_langchain_id_field_and_missing_ids():
    fn = adapt_retriever(_ListedDocs(), label="app:docs")
    assert _run(fn, "cats", 5) == ["internal"]
    by_source = adapt_retriever(_ListedDocs(), id_field="source", label="app:docs")
    assert _run(by_source, "cats", 5) == ["d1"]

    missing = adapt_retriever(_NoId(), label="app:docs")
    with pytest.raises(RetrieverResponseError, match=r"result\[0\] has no string id"):
        _run(missing, "cats", 1)
    missing_meta = adapt_retriever(_ListedDocs(), id_field="missing", label="app:docs")
    with pytest.raises(RetrieverResponseError, match=r"metadata field 'missing'"):
        _run(missing_meta, "cats", 1)

    case = Case.model_validate({"id": "q1", "query": "cats", "relevant": ["d1"]})
    with pytest.raises(ConfigError, match=r"case q1: result\[0\] has no string id"):
        build_report(
            [case],
            missing,
            1,
            dataset="cases.jsonl",
            retriever_name="app:docs",
            thresholds={},
        )


def test_langchain_non_list_names_the_case():
    case = Case.model_validate({"id": "q1", "query": "cats", "relevant": ["d1"]})
    fn = adapt_retriever(_NotAList(), label="app:docs")
    with pytest.raises(ConfigError, match="case q1: retriever must return a list"):
        build_report(
            [case],
            fn,
            1,
            dataset="cases.jsonl",
            retriever_name="app:docs",
            thresholds={},
        )


def test_llamaindex_index_uses_node_id_and_top_k():
    index = _index(
        TextNode(text="one", id_="d1"),
        TextNode(text="two", id_="d2"),
        TextNode(text="three", id_="d3"),
        TextNode(text="four", id_="d4"),
    )
    fn = adapt_retriever(index, label="app:index")
    ids = _run(fn, "one", 4)
    assert set(ids) == {"d1", "d2", "d3", "d4"}
    assert len(_run(fn, "one", 1)) == 1


def test_llamaindex_retriever_sets_top_k_for_the_call_and_restores_it():
    index = _index(*(TextNode(text=f"doc {i}", id_=f"d{i}") for i in range(1, 5)))
    retriever = index.as_retriever(similarity_top_k=2)
    seen: list[tuple[str, int]] = []
    real = retriever.aretrieve

    async def spy(query: str):
        seen.append(("aretrieve", retriever.similarity_top_k))
        return await real(query)

    retriever.aretrieve = spy
    fn = adapt_retriever(retriever, adapter="llamaindex", label="app:retriever")
    ids = _run(fn, "doc", 4)
    assert len(ids) == 4
    assert set(ids) == {"d1", "d2", "d3", "d4"}
    assert seen == [("aretrieve", 4)]
    assert retriever.similarity_top_k == 2


def test_llamaindex_id_fields_and_sync_fallback():
    node = TextNode(
        text="alpha",
        id_="chunk-1",
        metadata={"source": "doc-a"},
        relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="doc-a")},
    )
    index = _index(node)
    assert _run(adapt_retriever(index, label="app:index"), "alpha", 1) == ["chunk-1"]
    assert _run(adapt_retriever(index, id_field="ref_doc_id", label="app:index"), "alpha", 1) == [
        "doc-a"
    ]
    assert _run(adapt_retriever(index, id_field="source", label="app:index"), "alpha", 1) == [
        "doc-a"
    ]

    bare = _index(TextNode(text="alpha", id_="chunk-1"))
    missing_ref = adapt_retriever(bare, id_field="ref_doc_id", label="app:index")
    with pytest.raises(RetrieverResponseError, match=r"result\[0\] has no string ref_doc_id"):
        _run(missing_ref, "alpha", 1)
    missing_meta = adapt_retriever(bare, id_field="source", label="app:index")
    with pytest.raises(RetrieverResponseError, match=r"metadata field 'source'"):
        _run(missing_meta, "alpha", 1)

    retriever = index.as_retriever(similarity_top_k=2)
    sync_calls: list[str] = []
    real = retriever.retrieve

    def sync_retrieve(query: str):
        sync_calls.append(query)
        return real(query)

    def not_async(*args, **kwargs):
        raise AssertionError("aretrieve was used")

    retriever.aretrieve = not_async
    retriever.retrieve = sync_retrieve
    fn = adapt_retriever(retriever, label="app:retriever")
    assert _run(fn, "alpha", 1) == ["chunk-1"]
    assert sync_calls == ["alpha"]


def test_llamaindex_truncates_when_top_k_is_not_supported():
    fn = adapt_retriever(_LlamaFixed(), label="app:retriever")
    assert _run(fn, "q", 1) == ["d1"]
    case = Case.model_validate({"id": "q9", "query": "q", "relevant": ["d1"]})
    report = build_report(
        [case],
        fn,
        2,
        dataset="cases.jsonl",
        retriever_name="app:retriever",
        thresholds={},
    )
    assert report["results"][0]["retrieved"] == ["d1", "d2"]


def test_plain_callable_is_unchanged_and_does_not_import_frameworks(monkeypatch):
    real = importlib.import_module

    def blocked(name, package=None):
        root = name.split(".", 1)[0]
        if root in {"langchain_core", "llama_index"}:
            raise AssertionError(name)
        return real(name, package)

    monkeypatch.setattr("retrieval_gate.adapters.importlib.import_module", blocked)
    fn = load_retriever("tests.fakes:retrieve")
    assert fn is fakes.retrieve
    with pytest.raises(ConfigError, match="is not callable"):
        adapt_retriever(object(), label="app:obj")
    with pytest.raises(ConfigError, match="id_field applies"):
        load_retriever("tests.fakes:retrieve", id_field="source")


def test_missing_framework_extra_names_the_install(monkeypatch):
    real = importlib.import_module

    def missing(name, package=None):
        root = name.split(".", 1)[0]
        if root in {"langchain_core", "llama_index"}:
            raise ModuleNotFoundError(name)
        return real(name, package)

    monkeypatch.setattr("retrieval_gate.adapters.importlib.import_module", missing)

    class Store:
        pass

    Store.__module__ = "langchain_core.vectorstores.in_memory"
    with pytest.raises(ConfigError, match=r"retrieval-gate\[langchain\]"):
        adapt_retriever(Store(), label="app:store")

    class Index:
        pass

    Index.__module__ = "llama_index.core.indices.vector_store.base"
    with pytest.raises(ConfigError, match=r"retrieval-gate\[llamaindex\]"):
        adapt_retriever(Index(), adapter="auto", label="app:index")
    with pytest.raises(ConfigError, match=r"retrieval-gate\[langchain\]"):
        adapt_retriever(object(), adapter="langchain", label="app:store")
    with pytest.raises(ConfigError, match=r"retrieval-gate\[llamaindex\]"):
        adapt_retriever(object(), adapter="llamaindex", label="app:index")


def test_explicit_adapter_rejects_the_wrong_object():
    with pytest.raises(ConfigError, match="not a LangChain retriever or vector store"):
        load_retriever("tests.fakes:retrieve", adapter="langchain")
    with pytest.raises(ConfigError, match="not a LlamaIndex retriever or index"):
        load_retriever("tests.fakes:retrieve", adapter="llamaindex")
    with pytest.raises(ConfigError, match="adapter must be auto, langchain, or llamaindex"):
        load_retriever("tests.fakes:retrieve", adapter="bm25")


def test_load_retriever_detects_framework_attributes():
    fakes.lc_store = _store(Document(id="d1", page_content="cats", metadata={"source": "s1"}))
    fakes.li_index = _index(TextNode(text="cats", id_="d9", metadata={"source": "s9"}))
    try:
        lc = load_retriever("tests.fakes:lc_store")
        assert _run(lc, "cats", 1) == ["d1"]
        li = load_retriever("tests.fakes:li_index", id_field="source")
        assert _run(li, "cats", 1) == ["s9"]
    finally:
        del fakes.lc_store
        del fakes.li_index


def _write_case(path, relevant: str) -> None:
    path.write_text(
        f'{{"id": "q1", "query": "cats", "relevant": ["{relevant}"]}}\n',
        encoding="utf-8",
    )


def test_cli_scores_langchain_and_llamaindex_attributes(tmp_path):
    dataset = tmp_path / "cases.jsonl"
    _write_case(dataset, "d1")
    fakes.lc_store = _store(Document(id="d1", page_content="cats", metadata={"source": "src"}))
    fakes.li_index = _index(TextNode(text="cats", id_="d1", metadata={"source": "src"}))
    try:
        langchain = runner.invoke(
            app,
            [
                "run",
                "--dataset",
                str(dataset),
                "--retriever",
                "tests.fakes:lc_store",
                "--adapter",
                "langchain",
                "--k",
                "1",
            ],
        )
        assert langchain.exit_code == 0, _output(langchain)
        assert "hit_rate   1.0000" in (langchain.stdout or "")

        llama = runner.invoke(
            app,
            [
                "run",
                "--dataset",
                str(dataset),
                "--retriever",
                "tests.fakes:li_index",
                "--k",
                "1",
            ],
        )
        assert llama.exit_code == 0, _output(llama)
        assert "hit_rate   1.0000" in (llama.stdout or "")
    finally:
        del fakes.lc_store
        del fakes.li_index


def test_cli_id_field_and_adapter_errors(tmp_path):
    dataset = tmp_path / "cases.jsonl"
    _write_case(dataset, "src")
    fakes.lc_store = _store(
        Document(id="internal", page_content="cats", metadata={"source": "src"})
    )
    fakes.bare = _NoId()
    try:
        sourced = runner.invoke(
            app,
            [
                "run",
                "--dataset",
                str(dataset),
                "--retriever",
                "tests.fakes:lc_store",
                "--id-field",
                "source",
                "--k",
                "1",
                "--fail-under",
                "recall=1",
            ],
        )
        assert sourced.exit_code == 0, _output(sourced)

        missing = runner.invoke(
            app,
            [
                "run",
                "--dataset",
                str(dataset),
                "--retriever",
                "tests.fakes:bare",
                "--adapter",
                "langchain",
            ],
        )
        assert missing.exit_code == 2
        assert "case q1" in _output(missing)
        assert "has no string id" in _output(missing)
    finally:
        del fakes.lc_store
        del fakes.bare

    plain = runner.invoke(
        app,
        [
            "run",
            "--dataset",
            str(dataset),
            "--retriever",
            "tests.fakes:retrieve",
            "--id-field",
            "source",
        ],
    )
    assert plain.exit_code == 2
    assert "id_field applies" in _output(plain)

    bad_adapter = runner.invoke(
        app,
        [
            "run",
            "--dataset",
            str(dataset),
            "--retriever",
            "tests.fakes:retrieve",
            "--adapter",
            "bm25",
        ],
    )
    assert bad_adapter.exit_code == 2
    assert "adapter must be auto, langchain, or llamaindex" in _output(bad_adapter)

    with_url = runner.invoke(
        app,
        [
            "run",
            "--dataset",
            str(dataset),
            "--retriever-url",
            "http://127.0.0.1/search",
            "--adapter",
            "langchain",
        ],
    )
    assert with_url.exit_code == 2
    assert "adapter requires a retriever import path" in _output(with_url)


def test_config_adapter_and_id_field(tmp_path):
    dataset = tmp_path / "cases.jsonl"
    _write_case(dataset, "src")
    fakes.lc_store = _store(
        Document(id="internal", page_content="cats", metadata={"source": "src"})
    )
    config = tmp_path / "gate.yml"
    config.write_text(
        "dataset: cases.jsonl\n"
        "retriever: tests.fakes:lc_store\n"
        "adapter: langchain\n"
        "id_field: source\n"
        "k: 1\n"
        "fail_under:\n"
        "  recall: 1\n",
        encoding="utf-8",
    )
    try:
        result = runner.invoke(app, ["run", "--config", str(config)])
        assert result.exit_code == 0, _output(result)
        assert "retriever: tests.fakes:lc_store" in (result.stdout or "")
    finally:
        del fakes.lc_store

    config.write_text(
        "dataset: cases.jsonl\nretriever: tests.fakes:retrieve\nadapter: nope\n",
        encoding="utf-8",
    )
    bad = runner.invoke(app, ["run", "--config", str(config)])
    assert bad.exit_code == 2
    assert "adapter must be auto, langchain, or llamaindex" in _output(bad)

    config.write_text(
        "dataset: cases.jsonl\nretriever: tests.fakes:retrieve\nid_field: source\n",
        encoding="utf-8",
    )
    plain = runner.invoke(app, ["run", "--config", str(config)])
    assert plain.exit_code == 2
    assert "id_field applies" in _output(plain)

    config.write_text(
        "dataset: cases.jsonl\nretriever_url: http://127.0.0.1/search\nadapter: auto\n",
        encoding="utf-8",
    )
    url = runner.invoke(app, ["run", "--config", str(config)])
    assert url.exit_code == 2
    assert "adapter requires a retriever import path" in _output(url)


def test_langchain_k_values_searches_once_at_the_largest_k():
    store = _store(
        Document(id="d1", page_content="cats"),
        Document(id="d2", page_content="cats"),
        Document(id="d3", page_content="cats"),
    )
    seen: list[int] = []
    real = store.asimilarity_search

    async def spy(query: str, k: int = 4, **kwargs):
        seen.append(k)
        return await real(query, k=k, **kwargs)

    store.asimilarity_search = spy
    fn = adapt_retriever(store, label="app:store")
    case = Case.model_validate({"id": "q1", "query": "cats", "relevant": ["d1"]})
    report = build_report(
        [case],
        fn,
        3,
        dataset="cases.jsonl",
        retriever_name="app:store",
        thresholds={},
        k_values=[1, 3],
    )
    assert seen == [3]
    assert report["k"] == 3
    assert [row["k"] for row in report["cutoffs"]] == [1, 3]
    retrieved = report["results"][0]["retrieved"]
    assert len(retrieved) == 3
    hit = 1.0 if retrieved[0] == "d1" else 0.0
    assert report["cutoffs"][0]["metrics"]["hit_rate"] == hit
    assert report["cutoffs"][1]["metrics"]["precision"] == round(1 / 3, 6)


def test_llamaindex_k_values_sets_top_k_once():
    index = _index(*(TextNode(text=f"doc {i}", id_=f"d{i}") for i in range(1, 5)))
    retriever = index.as_retriever(similarity_top_k=2)
    seen: list[tuple[str, int]] = []
    real = retriever.aretrieve

    async def spy(query: str):
        seen.append(("aretrieve", retriever.similarity_top_k))
        return await real(query)

    retriever.aretrieve = spy
    fn = adapt_retriever(retriever, adapter="llamaindex", label="app:retriever")
    case = Case.model_validate({"id": "q1", "query": "doc", "relevant": ["d1"]})
    report = build_report(
        [case],
        fn,
        4,
        dataset="cases.jsonl",
        retriever_name="app:retriever",
        thresholds={},
        k_values=[1, 4],
    )
    assert seen == [("aretrieve", 4)]
    assert retriever.similarity_top_k == 2
    assert report["k"] == 4
    assert [row["k"] for row in report["cutoffs"]] == [1, 4]
    assert len(report["results"][0]["retrieved"]) == 4


def test_llamaindex_fixed_list_is_scored_at_each_cutoff():
    fn = adapt_retriever(_LlamaFixed(), label="app:retriever")
    case = Case.model_validate({"id": "q1", "query": "q", "relevant": ["d1"]})
    report = build_report(
        [case],
        fn,
        3,
        dataset="cases.jsonl",
        retriever_name="app:retriever",
        thresholds={"precision": 0.5},
        k_values=[1, 3],
        thresholds_by_k={1: {}, 3: {"precision": 0.5}},
    )
    assert report["results"][0]["retrieved"] == ["d1", "d2", "d3"]
    assert report["cutoffs"][0]["metrics"]["precision"] == 1.0
    assert report["cutoffs"][0]["metrics"]["recall"] == 1.0
    assert report["cutoffs"][1]["metrics"]["precision"] == round(1 / 3, 6)
    assert report["failed"] == ["precision"]
