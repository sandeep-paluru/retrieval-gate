"""LangChain in-memory vector store over the example documents.

The embedding is a deterministic hashing trick. It makes no model call and
no network call. Point ``--retriever`` at ``store`` (or at ``retriever``).
"""

import hashlib
import math

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore

from examples.retriever import DOCS, tokenize

_DIM = 1024


def _embed(text: str) -> list[float]:
    vector = [0.0] * _DIM
    for token in tokenize(text):
        digest = hashlib.sha256(token.encode()).digest()
        bucket = int.from_bytes(digest[:4], "little") % _DIM
        vector[bucket] += 1.0
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


class HashEmbedding(Embeddings):
    """Bag of tokens, hashed into a fixed vector."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [_embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return _embed(text)


store = InMemoryVectorStore(embedding=HashEmbedding())
store.add_documents(
    [
        Document(id=doc["id"], page_content=doc["text"], metadata={"source": doc["id"]})
        for doc in DOCS
    ]
)
retriever = store.as_retriever()
