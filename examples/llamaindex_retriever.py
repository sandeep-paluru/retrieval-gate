"""LlamaIndex in-memory index over the example documents.

The embedding is a deterministic hashing trick. It makes no model call and
no network call. Nodes keep the example document id as ``node_id``. Point
``--retriever`` at ``index`` (or at ``retriever``).
"""

import hashlib
import math

from llama_index.core import VectorStoreIndex
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.schema import TextNode

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


class HashEmbedding(BaseEmbedding):
    """Bag of tokens, hashed into a fixed vector."""

    def _get_text_embedding(self, text: str) -> list[float]:
        return _embed(text)

    def _get_query_embedding(self, query: str) -> list[float]:
        return _embed(query)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return _embed(query)


index = VectorStoreIndex(
    [
        TextNode(text=doc["text"], id_=doc["id"], metadata={"source": doc["id"]})
        for doc in DOCS
    ],
    embed_model=HashEmbedding(),
)
retriever = index.as_retriever()
