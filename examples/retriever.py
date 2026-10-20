"""Small BM25 retriever over a fixed 10-document corpus.

This module is the quickstart retriever. It has no dependencies beyond
the standard library. Parameters are k1 = 1.5 and b = 0.75. Inverse
document frequency for a term that appears in ``df`` of ``n`` documents is
``log(1 + (n - df + 0.5) / (df + 0.5))``.
"""

import math
import re
from collections import Counter

_K1 = 1.5
_B = 0.75
_TOKEN = re.compile(r"[a-z0-9]+")

DOCS = [
    {
        "id": "d01",
        "text": (
            "To reset your password, open settings, choose security, "
            "and send a reset link by email."
        ),
    },
    {
        "id": "d02",
        "text": (
            "Turn on two factor authentication with an authenticator app "
            "or an SMS code at login."
        ),
    },
    {
        "id": "d03",
        "text": "Download each monthly invoice from the billing page as a PDF.",
    },
    {
        "id": "d04",
        "text": "Refunds go back to the original payment method within 5 to 10 business days.",
    },
    {
        "id": "d05",
        "text": (
            "Create an API key on the developer page. The key is shown once, "
            "so store it safely."
        ),
    },
    {
        "id": "d06",
        "text": (
            "The API allows 100 requests per minute. "
            "A 429 status means the rate limit was hit."
        ),
    },
    {
        "id": "d07",
        "text": (
            "Single sign on uses SAML. Upload the identity provider "
            "metadata in the admin console."
        ),
    },
    {
        "id": "d08",
        "text": (
            "Export account data as CSV or JSON from the export page. "
            "Large exports arrive by email."
        ),
    },
    {
        "id": "d09",
        "text": "Invite a teammate by email and assign the role admin, member, or viewer.",
    },
    {
        "id": "d10",
        "text": "Webhooks POST JSON to your URL when an invoice is paid or a user is invited.",
    },
]


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def _index():
    tokenized = [(doc["id"], tokenize(doc["text"])) for doc in DOCS]
    doc_freq: Counter[str] = Counter()
    for _, tokens in tokenized:
        doc_freq.update(set(tokens))
    avgdl = sum(len(tokens) for _, tokens in tokenized) / len(tokenized)
    n = len(tokenized)
    idf = {
        term: math.log(1.0 + (n - freq + 0.5) / (freq + 0.5))
        for term, freq in doc_freq.items()
    }
    return tokenized, idf, avgdl


_DOCS, _IDF, _AVGDL = _index()


def _score(terms: list[str], tokens: list[str]) -> float:
    tf = Counter(tokens)
    length = len(tokens)
    total = 0.0
    for term in dict.fromkeys(terms):
        freq = tf.get(term, 0)
        if freq == 0:
            continue
        denom = freq + _K1 * (1 - _B + _B * length / _AVGDL)
        total += _IDF[term] * (freq * (_K1 + 1) / denom)
    return total


def retrieve(query: str, k: int) -> list[str]:
    """Return up to ``k`` document ids, best match first."""

    if k < 1:
        return []
    terms = tokenize(query)
    scored = []
    for doc_id, tokens in _DOCS:
        value = _score(terms, tokens)
        if value > 0:
            scored.append((value, doc_id))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [doc_id for _, doc_id in scored[:k]]
