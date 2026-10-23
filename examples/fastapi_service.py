"""Search API over the example BM25 retriever.

Start it from the repository root:

    uvicorn examples.fastapi_service:app --port 8000

POST /search takes JSON {"query": "...", "k": 5}. GET /search takes the
same two fields as query parameters. Both return {"results": [{"id": "..."}]}.
"""

from fastapi import FastAPI
from pydantic import BaseModel

from examples.retriever import retrieve

app = FastAPI()


class SearchRequest(BaseModel):
    query: str
    k: int = 5


def _results(query: str, k: int) -> dict[str, list[dict[str, str]]]:
    return {"results": [{"id": doc_id} for doc_id in retrieve(query, k)]}


@app.get("/search")
def search_get(query: str, k: int = 5) -> dict[str, list[dict[str, str]]]:
    return _results(query, k)


@app.post("/search")
def search_post(body: SearchRequest) -> dict[str, list[dict[str, str]]]:
    return _results(body.query, body.k)
