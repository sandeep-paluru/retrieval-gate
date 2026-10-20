"""Retriever callables used by the tests."""

calls: list[tuple[str, int]] = []

LIBRARY = {
    "cats": ["d1", "d2", "d3"],
    "none": ["d8"],
    "dupes": ["d1", "d1", "d2"],
}


def retrieve(query: str, k: int) -> list[str]:
    calls.append((query, k))
    return list(LIBRARY[query])


async def retrieve_async(query: str, k: int) -> list[dict[str, str]]:
    return [{"id": "d1"}, {"id": "d2"}]


def retrieve_bad(query: str, k: int) -> str:
    return "nope"


def retrieve_boom(query: str, k: int) -> list[str]:
    raise RuntimeError("backend down")
