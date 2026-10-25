"""Retriever callables used by the tests."""

import contextvars
import threading

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


class ScriptedClock:
    """Clock whose start/end pair yields the next duration.

    The first read in a pair returns 0 and the second returns the
    duration, so the difference is that duration even when calls
    overlap. Each duration is consumed once, in start order.
    """

    def __init__(self, durations_s: list[float]) -> None:
        self._durations = [float(item) for item in durations_s]
        self._index = 0
        self._lock = threading.Lock()
        self._pending = contextvars.ContextVar("rg_scripted_clock_end", default=None)

    def __call__(self) -> float:
        scheduled = self._pending.get()
        if scheduled is None:
            with self._lock:
                if self._index >= len(self._durations):
                    raise AssertionError("clock was read more times than the test scheduled")
                duration = self._durations[self._index]
                self._index += 1
            self._pending.set(duration)
            return 0.0
        self._pending.set(None)
        return float(scheduled)

    def remaining(self) -> int:
        with self._lock:
            return len(self._durations) - self._index


def scripted_clock(durations_s: list[float]) -> ScriptedClock:
    return ScriptedClock(durations_s)
