"""Lightweight in-memory run log shared across the pipeline.

Keeps a bounded, process-global list of timestamped entries capturing what
happens behind the scenes - document loading, unit indexing, deterministic
extraction, the single AI call and per-step timings. Used for internal
diagnostics (and surfaced in server logs); no external dependencies, so any
module can import it safely.
"""

from __future__ import annotations

import datetime as _dt
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, List

# Cap the buffer so a long session never grows without bound.
_MAX_ENTRIES = 1000


@dataclass
class LogEntry:
    ts: float       # epoch seconds (time.time)
    level: str      # INFO | TIMING | WARN | ERROR
    msg: str

    @property
    def stamp(self) -> str:
        return _dt.datetime.fromtimestamp(self.ts).strftime("%H:%M:%S")


_ENTRIES: List[LogEntry] = []


def log(msg: str, level: str = "INFO") -> None:
    """Append one entry, trimming the oldest if the buffer is full."""
    _ENTRIES.append(LogEntry(ts=time.time(), level=level, msg=str(msg)))
    overflow = len(_ENTRIES) - _MAX_ENTRIES
    if overflow > 0:
        del _ENTRIES[:overflow]


def warn(msg: str) -> None:
    log(msg, level="WARN")


def error(msg: str) -> None:
    log(msg, level="ERROR")


@contextmanager
def timed(label: str) -> Iterator[None]:
    """Log start/finish of a step plus its wall-clock duration."""
    start = time.perf_counter()
    log(f"{label} - started")
    try:
        yield
    except Exception as exc:  # noqa: BLE001 - re-raised after logging
        log(f"{label} - FAILED after {time.perf_counter() - start:.2f}s: {exc}",
            level="ERROR")
        raise
    else:
        log(f"{label} - done in {time.perf_counter() - start:.2f}s", level="TIMING")
