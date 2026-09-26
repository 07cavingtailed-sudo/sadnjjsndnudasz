"""Tape utilities: summaries, previews and export to a frame-indexed macro.

A *tape* is one ``0/1`` per physics tick.  A *macro* is the same information as
a list of press/release events keyed by frame, which is the shape every replay
tool uses; converting between the two is lossless.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class TapeStats:
    ticks: int
    seconds: float
    presses: int
    held_fraction: float
    shortest_press: int
    shortest_gap: int

    def __str__(self) -> str:
        return (
            f"{self.ticks} ticks ({self.seconds:.1f}s), {self.presses} presses, "
            f"held {self.held_fraction * 100:.1f}% of the time, "
            f"shortest press {self.shortest_press} tick(s), shortest gap {self.shortest_gap}"
        )


def _runs(tape: np.ndarray, value: int) -> list[int]:
    out: list[int] = []
    run = 0
    for a in tape:
        if a == value:
            run += 1
        elif run:
            out.append(run)
            run = 0
    if run:
        out.append(run)
    return out


def stats(tape: np.ndarray, fps: int = 60) -> TapeStats:
    arr = np.asarray(tape, dtype=np.uint8).reshape(-1)
    presses = _runs(arr, 1)
    # gaps between presses only, not the leading/trailing idle stretches
    inner = arr[np.argmax(arr == 1) : len(arr) - np.argmax(arr[::-1] == 1)] if presses else arr[:0]
    gaps = _runs(inner, 0)
    return TapeStats(
        ticks=int(arr.size),
        seconds=arr.size / fps,
        presses=len(presses),
        held_fraction=float(arr.mean()) if arr.size else 0.0,
        shortest_press=min(presses) if presses else 0,
        shortest_gap=min(gaps) if gaps else 0,
    )


def to_events(tape: np.ndarray) -> list[dict]:
    """Frame-indexed press/release events, starting from a released button."""
    arr = np.asarray(tape, dtype=np.uint8).reshape(-1)
    events: list[dict] = []
    prev = 0
    for frame, a in enumerate(arr):
        if a != prev:
            events.append({"frame": int(frame), "down": bool(a)})
            prev = int(a)
    if prev:
        events.append({"frame": int(arr.size), "down": False})
    return events


def from_events(events: list[dict], ticks: int | None = None) -> np.ndarray:
    """Inverse of :func:`to_events`."""
    end = ticks if ticks is not None else (max((e["frame"] for e in events), default=0) + 1)
    tape = np.zeros(end, dtype=np.uint8)
    state, last = 0, 0
    for e in sorted(events, key=lambda e: e["frame"]):
        f = min(int(e["frame"]), end)
        tape[last:f] = state
        state, last = int(bool(e["down"])), f
    tape[last:end] = state
    return tape


def export_macro(tape: np.ndarray, path: str | Path, *, fps: int = 60, level: str = "") -> Path:
    """Write a macro file: ``{"fps", "ticks", "level", "inputs": [{frame, down}]}``.

    Deliberately a plain, documented structure rather than a guess at a particular
    replay mod's binary format.  Every mod's format is this list of events plus a
    header, so converting is a few lines -- and a wrong guess at someone else's
    format would silently produce a macro that desyncs.
    """
    arr = np.asarray(tape, dtype=np.uint8).reshape(-1)
    payload = {
        "format": "gdbot-macro/1",
        "fps": int(fps),
        "ticks": int(arr.size),
        "level": level,
        "inputs": to_events(arr),
    }
    out = Path(path)
    out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    return out


def preview(tape: np.ndarray, width: int = 100) -> str:
    """One-line picture of when the button is held: ``_`` released, ``#`` held."""
    arr = np.asarray(tape, dtype=np.uint8).reshape(-1)
    if arr.size == 0:
        return ""
    cols = np.array_split(arr, min(width, arr.size))
    return "".join("#" if c.any() else "_" for c in cols)
