"""Level geometry: object list, JSON serialisation and a spatial index."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from gdbot.types import Mode

#: One Geometry Dash grid cell, in world units.  Every length here is a multiple
#: of this so levels stay readable.
BLOCK = 30.0

#: Horizontal player speed per speed portal, in world units per second -- the
#: figures commonly cited for the real game.  The simulator only needs them to be
#: in the right proportion to each other and to the jump arc.
SPEEDS: dict[str, float] = {
    "0.5x": 251.16,
    "1x": 311.58,
    "2x": 387.42,
    "3x": 468.00,
    "4x": 576.00,
}
SPEED_NAMES = list(SPEEDS)


class ObjType(IntEnum):
    """Object kinds the simulator understands."""

    SOLID = 0          #: block: stand on top, dying on a side hit
    SPIKE = 1          #: hazard: any overlap kills
    ORB = 2            #: jump ring: pressing while overlapping gives an impulse
    PAD = 3            #: jump pad: impulse on contact, no input needed
    PORTAL_GRAVITY = 4  #: param 0 = normal, 1 = flipped, 2 = toggle
    PORTAL_MODE = 5     #: param = Mode value
    PORTAL_SPEED = 6    #: param = index into SPEED_NAMES


#: Hitbox scale applied to the player when testing hazards.  The real game uses
#: a noticeably smaller hazard hitbox than the visual sprite, which is why
#: "impossible looking" gaps are passable.
HAZARD_SHRINK = 0.55


@dataclass(slots=True)
class Obj:
    """One axis-aligned level object. ``x``/``y`` are its lower-left corner."""

    t: ObjType
    x: float
    y: float
    w: float = BLOCK
    h: float = BLOCK
    param: int = 0

    def as_tuple(self) -> tuple[float, float, float, float, int, int]:
        return (self.x, self.y, self.w, self.h, int(self.t), int(self.param))

    def to_json(self) -> dict:
        d = {"t": ObjType(self.t).name.lower(), "x": self.x, "y": self.y}
        if self.w != BLOCK:
            d["w"] = self.w
        if self.h != BLOCK:
            d["h"] = self.h
        if self.param:
            d["param"] = self.param
        return d

    @classmethod
    def from_json(cls, d: dict) -> "Obj":
        return cls(
            t=ObjType[str(d["t"]).upper()],
            x=float(d["x"]),
            y=float(d["y"]),
            w=float(d.get("w", BLOCK)),
            h=float(d.get("h", BLOCK)),
            param=int(d.get("param", 0)),
        )


@dataclass
class Level:
    """A level: a bag of objects plus the player's starting configuration."""

    name: str
    length: float
    objects: list[Obj] = field(default_factory=list)
    speed: str = "1x"
    mode: Mode = Mode.CUBE
    ceiling: float = BLOCK * 10
    #: Optional author solution, used by tests and as a curriculum baseline.
    solution: list[int] | None = None

    # --- spatial index -----------------------------------------------------
    #: Objects bucketed by ``floor(x / BUCKET)`` so a tick only tests nearby
    #: geometry instead of the whole level.
    BUCKET = BLOCK * 4

    def __post_init__(self) -> None:
        self.mode = Mode(self.mode)
        if self.speed not in SPEEDS:
            raise ValueError(f"unknown speed {self.speed!r}, expected one of {SPEED_NAMES}")
        self.reindex()

    def reindex(self) -> None:
        """Rebuild the bucket index and the packed object array."""
        self._buckets: dict[int, list[int]] = {}
        for i, o in enumerate(self.objects):
            lo = int(o.x // self.BUCKET)
            hi = int((o.x + o.w) // self.BUCKET)
            for b in range(lo, hi + 1):
                self._buckets.setdefault(b, []).append(i)
        if self.objects:
            self._packed = np.array([o.as_tuple() for o in self.objects], dtype=np.float64)
        else:
            self._packed = np.zeros((0, 6), dtype=np.float64)

    def near(self, x: float, reach: float = BLOCK) -> list[Obj]:
        """Objects whose bucket overlaps ``[x - reach, x + reach]``."""
        lo = int((x - reach) // self.BUCKET)
        hi = int((x + reach) // self.BUCKET)
        seen: set[int] = set()
        out: list[Obj] = []
        for b in range(lo, hi + 1):
            for i in self._buckets.get(b, ()):
                if i not in seen:
                    seen.add(i)
                    out.append(self.objects[i])
        return out

    def near_idx(self, x: float, reach: float = BLOCK) -> list[int]:
        """Indices of objects whose bucket overlaps ``[x - reach, x + reach]``.

        The physics loop needs indices so it can latch one-shot objects (portals,
        pads) that would otherwise re-trigger on every tick of an overlap.
        """
        lo = int((x - reach) // self.BUCKET)
        hi = int((x + reach) // self.BUCKET)
        seen: set[int] = set()
        for b in range(lo, hi + 1):
            seen.update(self._buckets.get(b, ()))
        return sorted(seen)

    @property
    def packed(self) -> np.ndarray:
        """``(n, 6)`` array of ``(x, y, w, h, type, param)`` for vectorised use."""
        return self._packed

    def add(self, *objs: Obj) -> "Level":
        self.objects.extend(objs)
        self.reindex()
        return self

    def extend(self, objs: Iterable[Obj]) -> "Level":
        self.objects.extend(objs)
        self.reindex()
        return self

    # --- IO ----------------------------------------------------------------
    def to_json(self) -> dict:
        d = {
            "name": self.name,
            "length": self.length,
            "speed": self.speed,
            "mode": Mode(self.mode).name.lower(),
            "ceiling": self.ceiling,
            "objects": [o.to_json() for o in self.objects],
        }
        if self.solution is not None:
            d["solution"] = _rle_encode(self.solution)
        return d

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_json(), indent=1), encoding="utf-8")

    @classmethod
    def from_json(cls, d: dict) -> "Level":
        sol = d.get("solution")
        return cls(
            name=str(d.get("name", "level")),
            length=float(d["length"]),
            objects=[Obj.from_json(o) for o in d.get("objects", [])],
            speed=str(d.get("speed", "1x")),
            mode=Mode[str(d.get("mode", "cube")).upper()],
            ceiling=float(d.get("ceiling", BLOCK * 10)),
            solution=_rle_decode(sol) if sol is not None else None,
        )

    @classmethod
    def load(cls, path: str | Path) -> "Level":
        return cls.from_json(json.loads(Path(path).read_text(encoding="utf-8")))

    def __len__(self) -> int:
        return len(self.objects)

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return (
            f"Level({self.name!r}, length={self.length:.0f}u"
            f"={self.length / BLOCK:.0f} blocks, objects={len(self.objects)})"
        )


def _rle_encode(actions: Sequence[int]) -> list[int]:
    """Run-length encode an input tape: ``[first_value, run, run, ...]``.

    Tapes are mostly long runs of the same value, so this keeps level files small
    and human-scannable.
    """
    arr = [1 if a else 0 for a in actions]
    if not arr:
        return [0]
    out = [arr[0]]
    run = 1
    for prev, cur in zip(arr, arr[1:]):
        if cur == prev:
            run += 1
        else:
            out.append(run)
            run = 1
    out.append(run)
    return out


def _rle_decode(data: Sequence[int]) -> list[int]:
    if not data:
        return []
    value = int(data[0])
    out: list[int] = []
    for run in data[1:]:
        out.extend([value] * int(run))
        value ^= 1
    return out
