"""Beam search over input tapes, with state deduplication.

Why this and not "just reinforcement learning": Geometry Dash is deterministic
and has a one-bit action space.  That makes clearing a level a *search* problem
over ``{0,1}^ticks`` rather than a control problem, and a search that keeps one
representative per distinguishable physical state explores it in time linear in
the level length instead of exponential.

The search keeps, for every tick, a bounded frontier of surviving player states.
Two states are treated as the same when their quantised position, velocity, mode,
gravity and consumed-object set agree.  Dedup only ever *loses* candidate paths
-- it never invents one -- so any tape this returns is genuinely valid, and the
solver replays it to confirm.

This needs ``World.snapshot`` / ``World.restore``, so it runs in the simulator.
The real game's equivalent is :mod:`gdbot.agents.ga_tape`, which only needs
"replay a tape and report how far it got".
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from gdbot.sim.physics import Player, Snapshot, World
from gdbot.types import Outcome


@dataclass
class BeamConfig:
    """Search budget and how finely states are distinguished."""

    #: Maximum surviving states carried from one tick to the next.  Wider finds
    #: harder paths and costs linearly more time.
    width: int = 600
    #: Quantisation of the dedup key.  Finer = more states kept = slower but
    #: less likely to discard the one path that works.
    dedup_x: float = 4.0
    dedup_y: float = 2.0
    dedup_vy: float = 20.0
    max_ticks: int = 60 * 240
    #: Stop as soon as progress reaches this (1.0 = clear the level).
    goal_progress: float = 1.0
    #: Wall-clock budget in seconds; 0 disables.
    time_budget: float = 0.0
    #: Call back with (tick, frontier_size, progress) every N ticks; 0 disables.
    log_every: int = 0


@dataclass
class BeamResult:
    tape: np.ndarray
    outcome: Outcome
    progress: float
    ticks: int
    expanded: int
    peak_frontier: int
    elapsed: float = 0.0

    @property
    def solved(self) -> bool:
        return self.outcome is Outcome.COMPLETE

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return (
            f"BeamResult({self.outcome.name}, progress={self.progress:.4f}, "
            f"ticks={self.ticks}, expanded={self.expanded:,}, "
            f"peak_frontier={self.peak_frontier}, {self.elapsed:.1f}s)"
        )


class _TapeTree:
    """Parent-pointer tree of actions.

    Storing whole tapes per frontier state would be quadratic in level length;
    storing one ``(parent, action)`` pair per surviving state is linear.
    """

    __slots__ = ("parent", "action", "n")

    def __init__(self, capacity: int = 1 << 16) -> None:
        self.parent = np.full(capacity, -1, dtype=np.int32)
        self.action = np.zeros(capacity, dtype=np.uint8)
        self.n = 0

    def _grow(self) -> None:
        cap = len(self.parent) * 2
        self.parent = np.resize(self.parent, cap)
        self.action = np.resize(self.action, cap)

    def add(self, parent: int, action: int) -> int:
        if self.n >= len(self.parent):
            self._grow()
        idx = self.n
        self.parent[idx] = parent
        self.action[idx] = action
        self.n += 1
        return idx

    def path(self, idx: int) -> np.ndarray:
        out: list[int] = []
        while idx >= 0:
            out.append(int(self.action[idx]))
            idx = int(self.parent[idx])
        out.reverse()
        return np.asarray(out, dtype=np.uint8)


def _state_key(w: World, cfg: BeamConfig) -> tuple:
    """Everything about the world that can affect the future, quantised.

    ``_used`` is included verbatim: two runs that consumed different portals are
    genuinely different futures, and a frozenset hashes in constant time.
    """
    p: Player = w.player
    return (
        int(p.x / cfg.dedup_x),
        int(p.y / cfg.dedup_y),
        int(p.vy / cfg.dedup_vy),
        int(p.mode),
        int(p.grav),
        p.on_ground,
        p.hold,
        int(p.speed),
        w._used,
    )


def _select(items: list[tuple[tuple, tuple]], width: int) -> list[tuple[tuple, tuple]]:
    """Trim the frontier to ``width`` entries while keeping it spread out.

    At a given tick nearly every surviving state shares the same ``x``, so
    ranking by progress alone would pick an arbitrary clump.  Sorting by
    ``(-x, key)`` groups by progress and then orders by height, and taking a
    uniform stride through that keeps candidates from across the whole range of
    heights and velocities instead of a single cluster.
    """
    if len(items) <= width:
        return items
    items.sort(key=lambda kv: (-kv[1][3], kv[0][:8]))
    step = len(items) / width
    return [items[min(len(items) - 1, int(i * step))] for i in range(width)]


def beam_search(
    world: World,
    cfg: BeamConfig | None = None,
    *,
    start: Snapshot | None = None,
    on_progress: Callable[[int, int, float], None] | None = None,
) -> BeamResult:
    """Search for an input tape that reaches ``cfg.goal_progress``.

    ``start`` lets the curriculum resume from a checkpoint instead of the level
    start, which is what makes training the back half of a long level affordable.
    """
    cfg = cfg or BeamConfig()
    t0 = time.perf_counter()
    tree = _TapeTree()

    if start is None:
        world.reset()
    else:
        world.restore(start)
    root_snap = world.snapshot()
    base_tick = world.tick

    frontier: list[tuple[int, Snapshot]] = [(-1, root_snap)]
    best_node, best_progress = -1, world.progress
    expanded = 0
    peak = 1

    for step_i in range(cfg.max_ticks):
        # key -> (parent_node, action, snapshot, x)
        candidates: dict[tuple, tuple[int, int, Snapshot, float]] = {}
        for node, snap in frontier:
            for action in (0, 1):
                world.restore(snap)
                outcome = world.step(action)
                expanded += 1
                if outcome is Outcome.DEAD:
                    continue
                progress = world.progress
                if progress > best_progress:
                    best_progress = progress
                    best_node = tree.add(node, action)
                if outcome is Outcome.COMPLETE or progress >= cfg.goal_progress:
                    leaf = tree.add(node, action)
                    return BeamResult(
                        tape=tree.path(leaf),
                        outcome=Outcome.COMPLETE,
                        progress=progress,
                        ticks=world.tick - base_tick,
                        expanded=expanded,
                        peak_frontier=peak,
                        elapsed=time.perf_counter() - t0,
                    )
                if outcome is Outcome.TIMEOUT:
                    continue
                key = _state_key(world, cfg)
                prev = candidates.get(key)
                if prev is None or world.player.x > prev[3]:
                    candidates[key] = (node, action, world.snapshot(), world.player.x)

        if not candidates:
            break

        chosen = _select(list(candidates.items()), cfg.width)
        frontier = [(tree.add(parent, action), snap) for _, (parent, action, snap, _x) in chosen]
        peak = max(peak, len(frontier))

        if cfg.log_every and step_i % cfg.log_every == 0:
            prog = max(c[1][3] for c in chosen) / world.level.length
            if on_progress is not None:
                on_progress(step_i, len(frontier), prog)
        if cfg.time_budget and (time.perf_counter() - t0) > cfg.time_budget:
            break

    tape = tree.path(best_node) if best_node >= 0 else np.zeros(0, dtype=np.uint8)
    return BeamResult(
        tape=tape,
        outcome=Outcome.DEAD if best_progress < cfg.goal_progress else Outcome.COMPLETE,
        progress=best_progress,
        ticks=len(tape),
        expanded=expanded,
        peak_frontier=peak,
        elapsed=time.perf_counter() - t0,
    )
