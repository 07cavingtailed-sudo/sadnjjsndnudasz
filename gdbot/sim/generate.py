"""Procedural level generator with solvability guaranteed by construction.

The real Skeletal Shenanigans cannot be shipped here -- its geometry is not ours
to redistribute, and the agent reads the real level off the screen anyway.  What
this module provides is an honest *benchmark*: levels with the same structural
features a demon has (tight timing windows, ship and wave corridors, gravity
flips, speed changes, two minutes of it) that the solver can be measured against
reproducibly.

Solvability is not hoped for, it is proven while generating.  After each chunk is
appended, a beam search runs from the state the player was actually in at the end
of the previous chunk.  If the chunk cannot be cleared it is thrown away and a
different one is drawn.  The concatenation of the per-chunk tapes is therefore a
verified solution for the whole level, which is stored in the file and used by
the tests as ground truth.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from gdbot.agents.beam import BeamConfig, beam_search
from gdbot.sim.level import BLOCK as B
from gdbot.sim.level import SPEED_NAMES, Level, Obj, ObjType
from gdbot.sim.physics import TPS, World
from gdbot.types import Mode, Outcome

#: A chunk builder takes the x where it starts plus a difficulty in [0, 1] and
#: returns its objects and its width in world units.
Builder = Callable[[float, float, np.random.Generator], tuple[list[Obj], float]]


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * max(0.0, min(1.0, t))


# --------------------------------------------------------------------- chunks
def spike_run(x: float, d: float, rng: np.random.Generator) -> tuple[list[Obj], float]:
    """Floor spikes that have to be jumped, spaced near the jump length."""
    n = int(rng.integers(3, 6))
    spacing = _lerp(5.4, 4.2, d)
    objs = []
    for i in range(n):
        objs.append(Obj(ObjType.SPIKE, x + (1.5 + i * spacing) * B, 0.0))
        if d > 0.55 and rng.random() < 0.4:
            objs.append(Obj(ObjType.SPIKE, x + (2.5 + i * spacing) * B, 0.0))
    return objs, (1.5 + n * spacing + 2.0) * B


def platform_gaps(x: float, d: float, rng: np.random.Generator) -> tuple[list[Obj], float]:
    """Floating platforms over a spike pit; land each one or die."""
    n = int(rng.integers(3, 6))
    gap = _lerp(2.2, 3.3, d)
    plat_w = _lerp(3.0, 1.6, d)
    objs: list[Obj] = []
    cx = x + 2.0 * B
    height = 2.0
    pit_from = cx
    for i in range(n):
        objs.append(Obj(ObjType.SOLID, cx, height * B, w=plat_w * B, h=B))
        cx += (plat_w + gap) * B
        # keep steps within reach: a jump peaks at ~2 blocks and lands 4.3 away
        height = float(np.clip(height + rng.choice([-1.0, 0.0, 0.0, 1.0]), 1.0, 4.0))
    pit_w = cx - pit_from
    for k in range(int(pit_w // B)):
        objs.append(Obj(ObjType.SPIKE, pit_from + k * B, 0.0))
    return objs, (cx - x) + 2.5 * B


def orb_chain(x: float, d: float, rng: np.random.Generator) -> tuple[list[Obj], float]:
    """Rings over a pit: the jump has to come from mid-air, on time."""
    n = int(rng.integers(2, 4))
    spacing = _lerp(4.4, 3.4, d)
    objs: list[Obj] = []
    cx = x + 3.0 * B
    span = (n + 1) * spacing * B
    for k in range(int(span // B)):
        objs.append(Obj(ObjType.SPIKE, x + 2.0 * B + k * B, 0.0))
    y = 2.0
    for i in range(n):
        objs.append(Obj(ObjType.ORB, cx + i * spacing * B, y * B, w=B, h=B))
        y = float(np.clip(y + rng.choice([-0.5, 0.0, 0.5]), 1.5, 4.0))
    return objs, span + 4.0 * B


def pad_launch(x: float, d: float, rng: np.random.Generator) -> tuple[list[Obj], float]:
    """A jump pad under a ceiling of spikes."""
    objs = [Obj(ObjType.PAD, x + 2.0 * B, 0.0, w=B, h=B * 0.4)]
    for k in range(int(_lerp(3, 6, d))):
        objs.append(Obj(ObjType.SPIKE, x + (4.0 + k) * B, 0.0))
    return objs, _lerp(9.0, 12.0, d) * B


def _corridor(
    x: float,
    d: float,
    rng: np.random.Generator,
    *,
    mode: Mode,
    clearance_easy: float,
    clearance_hard: float,
    max_slope: float,
    ceiling: float,
) -> tuple[list[Obj], float]:
    """Shared builder for ship and wave passages."""
    cols = int(rng.integers(14, 26))
    clearance = _lerp(clearance_easy, clearance_hard, d)
    objs: list[Obj] = [
        Obj(ObjType.PORTAL_MODE, x + 1.0 * B, 0.0, w=B * 0.6, h=ceiling, param=int(mode)),
    ]
    floor = 2.0
    top_limit = ceiling / B
    for i in range(cols):
        cx = x + (2.0 + i) * B
        floor = float(np.clip(floor + rng.uniform(-max_slope, max_slope), 0.0, top_limit - clearance - 1.0))
        if floor > 0.0:
            objs.append(Obj(ObjType.SOLID, cx, 0.0, w=B, h=floor * B))
        cap_y = (floor + clearance) * B
        if cap_y < ceiling:
            objs.append(Obj(ObjType.SOLID, cx, cap_y, w=B, h=ceiling - cap_y))
    # back to cube, then flat ground to land on
    exit_x = x + (2.0 + cols) * B
    objs.append(Obj(ObjType.PORTAL_MODE, exit_x, 0.0, w=B * 0.6, h=ceiling, param=int(Mode.CUBE)))
    return objs, (exit_x - x) + 6.0 * B


def ship_corridor(x: float, d: float, rng: np.random.Generator) -> tuple[list[Obj], float]:
    return _corridor(
        x, d, rng, mode=Mode.SHIP,
        clearance_easy=4.5, clearance_hard=2.8, max_slope=0.8, ceiling=B * 10,
    )


def wave_corridor(x: float, d: float, rng: np.random.Generator) -> tuple[list[Obj], float]:
    return _corridor(
        x, d, rng, mode=Mode.WAVE,
        clearance_easy=3.6, clearance_hard=2.2, max_slope=0.6, ceiling=B * 10,
    )


def ball_flip(x: float, d: float, rng: np.random.Generator) -> tuple[list[Obj], float]:
    """Ball mode between a floor and a roof, with hazards on both."""
    cols = int(rng.integers(10, 18))
    ceiling = B * 10
    roof = _lerp(6.0, 4.5, d)
    objs: list[Obj] = [
        Obj(ObjType.PORTAL_MODE, x + 1.0 * B, 0.0, w=B * 0.6, h=ceiling, param=int(Mode.BALL)),
        Obj(ObjType.SOLID, x + 2.0 * B, roof * B, w=cols * B, h=B),
    ]
    for i in range(cols):
        cx = x + (2.0 + i) * B
        if rng.random() < _lerp(0.12, 0.3, d):
            objs.append(Obj(ObjType.SPIKE, cx, 0.0))
        if rng.random() < _lerp(0.12, 0.3, d):
            objs.append(Obj(ObjType.SPIKE, cx, (roof - 1.0) * B))
    objs.append(Obj(ObjType.PORTAL_MODE, x + (2.0 + cols) * B, 0.0, w=B * 0.6, h=ceiling, param=int(Mode.CUBE)))
    return objs, (cols + 8.0) * B


def speed_shift(x: float, d: float, rng: np.random.Generator) -> tuple[list[Obj], float]:
    """A speed portal, some spikes at the new speed, then back to the base."""
    faster = int(rng.integers(2, 4 if d < 0.7 else 5))  # index into SPEED_NAMES
    ceiling = B * 10
    objs: list[Obj] = [
        Obj(ObjType.PORTAL_SPEED, x + 1.0 * B, 0.0, w=B * 0.6, h=ceiling, param=faster)
    ]
    n = int(rng.integers(3, 6))
    spacing = _lerp(7.0, 5.5, d) * (1.0 + 0.25 * faster)
    for i in range(n):
        objs.append(Obj(ObjType.SPIKE, x + (4.0 + i * spacing) * B, 0.0))
    end = 4.0 + n * spacing + 2.0
    objs.append(Obj(ObjType.PORTAL_SPEED, x + end * B, 0.0, w=B * 0.6, h=ceiling, param=1))
    return objs, (end + 3.0) * B


def gravity_flip(x: float, d: float, rng: np.random.Generator) -> tuple[list[Obj], float]:
    """Upside-down cube run along the ceiling."""
    ceiling = B * 10
    cols = int(rng.integers(10, 16))
    objs: list[Obj] = [
        Obj(ObjType.PORTAL_GRAVITY, x + 1.0 * B, 0.0, w=B * 0.6, h=ceiling, param=1),
    ]
    for i in range(cols):
        cx = x + (3.0 + i * _lerp(5.2, 4.2, d)) * B
        if rng.random() < _lerp(0.35, 0.7, d):
            # spikes hang from the ceiling when gravity is flipped
            objs.append(Obj(ObjType.SPIKE, cx, ceiling - B))
    objs.append(
        Obj(ObjType.PORTAL_GRAVITY, x + (4.0 + cols * _lerp(5.2, 4.2, d)) * B, 0.0,
            w=B * 0.6, h=ceiling, param=0)
    )
    return objs, (8.0 + cols * _lerp(5.2, 4.2, d)) * B


#: Chunk pool with the difficulty band each one is drawn in.
CHUNKS: list[tuple[Builder, float, float]] = [
    (spike_run, 0.0, 1.0),
    (platform_gaps, 0.1, 1.0),
    (orb_chain, 0.2, 1.0),
    (pad_launch, 0.0, 0.8),
    (ship_corridor, 0.15, 1.0),
    (wave_corridor, 0.45, 1.0),
    (ball_flip, 0.3, 1.0),
    (speed_shift, 0.25, 1.0),
    (gravity_flip, 0.2, 1.0),
]


def _advance(world: World, snap, tape: Sequence[int]):
    """Replay ``tape`` from ``snap`` and return the state it ends in.

    The temporary finish line used to verify a chunk leaves the world in
    ``Outcome.COMPLETE``; that has to be cleared or the next chunk's search would
    start from an already-finished world and return an empty tape.
    """
    if snap is None:
        world.reset()
    else:
        world.restore(snap)
    for a in tape:
        world.step(bool(a))
    world.outcome = Outcome.RUNNING
    return world.snapshot()


@dataclass
class GenReport:
    """What generation had to do, so a level file can be trusted."""

    chunks_accepted: int = 0
    chunks_rejected: int = 0
    verify_expanded: int = 0
    solution_ticks: int = 0

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return (
            f"GenReport(accepted={self.chunks_accepted}, rejected={self.chunks_rejected}, "
            f"solution={self.solution_ticks} ticks "
            f"= {self.solution_ticks / TPS:.1f}s, expanded={self.verify_expanded:,})"
        )


def generate(
    name: str,
    *,
    n_chunks: int = 12,
    difficulty: float = 0.5,
    seed: int = 0,
    speed: str = "1x",
    verify_width: int = 320,
    tries_per_chunk: int = 6,
    ramp: bool = True,
) -> tuple[Level, GenReport]:
    """Build a level whose solution is found while building it.

    ``ramp`` makes difficulty climb across the level the way a real demon does,
    from the easy intro to the hardest section near the end.
    """
    rng = np.random.default_rng(seed)
    level = Level(name=name, length=6.0 * B, objects=[], speed=speed, mode=Mode.CUBE)
    report = GenReport()
    solution: list[int] = []
    snap = None
    x = 4.0 * B  # run-in: flat ground so the player settles before anything happens

    for k in range(n_chunks):
        d = difficulty * (_lerp(0.35, 1.0, k / max(1, n_chunks - 1)) if ramp else 1.0)
        pool = [b for b, lo, hi in CHUNKS if lo <= d <= hi] or [spike_run]
        accepted = False
        for _try in range(tries_per_chunk):
            builder = pool[int(rng.integers(0, len(pool)))]
            objs, width = builder(x, d, rng)
            level.objects.extend(objs)
            level.length = x + width
            level.reindex()

            world = World(level, max_ticks=TPS * 400)
            res = beam_search(
                world,
                BeamConfig(width=verify_width, max_ticks=int(width / 60) + 900, goal_progress=1.0),
                start=snap,
            )
            report.verify_expanded += res.expanded
            if res.solved:
                # Re-run the tape to capture the exact state the next chunk starts from.
                snap = _advance(world, snap, res.tape)
                solution.extend(int(a) for a in res.tape)
                # Start the next chunk ahead of the player's *actual* position, not
                # of the nominal chunk end.  A verified tape overshoots the finish
                # line by up to one tick, and a later chunk placing a full-height
                # portal there would retroactively invalidate the tape that was
                # already checked.
                x = max(x + width, snap[0] + 2.0 * B)
                report.chunks_accepted += 1
                accepted = True
                break
            # unsolvable as drawn: drop it and try a different pattern
            del level.objects[len(level.objects) - len(objs) :]
            level.reindex()
            report.chunks_rejected += 1
        if not accepted:
            # Fall back to a guaranteed-easy chunk so generation always terminates.
            objs, width = spike_run(x, 0.0, rng)
            level.objects.extend(objs)
            level.length = x + width
            level.reindex()
            world = World(level, max_ticks=TPS * 400)
            res = beam_search(
                world, BeamConfig(width=verify_width, max_ticks=int(width / 60) + 900), start=snap
            )
            if not res.solved:  # pragma: no cover - would mean the physics changed
                raise RuntimeError(f"fallback chunk at x={x:.0f} is unsolvable")
            snap = _advance(world, snap, res.tape)
            solution.extend(int(a) for a in res.tape)
            x = max(x + width, snap[0] + 2.0 * B)
            report.chunks_accepted += 1

    # outro: flat ground to the finish line
    level.length = x + 4.0 * B
    level.reindex()
    world = World(level, max_ticks=TPS * 400)
    res = beam_search(world, BeamConfig(width=verify_width, max_ticks=900), start=snap)
    if res.solved:
        solution.extend(int(a) for a in res.tape)

    # Final guarantee: the stored solution must clear the finished level when
    # replayed from a cold start.  Anything less and the file is not a benchmark.
    world = World(level, max_ticks=TPS * 400)
    check = world.run_tape(solution)
    if check.outcome is not Outcome.COMPLETE:
        full = beam_search(
            World(level, max_ticks=TPS * 400),
            BeamConfig(width=verify_width * 3, max_ticks=TPS * 400),
        )
        if not full.solved:  # pragma: no cover - generation is retried instead
            raise RuntimeError(
                f"generated level {name!r} is not solvable end to end "
                f"(best {full.progress:.3f}); regenerate with another seed"
            )
        solution = [int(a) for a in full.tape]

    level.solution = solution
    report.solution_ticks = len(solution)
    return level, report
