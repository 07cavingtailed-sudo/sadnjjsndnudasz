"""Deterministic Geometry Dash-style physics.

Determinism is the whole point.  Because the same input tape replayed from the
same state always produces the same run, "learning the level" reduces to
*searching for an input tape*, and a tape that works once works forever.  The
real game has this property too, which is what lets the same solver drive it.

Tuned to feel like Geometry Dash rather than to be a byte-exact clone: a cube
jump clears roughly two blocks of height and four of distance at 1x speed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Iterable, Sequence

import numpy as np

import gdbot.features as F
from gdbot.sim.level import BLOCK, HAZARD_SHRINK, SPEEDS, Level, Obj, ObjType
from gdbot.types import Attempt, Mode, Observation, Outcome

#: Simulator ticks per second.  The real game's rate is a config value
#: (``tick_rate``) and must match the FPS the game is set to run at.
TPS = 60
DT = 1.0 / TPS

# --- cube / ball -------------------------------------------------------------
GRAVITY = 2800.0      #: units/s^2
JUMP_V = 600.0        #: jump impulse -> ~2 blocks high, ~4.3 blocks across at 1x (see tests)
MAX_FALL = 1000.0

# --- rings and pads ----------------------------------------------------------
ORB_V = 620.0
PAD_V = 790.0

# --- ship --------------------------------------------------------------------
SHIP_THRUST = 1500.0
SHIP_GRAVITY = 1500.0
SHIP_MAX_V = 430.0

EPS = 1e-6
#: How close to a surface still counts as standing on it.
GROUND_EPS = 0.75


@dataclass(slots=True)
class Player:
    """Mutable player state.  Copyable, which is what makes checkpoints cheap."""

    x: float = 0.0
    y: float = 0.0
    vy: float = 0.0
    mode: Mode = Mode.CUBE
    #: +1 = normal gravity (down is -y), -1 = flipped.
    grav: float = 1.0
    speed: float = SPEEDS["1x"]
    on_ground: bool = True
    hold: bool = False
    size: float = BLOCK


#: An opaque, picklable checkpoint.  In the real game the equivalent is a
#: practice-mode checkpoint or a start-position object.
Snapshot = tuple


class World:
    """A level plus a player, advanced one tick at a time."""

    def __init__(self, level: Level, *, max_ticks: int = TPS * 180) -> None:
        self.level = level
        self.max_ticks = int(max_ticks)
        self.player = Player()
        self.tick = 0
        self.outcome = Outcome.RUNNING
        self._used: frozenset[int] = frozenset()
        self._near_bucket: int | None = None
        self._near: list[int] = []
        self.reset()

    # ------------------------------------------------------------------ setup
    def reset(self) -> None:
        lv = self.level
        mode = Mode(lv.mode)
        start_y = 0.0 if mode in (Mode.CUBE, Mode.BALL) else lv.ceiling * 0.4
        self.player = Player(
            x=0.0,
            y=start_y,
            vy=0.0,
            mode=mode,
            grav=1.0,
            speed=SPEEDS[lv.speed],
            on_ground=mode in (Mode.CUBE, Mode.BALL),
            hold=False,
        )
        self.tick = 0
        self.outcome = Outcome.RUNNING
        self._used = frozenset()
        self._near_bucket = None
        self._near = []

    def spawn_at(self, progress: float) -> None:
        """Place the player partway through the level, like a start-position object.

        Geometry Dash lets you drop a start position anywhere and it carries the
        game mode, gravity and speed that apply there; practising the back half of
        a level is exactly this.  The reconstruction scans every portal before the
        target x and replays its effect, then stands the player on the first
        surface that does not clip geometry.

        This is an *approximation* of the state a real run arrives in -- the real
        run gets there with some velocity and mid-air -- so it is used for policy
        practice, never for stitching the final tape.  The tape is chained forward
        from verified states instead.
        """
        self.reset()
        lv = self.level
        p = self.player
        target = max(0.0, min(lv.length - p.size, progress * lv.length))

        mode, grav, speed = Mode(lv.mode), 1.0, SPEEDS[lv.speed]
        used: set[int] = set()
        for i, o in enumerate(lv.objects):
            if o.x + o.w > target:
                continue
            if o.t == ObjType.PORTAL_MODE:
                mode = Mode(o.param)
            elif o.t == ObjType.PORTAL_GRAVITY:
                grav = -grav if o.param == 2 else (1.0 if o.param == 0 else -1.0)
            elif o.t == ObjType.PORTAL_SPEED:
                speed = SPEEDS[list(SPEEDS)[int(o.param)]]
            if o.t >= ObjType.PORTAL_GRAVITY or o.t == ObjType.PAD:
                used.add(i)

        p.x, p.mode, p.grav, p.speed, p.vy, p.hold = target, mode, grav, speed, 0.0, False
        self._used = frozenset(used)
        self._near_bucket = None

        if mode in (Mode.SHIP, Mode.WAVE):
            p.y = lv.ceiling * 0.4
            p.on_ground = False
            return

        near = self.level.near_idx(target, reach=self.level.BUCKET * 1.5)
        objs = lv.objects
        if grav > 0:
            levels = [0.0] + [
                objs[i].y + objs[i].h
                for i in near
                if objs[i].t == ObjType.SOLID and self._x_overlap(p, objs[i])
            ]
            levels.sort()
        else:
            levels = [lv.ceiling - p.size] + [
                objs[i].y - p.size
                for i in near
                if objs[i].t == ObjType.SOLID and self._x_overlap(p, objs[i])
            ]
            levels.sort(reverse=True)
        for cand in levels:
            p.y = cand
            if not any(
                objs[i].t == ObjType.SOLID and self._overlaps(p, objs[i]) for i in near
            ):
                break
        p.on_ground = True

    # -------------------------------------------------------------- snapshots
    def snapshot(self) -> Snapshot:
        p = self.player
        return (
            p.x, p.y, p.vy, int(p.mode), p.grav, p.speed, p.on_ground, p.hold, p.size,
            self.tick, int(self.outcome), self._used,
        )

    def restore(self, snap: Snapshot) -> None:
        (x, y, vy, mode, grav, speed, on_ground, hold, size,
         tick, outcome, used) = snap
        self.player = Player(x, y, vy, Mode(mode), grav, speed, on_ground, hold, size)
        self.tick = tick
        self.outcome = Outcome(outcome)
        self._used = used
        # The near-object cache is keyed by bucket, so it stays valid across a
        # restore into the same bucket -- a large win for the beam search.

    # --------------------------------------------------------------- geometry
    def _refresh_near(self) -> list[int]:
        """Objects around the player, recomputed only when crossing a bucket."""
        bucket = int(self.player.x // self.level.BUCKET)
        if bucket != self._near_bucket:
            self._near_bucket = bucket
            self._near = self.level.near_idx(self.player.x, reach=self.level.BUCKET * 1.5)
        return self._near

    def _x_overlap(self, p: Player, o: Obj) -> bool:
        return p.x < o.x + o.w - EPS and o.x < p.x + p.size - EPS

    def _overlaps(self, p: Player, o: Obj, shrink: float = 1.0) -> bool:
        side = p.size * shrink
        off = (p.size - side) * 0.5
        px, py = p.x + off, p.y + off
        return (
            px < o.x + o.w - EPS
            and o.x < px + side - EPS
            and py < o.y + o.h - EPS
            and o.y < py + side - EPS
        )

    def _sweep_y(self, p: Player, dy: float, near: Sequence[int]) -> tuple[float, float, int, bool]:
        """Move vertically, stopping at the first surface crossed.

        Returns ``(new_y, new_vy, hit_sign, hit_was_block)`` where ``hit_sign`` is
        -1 for a surface below, +1 for one above and 0 for no contact.
        """
        objs = self.level.objects
        y0 = p.y
        y1 = y0 + dy
        if dy < 0.0:
            best: float | None = None
            best_block = False
            for i in near:
                o = objs[i]
                if o.t != ObjType.SOLID or not self._x_overlap(p, o):
                    continue
                top = o.y + o.h
                if y0 >= top - EPS and y1 < top and (best is None or top > best):
                    best, best_block = top, True
            if y0 >= -EPS and y1 < 0.0 and (best is None or best < 0.0):
                best, best_block = 0.0, False
            if best is not None:
                return best, 0.0, -1, best_block
        elif dy > 0.0:
            best = None
            best_block = False
            top0 = y0 + p.size
            for i in near:
                o = objs[i]
                if o.t != ObjType.SOLID or not self._x_overlap(p, o):
                    continue
                if top0 <= o.y + EPS and y1 + p.size > o.y and (best is None or o.y < best):
                    best, best_block = o.y, True
            ceil = self.level.ceiling
            if top0 <= ceil + EPS and y1 + p.size > ceil and (best is None or ceil < best):
                best, best_block = ceil, False
            if best is not None:
                return best - p.size, 0.0, +1, best_block
        return y1, p.vy, 0, False

    def _probe_ground(self, p: Player, near: Sequence[int]) -> bool:
        """True when a surface sits directly in the gravity direction."""
        objs = self.level.objects
        if p.grav > 0.0:
            bottom = p.y
            if abs(bottom) <= GROUND_EPS:
                return True
            for i in near:
                o = objs[i]
                if o.t == ObjType.SOLID and self._x_overlap(p, o):
                    if abs(bottom - (o.y + o.h)) <= GROUND_EPS:
                        return True
        else:
            top = p.y + p.size
            if abs(top - self.level.ceiling) <= GROUND_EPS:
                return True
            for i in near:
                o = objs[i]
                if o.t == ObjType.SOLID and self._x_overlap(p, o):
                    if abs(top - o.y) <= GROUND_EPS:
                        return True
        return False

    # ------------------------------------------------------------------- tick
    def step(self, hold: bool) -> Outcome:
        """Advance one physics tick. Cheap: computes no observation."""
        if self.outcome is not Outcome.RUNNING:
            return self.outcome

        p = self.player
        objs = self.level.objects
        hold = bool(hold)
        press_edge = hold and not p.hold
        p.hold = hold
        near = self._refresh_near()

        # 1. one-shot interactables, evaluated before movement
        boosted = False
        for i in near:
            o = objs[i]
            if o.t == ObjType.PAD and i not in self._used and self._overlaps(p, o):
                p.vy = PAD_V * p.grav
                self._used |= {i}
                boosted = True
            elif o.t == ObjType.ORB and press_edge and self._overlaps(p, o):
                p.vy = ORB_V * p.grav
                boosted = True

        # 2. mode dynamics
        if p.mode is Mode.CUBE:
            if not boosted and p.on_ground and hold:
                p.vy = JUMP_V * p.grav
            p.vy -= GRAVITY * DT * p.grav
            p.vy = max(-MAX_FALL, min(MAX_FALL, p.vy))
        elif p.mode is Mode.BALL:
            if not boosted and p.on_ground and press_edge:
                p.grav = -p.grav
                p.vy = 0.0
            p.vy -= GRAVITY * DT * p.grav
            p.vy = max(-MAX_FALL, min(MAX_FALL, p.vy))
        elif p.mode is Mode.SHIP:
            accel = (SHIP_THRUST if hold else -SHIP_GRAVITY) * p.grav
            p.vy = max(-SHIP_MAX_V, min(SHIP_MAX_V, p.vy + accel * DT))
        elif p.mode is Mode.WAVE:
            p.vy = (p.speed if hold else -p.speed) * p.grav

        # 3. vertical movement
        p.y, p.vy, hit, hit_block = self._sweep_y(p, p.vy * DT, near)
        if p.mode is Mode.WAVE and hit_block:
            return self._die()

        # 4. horizontal movement: any overlap with a block is a side hit
        p.x += p.speed * DT
        near = self._refresh_near()
        for i in near:
            o = objs[i]
            if o.t == ObjType.SOLID and self._overlaps(p, o):
                return self._die()

        # 5. hazards
        for i in near:
            o = objs[i]
            if o.t == ObjType.SPIKE and self._overlaps(p, o, HAZARD_SHRINK):
                return self._die()

        # 6. portals
        for i in near:
            o = objs[i]
            if i in self._used or o.t < ObjType.PORTAL_GRAVITY:
                continue
            if not self._overlaps(p, o):
                continue
            self._used |= {i}
            if o.t == ObjType.PORTAL_GRAVITY:
                p.grav = -p.grav if o.param == 2 else (1.0 if o.param == 0 else -1.0)
            elif o.t == ObjType.PORTAL_MODE:
                p.mode = Mode(o.param)
                if p.mode in (Mode.SHIP, Mode.WAVE):
                    p.vy = 0.0
            elif o.t == ObjType.PORTAL_SPEED:
                p.speed = SPEEDS[list(SPEEDS)[int(o.param)]]

        p.on_ground = self._probe_ground(p, near)

        # 7. bookkeeping
        self.tick += 1
        if p.x >= self.level.length:
            self.outcome = Outcome.COMPLETE
        elif self.tick >= self.max_ticks:
            self.outcome = Outcome.TIMEOUT
        return self.outcome

    def _die(self) -> Outcome:
        self.tick += 1
        self.outcome = Outcome.DEAD
        return self.outcome

    # ----------------------------------------------------------- observations
    @property
    def progress(self) -> float:
        return float(min(1.0, max(0.0, self.player.x / self.level.length)))

    def occupancy(self) -> np.ndarray:
        """Binary ``(GRID_H, GRID_W)`` map of blocking / lethal geometry ahead.

        Rings and pads are deliberately *not* drawn: the real game's vision stack
        cannot tell them apart from decoration reliably, so the reactive policy
        never learns to depend on something it will not see.  Sections built on
        ring timing are cracked by the tape search instead.
        """
        p = self.player
        grid = np.zeros((F.GRID_H, F.GRID_W), dtype=np.float32)
        x0 = p.x + p.size * 0.5
        span_x = F.REACH_X_BLOCKS * BLOCK
        # Vertically the strip covers the play area, exactly like the rows of a
        # screen grab do, so both sides of the feature contract agree.
        span_y = F.VERTICAL_BLOCKS * BLOCK
        y_top = span_y
        cw = span_x / F.GRID_W
        ch = span_y / F.GRID_H

        def fill(ox: float, ow: float, oy: float, oh: float) -> None:
            c0 = int(math.floor((ox - x0) / cw))
            c1 = int(math.floor((ox + ow - EPS - x0) / cw))
            # rows run top-down, so the object's top edge gives the first row
            r0 = int(math.floor((y_top - (oy + oh)) / ch))
            r1 = int(math.floor((y_top - oy - EPS) / ch))
            c0, c1 = max(0, c0), min(F.GRID_W - 1, c1)
            r0, r1 = max(0, r0), min(F.GRID_H - 1, r1)
            if c0 <= c1 and r0 <= r1:
                grid[r0 : r1 + 1, c0 : c1 + 1] = 1.0

        for i in self.level.near_idx(x0 + span_x * 0.5, reach=span_x * 0.5 + BLOCK * 2):
            o = self.level.objects[i]
            if o.t in (ObjType.SOLID, ObjType.SPIKE):
                fill(o.x, o.w, o.y, o.h)
        # The strip spans exactly [0, ceiling], so the ground below and the void
        # above fall outside it by construction; ``on_ground`` carries that
        # information instead.  Ground-level hazards sit in the bottom row.
        return grid

    def observe(self) -> Observation:
        p = self.player
        grid = self.occupancy()
        feats = F.pack(
            grid,
            y_norm=(p.y / max(self.level.ceiling, 1.0)) * 2.0 - 1.0,
            vy=p.vy,
            on_ground=p.on_ground,
            gravity_sign=p.grav,
            speed=p.speed,
            hold=p.hold,
            mode=p.mode,
            progress=self.progress,
        )
        return Observation(
            features=feats,
            progress=self.progress,
            alive=self.outcome is not Outcome.DEAD,
            tick=self.tick,
        )

    # ------------------------------------------------------------- rollouts
    def run_tape(
        self,
        tape: Iterable[int],
        *,
        snapshot: Snapshot | None = None,
        max_extra_ticks: int = 0,
    ) -> Attempt:
        """Replay an input tape and report what happened.

        This is the hot path for the tape search: no observations are built, so a
        tick costs only physics.
        """
        if snapshot is None:
            self.reset()
        else:
            self.restore(snapshot)
        start = self.progress
        arr = np.asarray(tape, dtype=np.uint8).reshape(-1)
        t0 = self.tick
        for a in arr:
            if self.step(bool(a)) is not Outcome.RUNNING:
                break
        else:
            for _ in range(max_extra_ticks):
                if self.step(False) is not Outcome.RUNNING:
                    break
        outcome = self.outcome
        if outcome is Outcome.RUNNING:
            outcome = Outcome.TIMEOUT
        return Attempt(
            actions=arr[: max(1, self.tick - t0)],
            start_progress=start,
            end_progress=self.progress,
            outcome=outcome,
            reward=0.0,
            ticks=self.tick - t0,
        )
