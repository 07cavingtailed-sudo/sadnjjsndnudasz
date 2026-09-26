"""The simulator as an :class:`~gdbot.env.base.AttemptEnv`."""

from __future__ import annotations

from typing import Any

from gdbot.env.base import AttemptEnv
from gdbot.sim.level import Level
from gdbot.sim.physics import TPS, Snapshot, World
from gdbot.types import Observation, Outcome, StepResult


class SimEnv(AttemptEnv):
    """Exact, fast and fully checkpointable."""

    supports_snapshots = True

    def __init__(self, level: Level, *, max_ticks: int = TPS * 240) -> None:
        self.world = World(level, max_ticks=max_ticks)
        self._last_progress = 0.0

    @property
    def level(self) -> Level:
        return self.world.level

    def reset(self) -> Observation:
        self.world.reset()
        self._last_progress = self.world.progress
        return self.world.observe()

    def step(self, action: int) -> StepResult:
        before = self.world.progress
        outcome = self.world.step(bool(action))
        gained = self.world.progress - before
        # A dense per-tick reward is only used by the torch PPO agent; the search
        # algorithms score whole attempts (see shape_reward).
        reward = gained * 100.0 + (-1.0 if outcome is Outcome.DEAD else 0.0)
        self._last_progress = self.world.progress
        return StepResult(obs=self.world.observe(), reward=reward, outcome=outcome)

    @property
    def progress(self) -> float:
        return self.world.progress

    def observe(self) -> Observation:
        return self.world.observe()

    def snapshot(self) -> Snapshot:
        return self.world.snapshot()

    def restore(self, snap: Snapshot) -> None:
        self.world.restore(snap)
        self._last_progress = self.world.progress

    def spawn_at(self, progress: float) -> Observation:
        self.world.spawn_at(progress)
        self._last_progress = self.world.progress
        return self.world.observe()

    def run_tape(self, tape, *, start: Any | None = None):
        # Delegate to the physics loop, which skips building observations and is
        # roughly an order of magnitude faster than the generic implementation.
        return self.world.run_tape(tape, snapshot=start)
