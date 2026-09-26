"""Shared value types.

Geometry Dash has exactly one input: the jump button.  Every game mode is a
different interpretation of "is the button down right now", so the whole agent
works with a binary action.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import IntEnum
from typing import Iterable, Sequence

import numpy as np

HOLD = 1
RELEASE = 0
N_ACTIONS = 2


class Mode(IntEnum):
    """Player game mode. Portals switch between these."""

    CUBE = 0
    SHIP = 1
    BALL = 2
    WAVE = 3


class Outcome(IntEnum):
    """How an attempt ended."""

    RUNNING = 0
    DEAD = 1
    COMPLETE = 2
    TIMEOUT = 3


@dataclass(slots=True)
class Observation:
    """What the agent sees on one tick.

    ``features`` is the low-dimensional vector the policies consume.  It has the
    same layout whether it came from the simulator or from a screen grab of the
    real game, which is what lets a simulator-pretrained policy transfer.
    """

    features: np.ndarray
    progress: float
    alive: bool
    tick: int
    frame: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.features = np.asarray(self.features, dtype=np.float32).reshape(-1)


@dataclass(slots=True)
class StepResult:
    obs: Observation
    reward: float
    outcome: Outcome

    @property
    def done(self) -> bool:
        return self.outcome is not Outcome.RUNNING


@dataclass(slots=True)
class Attempt:
    """One complete run, from a start point to death or completion.

    ``actions`` is the frame-indexed input tape actually issued, which is what
    makes a successful attempt replayable in a deterministic game.
    """

    actions: np.ndarray
    start_progress: float
    end_progress: float
    outcome: Outcome
    reward: float
    ticks: int
    seed: int = 0

    def __post_init__(self) -> None:
        self.actions = np.asarray(self.actions, dtype=np.uint8).reshape(-1)

    @property
    def cleared(self) -> bool:
        return self.outcome is Outcome.COMPLETE

    @property
    def gained(self) -> float:
        return max(0.0, self.end_progress - self.start_progress)


@dataclass(slots=True)
class Segment:
    """A slice of the level the curriculum trains on.

    ``start`` / ``end`` are progress fractions in ``[0, 1]``.
    """

    index: int
    start: float
    end: float
    solved: bool = False
    attempts: int = 0
    best_progress: float = 0.0
    tape: np.ndarray | None = None

    def contains(self, progress: float) -> bool:
        return self.start <= progress < self.end

    def with_tape(self, tape: Sequence[int]) -> "Segment":
        return replace(self, tape=np.asarray(tape, dtype=np.uint8), solved=True)


@dataclass(slots=True)
class TrainingStats:
    """Rolling counters surfaced in the CLI and written to the run log."""

    attempts: int = 0
    deaths: int = 0
    clears: int = 0
    ticks: int = 0
    best_progress: float = 0.0
    history: list[float] = field(default_factory=list)

    def record(self, attempt: Attempt) -> None:
        self.attempts += 1
        self.ticks += attempt.ticks
        if attempt.cleared:
            self.clears += 1
        elif attempt.outcome is Outcome.DEAD:
            self.deaths += 1
        self.best_progress = max(self.best_progress, attempt.end_progress)
        self.history.append(attempt.end_progress)

    def recent_mean(self, n: int = 50) -> float:
        if not self.history:
            return 0.0
        window = self.history[-n:]
        return float(sum(window) / len(window))


def tape(actions: Iterable[int]) -> np.ndarray:
    """Normalise any iterable of 0/1 into a compact input tape."""
    arr = np.fromiter((1 if a else 0 for a in actions), dtype=np.uint8)
    return arr
