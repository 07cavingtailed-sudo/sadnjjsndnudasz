"""The attempt-based environment interface.

Geometry Dash does not offer a step-reset API; it offers attempts.  Modelling it
that way -- rather than forcing it into a per-step RL mould it does not fit -- is
what lets the identical solver run against the simulator and against the real
game, where the only things available are "press the button" and "watch the
progress bar".
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable, Sequence

import numpy as np

import gdbot.features as F
from gdbot.agents.policy import MLPPolicy
from gdbot.config import TrainConfig
from gdbot.types import Attempt, Observation, Outcome, StepResult


def shape_reward(attempt: Attempt, cfg: TrainConfig) -> float:
    """Turn an attempt into the single scalar the search algorithms optimise.

    Progress dominates, dying costs a little and dawdling costs a little, so that
    between two tapes that reach the same place the tighter one wins.
    """
    reward = cfg.progress_reward * attempt.gained
    reward -= cfg.tick_cost * attempt.ticks
    if attempt.outcome is Outcome.DEAD:
        reward -= cfg.death_penalty
    elif attempt.outcome is Outcome.COMPLETE:
        reward += cfg.clear_bonus
    return float(reward)


class AttemptEnv(ABC):
    """One episode = one attempt at (a section of) the level."""

    #: Length of the feature vector, fixed by :mod:`gdbot.features`.
    n_features: int = F.N_FEATURES
    #: Whether the environment can save and restore exact state.  True for the
    #: simulator; the real game only offers practice-mode checkpoints.
    supports_snapshots: bool = False

    # ---------------------------------------------------------------- core API
    @abstractmethod
    def reset(self) -> Observation:
        """Start a fresh attempt from the beginning of the level."""

    @abstractmethod
    def step(self, action: int) -> StepResult:
        """Hold (1) or release (0) the button for one tick."""

    @property
    @abstractmethod
    def progress(self) -> float:
        """How far through the level the current attempt has got, 0..1."""

    @abstractmethod
    def observe(self) -> Observation:
        """The current observation without advancing time."""

    # ------------------------------------------------------- optional capabilities
    def snapshot(self) -> Any:
        raise NotImplementedError(f"{type(self).__name__} cannot snapshot state")

    def restore(self, snap: Any) -> None:
        raise NotImplementedError(f"{type(self).__name__} cannot restore state")

    def spawn_at(self, progress: float) -> Observation:
        raise NotImplementedError(f"{type(self).__name__} cannot spawn mid-level")

    def close(self) -> None:
        """Release any resources (capture handles, held keys)."""

    def __enter__(self) -> "AttemptEnv":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------- rollouts
    def _begin(self, start: Any | None) -> None:
        if start is None:
            self.reset()
        else:
            self.restore(start)

    def run_tape(self, tape: Iterable[int], *, start: Any | None = None) -> Attempt:
        """Play a fixed input tape and report the attempt."""
        self._begin(start)
        arr = np.asarray(tape, dtype=np.uint8).reshape(-1)
        start_progress = self.progress
        ticks = 0
        outcome = Outcome.RUNNING
        for action in arr:
            result = self.step(int(action))
            ticks += 1
            outcome = result.outcome
            if result.done:
                break
        if outcome is Outcome.RUNNING:
            outcome = Outcome.TIMEOUT
        return Attempt(
            actions=arr[:ticks] if ticks else arr[:0],
            start_progress=start_progress,
            end_progress=self.progress,
            outcome=outcome,
            reward=0.0,
            ticks=ticks,
        )

    def run_hybrid(
        self,
        tape: Iterable[int],
        policy: MLPPolicy,
        *,
        max_ticks: int = 60 * 240,
        start: Any | None = None,
    ) -> Attempt:
        """Replay a known-good prefix, then hand control to the policy.

        This is how the curriculum proposes a seed for the next unsolved section
        when it cannot checkpoint: the real game has no way to drop you into the
        middle of a level with the right velocity, but replaying the inputs that
        got you there does exactly that.
        """
        self._begin(start)
        start_progress = self.progress
        actions: list[int] = []
        outcome = Outcome.RUNNING
        for action in np.asarray(tape, dtype=np.uint8).reshape(-1):
            actions.append(int(action))
            result = self.step(int(action))
            outcome = result.outcome
            if result.done:
                break
        if outcome is Outcome.RUNNING:
            for _ in range(max(0, max_ticks - len(actions))):
                action = policy.act(self.observe().features)
                actions.append(action)
                result = self.step(action)
                outcome = result.outcome
                if result.done:
                    break
        if outcome is Outcome.RUNNING:
            outcome = Outcome.TIMEOUT
        return Attempt(
            actions=np.asarray(actions, dtype=np.uint8),
            start_progress=start_progress,
            end_progress=self.progress,
            outcome=outcome,
            reward=0.0,
            ticks=len(actions),
        )

    def run_policy(
        self,
        policy: MLPPolicy,
        *,
        start: Any | None = None,
        max_ticks: int = 60 * 240,
        rng: np.random.Generator | None = None,
        temperature: float = 0.0,
    ) -> Attempt:
        """Let a reactive policy play an attempt, recording the tape it produced.

        With ``temperature`` at 0 the policy is greedy, so the attempt is
        reproducible and its tape can be handed straight to the tape search.
        """
        self._begin(start)
        start_progress = self.progress
        actions: list[int] = []
        outcome = Outcome.RUNNING
        for _ in range(max_ticks):
            obs = self.observe()
            if temperature > 0.0:
                if rng is None:
                    raise ValueError("temperature > 0 requires an rng")
                action = policy.act_stochastic(obs.features, rng, temperature)
            else:
                action = policy.act(obs.features)
            actions.append(action)
            result = self.step(action)
            outcome = result.outcome
            if result.done:
                break
        if outcome is Outcome.RUNNING:
            outcome = Outcome.TIMEOUT
        return Attempt(
            actions=np.asarray(actions, dtype=np.uint8),
            start_progress=start_progress,
            end_progress=self.progress,
            outcome=outcome,
            reward=0.0,
            ticks=len(actions),
        )
