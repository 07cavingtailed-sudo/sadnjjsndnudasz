"""Deciding, from a stream of progress readings, when an attempt ends."""

from __future__ import annotations

from dataclasses import dataclass, field

from gdbot.config import VisionConfig
from gdbot.types import Outcome


@dataclass
class AttemptTracker:
    """Progress readings in, attempt outcomes out.

    Template matching on the death animation or the level-complete screen would
    need per-user calibration and would break on any texture pack.  Progress
    alone is enough: it only ever goes up during a run, so a drop means the run
    restarted, and it stops changing when the game is not playing.
    """

    cfg: VisionConfig = field(default_factory=VisionConfig)
    progress: float = 0.0
    peak: float = 0.0
    outcome: Outcome = Outcome.RUNNING
    _stalled: int = 0
    _at_end: int = 0

    def reset(self, progress: float = 0.0) -> None:
        self.progress = progress
        self.peak = progress
        self.outcome = Outcome.RUNNING
        self._stalled = 0
        self._at_end = 0

    def update(self, progress: float) -> Outcome:
        """Feed one reading and get the current outcome."""
        cfg = self.cfg
        previous = self.progress
        self.progress = progress
        self.peak = max(self.peak, progress)

        if progress >= cfg.complete_progress:
            self._at_end += 1
            if self._at_end >= cfg.complete_hold_ticks:
                self.outcome = Outcome.COMPLETE
                return self.outcome
        else:
            self._at_end = 0

        # The bar snapping backwards is the game having restarted the level.
        if previous - progress > cfg.death_progress_drop:
            self.outcome = Outcome.DEAD
            return self.outcome

        if abs(progress - previous) < 1e-6:
            self._stalled += 1
            if self._stalled >= cfg.stall_ticks:
                # Frozen bar with progress already made: death freeze or a menu.
                self.outcome = Outcome.DEAD if self.peak > 0.0 else Outcome.RUNNING
                return self.outcome
        else:
            self._stalled = 0

        self.outcome = Outcome.RUNNING
        return self.outcome
