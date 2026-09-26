"""The real-game environment, driven by a fake screen and a fake keyboard."""

import time

import numpy as np

from gdbot.config import Config
from gdbot.env.gd_env import GDEnv
from gdbot.inputs import NullPresser
from gdbot.types import Outcome


class FakeScreen:
    """Level restarts (camera cut) at 0.1 s; the bar fills at ``rate`` per second.

    If ``die_at`` is set the bar snaps back to zero there, like a death.
    """

    def __init__(self, rate: float = 3.0, die_at: float | None = None) -> None:
        self.t0: float | None = None
        self.rate = rate
        self.die_at = die_at

    def grab(self):
        now = time.perf_counter()
        if self.t0 is None:
            self.t0 = now
        el = now - self.t0
        play = np.full((72, 128, 3), 20 if el < 0.1 else 200, np.uint8)
        progress = 0.0 if el < 0.1 else min(1.0, (el - 0.1) * self.rate)
        if self.die_at is not None and progress >= self.die_at:
            progress = 0.0
        bar = np.zeros((8, 200, 3), np.uint8)
        bar[:, : int(progress * 200)] = 255
        return play, bar

    def close(self):
        pass


def _env(screen) -> GDEnv:
    cfg = Config.from_dict({"tick_rate": 120, "vision": {"complete_hold_ticks": 2}})
    return GDEnv(cfg, capture=screen, presser=NullPresser())


def test_tape_is_played_and_completion_detected():
    env = _env(FakeScreen(rate=3.0))
    tape = np.tile(np.array([0, 0, 1, 1], np.uint8), 200)
    attempt = env.run_tape(tape)
    assert attempt.outcome is Outcome.COMPLETE
    assert attempt.end_progress == 1.0
    presses = env.presser.events.count(True)
    assert presses == int(np.count_nonzero(np.diff(np.concatenate(([0], tape[: attempt.ticks]))) == 1))
    assert not env.presser.held  # key released once the attempt ends
    env.close()


def test_death_reports_the_peak_not_the_reset_bar():
    env = _env(FakeScreen(rate=3.0, die_at=0.4))
    attempt = env.run_tape(np.zeros(2000, np.uint8))
    assert attempt.outcome is Outcome.DEAD
    assert 0.35 < attempt.end_progress <= 0.41
    env.close()
