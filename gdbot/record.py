"""Recording a human's attempts, so the bot can start from how *you* play.

Every recorded attempt is a tape plus the progress it reached.  The best one is a
far better seed for the tape search than anything a freshly initialised policy
proposes: it already clears the sections you can clear, and the search only has
to fix the part where you died.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np

from gdbot.config import Config
from gdbot.env.gd_env import GDEnv
from gdbot.inputs.keys import NullPresser
from gdbot.types import Outcome


@dataclass
class Recording:
    tape: np.ndarray
    progress: float
    outcome: Outcome


class _KeyWatcher:
    """Tracks whether the jump key is physically held, via pynput."""

    def __init__(self, key: str) -> None:
        try:
            from pynput import keyboard
        except ImportError as exc:  # pragma: no cover - depends on the host
            raise RuntimeError("recording needs 'pynput' (pip install pynput)") from exc
        self._keyboard = keyboard
        self._target = {"space": keyboard.Key.space, "up": keyboard.Key.up,
                        "enter": keyboard.Key.enter}.get(key, keyboard.KeyCode.from_char(key))
        self.held = False
        self._lock = threading.Lock()
        self._listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)

    def _on_press(self, k) -> None:
        if k == self._target:
            with self._lock:
                self.held = True

    def _on_release(self, k) -> None:
        if k == self._target:
            with self._lock:
                self.held = False

    def __enter__(self) -> "_KeyWatcher":
        self._listener.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._listener.stop()


def record_attempts(cfg: Config, attempts: int = 5, *, verbose: bool = True) -> list[Recording]:  # pragma: no cover - needs the game
    """Watch the player for ``attempts`` runs and return what they did.

    The environment is built with a presser that sends nothing -- the human is
    playing -- but it still does the attempt bookkeeping: tick 0 from the camera
    cut, ticks from the clock, death and completion from the progress bar.
    """
    recordings: list[Recording] = []
    env = GDEnv(cfg, presser=NullPresser(cfg.inputs.key), live_run_timeout=900.0)
    try:
        with _KeyWatcher(cfg.inputs.key) as keys:
            for i in range(attempts):
                if verbose:
                    print(f"  attempt {i + 1}/{attempts}: play when the level restarts...", flush=True)
                env.reset()
                tape: list[int] = []
                while True:
                    tape.append(int(keys.held))
                    result = env.step(0)  # NullPresser: observe only
                    if result.done:
                        break
                rec = Recording(np.asarray(tape, dtype=np.uint8), env.progress, result.outcome)
                recordings.append(rec)
                if verbose:
                    print(f"    -> {rec.outcome.name.lower()} at {rec.progress * 100:.1f}%", flush=True)
    finally:
        env.close()
    return recordings
