"""The real Geometry Dash, driven through the screen and the keyboard.

What this can and cannot know
-----------------------------
It reads the progress bar (reliable) and an edge-density occupancy grid
(approximate).  It does not know the player's velocity, game mode or gravity;
those feature slots are estimated or left neutral.  It cannot save or restore
state, so every attempt starts from the beginning of the level -- the curriculum
copes with that by replaying the known-good prefix of the tape.

Timing is the hard part
-----------------------
A tape only reproduces a run if input *n* lands on game frame *n*.  Two things
make that true here:

* **Tick 0 is found from the picture, not the progress bar.**  When Geometry
  Dash restarts a level the camera cuts back to the start in a single frame, so
  the mean frame difference spikes.  That spike pins tick 0 to within one frame.
  The progress bar would only pin it to within a dozen frames: at the start of a
  two-minute level the bar has not yet filled its first pixel.
* **Ticks are scheduled from tick 0, not from each other**, by
  :class:`~gdbot.env.clock.TickClock`, so lateness never accumulates.

Even so, capture latency varies by machine.  :attr:`GDEnv.clock` records the
jitter, ``gdbot verify`` reports how many of N replays of the final tape clear,
and a tape learned on the real game is learned *with* that jitter -- fragile
timings fail and get mutated away, so what survives is inherently the tolerant
version of the solution.
"""

from __future__ import annotations

import time

import numpy as np

import gdbot.features as F
from gdbot.capture.screen import ScreenCapture
from gdbot.config import Config
from gdbot.env.base import AttemptEnv
from gdbot.env.clock import TickClock
from gdbot.inputs.keys import KeyPresser, make_presser
from gdbot.types import Mode, Observation, Outcome, StepResult
from gdbot.vision.occupancy import estimate_player_y, occupancy_from_frame
from gdbot.vision.progress import read_progress_bar
from gdbot.vision.state import AttemptTracker


class GameNotResponding(RuntimeError):
    """The screen is not changing the way a running level should."""


class GDEnv(AttemptEnv):
    """Attempts against the live game."""

    supports_snapshots = False

    #: Mean absolute frame difference (0-255) that counts as a camera cut.
    CUT_THRESHOLD = 38.0
    #: How long to wait for a restart before giving up, seconds.
    RESTART_TIMEOUT = 12.0
    #: How long holding jump may take to end an unwanted run, seconds.
    ABORT_TIMEOUT = 20.0
    #: After a clear the game shows its level-complete screen and does not
    #: restart by itself, so someone has to press restart; wait much longer.
    AFTER_CLEAR_TIMEOUT = 180.0

    def __init__(
        self,
        cfg: Config,
        *,
        capture: ScreenCapture | None = None,
        presser: KeyPresser | None = None,
    ) -> None:
        self.cfg = cfg
        self.capture = capture or ScreenCapture(cfg.capture)
        self.presser = presser or make_presser(cfg.inputs)
        self.tracker = AttemptTracker(cfg.vision)
        self.clock = TickClock(cfg.tick_rate)
        self._play: np.ndarray | None = None
        self._prev_gray: np.ndarray | None = None
        self._reading = 0.0
        self._y = -1.0
        self._vy = 0.0
        self._tick = 0

    # --------------------------------------------------------------- reading
    def _grab(self) -> float:
        play, bar = self.capture.grab()
        self._play = play
        cc = self.cfg.capture
        self._reading = read_progress_bar(
            bar, fill_rgb=cc.bar_fill_rgb, tolerance=cc.bar_fill_tolerance
        )
        return self._reading

    def _frame_delta(self) -> float:
        """Mean absolute difference from the previous frame, downsampled."""
        if self._play is None:
            return 0.0
        gray = self._play[::8, ::8, :3].astype(np.float32).mean(axis=2)
        prev, self._prev_gray = self._prev_gray, gray
        if prev is None or prev.shape != gray.shape:
            return 0.0
        return float(np.abs(gray - prev).mean())

    # ------------------------------------------------------------ attempt flow
    def _end_live_run(self) -> None:
        """Get rid of a run that is still going: hold jump until it dies.

        Geometry Dash has no restart key.  Holding jump makes a cube bounce into
        the first obstacle within a few seconds in practically every level, which
        is far faster than waiting for the run to end on its own.
        """
        deadline = time.perf_counter() + self.ABORT_TIMEOUT
        start = self._grab()
        self.presser.set(True)
        try:
            while time.perf_counter() < deadline:
                time.sleep(0.01)
                reading = self._grab()
                if start - reading > self.cfg.vision.death_progress_drop:
                    return
                start = max(start, reading)
        finally:
            self.presser.release()
        raise GameNotResponding(
            "could not end the current run by holding jump. Check that the game "
            "window has focus and that the progress bar region is calibrated "
            "(gdbot calibrate)."
        )

    def _wait_for_restart(self, timeout: float | None = None) -> float:
        """Block until a new attempt begins; return the perf_counter of tick 0."""
        deadline = time.perf_counter() + (self.RESTART_TIMEOUT if timeout is None else timeout)
        self._prev_gray = None
        saw_low = False
        while time.perf_counter() < deadline:
            reading = self._grab()
            delta = self._frame_delta()
            now = time.perf_counter()
            if reading <= 0.01:
                saw_low = True
            # The camera cut back to the start of the level: that frame is tick 0.
            if delta >= self.CUT_THRESHOLD and reading <= 0.02:
                return now
            # Fallback: the bar has started filling again after being empty.
            if saw_low and 0.0 < reading < 0.05:
                return now
            time.sleep(0.002)
        raise GameNotResponding(
            "no new attempt started. Make sure auto-retry is enabled in Geometry "
            "Dash's settings, the level is open (not paused), and the capture "
            "regions are calibrated (gdbot calibrate)."
        )

    def reset(self) -> Observation:
        self.presser.release()
        timeout = None
        if self.tracker.outcome is Outcome.COMPLETE:
            print("  level cleared -- press Restart in the game to start the next attempt", flush=True)
            timeout = self.AFTER_CLEAR_TIMEOUT
        else:
            reading = self._grab()
            if reading > 0.02 and self.tracker.outcome is Outcome.RUNNING:
                self._end_live_run()
        tick0 = self._wait_for_restart(timeout)
        self.clock.start(at=tick0)
        self._tick = 0
        self._y, self._vy = -1.0, 0.0
        self.tracker.reset(self._reading)
        return self.observe()

    def step(self, action: int) -> StepResult:
        before = self.progress
        self.presser.set(bool(action))
        self.clock.wait_next()
        self._tick += 1
        reading = self._grab()
        outcome = self.tracker.update(reading)
        if outcome is not Outcome.RUNNING:
            self.presser.release()
        # No features here: the tape search -- nearly every real attempt -- never
        # looks at them, and edge detection on a full frame every tick would eat
        # into the 16 ms tick budget and make input timing worse.  Policies call
        # observe() when they actually need to look.
        obs = Observation(
            features=F.empty(),
            progress=self.progress,
            alive=outcome is not Outcome.DEAD,
            tick=self._tick,
            frame=self._play,
        )
        reward = (self.progress - before) * 100.0 - (1.0 if outcome is Outcome.DEAD else 0.0)
        return StepResult(obs=obs, reward=reward, outcome=outcome)

    @property
    def progress(self) -> float:
        # After a death the bar snaps to zero; the attempt's result is its peak.
        return self.tracker.peak

    def observe(self) -> Observation:
        if self._play is None:
            self._grab()
        assert self._play is not None
        # The grid is 7x10 cells: a quarter-resolution frame loses nothing that
        # matters and costs a sixteenth of the time.
        stride = max(1, self._play.shape[0] // 270)
        small = self._play[::stride, ::stride]
        grid = occupancy_from_frame(small, self.cfg.vision)
        y = estimate_player_y(small)
        # velocity from successive height estimates: y spans the play area, which
        # is VERTICAL_BLOCKS cells of 30 units
        half_height_units = F.VERTICAL_BLOCKS * 30.0 / 2.0
        self._vy = (y - self._y) * half_height_units * self.cfg.tick_rate
        self._y = y
        feats = F.pack(
            grid,
            y_norm=y,
            vy=self._vy,
            on_ground=abs(self._vy) < 30.0,
            gravity_sign=1.0,
            speed=311.58,
            hold=self.presser.held,
            mode=Mode.CUBE,
            progress=self.progress,
        )
        return Observation(
            features=feats,
            progress=self.progress,
            alive=self.tracker.outcome is not Outcome.DEAD,
            tick=self._tick,
            frame=self._play,
        )

    def close(self) -> None:
        try:
            self.presser.close()
        finally:
            self.clock.close()
            self.capture.close()
