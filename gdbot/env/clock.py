"""A tick clock that holds its rate under an ordinary desktop OS.

``time.sleep`` on Windows has ~15 ms granularity by default, which is an entire
frame.  The clock asks for 1 ms timer resolution where it can, sleeps until just
before the deadline and spins for the rest.  Deadlines are computed from the
start time rather than from the previous tick, so lateness never accumulates
into drift.
"""

from __future__ import annotations

import platform
import time
from dataclasses import dataclass, field


@dataclass
class JitterStats:
    """How far actual ticks landed from their deadlines, in milliseconds."""

    samples: list[float] = field(default_factory=list)

    def add(self, late_ms: float) -> None:
        self.samples.append(late_ms)
        if len(self.samples) > 20000:
            del self.samples[:10000]

    @property
    def mean(self) -> float:
        return sum(self.samples) / len(self.samples) if self.samples else 0.0

    @property
    def p99(self) -> float:
        if not self.samples:
            return 0.0
        ordered = sorted(self.samples)
        return ordered[min(len(ordered) - 1, int(0.99 * len(ordered)))]

    def summary(self, tick_ms: float) -> str:
        return (
            f"tick jitter mean {self.mean:.2f} ms, p99 {self.p99:.2f} ms "
            f"(one tick = {tick_ms:.2f} ms)"
        )


class TickClock:
    """Waits for successive tick deadlines at ``rate`` Hz."""

    SPIN_MS = 1.5

    def __init__(self, rate: int = 60) -> None:
        self.rate = int(rate)
        self.period = 1.0 / self.rate
        self.jitter = JitterStats()
        self._t0 = 0.0
        self._n = 0
        self._raised_resolution = False
        if platform.system() == "Windows":  # pragma: no cover - Windows only
            try:
                import ctypes

                ctypes.WinDLL("winmm").timeBeginPeriod(1)
                self._raised_resolution = True
            except OSError:
                pass

    def start(self, at: float | None = None) -> None:
        """Declare that tick 0 happened at ``at`` (default: now)."""
        self._t0 = time.perf_counter() if at is None else at
        self._n = 0

    @property
    def tick(self) -> int:
        return self._n

    def wait_next(self) -> None:
        """Block until the next tick's deadline."""
        self._n += 1
        deadline = self._t0 + self._n * self.period
        remaining = deadline - time.perf_counter()
        if remaining > self.SPIN_MS / 1000.0:
            time.sleep(remaining - self.SPIN_MS / 1000.0)
        while time.perf_counter() < deadline:
            pass
        self.jitter.add((time.perf_counter() - deadline) * 1000.0)

    def close(self) -> None:
        if self._raised_resolution:  # pragma: no cover - Windows only
            import ctypes

            ctypes.WinDLL("winmm").timeEndPeriod(1)
            self._raised_resolution = False
