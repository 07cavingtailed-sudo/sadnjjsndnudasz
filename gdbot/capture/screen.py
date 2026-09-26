"""Grabbing the game window.

One grab per tick, of the smallest rectangle that contains both regions of
interest, then sliced.  Two separate grabs would double the per-tick syscall
cost, and at 60 Hz the whole tick budget is 16 ms.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from gdbot.config import CaptureConfig


def union_rect(*rects: Sequence[int]) -> tuple[int, int, int, int]:
    """Smallest ``(x, y, w, h)`` containing all the given ``[x, y, w, h]``."""
    xs0 = min(r[0] for r in rects)
    ys0 = min(r[1] for r in rects)
    xs1 = max(r[0] + r[2] for r in rects)
    ys1 = max(r[1] + r[3] for r in rects)
    return xs0, ys0, xs1 - xs0, ys1 - ys0


class ScreenCapture:
    """Reads the play area and the progress bar out of one screen grab."""

    def __init__(self, cfg: CaptureConfig) -> None:
        try:
            import mss  # noqa: PLC0415 - optional dependency, game machine only
        except ImportError as exc:  # pragma: no cover - depends on the host
            raise RuntimeError(
                "screen capture needs 'mss'. Install the game extras:\n"
                "    pip install 'gdbot[game]'"
            ) from exc
        self.cfg = cfg
        self._sct = mss.mss()
        self.play = tuple(cfg.play_area)
        self.bar = tuple(cfg.progress_bar)
        self.region = union_rect(self.play, self.bar)
        # Rectangles are measured on the calibration screenshot, i.e. relative to
        # the chosen monitor; mss grabs in virtual-desktop coordinates, which only
        # coincide for a monitor whose corner is at (0, 0).
        monitor = self._sct.monitors[cfg.monitor]
        self._mon = {
            "left": monitor["left"] + self.region[0],
            "top": monitor["top"] + self.region[1],
            "width": self.region[2],
            "height": self.region[3],
        }

    def grab(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(play_area_rgb, progress_bar_rgb)`` as uint8 arrays."""
        raw = np.asarray(self._sct.grab(self._mon))  # BGRA
        frame = raw[:, :, 2::-1]  # -> RGB
        return self._slice(frame, self.play), self._slice(frame, self.bar)

    def _slice(self, frame: np.ndarray, rect: Sequence[int]) -> np.ndarray:
        ox, oy = self.region[0], self.region[1]
        x, y, w, h = rect
        return frame[y - oy : y - oy + h, x - ox : x - ox + w]

    def close(self) -> None:
        self._sct.close()

    def __enter__(self) -> "ScreenCapture":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
