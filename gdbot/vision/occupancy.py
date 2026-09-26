"""Building the occupancy grid from a frame.

This is the least reliable part of the whole system and it is written to be
honest about that.  Geometry Dash has animated backgrounds, glow, moving
decoration and user-chosen colour channels; no fixed rule separates "obstacle"
from "scenery" in every level.  What works well enough is edge density: hazards
and blocks are drawn with hard outlines, backgrounds are comparatively smooth.

Because it is unreliable, nothing load-bearing depends on it.  It feeds the
reactive policy, which is a *proposer*.  The tape search that actually clears the
level reads only the progress bar.
"""

from __future__ import annotations

import numpy as np

import gdbot.features as F
from gdbot.config import VisionConfig

#: Geometry Dash keeps the player at a fixed fraction of the screen width.
PLAYER_X_FRAC = 0.25


def _gradient_magnitude(gray: np.ndarray) -> np.ndarray:
    """Cheap Sobel-ish edge magnitude in plain numpy (no opencv needed)."""
    gy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
    gx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
    return gx + gy


def _block_max(arr: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Max-pool ``arr`` down to ``(rows, cols)`` for any input size.

    Max rather than mean: a thin spike occupies few pixels in a cell but must
    still light the cell up.  Averaging would hide exactly the objects that kill.
    """
    h, w = arr.shape
    row_edges = np.linspace(0, h, rows + 1).astype(int)
    col_edges = np.linspace(0, w, cols + 1).astype(int)
    out = np.zeros((rows, cols), dtype=np.float32)
    for r in range(rows):
        r0, r1 = row_edges[r], max(row_edges[r] + 1, row_edges[r + 1])
        band = arr[r0:r1]
        for c in range(cols):
            c0, c1 = col_edges[c], max(col_edges[c] + 1, col_edges[c + 1])
            cell = band[:, c0:c1]
            if cell.size:
                out[r, c] = cell.max()
    return out


def occupancy_from_frame(
    frame_rgb: np.ndarray,
    cfg: VisionConfig | None = None,
    *,
    player_x_frac: float = PLAYER_X_FRAC,
) -> np.ndarray:
    """``(GRID_H, GRID_W)`` binary occupancy of the strip ahead of the player.

    The strip is a fixed rectangle on screen: horizontally from the player's
    column forward, vertically the whole play area.  That is the same region the
    simulator draws, which is what makes a simulator-trained policy applicable.
    """
    cfg = cfg or VisionConfig()
    if frame_rgb.ndim != 3:
        raise ValueError(f"expected an HxWx3 frame, got {frame_rgb.shape}")
    h, w = frame_rgb.shape[:2]
    x0 = int(player_x_frac * w)
    x1 = min(w, x0 + max(1, int(cfg.ray_reach * w)))
    strip = frame_rgb[:, x0:x1, :3].astype(np.float32)
    gray = strip.mean(axis=2)
    if gray.size == 0:
        return np.zeros((F.GRID_H, F.GRID_W), dtype=np.float32)
    magnitude = _gradient_magnitude(gray)
    pooled = _block_max(magnitude, F.GRID_H, F.GRID_W)
    return (pooled >= cfg.edge_threshold).astype(np.float32)


def estimate_player_y(
    frame_rgb: np.ndarray,
    *,
    player_x_frac: float = PLAYER_X_FRAC,
    band_frac: float = 0.035,
) -> float:
    """Rough player height in ``[-1, 1]`` (-1 = bottom of the play area).

    Found as the most-textured row inside the narrow column band where the game
    keeps the player.  It is an estimate; when it is wrong the policy loses a
    scalar, and the tape search does not care.
    """
    h, w = frame_rgb.shape[:2]
    cx = int(player_x_frac * w)
    half = max(2, int(band_frac * w))
    band = frame_rgb[:, max(0, cx - half) : min(w, cx + half), :3].astype(np.float32).mean(axis=2)
    if band.size == 0:
        return 0.0
    row_energy = _gradient_magnitude(band).sum(axis=1)
    row = int(np.argmax(row_energy))
    # rows run top-down; features expect -1 at the bottom
    return float(1.0 - 2.0 * row / max(1, h - 1))
