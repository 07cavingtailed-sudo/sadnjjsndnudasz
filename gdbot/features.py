"""The observation contract shared by the simulator and the real game.

A policy pretrained in the simulator is only useful against the real game if
both produce *the same* feature vector.  This module is the single definition of
that vector, so neither side can drift.

Layout (``N_FEATURES`` floats, all roughly in ``[-1, 1]``):

===============  ==========================================================
slice            meaning
===============  ==========================================================
``0 : GRID``     ``GRID_H x GRID_W`` binary occupancy of the strip ahead of the
                 player, row 0 = top.  1.0 means "something is there".
                 Horizontally the strip starts at the player and reaches
                 ``REACH_X_BLOCKS`` ahead; vertically it spans the whole play
                 area, so a row means the same thing as a band of screen rows
                 and the real game's vision does not have to track the player
                 icon to build it.
``GRID + 0``     player height, normalised by the play area height
``GRID + 1``     vertical velocity, normalised
``GRID + 2``     1.0 when standing on a surface
``GRID + 3``     gravity sign (+1 normal, -1 flipped)
``GRID + 4``     horizontal speed, normalised
``GRID + 5``     whether the button was held on the previous tick
``GRID + 6:10``  one-hot game mode (cube, ship, ball, wave)
``GRID + 10``    progress through the level, 0..1
===============  ==========================================================
"""

from __future__ import annotations

import numpy as np

from gdbot.types import Mode

#: Occupancy window resolution.  Small on purpose: the search algorithms work
#: far better on a few dozen inputs than on raw pixels.
GRID_W = 10
GRID_H = 7

#: How far ahead the strip looks, in Geometry Dash grid cells.
REACH_X_BLOCKS = 7.0
#: Height of the play area in grid cells.  Geometry Dash shows ten cells
#: vertically, which is also :attr:`gdbot.sim.level.Level.ceiling` by default.
VERTICAL_BLOCKS = 10.0

GRID_SIZE = GRID_W * GRID_H
N_SCALARS = 11
N_FEATURES = GRID_SIZE + N_SCALARS

#: Normalisation constants, chosen so typical values land inside [-1, 1].
VY_SCALE = 900.0
SPEED_SCALE = 600.0

IDX_Y = GRID_SIZE + 0
IDX_VY = GRID_SIZE + 1
IDX_ON_GROUND = GRID_SIZE + 2
IDX_GRAVITY = GRID_SIZE + 3
IDX_SPEED = GRID_SIZE + 4
IDX_HOLD = GRID_SIZE + 5
IDX_MODE = GRID_SIZE + 6
IDX_PROGRESS = GRID_SIZE + 10


def empty() -> np.ndarray:
    """An all-zero feature vector of the right shape."""
    return np.zeros(N_FEATURES, dtype=np.float32)


def pack(
    grid: np.ndarray,
    *,
    y_norm: float,
    vy: float,
    on_ground: bool,
    gravity_sign: float,
    speed: float,
    hold: bool,
    mode: Mode,
    progress: float,
) -> np.ndarray:
    """Assemble a feature vector from an occupancy grid plus player scalars."""
    if grid.shape != (GRID_H, GRID_W):
        raise ValueError(f"grid must be {(GRID_H, GRID_W)}, got {grid.shape}")
    out = empty()
    out[:GRID_SIZE] = np.clip(grid.reshape(-1), 0.0, 1.0)
    out[IDX_Y] = float(np.clip(y_norm, -2.0, 2.0))
    out[IDX_VY] = float(np.clip(vy / VY_SCALE, -2.0, 2.0))
    out[IDX_ON_GROUND] = 1.0 if on_ground else 0.0
    out[IDX_GRAVITY] = float(gravity_sign)
    out[IDX_SPEED] = float(speed / SPEED_SCALE)
    out[IDX_HOLD] = 1.0 if hold else 0.0
    out[IDX_MODE + int(mode)] = 1.0
    out[IDX_PROGRESS] = float(np.clip(progress, 0.0, 1.0))
    return out


def grid_of(features: np.ndarray) -> np.ndarray:
    """Recover the occupancy grid from a packed feature vector (for debugging)."""
    return np.asarray(features[:GRID_SIZE], dtype=np.float32).reshape(GRID_H, GRID_W)


def render_ascii(features: np.ndarray) -> str:
    """Human-readable dump of the occupancy window; handy when calibrating."""
    grid = grid_of(features)
    rows = ["".join("#" if v > 0.5 else "." for v in row) for row in grid]
    mode = Mode(int(np.argmax(features[IDX_MODE : IDX_MODE + 4])))
    rows.append(
        f"y={features[IDX_Y]:+.2f} vy={features[IDX_VY]:+.2f} "
        f"ground={features[IDX_ON_GROUND]:.0f} grav={features[IDX_GRAVITY]:+.0f} "
        f"mode={mode.name.lower()} progress={features[IDX_PROGRESS]:.3f}"
    )
    return "\n".join(rows)
