"""Reading the progress bar."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def read_progress_bar(
    bar_rgb: np.ndarray,
    *,
    fill_rgb: Sequence[int] = (255, 255, 255),
    tolerance: float = 70.0,
    column_ratio: float = 0.5,
) -> float:
    """Fraction of the bar that is filled, in ``[0, 1]``.

    Measured as the length of the *leading run* of filled columns rather than the
    total count of them.  Geometry Dash draws other bright things near the bar
    (the attempt counter, the pause button), and counting every matching column
    would read those as progress; a leading run cannot.
    """
    if bar_rgb.ndim != 3 or bar_rgb.shape[2] < 3:
        raise ValueError(f"expected an HxWx3 image, got {bar_rgb.shape}")
    if bar_rgb.size == 0:
        return 0.0
    target = np.asarray(fill_rgb, dtype=np.int16).reshape(1, 1, 3)
    diff = np.abs(bar_rgb[:, :, :3].astype(np.int16) - target).max(axis=2)
    matched = diff <= tolerance
    filled_columns = matched.mean(axis=0) >= column_ratio
    if not filled_columns.any():
        return 0.0
    # length of the run that starts at the left edge
    if not filled_columns[0]:
        return 0.0
    gaps = np.flatnonzero(~filled_columns)
    run = int(gaps[0]) if gaps.size else filled_columns.size
    return float(run / filled_columns.size)
