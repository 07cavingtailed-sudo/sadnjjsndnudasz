import numpy as np
import pytest

import gdbot.features as F
from gdbot.config import VisionConfig
from gdbot.sim.level import BLOCK, Level, Obj, ObjType
from gdbot.sim.physics import World
from gdbot.types import Outcome
from gdbot.vision import AttemptTracker, estimate_player_y, occupancy_from_frame, read_progress_bar


def _bar(fill: float, width: int = 300) -> np.ndarray:
    bar = np.full((16, width, 3), 40, np.uint8)
    bar[:, : int(round(fill * width))] = 255
    return bar


@pytest.mark.parametrize("fill", [0.0, 0.1, 0.37, 0.5, 0.999, 1.0])
def test_progress_bar_reading(fill):
    assert read_progress_bar(_bar(fill)) == pytest.approx(fill, abs=1 / 300)


def test_progress_bar_ignores_bright_ui_detached_from_the_fill():
    bar = _bar(0.2)
    bar[:, 250:270] = 255  # e.g. the attempt counter drawn over the bar's area
    assert read_progress_bar(bar) == pytest.approx(0.2, abs=1 / 300)


def test_tracker_death_completion_and_stall():
    t = AttemptTracker(VisionConfig(stall_ticks=4, complete_hold_ticks=3))
    assert [t.update(p) for p in (0.0, 0.1, 0.2)] == [Outcome.RUNNING] * 3
    assert t.update(0.0) is Outcome.DEAD and t.peak == pytest.approx(0.2)

    t.reset()
    outcomes = [t.update(p) for p in (0.5, 0.996, 0.997, 0.998)]
    assert outcomes[-1] is Outcome.COMPLETE

    t.reset()
    # a frozen bar before any progress is a paused/menu screen, not a death
    assert all(t.update(0.0) is Outcome.RUNNING for _ in range(10))
    t.update(0.3)
    assert [t.update(0.3) for _ in range(4)][-1] is Outcome.DEAD


def test_occupancy_sees_hard_edges_not_smooth_backgrounds():
    frame = np.tile(np.linspace(20, 80, 360, dtype=np.float32)[:, None, None], (1, 640, 3)).astype(np.uint8)
    assert occupancy_from_frame(frame, VisionConfig(edge_threshold=40)).sum() == 0
    frame[250:310, 300:360] = 255
    grid = occupancy_from_frame(frame, VisionConfig(edge_threshold=40))
    assert grid.shape == (F.GRID_H, F.GRID_W)
    assert grid.sum() > 0 and grid[: F.GRID_H // 2].sum() == 0  # object is in the lower half


def test_player_height_estimate_is_normalised():
    frame = np.zeros((300, 400, 3), np.uint8)
    frame[240:270, 95:105] = 255  # a bright square near the bottom, in the player column
    y = estimate_player_y(frame)
    assert -1.0 <= y <= 1.0 and y < 0.0


def test_simulator_grid_puts_ground_hazards_in_the_bottom_row():
    lv = Level("g", length=30 * BLOCK).add(Obj(ObjType.SPIKE, 3 * BLOCK, 0.0))
    grid = F.grid_of(World(lv).observe().features)
    assert grid[-1].sum() > 0 and grid[:-1].sum() == 0
