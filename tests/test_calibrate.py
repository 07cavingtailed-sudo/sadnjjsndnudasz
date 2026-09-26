import numpy as np
import pytest

from gdbot.calibrate import fit_progress_bar, play_area_from, update_config_text
from gdbot.config import loads
from gdbot.vision import read_progress_bar


def _screen(fill_to: int = 870) -> np.ndarray:
    """1080p frame with a bordered bar at y 30..47; filled green up to ``fill_to``."""
    img = np.full((1080, 1920, 3), 25, np.uint8)
    img[29:49, 659:1261] = 120
    img[30:48, 660:1260] = 40
    img[30:48, 660:fill_to] = (80, 230, 90)
    return img


def test_bar_fit_finds_rectangle_and_fill_colour():
    img = _screen()
    fit = fit_progress_bar(img, (663, 36), (1256, 41))
    assert fit.confident
    assert fit.fill_rgb == [80, 230, 90]  # measured, not assumed to be white
    x, y, w, h = fit.rect
    assert 30 <= y and y + h <= 48  # strictly inside the border
    reading = read_progress_bar(img[y : y + h, x : x + w], fill_rgb=fit.fill_rgb, tolerance=70)
    assert reading == pytest.approx(0.35, abs=0.01)


def test_bar_fit_flags_an_empty_bar():
    fit = fit_progress_bar(_screen(fill_to=660), (663, 36), (1256, 41))
    assert not fit.confident and "empty" in fit.reason


def test_bar_fit_flags_a_click_that_missed_the_bar():
    fit = fit_progress_bar(_screen(), (663, 400), (1256, 400))  # middle of the level
    assert not fit.confident


def test_play_area_accepts_corners_in_any_order():
    assert play_area_from((1900, 1000), (20, 10)) == [20, 10, 1880, 990]


def test_config_update_keeps_comments_and_other_values():
    text = """{
  // how the game is seen
  "capture": {
    "play_area": [0, 0, 1920, 1080],   // whole screen
    "progress_bar": [660, 30, 600, 18],
    "bar_fill_rgb": [255, 255, 255],
  },
}"""
    new = update_config_text(text, {"progress_bar": [1, 2, 3, 4], "bar_fill_rgb": [9, 8, 7]})
    assert "// how the game is seen" in new and "// whole screen" in new
    data = loads(new)["capture"]
    assert data == {"play_area": [0, 0, 1920, 1080], "progress_bar": [1, 2, 3, 4], "bar_fill_rgb": [9, 8, 7]}
    with pytest.raises(KeyError):
        update_config_text(text, {"missing": [1]})
