"""Guided screen calibration.

Measuring rectangles on a screenshot and typing them into a JSON file is where a
first-time user gets stuck, so this does it for them: take a screenshot while
the level is running, have the user click the two ends of the progress bar and
two corners of the game, then work out the rest from the pixels and write it
into the config file.

The only things the clicks cannot give precisely are the bar's thickness and its
fill colour, and those are read off the screenshot: the colour just inside the
bar's left end *is* the fill (the level is running, so the bar is partly
filled), and the band of rows around the click that share that colour is the
bar's inside.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

#: A bar thicker than this is not a bar: the click probably hit the background.
MAX_BAR_HEIGHT = 60

PROMPTS = (
    "1/4: щёлкните по ЛЕВОМУ краю полосы прогресса (вверху экрана)",
    "2/4: щёлкните по ПРАВОМУ краю полосы прогресса",
    "3/4: щёлкните по ЛЕВОМУ ВЕРХНЕМУ углу игрового поля",
    "4/4: щёлкните по ПРАВОМУ НИЖНЕМУ углу игрового поля",
)


@dataclass
class BarFit:
    rect: list[int]
    fill_rgb: list[int]
    #: False when the picture does not look like a partly filled bar, e.g. the
    #: screenshot was taken at 0 % or the clicks missed the bar.
    confident: bool
    reason: str = ""


def _color_at(img: np.ndarray, x: int, y: int, r: int = 1) -> np.ndarray:
    h, w = img.shape[:2]
    patch = img[max(0, y - r) : min(h, y + r + 1), max(0, x - r) : min(w, x + r + 1), :3]
    return np.median(patch.reshape(-1, 3), axis=0)


def fit_progress_bar(
    img: np.ndarray, left: Sequence[int], right: Sequence[int], *, tolerance: float = 40.0
) -> BarFit:
    """Turn two clicks on the bar's ends into its inner rectangle and fill colour."""
    h, w = img.shape[:2]
    x0, x1 = sorted((int(left[0]), int(right[0])))
    x0, x1 = max(0, x0), min(w - 1, x1)
    y = int(round((int(left[1]) + int(right[1])) / 2))
    y = max(0, min(h - 1, y))
    if x1 - x0 < 20:
        return BarFit([x0, y, max(1, x1 - x0), 1], [255, 255, 255], False, "clicks are too close together")

    # Just inside the left end: filled, because the level is running.
    probe_x = x0 + max(3, (x1 - x0) // 40)
    fill = _color_at(img, probe_x, y)

    def matches(yy: int) -> bool:
        return bool(np.abs(img[yy, probe_x, :3].astype(np.float32) - fill).max() <= tolerance)

    top = y
    while top - 1 >= 0 and matches(top - 1) and y - top < MAX_BAR_HEIGHT:
        top -= 1
    bottom = y
    while bottom + 1 < h and matches(bottom + 1) and bottom - y < MAX_BAR_HEIGHT:
        bottom += 1
    height = bottom - top + 1
    # Use the middle of the band: its edges are anti-aliased into the border.
    trim = height // 4
    band_top, band_h = top + trim, max(1, height - 2 * trim)
    rect = [x0, band_top, x1 - x0, band_h]
    fill_rgb = [int(round(c)) for c in fill]

    if height > MAX_BAR_HEIGHT:
        return BarFit(rect, fill_rgb, False, "the colour at the left end fills a large area; the click probably missed the bar")
    empty = _color_at(img, x1 - max(3, (x1 - x0) // 40), y)
    if np.abs(empty - fill).max() <= tolerance:
        return BarFit(rect, fill_rgb, False, "both ends look the same; the bar was probably empty (0%) or full on the screenshot")
    return BarFit(rect, fill_rgb, True)


def play_area_from(corner_a: Sequence[int], corner_b: Sequence[int]) -> list[int]:
    xa, xb = sorted((int(corner_a[0]), int(corner_b[0])))
    ya, yb = sorted((int(corner_a[1]), int(corner_b[1])))
    return [xa, ya, max(1, xb - xa), max(1, yb - ya)]


def update_config_text(text: str, values: dict[str, list[int]]) -> str:
    """Replace ``"key": [...]`` entries in a config file, keeping its comments.

    Rewriting the file through ``json.dumps`` would throw away every comment that
    explains the settings, so the values are swapped in place instead.
    """
    for key, value in values.items():
        pattern = re.compile(r'("%s"\s*:\s*)\[[^\]]*\]' % re.escape(key))
        text, n = pattern.subn(lambda m: m.group(1) + json.dumps(value), text, count=1)
        if n == 0:
            raise KeyError(f"the config has no {key!r} entry to update")
    return text


def pick_points(png_path: str | Path, prompts: Sequence[str] = PROMPTS) -> list[tuple[int, int]] | None:  # pragma: no cover - GUI
    """Show the screenshot and collect one click per prompt. None if cancelled."""
    import tkinter as tk

    root = tk.Tk()
    root.title("gdbot: разметка экрана (Esc - отмена)")
    img = tk.PhotoImage(file=str(png_path))
    factor = max(
        1,
        math.ceil(max(img.width() / (root.winfo_screenwidth() * 0.9),
                      img.height() / (root.winfo_screenheight() * 0.8))),
    )
    shown = img.subsample(factor) if factor > 1 else img
    label = tk.Label(root, text=prompts[0], font=("Segoe UI", 14, "bold"), fg="white", bg="#202020", pady=6)
    label.pack(fill="x")
    canvas = tk.Canvas(root, width=shown.width(), height=shown.height(), highlightthickness=0, cursor="crosshair")
    canvas.pack()
    canvas.create_image(0, 0, image=shown, anchor="nw")
    points: list[tuple[int, int]] = []

    def on_click(event: "tk.Event") -> None:
        points.append((event.x * factor, event.y * factor))
        canvas.create_oval(event.x - 6, event.y - 6, event.x + 6, event.y + 6, outline="#ff3030", width=3)
        if len(points) < len(prompts):
            label.config(text=prompts[len(points)])
        else:
            label.config(text="Готово")
            root.after(500, root.destroy)

    canvas.bind("<Button-1>", on_click)
    root.bind("<Escape>", lambda _e: root.destroy())
    root.lift()
    root.attributes("-topmost", True)
    root.after(300, lambda: root.attributes("-topmost", False))
    root.mainloop()
    return points if len(points) == len(prompts) else None
