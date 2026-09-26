"""Configuration objects and a dependency-free config file loader.

Config files are JSON with ``//`` and ``/* */`` comments and trailing commas
allowed, so they read like YAML without pulling in a YAML parser.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Type, TypeVar, get_type_hints

_LINE_COMMENT = re.compile(r"(?<![:\\])//[^\n\r]*")
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.S)
_TRAILING_COMMA = re.compile(r",(\s*[}\]])")

T = TypeVar("T")


def loads(text: str) -> Any:
    """Parse relaxed JSON (comments + trailing commas)."""
    stripped = _BLOCK_COMMENT.sub("", text)
    stripped = _LINE_COMMENT.sub("", stripped)
    stripped = _TRAILING_COMMA.sub(r"\1", stripped)
    return json.loads(stripped)


def load_file(path: str | Path) -> Any:
    return loads(Path(path).read_text(encoding="utf-8"))


def _build(cls: Type[T], data: Mapping[str, Any]) -> T:
    """Recursively build a (possibly nested) dataclass from a mapping."""
    if not is_dataclass(cls):
        raise TypeError(f"{cls!r} is not a dataclass")
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"unknown option(s) for {cls.__name__}: {sorted(unknown)}")
    # ``from __future__ import annotations`` stores field types as strings, so
    # resolve them before checking for nested dataclasses.
    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for name, value in data.items():
        ftype = hints.get(name)
        if is_dataclass(ftype) and isinstance(value, Mapping):
            kwargs[name] = _build(ftype, value)
        else:
            kwargs[name] = value
    return cls(**kwargs)


@dataclass
class CaptureConfig:
    """Where on screen the game is, and how we read its state.

    All rectangles are ``[x, y, w, h]`` in screen pixels.  ``gdbot calibrate``
    fills these in interactively; they are the only numbers that depend on the
    user's monitor.
    """

    monitor: int = 1
    play_area: list[int] = field(default_factory=lambda: [0, 0, 1920, 1080])
    progress_bar: list[int] = field(default_factory=lambda: [660, 30, 600, 18])
    #: Colour distance below which a progress-bar pixel counts as "filled".
    bar_fill_tolerance: float = 70.0
    #: RGB of the filled part of the progress bar (GD default is near-white).
    bar_fill_rgb: list[int] = field(default_factory=lambda: [255, 255, 255])


@dataclass
class VisionConfig:
    """Thresholds for turning frames into game state."""

    #: A drop in progress larger than this means the attempt restarted (death).
    death_progress_drop: float = 0.02
    #: Progress at or above this, held for ``complete_hold_ticks``, means clear.
    complete_progress: float = 0.995
    complete_hold_ticks: int = 4
    #: Ticks with no progress change before we call the run stuck.
    stall_ticks: int = 45
    #: How far ahead of the player the rays reach, in fractions of screen width.
    ray_reach: float = 0.45
    #: Edge-detection threshold for obstacle detection (0-255 gradient).
    edge_threshold: float = 42.0


@dataclass
class InputConfig:
    """How we press the jump button."""

    backend: str = "auto"  # auto | sendinput | uinput | xdotool | pynput | sim
    key: str = "space"


@dataclass
class TrainConfig:
    """Learning-loop hyper-parameters."""

    #: CEM population and elite fraction.
    population: int = 24
    elite_frac: float = 0.25
    init_sigma: float = 0.5
    sigma_decay: float = 0.995
    min_sigma: float = 0.02
    #: Tape search (used against the real game): population per generation, and
    #: the probability that a child gets a second edit on top of its first.
    tape_population: int = 32
    tape_mutation: float = 0.04
    #: Hidden layer sizes of the numpy policy network.
    hidden: list[int] = field(default_factory=lambda: [64, 64])
    seed: int = 0
    #: Reward shaping.
    progress_reward: float = 100.0
    death_penalty: float = 1.0
    tick_cost: float = 0.0005
    clear_bonus: float = 50.0


@dataclass
class CurriculumConfig:
    """How the level is split into sections and how hard each is tried."""

    n_segments: int = 24
    #: Attempts to spend on one section before retreating one section back.
    attempts_per_segment: int = 1500
    #: Verify the stitched full-level tape this many times before declaring done.
    verify_runs: int = 3
    #: Give up on a section early -- and retreat -- when after this many
    #: generations no tape has got even one second of play past the end of the
    #: verified prefix.  That pattern means the prefix leaves the player in a
    #: state nothing can save, and the attempts it would burn are the longest
    #: ones of all: they replay the whole prefix first.
    doomed_generations: int = 15


@dataclass
class Config:
    name: str = "default"
    level: str = "gdbot/sim/levels/tutorial.json"
    run_dir: str = "runs/default"
    tick_rate: int = 60
    max_ticks: int = 60 * 180
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    inputs: InputConfig = field(default_factory=InputConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)

    @classmethod
    def from_file(cls, path: str | Path) -> "Config":
        return _build(cls, load_file(path))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Config":
        return _build(cls, data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
