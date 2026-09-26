import json

import numpy as np
import pytest

import gdbot.features as F
from gdbot.cli import main
from gdbot.config import Config, loads
from gdbot.macro import export_macro, from_events, stats, to_events
from gdbot.store import RunStore
from gdbot.types import Mode

from pathlib import Path

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_relaxed_json_and_strict_keys():
    assert loads('{"a": 1, // note\n "b": [1, 2,], /* x */}') == {"a": 1, "b": [1, 2]}
    cfg = Config.from_dict({"train": {"population": 8}})
    assert cfg.train.population == 8 and cfg.tick_rate == 60
    with pytest.raises(ValueError):
        Config.from_dict({"trian": {}})


@pytest.mark.parametrize("path", sorted(CONFIGS.glob("*.json")), ids=lambda p: p.name)
def test_shipped_configs_load(path):
    Config.from_file(path)


def test_macro_events_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(50):
        tape = (rng.random(int(rng.integers(1, 300))) < 0.3).astype(np.uint8)
        assert np.array_equal(from_events(to_events(tape), len(tape)), tape)


def test_export_macro(tmp_path):
    tape = np.array([0, 1, 1, 0, 1], np.uint8)
    data = json.loads(export_macro(tape, tmp_path / "m.json", fps=60).read_text())
    assert data["ticks"] == 5 and data["fps"] == 60
    assert data["inputs"] == [
        {"frame": 1, "down": True}, {"frame": 3, "down": False},
        {"frame": 4, "down": True}, {"frame": 5, "down": False},
    ]
    assert stats(tape).presses == 2


def test_store_roundtrip(tmp_path):
    store = RunStore(tmp_path)
    tape = np.array([0, 1, 1, 0], np.uint8)
    store.save_tape("t", tape, progress=0.5)
    assert np.array_equal(store.load_tape("t"), tape)
    store.log("evt", value=np.float32(1.5))
    assert list(store.read_log())[0]["value"] == 1.5


def test_feature_vector_layout():
    grid = np.zeros((F.GRID_H, F.GRID_W), np.float32)
    v = F.pack(grid, y_norm=0.0, vy=0.0, on_ground=True, gravity_sign=-1.0, speed=311.58,
               hold=True, mode=Mode.WAVE, progress=0.25)
    assert v.shape == (F.N_FEATURES,)
    assert v[F.IDX_GRAVITY] == -1.0 and v[F.IDX_HOLD] == 1.0
    assert v[F.IDX_MODE + int(Mode.WAVE)] == 1.0 and v[F.IDX_PROGRESS] == 0.25


def test_cli_solve_play_export(tmp_path, capsys):
    run = str(tmp_path / "run")
    assert main(["solve", "--level", "tutorial", "--run-dir", run]) == 0
    assert main(["play", "--level", "tutorial", "--run-dir", run, "--runs", "2"]) == 0
    assert main(["export-macro", "--run-dir", run, "--out", str(tmp_path / "m.json")]) == 0
    assert main(["show", "--level", "skeletal_proxy"]) == 0
    out = capsys.readouterr().out
    assert "CLEARED" in out and "2/2 clean runs" in out
