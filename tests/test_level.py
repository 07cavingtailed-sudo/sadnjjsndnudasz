import numpy as np
import pytest

from gdbot.sim.level import BLOCK, Level, Obj, ObjType, _rle_decode, _rle_encode
from gdbot.sim.physics import TPS, World
from gdbot.types import Outcome
from gdbot.cli import LEVELS


def test_json_roundtrip(tmp_path, spikes):
    spikes.solution = [0, 0, 1, 1, 0, 1]
    path = tmp_path / "lv.json"
    spikes.save(path)
    back = Level.load(path)
    assert back.length == spikes.length
    assert [o.as_tuple() for o in back.objects] == [o.as_tuple() for o in spikes.objects]
    assert back.solution == spikes.solution


@pytest.mark.parametrize("tape", [[], [0], [1], [1, 1, 0, 0, 0, 1], [0] * 50 + [1] * 3])
def test_rle_roundtrip(tape):
    assert _rle_decode(_rle_encode(tape)) == tape


def test_wide_objects_are_found_from_every_bucket_they_span():
    lv = Level("wide", length=100 * BLOCK).add(Obj(ObjType.SOLID, 10 * BLOCK, 0.0, w=40 * BLOCK))
    for x in np.linspace(10 * BLOCK, 50 * BLOCK, 9):
        assert 0 in lv.near_idx(float(x))


@pytest.mark.parametrize("path", sorted(LEVELS.glob("*.json")), ids=lambda p: p.stem)
def test_bundled_levels_are_solvable(path):
    """The stored author solution must clear every bundled level from a cold start."""
    level = Level.load(path)
    assert level.solution, f"{path.name} ships without a solution"
    attempt = World(level, max_ticks=TPS * 400).run_tape(level.solution)
    assert attempt.outcome is Outcome.COMPLETE
