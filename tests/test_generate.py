import pytest

from gdbot.sim.generate import generate
from gdbot.sim.physics import TPS, World
from gdbot.types import Outcome


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_generated_levels_ship_a_working_solution(seed):
    # Regression: chunk tapes used to be concatenated from snapshots that (a) kept
    # the temporary finish line's COMPLETE outcome and (b) predated full-height
    # portals placed later -- both produced solutions that died on replay.
    level, report = generate("t", n_chunks=5, difficulty=0.6, seed=seed)
    assert report.chunks_accepted == 5
    attempt = World(level, max_ticks=TPS * 400).run_tape(level.solution)
    assert attempt.outcome is Outcome.COMPLETE


def test_generation_is_deterministic():
    a, _ = generate("t", n_chunks=4, difficulty=0.5, seed=9)
    b, _ = generate("t", n_chunks=4, difficulty=0.5, seed=9)
    assert [o.as_tuple() for o in a.objects] == [o.as_tuple() for o in b.objects]
    assert a.solution == b.solution
