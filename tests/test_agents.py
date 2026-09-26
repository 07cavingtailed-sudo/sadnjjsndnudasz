import numpy as np
import pytest

import gdbot.features as F
from gdbot.agents.beam import BeamConfig, beam_search
from gdbot.agents.cem import CEM, CEMConfig
from gdbot.agents.ga_tape import TapeGA, TapeGAConfig
from gdbot.agents.policy import MLPPolicy
from gdbot.sim.generate import generate
from gdbot.sim.physics import World
from gdbot.types import Outcome


# ---------------------------------------------------------------- beam search
def test_beam_finds_a_tape_that_replays(spikes):
    result = beam_search(World(spikes), BeamConfig(width=64, max_ticks=600))
    assert result.solved
    assert World(spikes).run_tape(result.tape).outcome is Outcome.COMPLETE


def test_beam_resumes_from_a_snapshot(spikes):
    w = World(spikes)
    ref = beam_search(w, BeamConfig(width=64, max_ticks=600))
    w.reset()
    for a in ref.tape[:120]:
        w.step(bool(a))
    snap = w.snapshot()
    tail = beam_search(w, BeamConfig(width=64, max_ticks=600), start=snap)
    assert tail.solved
    full = np.concatenate([ref.tape[:120], tail.tape])
    assert World(spikes).run_tape(full).outcome is Outcome.COMPLETE


def test_beam_reports_best_progress_when_stuck():
    from gdbot.sim.level import BLOCK, Level, Obj, ObjType

    # a wall three blocks tall cannot be jumped
    lv = Level("wall", length=20 * BLOCK).add(Obj(ObjType.SOLID, 8 * BLOCK, 0.0, w=BLOCK, h=3 * BLOCK))
    result = beam_search(World(lv), BeamConfig(width=32, max_ticks=400))
    assert not result.solved
    assert 0.2 < result.progress < 0.45


# --------------------------------------------------------------------- tape GA
def test_mutation_never_touches_the_frozen_prefix():
    ga = TapeGA(lambda t: None, TapeGAConfig(freeze_before=40, seed=1))
    rng = np.random.default_rng(0)
    base = (rng.random(120) < 0.2).astype(np.uint8)
    for _ in range(2000):
        child = ga._mutate(base, death_tick=int(rng.integers(0, 120)), reach=300)
        assert child.size == base.size
        assert np.array_equal(child[:40], base[:40])


def test_suffix_shift_keeps_press_spacing():
    ga = TapeGA(lambda t: None, TapeGAConfig(seed=3))
    base = np.zeros(60, np.uint8)
    base[[10, 20, 27, 35]] = 1
    for _ in range(200):
        out = base.copy()
        ga._shift_suffix(out, 18, 0)
        presses = np.flatnonzero(np.diff(np.concatenate(([0], out))) == 1)
        later = presses[presses > 12]
        # the presses after the shifted edge keep their 7- and 8-tick spacing
        if len(later) == 3:
            assert list(np.diff(later)) == [7, 8]


def test_ga_solves_from_attempt_outcomes_only(spikes):
    world = World(spikes)
    ga = TapeGA(lambda t: world.run_tape(t), TapeGAConfig(population=24, generations=150, seed=0))
    result = ga.run(np.zeros(260, np.uint8))
    assert result.solved
    assert World(spikes).run_tape(result.tape).outcome is Outcome.COMPLETE


def test_ga_crosses_a_ring_chain():
    """Regression: a chain of three rings over a pit is a deceptive plateau.

    Hitting ring k earlier means ring k+1 must be hit earlier too, so every single
    edit makes the run worse.  Without the suffix shift the search sat at 7.6 %
    forever; with it this clears in a few thousand attempts.
    """
    level, _ = generate("medium", n_chunks=10, difficulty=0.7, seed=5)
    world = World(level, max_ticks=60 * 120)
    ga = TapeGA(lambda t: world.run_tape(t),
                TapeGAConfig(population=32, generations=250, goal_progress=1 / 12, seed=2))
    result = ga.run(np.zeros(200, np.uint8))
    assert result.progress >= 1 / 12


# ------------------------------------------------------------------------- CEM
def test_cem_maximises_a_simple_function():
    target = np.linspace(-1, 1, 10).astype(np.float32)
    cem = CEM(10, CEMConfig(population=32, init_sigma=1.0, seed=0))
    for _ in range(80):
        pop = cem.ask()
        cem.tell(-((pop - target) ** 2).sum(axis=1))
    assert np.abs(cem.mean - target).max() < 0.1


def test_cem_sigma_shrinks():
    # Regression: taking max(new_fit, old_sigma) ratcheted sigma upwards
    # (0.5 -> 4.3 over 40 generations) instead of converging.
    cem = CEM(50, CEMConfig(population=24, init_sigma=0.5, seed=0))
    rng = np.random.default_rng(0)
    for _ in range(40):
        cem.ask()
        cem.tell(rng.random(24))
    assert cem.sigma_mean < 0.5


# ---------------------------------------------------------------------- policy
def test_policy_params_roundtrip_and_determinism():
    pol = MLPPolicy(hidden=(16,), seed=0)
    assert pol.n_params == F.N_FEATURES * 16 + 16 + 16 * 2 + 2
    params = np.random.default_rng(1).standard_normal(pol.n_params).astype(np.float32)
    pol.set_params(params)
    assert np.array_equal(pol.get_params(), params)
    x = np.random.default_rng(2).standard_normal(F.N_FEATURES).astype(np.float32)
    assert pol.act(x) == pol.act(x) == pol.clone().act(x)
    with pytest.raises(ValueError):
        pol.set_params(params[:-1])
