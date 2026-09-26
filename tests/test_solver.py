import numpy as np

from gdbot.config import Config
from gdbot.env import SimEnv
from gdbot.env.base import AttemptEnv
from gdbot.sim.level import Level
from gdbot.solver import Solver
from gdbot.store import RunStore
from gdbot.types import Outcome
from gdbot.cli import LEVELS


class RealGameRules(AttemptEnv):
    """The simulator restricted to what the real game allows: no snapshots, no spawning."""

    supports_snapshots = False

    def __init__(self, level):
        self._e = SimEnv(level)

    def reset(self):
        return self._e.reset()

    def step(self, a):
        return self._e.step(a)

    @property
    def progress(self):
        return self._e.progress

    def observe(self):
        return self._e.observe()

    def run_tape(self, tape, *, start=None):
        assert start is None, "the real game cannot start mid-level"
        return self._e.run_tape(tape)


def _cfg(tmp_path, **curriculum):
    return Config.from_dict({
        "run_dir": str(tmp_path),
        "max_ticks": 60 * 60,
        "curriculum": {"n_segments": 6, "verify_runs": 2, **curriculum},
    })


def test_simulator_path_clears_and_verifies(tmp_path):
    level = Level.load(LEVELS / "tutorial.json")
    level.solution = None
    cfg = _cfg(tmp_path)
    report = Solver(SimEnv(level, max_ticks=cfg.max_ticks), cfg, store=RunStore(tmp_path), verbose=False).solve()
    assert report.solved and report.verified == 2
    saved = RunStore(tmp_path).load_tape("best")
    assert np.array_equal(saved, report.tape)


def test_real_game_path_clears_with_attempts_only(tmp_path, spikes):
    cfg = _cfg(tmp_path, attempts_per_segment=2000)
    env = RealGameRules(spikes)
    solver = Solver(env, cfg, store=RunStore(tmp_path), verbose=False)
    report = solver.solve()
    assert report.method == "tape-ga"
    assert report.solved and report.verified == 2
    assert env.run_tape(report.tape).outcome is Outcome.COMPLETE


def test_seed_tape_is_used(tmp_path, spikes):
    # a known answer handed in as the "recorded human attempt" should need no search
    solution = np.zeros(300, np.uint8)
    for t in (22, 62, 103, 143):
        solution[t] = 1
    env = RealGameRules(spikes)
    assert env.run_tape(solution).outcome is Outcome.COMPLETE
    solver = Solver(env, _cfg(tmp_path), store=RunStore(tmp_path), verbose=False, seed_tape=solution)
    report = solver.solve()
    assert report.solved
    assert solver.stats.attempts < 40


def test_cem_pretraining_learns_the_tutorial(tmp_path):
    level = Level.load(LEVELS / "tutorial.json")
    level.solution = None
    cfg = _cfg(tmp_path)
    env = SimEnv(level, max_ticks=cfg.max_ticks)
    policy = Solver(env, cfg, store=RunStore(tmp_path), verbose=False).pretrain_policy(iterations=20)
    assert env.run_policy(policy, max_ticks=cfg.max_ticks).end_progress > 0.5
    assert RunStore(tmp_path).has_policy("policy")
