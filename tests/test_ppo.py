import numpy as np
import pytest

torch = pytest.importorskip("torch")

import gdbot.features as F  # noqa: E402
from gdbot.agents.ppo import ActorCritic, PPOConfig, to_numpy_policy, train_ppo  # noqa: E402
from gdbot.env import SimEnv  # noqa: E402
from gdbot.sim.level import Level  # noqa: E402
from gdbot.cli import LEVELS  # noqa: E402


def test_numpy_export_matches_torch():
    model = ActorCritic(F.N_FEATURES, (32,))
    x = np.random.default_rng(0).standard_normal((8, F.N_FEATURES)).astype(np.float32)
    expected = model.actor(torch.as_tensor(x)).detach().numpy()
    assert np.allclose(to_numpy_policy(model, F.N_FEATURES).logits(x), expected, atol=1e-5)


def test_ppo_learns_past_the_first_spike():
    # Across seeds 0-3 the greedy policy first clears spike 1 at update 15-18;
    # 22 updates leaves margin without making this the slowest test by far.
    level = Level.load(LEVELS / "tutorial.json")
    env = SimEnv(level)
    never_press = env.run_policy(to_numpy_policy(ActorCritic(F.N_FEATURES, (32,), 50.0), F.N_FEATURES))
    _, history = train_ppo(env, PPOConfig(updates=22, steps_per_update=1024, seed=0))
    assert max(h["greedy_progress"] for h in history) > never_press.end_progress + 0.1
