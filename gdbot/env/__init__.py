"""Environments.

The solver is written against :class:`gdbot.env.base.AttemptEnv`, which models
the game the way it actually behaves: you get one attempt, it ends, you get
another.  :class:`gdbot.env.sim_env.SimEnv` implements it on the simulator and
:class:`gdbot.env.gd_env.GDEnv` on the real game over screen capture, so the
same curriculum code drives both.
"""

from gdbot.env.base import AttemptEnv, shape_reward
from gdbot.env.sim_env import SimEnv

__all__ = ["AttemptEnv", "shape_reward", "SimEnv"]


def __getattr__(name: str):  # pragma: no cover - optional heavy import
    # GDEnv pulls in mss / opencv, which are only installed on the machine that
    # actually runs Geometry Dash.  Import it lazily so the simulator side works
    # with numpy alone.
    if name == "GDEnv":
        from gdbot.env.gd_env import GDEnv

        return GDEnv
    raise AttributeError(name)
