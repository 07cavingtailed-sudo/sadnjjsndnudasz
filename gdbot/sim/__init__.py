"""A fast deterministic Geometry Dash-like simulator.

The simulator exists for one reason: the real game runs at 60 frames per second
and every attempt costs wall-clock seconds, while the simulator runs tens of
thousands of ticks per second.  Pretraining the policy here and only then
fine-tuning against the real game is what makes "thousands of attempts"
affordable.
"""

from gdbot.sim.level import Level, Obj, ObjType, SPEEDS, BLOCK
from gdbot.sim.physics import Player, World, TPS

__all__ = ["Level", "Obj", "ObjType", "SPEEDS", "BLOCK", "Player", "World", "TPS"]
