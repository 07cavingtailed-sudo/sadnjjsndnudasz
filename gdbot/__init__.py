"""gdbot - a self-learning Geometry Dash agent.

The package is split into three layers:

* ``gdbot.sim``      - a fast, deterministic Geometry Dash physics simulator used
                       for pretraining (millions of ticks per minute).
* ``gdbot.agents``   - the learning algorithms (CEM policy search, tape GA, PPO).
* ``gdbot.env``      - a common environment interface, implemented both by the
                       simulator and by the real game via screen capture.

``gdbot.solver`` ties them together into the forward-chained, backtracking
curriculum that actually clears a long level.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
