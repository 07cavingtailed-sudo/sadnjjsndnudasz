"""Learning and search algorithms.

Two very different tools, because a Geometry Dash level needs both:

* :mod:`gdbot.agents.beam` searches directly for an input *tape*.  It exploits
  the fact that the game is deterministic and is what actually clears a demon.
* :mod:`gdbot.agents.cem` and :mod:`gdbot.agents.ppo` learn a *reactive policy*
  that looks at the screen and decides.  Slower to perfect a level, but it
  generalises and it is what makes the bot able to attempt an unseen level.
* :mod:`gdbot.agents.ga_tape` refines a tape using nothing but "replay it and
  see how far you got", which is all the real game offers.
"""

from gdbot.agents.beam import BeamConfig, BeamResult, beam_search
from gdbot.agents.cem import CEM, CEMConfig
from gdbot.agents.ga_tape import TapeGA, TapeGAConfig
from gdbot.agents.policy import MLPPolicy

__all__ = [
    "BeamConfig",
    "BeamResult",
    "beam_search",
    "CEM",
    "CEMConfig",
    "TapeGA",
    "TapeGAConfig",
    "MLPPolicy",
]
