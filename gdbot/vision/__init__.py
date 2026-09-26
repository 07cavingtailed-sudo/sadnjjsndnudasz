"""Turning pixels into the game state the agent needs.

Three things have to be read off the screen, in descending order of how much the
bot depends on them:

1. **Progress** -- from the fill of the progress bar.  This is the learning
   signal, the death detector and the completion detector all at once, and it is
   read from a bar of solid colour, which is about the most robust thing on the
   screen.
2. **Attempt boundaries** -- derived from progress alone, no templates needed:
   progress falling back to zero means the run restarted.
3. **Obstacle occupancy** -- a coarse grid from edge density.  This is the weak
   link and it is deliberately not load-bearing: it feeds the reactive policy,
   while the tape search that actually clears the level needs only progress.
"""

from gdbot.vision.occupancy import estimate_player_y, occupancy_from_frame
from gdbot.vision.progress import read_progress_bar
from gdbot.vision.state import AttemptTracker

__all__ = [
    "read_progress_bar",
    "AttemptTracker",
    "occupancy_from_frame",
    "estimate_player_y",
]
