"""A small numpy MLP policy.

Deliberately not a deep convolutional network: the observation is already a
compact 81-float vector (see :mod:`gdbot.features`), and a few thousand
parameters is what gradient-free search can actually optimise.  The torch PPO
agent in :mod:`gdbot.agents.ppo` is available when a larger model is wanted.

The policy is deterministic by default.  That matters: a stochastic policy that
clears a level once cannot reproduce the run, while a deterministic one can be
recorded straight into a replayable tape.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

import gdbot.features as F


@dataclass
class MLPPolicy:
    """Tanh MLP mapping features to a hold/release decision."""

    n_in: int = F.N_FEATURES
    hidden: Sequence[int] = (32,)
    n_out: int = 2
    seed: int = 0

    def __post_init__(self) -> None:
        self.sizes = [int(self.n_in), *[int(h) for h in self.hidden], int(self.n_out)]
        rng = np.random.default_rng(self.seed)
        self._shapes: list[tuple[int, int]] = []
        for a, b in zip(self.sizes, self.sizes[1:]):
            self._shapes.append((a, b))
        self.n_params = sum(a * b + b for a, b in self._shapes)
        # Small random init keeps the initial policy near "never press", which is
        # a sane starting point: spamming the button dies immediately in GD.
        self.set_params(rng.standard_normal(self.n_params).astype(np.float32) * 0.1)

    # ------------------------------------------------------------ parameters
    def set_params(self, vec: np.ndarray) -> "MLPPolicy":
        vec = np.asarray(vec, dtype=np.float32).reshape(-1)
        if vec.size != self.n_params:
            raise ValueError(f"expected {self.n_params} params, got {vec.size}")
        self._params = vec.copy()
        self._w: list[np.ndarray] = []
        self._b: list[np.ndarray] = []
        i = 0
        for a, b in self._shapes:
            self._w.append(vec[i : i + a * b].reshape(a, b))
            i += a * b
            self._b.append(vec[i : i + b])
            i += b
        return self

    def get_params(self) -> np.ndarray:
        return self._params.copy()

    # --------------------------------------------------------------- forward
    def logits(self, x: np.ndarray) -> np.ndarray:
        h = np.asarray(x, dtype=np.float32)
        squeeze = h.ndim == 1
        if squeeze:
            h = h[None, :]
        for k, (w, b) in enumerate(zip(self._w, self._b)):
            h = h @ w + b
            if k < len(self._w) - 1:
                h = np.tanh(h)
        return h[0] if squeeze else h

    def act(self, features: np.ndarray) -> int:
        """Greedy action: 1 = hold the button."""
        z = self.logits(features)
        return int(z[1] > z[0])

    def act_stochastic(self, features: np.ndarray, rng: np.random.Generator, temp: float = 1.0) -> int:
        z = self.logits(features) / max(temp, 1e-6)
        z = z - z.max()
        p = np.exp(z)
        p /= p.sum()
        return int(rng.random() < p[1])

    def clone(self) -> "MLPPolicy":
        other = MLPPolicy(self.n_in, tuple(self.hidden), self.n_out, self.seed)
        other.set_params(self._params)
        return other
