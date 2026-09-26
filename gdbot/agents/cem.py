"""Cross-Entropy Method: gradient-free policy search.

CEM fits a diagonal Gaussian over policy parameters, samples a population,
keeps the best fraction and refits.  It is the right tool here because the only
learning signal the real game offers is "this attempt reached 37%" -- a single
scalar per attempt, no gradients, no per-step reward that means anything.  CEM
consumes exactly that, needs no GPU, and every attempt it spends is a real
attempt, which is what the "learn over thousands of attempts" loop looks like.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class CEMConfig:
    population: int = 24
    elite_frac: float = 0.25
    init_sigma: float = 0.5
    sigma_decay: float = 0.995
    min_sigma: float = 0.02
    #: Extra variance added to the elite fit; keeps the search from collapsing
    #: onto a local optimum after a lucky generation.
    noise_floor: float = 0.01
    #: Weight of the new elite fit versus the previous distribution.
    smoothing: float = 0.7
    seed: int = 0


class CEM:
    """Ask/tell optimiser over a flat parameter vector."""

    def __init__(self, n_params: int, cfg: CEMConfig | None = None, *, init: np.ndarray | None = None) -> None:
        self.cfg = cfg or CEMConfig()
        self.n_params = int(n_params)
        self.rng = np.random.default_rng(self.cfg.seed)
        self.mean = (
            np.zeros(self.n_params, dtype=np.float32)
            if init is None
            else np.asarray(init, dtype=np.float32).copy()
        )
        self.sigma = np.full(self.n_params, self.cfg.init_sigma, dtype=np.float32)
        self.n_elite = max(2, int(round(self.cfg.population * self.cfg.elite_frac)))
        self.generation = 0
        self.best_score = -np.inf
        self.best_params = self.mean.copy()
        self._population: np.ndarray | None = None

    def ask(self) -> np.ndarray:
        """Sample a population: ``(population, n_params)``."""
        noise = self.rng.standard_normal((self.cfg.population, self.n_params)).astype(np.float32)
        self._population = self.mean[None, :] + noise * self.sigma[None, :]
        return self._population

    def tell(self, scores: np.ndarray) -> None:
        """Refit the distribution from the elite members of the last ``ask``."""
        if self._population is None:
            raise RuntimeError("tell() called before ask()")
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        if scores.size != self._population.shape[0]:
            raise ValueError("one score per population member is required")
        order = np.argsort(-scores)
        elite = self._population[order[: self.n_elite]]
        a = self.cfg.smoothing
        self.mean = (a * elite.mean(axis=0) + (1.0 - a) * self.mean).astype(np.float32)
        fitted = elite.std(axis=0).astype(np.float32) + self.cfg.noise_floor
        # Smoothed refit, then decay.  Taking a max with the previous sigma instead
        # would ratchet: whenever the elites happen to be spread out the variance
        # grows, and it never gives that back.
        blended = a * fitted + (1.0 - a) * self.sigma
        self.sigma = np.maximum(self.cfg.min_sigma, blended * self.cfg.sigma_decay).astype(np.float32)
        top = float(scores[order[0]])
        if top > self.best_score:
            self.best_score = top
            self.best_params = self._population[order[0]].copy()
        self.generation += 1
        self._population = None

    @property
    def sigma_mean(self) -> float:
        return float(self.sigma.mean())
