"""Evolutionary refinement of an input tape from attempt outcomes alone.

This is the algorithm that works against the *real* game.  It never inspects
state, never needs a checkpoint it can restore programmatically and never needs
gradients -- the only thing it asks for is "play this tape and tell me how far
you got", which is exactly what a screen-capture bridge can provide.

Two ideas do the heavy lifting in :meth:`TapeGA._mutate`:

* **One surgical edit per child.**  A child differs from its parent by a single
  flip, a moved press edge, an inserted press or a deleted press.  Sprinkling
  several random bit flips per child sounds more exploratory but mostly breaks
  inputs that already worked.
* **Suffix shifts.**  In a chain of jump rings, hitting ring *k* one tick earlier
  starts the next arc one tick earlier, so ring *k+1* must also be hit one tick
  earlier.  Moving either press alone makes the run *worse*, and a search that
  only accepts improvements can never make the pair of moves.  Shifting a press
  together with every press after it preserves their spacing, which is exactly
  the coupled move a chain needs.
* **Edits aim at the death point, and reach further back the longer progress
  stalls.**  Most deaths are caused by the last input or two, so that is where
  edits concentrate.  But a run that keeps dying on the same spike is usually
  being set up to fail earlier -- a take-off one tick late lands you on the far
  side of a pit -- so each stalled generation widens how far back edits can land.
  Without that, the search sits at a local optimum forever.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from gdbot.types import Attempt, Outcome

#: Given a tape, play it and report the attempt.
Evaluator = Callable[[np.ndarray], Attempt]


@dataclass
class TapeGAConfig:
    population: int = 32
    generations: int = 400
    #: Probability that a child gets a second edit on top of the first.
    mutation: float = 0.2
    #: How far before the death point edits can land, in ticks, when progress
    #: is moving.  Grows while progress stalls (see ``stall_patience``).
    focus_ticks: int = 45
    #: Upper bound for the grown reach: five seconds of play.
    max_reach: int = 300
    #: Generations without improvement before the reach grows by ``focus_ticks``.
    stall_patience: int = 3
    #: Fraction of the population carried over unchanged.
    elite_frac: float = 0.15
    #: Probability of using crossover instead of mutation for a child.
    crossover_rate: float = 0.15
    #: Largest single move of a press edge, in ticks.
    max_shift: int = 6
    #: Longest press an insertion creates, in ticks.
    max_insert: int = 8
    #: Stop once a tape reaches this progress.
    goal_progress: float = 1.0
    #: Ticks at the start of the tape that must never be touched.  The curriculum
    #: sets this to the length of the already-solved prefix: those inputs are
    #: known good and perturbing them would throw away solved sections.
    freeze_before: int = 0
    seed: int = 0


@dataclass
class TapeGAResult:
    tape: np.ndarray
    progress: float
    outcome: Outcome
    attempts: int
    generations: int
    history: list[float] = field(default_factory=list)

    @property
    def solved(self) -> bool:
        return self.outcome is Outcome.COMPLETE


class TapeGA:
    """Population search over tapes, driven by a black-box evaluator."""

    def __init__(self, evaluate: Evaluator, cfg: TapeGAConfig | None = None) -> None:
        self.evaluate = evaluate
        self.cfg = cfg or TapeGAConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        self.attempts = 0

    # ------------------------------------------------------------- operators
    def _edit_once(self, out: np.ndarray, death_tick: int, reach: int) -> None:
        """Apply one local edit in place, aimed at ticks shortly before death."""
        cfg = self.cfg
        n = out.size
        floor = max(0, min(cfg.freeze_before, n - 1))
        # Triangular with its mode at the death tick: most edits land right
        # before it, a tail reaches back as far as ``reach``.
        back = int(self.rng.triangular(0.0, 0.0, max(1.0, float(reach))))
        e = death_tick - back + int(self.rng.integers(-2, 3))
        e = max(floor, min(n - 1, e))
        op = self.rng.random()

        if op < 0.22:
            out[e] ^= 1
        elif op < 0.42:
            self._shift_suffix(out, e, floor)
        elif op < 0.65:
            # Move the press edge nearest to e.  Timing, not presence, is what a
            # precision platformer usually gets wrong.
            lo = max(floor, e - cfg.max_shift * 2)
            hi = min(n, e + cfg.max_shift * 2)
            edges = floor + 1 + np.flatnonzero(np.diff(out[floor:].astype(np.int8)) != 0)
            edges = edges[(edges >= lo) & (edges < hi)]
            if edges.size == 0:
                out[e] ^= 1
                return
            edge = int(edges[np.argmin(np.abs(edges - e))])
            shift = int(self.rng.integers(1, cfg.max_shift + 1)) * (1 if self.rng.random() < 0.5 else -1)
            if shift > 0:
                out[edge : min(n, edge + shift)] = out[edge - 1]
            else:
                out[max(floor, edge + shift) : edge] = out[edge]
        elif op < 0.85:
            length = int(self.rng.integers(1, cfg.max_insert + 1))
            out[e : min(n, e + length)] = 1
        else:
            # delete the press that covers e (or the next one)
            ones = np.flatnonzero(out[e:] == 1)
            if ones.size == 0:
                return
            a = e + int(ones[0])
            zeros = np.flatnonzero(out[a:] == 0)
            b = a + int(zeros[0]) if zeros.size else n
            out[a:b] = 0

    def _shift_suffix(self, out: np.ndarray, e: int, floor: int) -> None:
        """Move everything from the press edge at/after ``e`` by a few ticks.

        Positive shifts delay the suffix (the gap is filled with whatever came
        just before), negative shifts advance it (the tail is padded with
        releases).  Tape length never changes, so the prefix stays aligned.
        """
        n = out.size
        edges = floor + 1 + np.flatnonzero(np.diff(out[floor:].astype(np.int8)) != 0)
        edges = edges[edges >= e]
        start = int(edges[0]) if edges.size else e
        if start <= floor or start >= n:
            return
        k = int(self.rng.integers(1, self.cfg.max_shift + 1))
        tail = out[start:].copy()
        if self.rng.random() < 0.5:
            k = min(k, n - start)
            out[start + k :] = tail[: n - start - k]
            out[start : start + k] = out[start - 1]
        else:
            k = min(k, start - floor)
            if k <= 0:
                return
            out[start - k : n - k] = tail
            out[n - k :] = 0

    def _mutate(self, tape: np.ndarray, death_tick: int, reach: int) -> np.ndarray:
        out = tape.copy()
        if out.size == 0:
            return out
        self._edit_once(out, death_tick, reach)
        if self.rng.random() < self.cfg.mutation:
            self._edit_once(out, death_tick, reach)
        return out

    def _crossover(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        n = min(a.size, b.size)
        lo = max(1, self.cfg.freeze_before)
        if n < 2 or lo >= n:
            return a.copy()
        cut = int(self.rng.integers(lo, n))
        out = a.copy()
        out[cut:n] = b[cut:n]
        return out

    # ------------------------------------------------------------------- run
    def run(
        self,
        seed_tape: np.ndarray,
        *,
        on_generation: Callable[[int, float, int], None] | None = None,
        should_stop: Callable[[int, float], bool] | None = None,
    ) -> TapeGAResult:
        """Evolve from ``seed_tape`` until the goal, the budget or ``should_stop``.

        ``should_stop(generation, best_progress)`` lets the caller abandon a search
        it has decided is hopeless, which on the real game saves hours.
        """
        cfg = self.cfg
        seed_tape = np.asarray(seed_tape, dtype=np.uint8).reshape(-1)
        n_elite = max(1, int(round(cfg.population * cfg.elite_frac)))

        population = [seed_tape.copy()]
        while len(population) < cfg.population:
            population.append(self._mutate(seed_tape, seed_tape.size, cfg.focus_ticks))

        best_tape = seed_tape.copy()
        best_progress = -1.0
        best_outcome = Outcome.DEAD
        death_tick = seed_tape.size
        stalled = 0
        history: list[float] = []

        for gen in range(cfg.generations):
            scored: list[tuple[float, int, float, np.ndarray]] = []
            improved = False
            for tape in population:
                attempt = self.evaluate(tape)
                self.attempts += 1
                # Fitness: progress first, then prefer tapes that survive longer;
                # a random tiebreak stops identical-scoring tapes from collapsing
                # the population onto a single one.
                scored.append((attempt.end_progress, attempt.ticks, float(self.rng.random()), tape))
                if attempt.end_progress > best_progress + 1e-9:
                    best_progress = attempt.end_progress
                    best_tape = tape.copy()
                    best_outcome = attempt.outcome
                    death_tick = attempt.ticks
                    improved = True
                if attempt.outcome is Outcome.COMPLETE or attempt.end_progress >= cfg.goal_progress:
                    history.append(best_progress)
                    return TapeGAResult(
                        tape=tape.copy(),
                        progress=attempt.end_progress,
                        outcome=Outcome.COMPLETE,
                        attempts=self.attempts,
                        generations=gen + 1,
                        history=history,
                    )

            scored.sort(key=lambda s: (-s[0], -s[1], s[2]))
            history.append(best_progress)
            stalled = 0 if improved else stalled + 1
            reach = min(cfg.max_reach, cfg.focus_ticks * (1 + stalled // max(1, cfg.stall_patience)))
            if on_generation is not None:
                on_generation(gen, best_progress, self.attempts)
            if should_stop is not None and should_stop(gen, best_progress):
                return TapeGAResult(
                    tape=best_tape,
                    progress=best_progress,
                    outcome=best_outcome,
                    attempts=self.attempts,
                    generations=gen + 1,
                    history=history,
                )

            elites = [s[3].copy() for s in scored[:n_elite]]
            population = list(elites)
            parents = [s[3] for s in scored[: max(2, len(scored) // 2)]]
            while len(population) < cfg.population:
                if len(parents) >= 2 and self.rng.random() < cfg.crossover_rate:
                    i, j = self.rng.choice(len(parents), size=2, replace=False)
                    child = self._crossover(parents[i], parents[j])
                else:
                    child = parents[int(self.rng.integers(0, len(parents)))]
                population.append(self._mutate(child, death_tick, reach))

        return TapeGAResult(
            tape=best_tape,
            progress=best_progress,
            outcome=best_outcome,
            attempts=self.attempts,
            generations=cfg.generations,
            history=history,
        )
