"""The orchestrator: learn a level, then clear it.

Strategy, and why it is this one.

A level like Skeletal Shenanigans is two minutes of frame-accurate input.  A
purely reactive policy will not produce that -- one mistimed press in seven
thousand ticks ends the run, so the probability of a clean run is the product of
seven thousand near-certainties, which is not near-certain.  What *does* produce
it is exploiting determinism: find an input tape section by section, never
disturbing a section that already works, and keep the tape.

So there are two learners and they do different jobs:

1. A reactive policy (CEM over a small network) learns to *play* -- it watches
   the geometry ahead and presses.  It generalises to level sections it has never
   seen, which makes it a good proposer.  It is not expected to clear the level.
2. A tape search takes over from the policy's proposal and makes a section
   actually work.  In the simulator that is a beam search with state dedup, which
   is complete up to its quantisation.  Against the real game, where state cannot
   be restored, it is a genetic search that mutates only the ticks around the
   point where the run died.

The curriculum is forward-chaining: section k is solved starting from the state
that the verified solution of sections 1..k-1 actually leaves the player in.  The
alternative -- solving sections from synthesised start positions and stitching
afterwards -- looks attractive but produces tapes that do not join up, because a
real run arrives at a section mid-air with velocity that a start position cannot
reproduce.  Forward chaining pays more per attempt and never has that problem.
When a section turns out to be unreachable from where the previous one left off,
the solver backtracks and re-solves both together.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from gdbot.agents.beam import BeamConfig, beam_search
from gdbot.agents.cem import CEM, CEMConfig
from gdbot.agents.ga_tape import TapeGA, TapeGAConfig
from gdbot.agents.policy import MLPPolicy
from gdbot.config import Config
from gdbot.env.base import AttemptEnv, shape_reward
from gdbot.store import RunStore
from gdbot.types import Attempt, Outcome, Segment, TrainingStats


@dataclass
class SegmentRecord:
    index: int
    start: float
    end: float
    reached: float
    method: str
    attempts: int
    ticks: int
    solved: bool
    elapsed: float


@dataclass
class SolveReport:
    solved: bool
    progress: float
    tape: np.ndarray
    attempts: int
    elapsed: float
    method: str
    segments: list[SegmentRecord] = field(default_factory=list)
    verified: int = 0
    verify_runs: int = 0

    def summary(self) -> str:
        head = "CLEARED" if self.solved else f"best {self.progress * 100:.2f}%"
        # A beam search expands simulator states, it does not play attempts;
        # calling those "attempts" would overstate how much playing happened.
        unit = "states explored" if self.method == "beam-search" else "attempts"
        return (
            f"{head} | {self.method} | {self.attempts:,} {unit} | "
            f"{len(self.tape)} ticks | {self.elapsed:.1f}s | "
            f"verified {self.verified}/{self.verify_runs}"
        )


class Solver:
    """Runs the whole pipeline against one environment."""

    def __init__(
        self,
        env: AttemptEnv,
        cfg: Config,
        *,
        store: RunStore | None = None,
        verbose: bool = True,
        seed_tape: np.ndarray | None = None,
    ) -> None:
        self.env = env
        self.cfg = cfg
        self.store = store or RunStore(cfg.run_dir)
        self.verbose = verbose
        #: A tape to propose before the policy's guess -- typically the best of
        #: the player's own recorded attempts.
        self.seed_tape = None if seed_tape is None else np.asarray(seed_tape, dtype=np.uint8)
        self.stats = TrainingStats()
        self.policy = MLPPolicy(
            n_in=env.n_features, hidden=tuple(cfg.train.hidden), seed=cfg.train.seed
        )
        self._ticks_per_progress: float | None = None

    # ------------------------------------------------------------------ utils
    def _say(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    def _record(self, attempt: Attempt) -> Attempt:
        self.stats.record(attempt)
        return attempt

    def segments(self) -> list[Segment]:
        n = max(1, self.cfg.curriculum.n_segments)
        edges = np.linspace(0.0, 1.0, n + 1)
        return [Segment(index=i, start=float(edges[i]), end=float(edges[i + 1])) for i in range(n)]

    def _estimate_ticks(self, progress_span: float) -> int:
        """How many ticks a span of the level takes, measured not guessed."""
        if self._ticks_per_progress is None:
            probe = self._record(self.env.run_tape(np.zeros(self.cfg.max_ticks, dtype=np.uint8)))
            gained = max(probe.end_progress - probe.start_progress, 1e-6)
            self._ticks_per_progress = probe.ticks / gained
        return int(progress_span * self._ticks_per_progress) + 90

    # --------------------------------------------------------------- pretrain
    def pretrain_policy(
        self,
        iterations: int = 60,
        *,
        practice: bool = True,
        on_generation: Callable[[int, float, float], None] | None = None,
    ) -> MLPPolicy:
        """Learn a reactive policy with the Cross-Entropy Method.

        Every candidate evaluation is one real attempt, so the attempt counter
        here is the honest "thousands of tries" number: ``iterations *
        population``.  ``practice`` spends part of the budget spawned at points
        the policy has already reached, which is the difference between learning
        the first ten seconds very well and learning the whole level.
        """
        tcfg = self.cfg.train
        cem = CEM(
            self.policy.n_params,
            CEMConfig(
                population=tcfg.population,
                elite_frac=tcfg.elite_frac,
                init_sigma=tcfg.init_sigma,
                sigma_decay=tcfg.sigma_decay,
                min_sigma=tcfg.min_sigma,
                seed=tcfg.seed,
            ),
            init=self.policy.get_params(),
        )
        rng = np.random.default_rng(tcfg.seed + 1)
        can_spawn = practice
        try:
            self.env.spawn_at(0.0)
        except NotImplementedError:
            can_spawn = False
        frontier = 0.0

        for it in range(iterations):
            # One spawn point per generation so the population is comparable.
            spawn = 0.0
            if can_spawn and frontier > 0.05 and it % 3 == 2:
                spawn = float(rng.uniform(0.0, frontier))
            population = cem.ask()
            scores = np.empty(len(population), dtype=np.float64)
            for i, params in enumerate(population):
                self.policy.set_params(params)
                if spawn > 0.0:
                    self.env.spawn_at(spawn)
                    start = self.env.snapshot() if self.env.supports_snapshots else None
                    attempt = self._record(
                        self.env.run_policy(self.policy, start=start, max_ticks=self.cfg.max_ticks)
                    )
                else:
                    attempt = self._record(
                        self.env.run_policy(self.policy, max_ticks=self.cfg.max_ticks)
                    )
                scores[i] = shape_reward(attempt, tcfg)
                if spawn == 0.0:
                    frontier = max(frontier, attempt.end_progress)
            cem.tell(scores)
            self.store.log(
                "cem",
                iteration=it,
                spawn=spawn,
                best=float(scores.max()),
                mean=float(scores.mean()),
                frontier=frontier,
                sigma=cem.sigma_mean,
                attempts=self.stats.attempts,
            )
            if on_generation is not None:
                on_generation(it, float(scores.max()), frontier)
            if self.verbose and (it % 5 == 0 or it == iterations - 1):
                self._say(
                    f"  CEM {it + 1:3d}/{iterations}  best={scores.max():7.2f}  "
                    f"frontier={frontier * 100:5.1f}%  sigma={cem.sigma_mean:.3f}  "
                    f"attempts={self.stats.attempts:,}"
                )

        self.policy.set_params(cem.best_params if np.isfinite(cem.best_score) else cem.mean)
        self.store.save_policy("policy", self.policy.get_params())
        self.store.log("cem_done", frontier=frontier, attempts=self.stats.attempts)
        return self.policy

    # ------------------------------------------------------------------ solve
    def solve(self) -> SolveReport:
        """Produce a tape that clears the level, or the best one found."""
        # The beam search needs exact state restore, which only the simulator has.
        if self.env.supports_snapshots and hasattr(self.env, "world"):
            report = self._solve_with_search()
        else:
            report = self._solve_blackbox()
        report.verify_runs = self.cfg.curriculum.verify_runs
        report.verified = self.verify(report.tape, report.verify_runs)
        self.store.save_tape(
            "best",
            report.tape,
            solved=report.solved,
            progress=report.progress,
            method=report.method,
            attempts=report.attempts,
            verified=f"{report.verified}/{report.verify_runs}",
        )
        self.store.save_json("report", report)
        return report

    def _solve_with_search(self) -> SolveReport:
        """Simulator path: beam search over tapes, widening until it gets through."""
        t0 = time.perf_counter()
        segs = self.segments()
        records: list[SegmentRecord] = []
        width = max(32, self.cfg.train.population * 8)
        best_tape = np.zeros(0, dtype=np.uint8)
        best_progress = 0.0
        expanded_total = 0

        for attempt_i in range(4):
            reached = {"seg": 0}

            def on_progress(tick: int, frontier: int, progress: float) -> None:
                while reached["seg"] < len(segs) and progress >= segs[reached["seg"]].end:
                    seg = segs[reached["seg"]]
                    records.append(
                        SegmentRecord(
                            index=seg.index, start=seg.start, end=seg.end, reached=progress,
                            method="beam", attempts=frontier, ticks=tick, solved=True,
                            elapsed=time.perf_counter() - t0,
                        )
                    )
                    reached["seg"] += 1
                    self._say(
                        f"  beam  section {seg.index + 1:2d}/{len(segs)} cleared "
                        f"at {progress * 100:5.1f}%  tick={tick:5d}  frontier={frontier}"
                    )

            result = beam_search(
                self.env.world,  # type: ignore[attr-defined]
                BeamConfig(
                    width=width,
                    max_ticks=self.cfg.max_ticks,
                    goal_progress=1.0,
                    log_every=30,
                ),
                on_progress=on_progress,
            )
            expanded_total += result.expanded
            if result.solved:
                on_progress(result.ticks, 1, 1.0)  # report the sections after the last log tick
            self.store.log(
                "beam",
                pass_=attempt_i, width=width, progress=result.progress,
                solved=result.solved, expanded=result.expanded, elapsed=result.elapsed,
            )
            if result.progress > best_progress:
                best_progress, best_tape = result.progress, result.tape
            self._say(
                f"  beam pass {attempt_i + 1} width={width}: "
                f"{'CLEARED' if result.solved else f'{result.progress * 100:.2f}%'} "
                f"({result.expanded:,} states, {result.elapsed:.1f}s)"
            )
            if result.solved:
                break
            width *= 3  # the section it stalled on needs a more diverse frontier

        return SolveReport(
            solved=best_progress >= 1.0,
            progress=best_progress,
            tape=best_tape,
            attempts=expanded_total,
            elapsed=time.perf_counter() - t0,
            method="beam-search",
            segments=records,
        )

    def _solve_blackbox(self) -> SolveReport:
        """Real-game path: only 'play this tape, report how far it got' is available.

        Each section is attacked with a genetic search over the ticks after the
        already-solved prefix.  A section that resists is merged with the previous
        one and both are re-solved, because the usual reason a section is
        impossible is that the previous one hands it the wrong velocity.
        """
        t0 = time.perf_counter()
        ccfg = self.cfg.curriculum
        tcfg = self.cfg.train
        segs = self.segments()
        n_total = len(segs)  # labels keep their original numbering through merges
        records: list[SegmentRecord] = []

        prefix = np.zeros(0, dtype=np.uint8)
        prefix_progress = 0.0
        # Stack of verified prefixes, so a retreat can step back to the state
        # before the previous section without re-deriving it.
        solved_stack: list[tuple[float, np.ndarray]] = [(0.0, prefix)]
        i = 0
        retreats = 0

        resumed = self._resume_point()
        if resumed is not None:
            prefix_progress, prefix, solved_stack = resumed
            i = next((k for k, sg in enumerate(segs) if sg.end > prefix_progress + 1e-9), len(segs))
            if i < len(segs):
                segs[i] = Segment(index=segs[i].index, start=prefix_progress, end=segs[i].end)

        while i < len(segs):
            seg = segs[i]
            goal = seg.end
            span = max(goal - prefix_progress, 1e-3)
            suffix_len = self._estimate_ticks(span)
            want = len(prefix) + suffix_len
            if self.seed_tape is not None and self.seed_tape.size > len(prefix):
                # The recorded run already knows how to play this stretch; the
                # search only has to repair where it went wrong.
                seed = np.concatenate([prefix, self.seed_tape[len(prefix) : want]])
            else:
                seed_attempt = self._record(
                    self.env.run_hybrid(prefix, self.policy, max_ticks=want)
                )
                seed = np.asarray(seed_attempt.actions, dtype=np.uint8)
            if seed.size < want:
                seed = np.concatenate([seed, np.zeros(want - seed.size, dtype=np.uint8)])

            budget = ccfg.attempts_per_segment
            ga = TapeGA(
                lambda tape: self._record(self.env.run_tape(tape)),
                TapeGAConfig(
                    population=tcfg.tape_population,
                    generations=max(1, budget // max(1, tcfg.tape_population)),
                    mutation=tcfg.tape_mutation,
                    goal_progress=goal,
                    freeze_before=len(prefix),
                    seed=tcfg.seed + i * 17 + retreats,
                ),
            )
            # One second of play, in progress units: the yardstick for "went nowhere".
            one_second = self.cfg.tick_rate / max(self._ticks_per_progress or 1.0, 1.0)
            floor = prefix_progress

            def doomed(gen: int, best: float, floor: float = floor, margin: float = one_second) -> bool:
                return i > 0 and gen + 1 >= ccfg.doomed_generations and best < floor + margin

            result = ga.run(seed, should_stop=doomed)
            solved = result.progress >= goal
            records.append(
                SegmentRecord(
                    index=seg.index, start=seg.start, end=seg.end, reached=result.progress,
                    method="tape-ga", attempts=result.attempts, ticks=int(result.tape.size),
                    solved=solved, elapsed=time.perf_counter() - t0,
                )
            )
            self.store.log(
                "segment",
                index=seg.index, goal=goal, reached=result.progress, solved=solved,
                attempts=result.attempts, total_attempts=self.stats.attempts,
            )
            self._say(
                f"  section {seg.index + 1:2d}/{n_total} -> "
                f"{'ok' if solved else 'STUCK'} at {result.progress * 100:5.1f}% "
                f"(goal {goal * 100:5.1f}%, {result.attempts} attempts, "
                f"{self.stats.attempts:,} total)"
            )

            if solved:
                prefix = self._trim_to(result.tape, goal)
                prefix_progress = goal
                solved_stack.append((goal, prefix))
                self.store.save_tape(f"prefix_{seg.index:03d}", prefix, progress=goal)
                i += 1
                continue

            # Stuck.  Almost always this means the previous section hands the
            # player the wrong velocity, so drop back and solve the pair as one.
            if i > 0 and retreats < len(segs):
                retreats += 1
                back = i - 1
                solved_stack.pop()  # discard the prefix that ends where we are stuck
                prefix_progress, prefix = solved_stack[-1]
                self._say(
                    f"  retreating to section {segs[back].index + 1} and solving it together "
                    f"with {seg.index + 1} (retreat {retreats})"
                )
                segs[back] = Segment(index=segs[back].index, start=prefix_progress, end=seg.end)
                del segs[back + 1 : i + 1]
                i = back
                continue

            return SolveReport(
                solved=False,
                progress=result.progress,
                tape=result.tape,
                attempts=self.stats.attempts,
                elapsed=time.perf_counter() - t0,
                method="tape-ga",
                segments=records,
            )

        final = self._record(self.env.run_tape(prefix))
        return SolveReport(
            solved=final.outcome is Outcome.COMPLETE,
            progress=final.end_progress,
            tape=prefix,
            attempts=self.stats.attempts,
            elapsed=time.perf_counter() - t0,
            method="tape-ga",
            segments=records,
        )

    def _resume_point(self) -> tuple[float, np.ndarray, list[tuple[float, np.ndarray]]] | None:
        """Pick up where an interrupted run stopped.

        Every solved section is saved as ``prefix_NNN.tape.json``.  Against the
        real game a run takes days, so it will be stopped and restarted; starting
        over would throw that away.  The furthest saved prefix is replayed once
        before it is trusted -- if the level, the config or the capture changed in
        between, it will not replay, and the run starts fresh instead of building
        on a tape that no longer works.
        """
        saved: list[tuple[float, str]] = []
        for path in self.store.root.glob("prefix_*.tape.json"):
            meta = json.loads(path.read_text(encoding="utf-8"))
            saved.append((float(meta.get("progress", 0.0)), path.name[: -len(".tape.json")]))
        if not saved:
            return None
        saved.sort()
        progress, name = saved[-1]
        tape = self.store.load_tape(name)
        check = self._record(self.env.run_tape(tape))
        if check.end_progress + 1e-9 < progress:
            self._say(
                f"  saved progress {progress * 100:.1f}% does not replay any more "
                f"(reached {check.end_progress * 100:.1f}%); starting from the beginning"
            )
            self.store.log("resume_rejected", saved=progress, reached=check.end_progress)
            return None
        stack = [(0.0, np.zeros(0, dtype=np.uint8))]
        stack += [(p, self.store.load_tape(n)) for p, n in saved]
        self._say(f"  resuming from {progress * 100:.1f}% (the saved tape replays)")
        self.store.log("resume", progress=progress)
        return progress, tape, stack

    def _trim_to(self, tape: np.ndarray, goal: float) -> np.ndarray:
        """Cut a tape at the first tick where it reaches ``goal`` progress.

        Keeping the tape exactly as long as the solved prefix is what makes
        ``freeze_before`` meaningful for the next section.
        """
        arr = np.asarray(tape, dtype=np.uint8)
        self.env.reset()
        outcome = Outcome.TIMEOUT
        cut = arr.size
        for k, action in enumerate(arr):
            result = self.env.step(int(action))
            if self.env.progress >= goal or result.done:
                cut, outcome = k + 1, result.outcome
                break
        # Against the real game this pass is a real attempt; count it as one.
        self._record(Attempt(actions=arr[:cut], start_progress=0.0,
                             end_progress=self.env.progress, outcome=outcome,
                             reward=0.0, ticks=cut))
        return arr[:cut].copy()

    # ----------------------------------------------------------------- verify
    def verify(self, tape: np.ndarray, runs: int = 3) -> int:
        """Replay the final tape and count how many runs actually clear.

        In the simulator this is a determinism check.  Against the real game it is
        the number that matters: it measures whether the tape survives capture and
        input jitter, which is the difference between a solution and a fluke.
        """
        if tape is None or len(tape) == 0 or runs <= 0:
            return 0
        ok = 0
        for r in range(runs):
            attempt = self._record(self.env.run_tape(tape))
            cleared = attempt.outcome is Outcome.COMPLETE
            ok += int(cleared)
            self.store.log("verify", run=r, cleared=cleared, progress=attempt.end_progress)
        self._say(f"  verification: {ok}/{runs} clean runs")
        return ok
