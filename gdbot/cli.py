"""Command line interface: ``gdbot <command> --help`` for each command."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from gdbot.config import Config

HERE = Path(__file__).resolve().parent
LEVELS = HERE / "sim" / "levels"


def _load_config(path: str | None, **overrides) -> Config:
    cfg = Config.from_file(path) if path else Config()
    for key, value in overrides.items():
        if value is not None:
            setattr(cfg, key, value)
    return cfg


def _resolve_level(name_or_path: str) -> Path:
    """A path that exists, else a bundled level by name -- from any directory."""
    p = Path(name_or_path)
    if p.exists():
        return p
    for bundled in (LEVELS / f"{name_or_path}.json", LEVELS / p.name, LEVELS / f"{p.stem}.json"):
        if bundled.exists():
            return bundled
    raise SystemExit(f"no such level: {name_or_path} (bundled: {', '.join(sorted(x.stem for x in LEVELS.glob('*.json')))})")


def _sim_env(cfg: Config, level_path: Path, *, blackbox: bool = False):
    from gdbot.env import SimEnv
    from gdbot.env.base import AttemptEnv
    from gdbot.sim.level import Level

    level = Level.load(level_path)
    level.solution = None  # never let the bot see the author's answer
    env = SimEnv(level, max_ticks=cfg.max_ticks)
    if not blackbox:
        return env

    class RealGameCapabilities(AttemptEnv):
        """The simulator restricted to what the real game offers: no restore, no spawn."""

        supports_snapshots = False

        def reset(self):
            return env.reset()

        def step(self, a):
            return env.step(a)

        @property
        def progress(self):
            return env.progress

        def observe(self):
            return env.observe()

        def run_tape(self, tape, *, start=None):
            return env.run_tape(tape)

    return RealGameCapabilities()


def _game_env(cfg: Config):
    from gdbot.env.gd_env import GDEnv

    return GDEnv(cfg)


def _countdown(seconds: int, message: str) -> None:  # pragma: no cover - interactive
    """Give the user time to bring the game window to the front.

    Key presses go to whichever window is active, and right after a command is
    typed that is the console, not the game.
    """
    if seconds <= 0:
        return
    print(message, flush=True)
    for left in range(seconds, 0, -1):
        print(f"  {left}...", flush=True)
        time.sleep(1.0)
    print("  поехали", flush=True)


GAME_FOCUS_MSG = (
    "Переключитесь в окно Geometry Dash и запустите уровень.\n"
    "Пока бот играет, не трогайте мышь и клавиатуру: нажатия идут в активное окно.\n"
    "Остановить: переключитесь в эту консоль и нажмите Ctrl+C (прогресс сохранится)."
)


# ---------------------------------------------------------------- commands
def cmd_generate(a: argparse.Namespace) -> int:
    from gdbot.sim.generate import generate

    t0 = time.perf_counter()
    level, report = generate(
        a.name, n_chunks=a.chunks, difficulty=a.difficulty, seed=a.seed, speed=a.speed
    )
    out = Path(a.out or f"{a.name}.json")
    level.save(out)
    print(f"{level}\n{report}\nsaved {out} ({time.perf_counter() - t0:.1f}s)")
    return 0


def cmd_train(a: argparse.Namespace) -> int:
    from gdbot.solver import Solver
    from gdbot.store import RunStore

    cfg = _load_config(a.config, run_dir=a.run_dir)
    level_path = _resolve_level(a.level or cfg.level)
    env = _sim_env(cfg, level_path)
    store = RunStore(cfg.run_dir)
    print(f"training a policy on {level_path.name} -> {cfg.run_dir}")
    if a.ppo:
        try:
            from gdbot.agents.ppo import PPOConfig, train_ppo
        except ImportError as exc:
            raise SystemExit(str(exc)) from exc
        policy, _ = train_ppo(
            env,
            PPOConfig(updates=a.iterations, seed=cfg.train.seed),
            hidden=tuple(cfg.train.hidden),
            on_update=lambda u, st: print(
                f"  PPO {u + 1:3d}/{a.iterations}  greedy={st['greedy_progress'] * 100:5.1f}%  "
                f"frontier={st['frontier'] * 100:5.1f}%", flush=True),
        )
        store.save_policy("policy", policy.get_params())
    else:
        solver = Solver(env, cfg, store=store)
        policy = solver.pretrain_policy(iterations=a.iterations)
    result = env.run_policy(policy, max_ticks=cfg.max_ticks)
    print(f"greedy policy reaches {result.end_progress * 100:.1f}% ({result.outcome.name.lower()}); "
          f"saved {Path(cfg.run_dir) / 'policy.policy.npy'}")
    return 0


def cmd_solve(a: argparse.Namespace) -> int:
    from gdbot.macro import export_macro, stats
    from gdbot.solver import Solver
    from gdbot.store import RunStore

    cfg = _load_config(a.config, run_dir=a.run_dir)
    store = RunStore(cfg.run_dir)
    if a.game:
        _countdown(a.delay, GAME_FOCUS_MSG)
        env = _game_env(cfg)
        where = "the running game"
    else:
        level_path = _resolve_level(a.level or cfg.level)
        env = _sim_env(cfg, level_path, blackbox=a.blackbox)
        where = f"{level_path.name}" + (" (real-game rules: no snapshots)" if a.blackbox else "")
    seed = store.load_tape(a.seed_tape) if a.seed_tape else None
    solver = Solver(env, cfg, store=store, seed_tape=seed)
    if store.has_policy("policy"):
        solver.policy.set_params(store.load_policy("policy"))
        print("using the policy already trained in this run directory")
    elif a.pretrain:
        print(f"pretraining a policy for {a.pretrain} iterations first")
        solver.pretrain_policy(iterations=a.pretrain)
    print(f"solving {where} -> {cfg.run_dir}")
    try:
        report = solver.solve()
    finally:
        env.close()
    print(f"\n{report.summary()}")
    played = solver.stats
    print(f"played {played.attempts:,} attempts = {played.ticks / cfg.tick_rate / 3600:.1f} h of game time "
          f"at {cfg.tick_rate} ticks/s (restart pauses not included)")
    print(f"tape: {stats(report.tape, cfg.tick_rate)}")
    macro = export_macro(report.tape, Path(cfg.run_dir) / "macro.json", fps=cfg.tick_rate, level=cfg.name)
    print(f"saved {Path(cfg.run_dir) / 'best.tape.json'} and {macro}")
    return 0 if report.solved else 1


def cmd_play(a: argparse.Namespace) -> int:
    from gdbot.store import RunStore

    cfg = _load_config(a.config, run_dir=a.run_dir)
    tape = RunStore(cfg.run_dir).load_tape(a.tape)
    if a.game:
        _countdown(a.delay, GAME_FOCUS_MSG)
    env = _game_env(cfg) if a.game else _sim_env(cfg, _resolve_level(a.level or cfg.level))
    try:
        ok = 0
        for i in range(a.runs):
            attempt = env.run_tape(tape)
            cleared = attempt.outcome.name == "COMPLETE"
            ok += cleared
            print(f"  run {i + 1}/{a.runs}: {attempt.outcome.name.lower()} at {attempt.end_progress * 100:.2f}%")
        if a.game:
            print(f"  {env.clock.jitter.summary(1000.0 / cfg.tick_rate)}")  # type: ignore[attr-defined]
    finally:
        env.close()
    print(f"{ok}/{a.runs} clean runs")
    return 0 if ok == a.runs else 1


def cmd_record(a: argparse.Namespace) -> int:  # pragma: no cover - needs the game
    from gdbot.record import record_attempts
    from gdbot.store import RunStore

    cfg = _load_config(a.config, run_dir=a.run_dir)
    _countdown(a.delay, "Переключитесь в окно Geometry Dash и играйте как обычно: бот записывает ваши попытки.")
    recs = record_attempts(cfg, attempts=a.attempts)
    best = max(recs, key=lambda r: r.progress)
    path = RunStore(cfg.run_dir).save_tape("human", best.tape, progress=best.progress,
                                           outcome=best.outcome.name)
    print(f"best recorded attempt: {best.progress * 100:.1f}% -> {path}")
    print("seed the solver with it:  gdbot solve --game --seed-tape human")
    return 0


def _enable_ansi() -> None:  # pragma: no cover - Windows only
    """Let cmd.exe interpret the escape codes the live readout redraws with."""
    if sys.platform != "win32":
        return
    import ctypes

    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
    mode = ctypes.c_uint32()
    if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
        kernel32.SetConsoleMode(handle, mode.value | 0x0004)  # VIRTUAL_TERMINAL_PROCESSING


def cmd_calibrate(a: argparse.Namespace) -> int:  # pragma: no cover - needs a display
    import gdbot.features as F
    from gdbot.calibrate import fit_progress_bar, pick_points, play_area_from, update_config_text
    from gdbot.capture.screen import ScreenCapture
    from gdbot.vision import occupancy_from_frame, read_progress_bar

    if not a.config:
        raise SystemExit("укажите конфиг игры: -c configs/skeletal_shenanigans.json")
    cfg = _load_config(a.config)
    try:
        import mss
        import mss.tools
    except ImportError:
        raise SystemExit("нужен пакет mss: python -m pip install -r requirements.txt")

    _countdown(a.delay, "Переключитесь в Geometry Dash и запустите уровень. Снимок экрана будет сделан, "
                        "когда уровень идёт и полоса прогресса уже немного заполнилась.")
    with mss.mss() as sct:
        shot = sct.grab(sct.monitors[cfg.capture.monitor])
        mss.tools.to_png(shot.rgb, shot.size, output=a.screenshot)
        img = np.asarray(shot)[:, :, 2::-1]  # BGRA -> RGB
    print(f"снимок экрана сохранён: {a.screenshot} ({shot.size.width}x{shot.size.height})")

    try:
        points = pick_points(a.screenshot)
    except ImportError:
        points = None
        print("нет tkinter, поэтому окно для щелчков не открыть. Откройте снимок в Paint и впишите в конфиг\n"
              "capture.play_area и capture.progress_bar = [x, y, ширина, высота] вручную.")
    if points:
        fit = fit_progress_bar(img, points[0], points[1])
        area = play_area_from(points[2], points[3])
        print(f"полоса прогресса: {fit.rect}, цвет заполнения {fit.fill_rgb}; игровое поле: {area}")
        if not fit.confident:
            print(f"ВНИМАНИЕ: не похоже на частично заполненную полосу ({fit.reason}).\n"
                  "Запустите calibrate ещё раз и сделайте снимок, когда уровень пройден хотя бы на 5-10%.")
        path = Path(a.config)
        path.write_text(update_config_text(path.read_text(encoding="utf-8"), {
            "play_area": area, "progress_bar": fit.rect, "bar_fill_rgb": fit.fill_rgb,
        }), encoding="utf-8")
        print(f"записано в {path}")
        cfg = _load_config(a.config)
    elif points is None:
        print("разметка отменена; конфиг не изменён")

    print("\nЖивые показания (Ctrl+C - выход). Во время попытки процент должен расти,\n"
          "а при смерти падать к нулю; сетка должна загораться там, где препятствия.\n")
    time.sleep(1.5)
    cap = ScreenCapture(cfg.capture)
    _enable_ansi()
    try:
        while True:
            play, bar = cap.grab()
            p = read_progress_bar(bar, fill_rgb=cfg.capture.bar_fill_rgb,
                                  tolerance=cfg.capture.bar_fill_tolerance)
            grid = occupancy_from_frame(play[:: max(1, play.shape[0] // 270), :: max(1, play.shape[0] // 270)],
                                        cfg.vision)
            rows = "\n".join("   " + "".join("#" if v else "." for v in r) for r in grid)
            sys.stdout.write(f"\x1b[2J\x1b[Hпрогресс {p * 100:6.2f}%   (сетка {F.GRID_H}x{F.GRID_W})\n{rows}\n")
            sys.stdout.flush()
            time.sleep(0.1)
    except KeyboardInterrupt:
        print()
    finally:
        cap.close()
    return 0


def cmd_show(a: argparse.Namespace) -> int:
    from gdbot.macro import preview, stats
    from gdbot.sim.level import BLOCK, Level, ObjType

    if a.tape:
        from gdbot.store import RunStore

        cfg = _load_config(a.config, run_dir=a.run_dir)
        tape = RunStore(cfg.run_dir).load_tape(a.tape)
        print(stats(tape, cfg.tick_rate))
        print(preview(tape, 100))
        return 0
    level = Level.load(_resolve_level(a.level))
    counts: dict[str, int] = {}
    for o in level.objects:
        counts[ObjType(o.t).name.lower()] = counts.get(ObjType(o.t).name.lower(), 0) + 1
    print(level)
    print("objects:", ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    if level.solution:
        print(f"author solution: {len(level.solution)} ticks ({len(level.solution) / 60:.1f}s)")
    print(f"length: {level.length / BLOCK:.0f} blocks at {level.speed}")
    return 0


def cmd_export(a: argparse.Namespace) -> int:
    from gdbot.macro import export_macro
    from gdbot.store import RunStore

    cfg = _load_config(a.config, run_dir=a.run_dir)
    tape = RunStore(cfg.run_dir).load_tape(a.tape)
    out = export_macro(tape, a.out, fps=cfg.tick_rate, level=cfg.name)
    print(f"wrote {out}")
    return 0


# ------------------------------------------------------------------ parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="gdbot", description="Self-learning Geometry Dash agent.")
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("-c", "--config", help="config file (JSON with comments)")
        sp.add_argument("--run-dir", help="where logs, tapes and policies go")

    g = sub.add_parser("generate", help="procedurally generate a verified-solvable level")
    g.add_argument("name")
    g.add_argument("--chunks", type=int, default=12)
    g.add_argument("--difficulty", type=float, default=0.6)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--speed", default="1x")
    g.add_argument("--out")
    g.set_defaults(fn=cmd_generate)

    t = sub.add_parser("train", help="learn a reactive policy in the simulator")
    common(t)
    t.add_argument("--level", help="bundled level name or path")
    t.add_argument("--iterations", type=int, default=60)
    t.add_argument("--ppo", action="store_true", help="use PPO (needs torch) instead of CEM")
    t.set_defaults(fn=cmd_train)

    s = sub.add_parser("solve", help="learn the level until it is cleared")
    common(s)
    s.add_argument("--level", help="bundled level name or path (simulator)")
    s.add_argument("--game", action="store_true", help="play the real game instead of the simulator")
    s.add_argument("--blackbox", action="store_true",
                   help="simulator, but restricted to real-game capabilities (no snapshots)")
    s.add_argument("--pretrain", type=int, default=0, help="CEM iterations to run first")
    s.add_argument("--seed-tape", help="name of a tape in the run dir to start from (e.g. 'human')")
    s.add_argument("--delay", type=int, default=5, help="seconds to switch to the game window (--game)")
    s.set_defaults(fn=cmd_solve)

    pl = sub.add_parser("play", help="replay a saved tape and count clean runs")
    common(pl)
    pl.add_argument("--tape", default="best")
    pl.add_argument("--level")
    pl.add_argument("--game", action="store_true")
    pl.add_argument("--runs", type=int, default=3)
    pl.add_argument("--delay", type=int, default=5, help="seconds to switch to the game window (--game)")
    pl.set_defaults(fn=cmd_play)

    r = sub.add_parser("record", help="record your own attempts to seed the solver")
    common(r)
    r.add_argument("--attempts", type=int, default=5)
    r.add_argument("--delay", type=int, default=5, help="seconds to switch to the game window")
    r.set_defaults(fn=cmd_record)

    cal = sub.add_parser("calibrate", help="click the progress bar on a screenshot; writes the config")
    common(cal)
    cal.add_argument("--screenshot", default="gd_screenshot.png")
    cal.add_argument("--delay", type=int, default=8, help="seconds to start the level before the screenshot")
    cal.set_defaults(fn=cmd_calibrate)

    sh = sub.add_parser("show", help="describe a level or a tape")
    common(sh)
    sh.add_argument("--level", default="skeletal_proxy")
    sh.add_argument("--tape")
    sh.set_defaults(fn=cmd_show)

    ex = sub.add_parser("export-macro", help="export a tape as frame-indexed press/release events")
    common(ex)
    ex.add_argument("--tape", default="best")
    ex.add_argument("--out", default="macro.json")
    ex.set_defaults(fn=cmd_export)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.fn(args) or 0)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
