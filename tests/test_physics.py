import numpy as np
import pytest

from gdbot.sim.level import BLOCK, Level, Obj, ObjType
from gdbot.sim.physics import TPS, World
from gdbot.types import Mode, Outcome


def test_flat_level_completes_without_input(flat):
    attempt = World(flat).run_tape(np.zeros(TPS * 10, dtype=np.uint8))
    assert attempt.outcome is Outcome.COMPLETE
    assert attempt.end_progress == pytest.approx(1.0)


def test_jump_is_about_two_blocks_high_and_four_across(flat):
    w = World(Level("long", length=200 * BLOCK))
    x0 = w.player.x
    w.step(True)
    apex = 0.0
    while not w.player.on_ground:
        w.step(False)
        apex = max(apex, w.player.y)
    assert 1.8 * BLOCK < apex < 2.2 * BLOCK
    assert 4.0 * BLOCK < w.player.x - x0 < 4.7 * BLOCK


def test_spike_kills_and_a_jump_clears_it(one_spike):
    assert World(one_spike).run_tape(np.zeros(400, np.uint8)).outcome is Outcome.DEAD
    tape = np.zeros(400, np.uint8)
    tape[22] = 1  # jump shortly before the spike
    assert World(one_spike).run_tape(tape).outcome is Outcome.COMPLETE


def test_block_side_kills_but_top_is_walkable():
    wall = Level("wall", length=20 * BLOCK).add(Obj(ObjType.SOLID, 6 * BLOCK, 0.0, w=6 * BLOCK, h=BLOCK))
    assert World(wall).run_tape(np.zeros(400, np.uint8)).outcome is Outcome.DEAD

    # Some take-off tick must land the cube on the ledge and let it run across.
    for takeoff in range(40):
        tape = np.zeros(400, np.uint8)
        tape[takeoff] = 1
        w = World(wall)
        stood_on_top = False
        for a in tape:
            if w.step(bool(a)) is not Outcome.RUNNING:
                break
            on_ledge = 6 * BLOCK - BLOCK < w.player.x < 12 * BLOCK
            stood_on_top |= on_ledge and w.player.on_ground and w.player.y == pytest.approx(BLOCK)
        if w.outcome is Outcome.COMPLETE and stood_on_top:
            return
    pytest.fail("no take-off tick lands on the ledge")


def test_same_tape_same_result(spikes):
    rng = np.random.default_rng(0)
    tape = (rng.random(600) < 0.05).astype(np.uint8)
    a, b = World(spikes), World(spikes)
    ra, rb = a.run_tape(tape), b.run_tape(tape)
    assert (ra.outcome, ra.ticks, ra.end_progress) == (rb.outcome, rb.ticks, rb.end_progress)
    assert a.snapshot() == b.snapshot()


def test_restore_continues_identically(spikes):
    tape = np.zeros(600, np.uint8)
    for t in (22, 64, 106, 148):
        tape[t] = 1
    w = World(spikes)
    for a in tape[:100]:
        w.step(bool(a))
    snap = w.snapshot()
    for a in tape[100:]:
        w.step(bool(a))
    first = w.snapshot()
    w.restore(snap)
    for a in tape[100:]:
        w.step(bool(a))
    assert w.snapshot() == first


def test_portals_change_mode_gravity_and_speed_once():
    lv = Level("portals", length=30 * BLOCK).add(
        Obj(ObjType.PORTAL_MODE, 3 * BLOCK, 0.0, w=18, h=300, param=int(Mode.SHIP)),
        Obj(ObjType.PORTAL_GRAVITY, 8 * BLOCK, 0.0, w=18, h=300, param=2),
        Obj(ObjType.PORTAL_SPEED, 12 * BLOCK, 0.0, w=18, h=300, param=3),
    )
    w = World(lv)
    for _ in range(int(20 * BLOCK / 5.0)):
        w.step(False)
    p = w.player
    assert p.mode is Mode.SHIP
    # a toggle portal overlaps the player for several ticks and must fire once
    assert p.grav == -1.0
    assert p.speed == pytest.approx(468.0)


def _first_arc_peak(level: Level, pressed) -> float:
    """Highest point reached before the cube lands again."""
    w = World(level)
    top, airborne = 0.0, False
    for t in range(200):
        w.step(pressed(t))
        top = max(top, w.player.y)
        airborne |= not w.player.on_ground
        if airborne and w.player.on_ground:
            break
    return top


def test_orb_needs_a_fresh_press_while_touching_it():
    lv = Level("orb", length=40 * BLOCK).add(Obj(ObjType.ORB, 8 * BLOCK, 1.5 * BLOCK))
    plain = _first_arc_peak(Level("none", length=40 * BLOCK), lambda t: t == 30)

    boosted = []
    for takeoff in range(20, 40):
        # holding from take-off straight through the ring: no new press edge
        held = _first_arc_peak(lv, lambda t, k=takeoff: t >= k)
        assert held == pytest.approx(plain, abs=1e-6)
        for repress in range(takeoff + 2, takeoff + 25):
            peak = _first_arc_peak(lv, lambda t, k=takeoff, r=repress: t == k or t == r)
            if peak > plain + 10.0:
                boosted.append((takeoff, repress))
    assert boosted, "a fresh press inside the ring never produced a second impulse"


def test_spawn_at_reconstructs_portal_state():
    lv = Level("spawn", length=40 * BLOCK).add(
        Obj(ObjType.PORTAL_GRAVITY, 5 * BLOCK, 0.0, w=18, h=300, param=1),
        Obj(ObjType.PORTAL_MODE, 15 * BLOCK, 0.0, w=18, h=300, param=int(Mode.WAVE)),
    )
    w = World(lv)
    w.spawn_at(0.3)
    assert w.player.grav == -1.0 and w.player.mode is Mode.CUBE and w.player.on_ground
    w.spawn_at(0.6)
    assert w.player.mode is Mode.WAVE
    # portals already passed are marked used and must not fire again
    assert len(w._used) == 2
