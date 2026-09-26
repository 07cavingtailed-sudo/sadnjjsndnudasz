"""Shared fixtures: tiny hand-built levels with known answers."""

from __future__ import annotations

import pytest

from gdbot.sim.level import BLOCK, Level, Obj, ObjType


@pytest.fixture
def flat() -> Level:
    return Level("flat", length=20 * BLOCK)


@pytest.fixture
def one_spike() -> Level:
    return Level("one_spike", length=20 * BLOCK).add(Obj(ObjType.SPIKE, 6 * BLOCK, 0.0))


@pytest.fixture
def spikes() -> Level:
    """Four spikes, each needing its own jump."""
    lv = Level("spikes", length=40 * BLOCK)
    for i in range(4):
        lv.add(Obj(ObjType.SPIKE, (6 + i * 7) * BLOCK, 0.0))
    return lv
