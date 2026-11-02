"""Regression: a policy-driven (external) ego starts at its scenario pose, not at the map origin.

The first T1 run and every closed-loop baseline on the UC1 junction started the ego at (0, 0), the junction centre,
which the red-light check counted as entering the junction on red at t = 0.05 s (EPDMS 0)."""
from __future__ import annotations

import math
from pathlib import Path

from tsf.closedloop.runner import ConstantVelocity
from tsf.sim.engine import TanitSim
from tsf.verify import ladder

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "uc1_left_turn_intersection.json"


def test_external_ego_starts_at_scenario_pose(tmp_path):
    rep = ladder.run(EXAMPLE.read_text(), tmp_path, qc=False)
    sim = TanitSim(rep._sim.sc, rep._sim.m, ego_policy=ConstantVelocity())
    ego = next(a for a in sim.agents.values() if a.role == "ego")
    x, y, h = ego.path.pose(ego.p, ego.d)
    assert math.hypot(ego.x - x, ego.y - y) < 1e-6 and abs(ego.yaw - h) < 1e-6
    for _ in range(20):                                  # 1 s at 12.5 m/s: still far before the stop line
        sim.step()
    assert ego.path.keys[ego.path.lane_index(ego.p)] == ego.path.keys[0]
