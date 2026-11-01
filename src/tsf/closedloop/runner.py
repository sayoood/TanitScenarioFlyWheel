"""Closed-loop simulation: TanitSim owns the world, a policy drives the ego.

Tiers (same loop, different sensor stage):
  T0 kinematic  : policy sees privileged ego state only (test policies, planner debugging)
  T1 neural     : Cosmos3-Nano-Sim-Auto renders the rig from WSM controls of the current world state
                  (tsf.cosmos), policy adapter consumes camera frames (e.g. TanitAD REFe SFT-4)
The loop is sim-time-decoupled: the world waits while the sensor stage generates the next chunk.
Scoring follows NAVSIM EPDMS structure (multipliers NC, DAC, TLC; weighted TTC, EP, comfort).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from shapely.geometry import Point
from shapely.ops import unary_union

from tsf.ir.scenario import Scenario
from tsf.sim.engine import SimLog, TanitSim
from tsf.sim.odr import OdrMap


# ----------------------------------------------------------------------------- test policies
@dataclass
class ConstantVelocity:
    """Keeps the initial speed and the current heading (no steering): the simplest closed-loop sanity check."""
    horizon: int = 8
    dt: float = 0.5
    v: float | None = None
    name: str = "const_vel"

    def __call__(self, sim: TanitSim, ego) -> dict:
        if self.v is None:
            self.v = ego.v
        return {"traj": [[self.v * self.dt * k, 0.0] for k in range(1, self.horizon + 1)], "dt": self.dt}


@dataclass
class AccelDecel:
    """Accelerate at `a_up` for `t_accel` s, then brake at `a_down` to standstill and hold, following the lane."""
    a_up: float = 1.5
    t_accel: float = 6.0
    a_down: float = 2.5
    horizon: int = 8
    dt: float = 0.5
    name: str = "accel_decel"

    def speed_at(self, t: float, v0: float) -> float:
        v_peak = v0 + self.a_up * self.t_accel
        if t <= self.t_accel:
            return v0 + self.a_up * t
        return max(0.0, v_peak - self.a_down * (t - self.t_accel))

    def __call__(self, sim: TanitSim, ego) -> dict:
        if not hasattr(self, "_v0"):
            self._v0 = ego.v
        pts, s = [], 0.0
        for k in range(1, self.horizon + 1):
            v = self.speed_at(sim.t + k * self.dt, self._v0)
            s += v * self.dt
            # follow the lane centre ahead (privileged map access in tier T0)
            x, y, _ = ego.path.pose(ego.p + s)
            c, sn = math.cos(-ego.yaw), math.sin(-ego.yaw)
            dx, dy = x - ego.x, y - ego.y
            pts.append([c * dx - sn * dy, sn * dx + c * dy])
        return {"traj": pts, "dt": self.dt}


# ----------------------------------------------------------------------------- scoring
@dataclass
class ClosedLoopResult:
    policy: str
    log: SimLog
    score: dict = field(default_factory=dict)


def drivable_area(m: OdrMap):
    from shapely.geometry import Polygon
    return unary_union([Polygon(np.concatenate([ln.left_border, ln.right_border[::-1]])).buffer(0)
                        for ln in m.lanes.values()]).buffer(0.3)


def epdms(sim: TanitSim, log: SimLog, ttc_threshold: float = 0.9) -> dict:
    """EPDMS-style score (NAVSIM v2 structure): NC*DAC*TLC * (5*TTC + 5*EP + 2*C) / 12."""
    mt = log.metrics
    e = log.roles.index("ego")
    S = log.states
    area = drivable_area(sim.m)
    off_road = [float(log.t[i]) for i in range(0, len(log.t), 5) if not area.contains(Point(S[i, e, 0], S[i, e, 1]))]
    nc = 0.0 if mt["collision"] else 1.0
    dac = 0.0 if off_road else 1.0
    tlc = 0.0 if mt["red_light_violations"] else 1.0
    ttc = 1.0 if mt["min_ttc"] is None or mt["min_ttc"] >= ttc_threshold else 0.0
    exp = max(1.0, float(np.mean(S[0, :, 3])) * float(log.t[-1]))  # progress expected at traffic speed
    ep = float(min(1.0, mt["ego_distance"] / exp))
    comfort = 1.0 if (mt["ego_max_decel"] <= 4.0 and mt["ego_max_accel"] <= 4.0 and mt["ego_max_abs_jerk"] <= 15) else 0.0
    score = nc * dac * tlc * (5 * ttc + 5 * ep + 2 * comfort) / 12
    return {"EPDMS": round(score, 4), "NC": nc, "DAC": dac, "TLC": tlc, "TTC": ttc, "EP": round(ep, 3),
            "C": comfort, "off_road_t": off_road[:5]}


def run(sc: Scenario, m: OdrMap, policy) -> ClosedLoopResult:
    sim = TanitSim(sc, m, ego_policy=policy)
    log = sim.run()
    res = ClosedLoopResult(getattr(policy, "name", type(policy).__name__), log, epdms(sim, log))
    res.sim = sim
    return res
