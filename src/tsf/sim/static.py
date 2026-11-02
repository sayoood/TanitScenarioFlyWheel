"""Static environment objects (TSIR `static_objects`) as TanitSim bodies.

A `StaticBody` is a stationary oriented box on the ground plane with a world pose resolved from its TSIR
`LanePosition` on the loaded map. The simulator treats every body as a standing obstacle: collisions and TTC
(`engine.metrics`), obstacle following in the driver model (`TanitSim._leader_gap`) and the lane-change safety
check. Renderers and exporters read them from `sim.static` / `log.static`.

Pose convention: `yaw` is the heading of the body's local x axis. For panels (traffic signs, construction signs)
the local x axis is the front-face normal, i.e. with `heading_deg` 0 it points against the lane's driving
direction, towards the approaching traffic; this is also the RDS-HQ cuboid convention.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from tsf.ir.scenario import SIGN_FACING_KINDS, Scenario


@dataclass
class StaticBody:
    id: str
    kind: str
    x: float                         # centre on the ground, sim frame
    y: float
    yaw: float                       # local x axis (front-face normal for panels)
    lwh: tuple[float, float, float]
    lane: tuple[str, int, int]       # OpenDRIVE lane key (road id, lane section, lane id)
    s: float                         # progress along that lane in driving direction (m)
    t: float                         # lateral offset from the lane centre (m, + = left of the driving direction)
    sign_type: str | None = None
    sign_value: float | None = None

    @property
    def is_panel(self) -> bool:
        return self.kind in SIGN_FACING_KINDS

    @property
    def road(self) -> str:
        return self.lane[0]

    def corners(self) -> np.ndarray:
        """Footprint corners (4, 2), counter-clockwise: l along the local x axis, w along local y."""
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        dx = np.array([self.lwh[0], self.lwh[0], -self.lwh[0], -self.lwh[0]]) / 2
        dy = np.array([self.lwh[1], -self.lwh[1], -self.lwh[1], self.lwh[1]]) / 2
        return np.stack([self.x + c * dx - s * dy, self.y + s * dx + c * dy], 1)

    def extent(self, heading: float) -> tuple[float, float]:
        """Half extents (along, across) of the footprint in a frame rotated to `heading` (rad)."""
        d = self.yaw - heading
        c, s = abs(math.cos(d)), abs(math.sin(d))
        return (c * self.lwh[0] + s * self.lwh[1]) / 2, (s * self.lwh[0] + c * self.lwh[1]) / 2

    def to_dict(self) -> dict:
        """JSON-ready record for scene exports."""
        d = asdict(self)
        d["lane"] = "/".join(map(str, self.lane))
        d["lwh"] = [round(float(v), 3) for v in self.lwh]
        for k in ("x", "y", "yaw", "s", "t"):
            d[k] = round(float(d[k]), 4)
        return {k: v for k, v in d.items() if v is not None}


def build_static(sc: Scenario, m) -> list[StaticBody]:
    """Resolve every TSIR static object on the loaded map `m`."""
    from tsf.sim.engine import tsir_lane_key         # engine imports this module

    out = []
    for o in sc.static_objects:
        key, prog = tsir_lane_key(sc, m, o.position)
        x, y, h = m.lanes[key].pose(prog, o.position.t_offset)
        yaw = h + math.radians(o.heading_deg) + (math.pi if o.kind in SIGN_FACING_KINDS else 0.0)
        out.append(StaticBody(o.id, o.kind, float(x), float(y), float(math.remainder(yaw, 2 * math.pi)),
                              tuple(float(v) for v in o.dims()), key, float(prog), float(o.position.t_offset),
                              o.sign_type, o.sign_value))
    return out


def explicit_speed_roads(static: list[StaticBody]) -> set[str]:
    """Roads that carry an explicit speed-limit sign: the map-derived default signs are left out there."""
    return {b.road for b in static if b.sign_type == "speed_limit"}
