"""TSIR road network -> ASAM OpenDRIVE (.xodr) via scenariogeneration.

Conventions (shared with xosc.py and TanitSim):
- linear road: road id 0, forward lanes = right lanes (-1..-n), backward lanes = left lanes (+1..+n).
- junction arm i: road id i, reference line starts at the junction and points outward along
  `heading_deg`; inbound lanes = left lanes (+1 innermost), outbound lanes = right lanes (-1 innermost).
  TSIR `s` on an inbound lane is the distance to the stop line, which equals road s.
"""
from __future__ import annotations

import math
from pathlib import Path

from scenariogeneration import xodr

from tsf.ir.scenario import Junction, LinearRoad, Scenario

JUNCTION_ID = 1000
SIGNAL_ID_BASE = 5000  # signal id of arm i = SIGNAL_ID_BASE + i


def turn_target(approaches, i: int, turn: str) -> int | None:
    """Arm index reached from inbound arm i for a turn choice (right-hand traffic)."""
    travel = approaches[i].heading_deg + 180.0
    want = {"straight": travel, "left": travel + 90.0, "right": travel - 90.0, "u_turn": travel + 180.0}[turn]
    best, err = None, 1e9
    for j, a in enumerate(approaches):
        if j == i and turn != "u_turn":
            continue
        e = abs(math.remainder(a.heading_deg - want, 360.0))
        if e < err:
            best, err = j, e
    return best if err < 50.0 else None


def lane_movements(sc: Scenario) -> list[tuple[int, int, int, int, str]]:
    """(from_arm, from_lane(+), to_arm, to_lane(-), turn) for every allowed junction movement."""
    road = sc.road
    assert isinstance(road, Junction)
    out = []
    for i, ap in enumerate(road.approaches):
        n_in = ap.inbound.driving_lanes
        for turn in ("left", "straight", "right"):
            j = turn_target(road.approaches, i, turn)
            if j is None:
                continue
            n_out = road.approaches[j].outbound.driving_lanes
            if turn == "left":
                pairs = [(1, 1)]
            elif turn == "right":
                pairs = [(n_in, n_out)]
            else:
                pairs = [(k, min(k, n_out)) for k in range(1, n_in + 1)]
            out += [(i, fl, j, tl, turn) for fl, tl in pairs]
    return out


def _linear(road: LinearRoad) -> xodr.OpenDrive:
    geoms = []
    for seg in road.segments:
        if seg.kind == "line":
            geoms.append(xodr.Line(seg.length))
        elif seg.kind == "arc":
            geoms.append(xodr.Arc(seg.curvature_start, length=seg.length))
        else:
            geoms.append(xodr.Spiral(seg.curvature_start, seg.curvature_end, length=seg.length))
    left = road.backward.driving_lanes if road.backward else 0
    r = xodr.create_road(geoms, id=0, left_lanes=left, right_lanes=road.forward.driving_lanes,
                         lane_width=road.forward.lane_width)
    _speed(r, road.speed_limit_kph)
    odr = xodr.OpenDrive("tsf_linear")
    odr.add_road(r)
    odr.adjust_roads_and_lanes()
    return odr


def _speed(r, kph: float):
    try:
        r.add_type(xodr.RoadType.town if kph <= 60 else xodr.RoadType.motorway, s=0, speed=kph / 3.6)
    except Exception:  # older/newer API variants: speed is optional metadata for us
        pass


def _junction(sc: Scenario) -> xodr.OpenDrive:
    road: Junction = sc.road
    roads = []
    for i, ap in enumerate(road.approaches):
        r = xodr.create_road(xodr.Line(ap.length), id=i, left_lanes=ap.inbound.driving_lanes,
                             right_lanes=ap.outbound.driving_lanes, lane_width=ap.inbound.lane_width)
        r.add_predecessor(xodr.ElementType.junction, JUNCTION_ID)
        _speed(r, ap.speed_limit_kph)
        if road.signalized:
            # one dynamic signal head facing inbound traffic at the stop line (left side of ref line)
            t = ap.inbound.driving_lanes * ap.inbound.lane_width + 1.0
            r.add_signal(xodr.Signal(s=0.5, t=t, country="DEU", Type="1000001", subtype="-1",
                                     dynamic=xodr.Dynamic.yes, orientation=xodr.Orientation.negative,
                                     zOffset=2.5, height=0.9, width=0.35, id=SIGNAL_ID_BASE + i,
                                     name=f"tl_arm{i}"))
        roads.append(r)
    radius = road.corner_radius + max(a.inbound.driving_lanes + a.outbound.driving_lanes
                                      for a in road.approaches) * 3.5 / 2
    jc = xodr.CommonJunctionCreator(id=JUNCTION_ID, name="tsf_junction", startnum=JUNCTION_ID + 1)
    for i, ap in enumerate(road.approaches):
        jc.add_incoming_road_circular_geometry(road_one=roads[i], radius=radius,
                                               angle=math.radians(ap.heading_deg), road_connection="predecessor")
    for fi, fl, ti, tl, _turn in lane_movements(sc):
        jc.add_connection(road_one_id=fi, road_two_id=ti, lane_one_id=fl, lane_two_id=-tl)
    odr = xodr.OpenDrive("tsf_junction")
    for r in roads:
        odr.add_road(r)
    odr.add_junction_creator(jc)
    odr.adjust_roads_and_lanes()
    return odr


def build(sc: Scenario) -> xodr.OpenDrive:
    if isinstance(sc.road, LinearRoad):
        return _linear(sc.road)
    if isinstance(sc.road, Junction):
        return _junction(sc)
    raise NotImplementedError("opendrive_file roads are loaded, not compiled")


def write(sc: Scenario, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if getattr(sc.road, "type", None) == "opendrive_file":
        path.write_bytes(Path(sc.road.path).read_bytes())
    else:
        build(sc).write_xml(str(path))
    return path
