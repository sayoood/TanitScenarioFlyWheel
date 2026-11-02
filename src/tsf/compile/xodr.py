"""TSIR road network -> ASAM OpenDRIVE 1.8 (.xodr), own writer (exact, link-continuous geometry).

Conventions (shared with xosc.py, osc2.py and TanitSim):
- linear road: road id 0, forward lanes = right lanes (-1..-n), backward lanes = left lanes (+1..+n).
- junction arm i: road id i, reference line starts on the junction boundary and points outward along
  `heading_deg`; inbound lanes = left lanes (+1 innermost), outbound lanes = right lanes (-1 innermost).
  TSIR `s` on an inbound lane is the distance to the stop line (= road s).
- connecting roads (ids JUNCTION_ID+1..): one lane (-1); reference line on the lane's left border;
  geometry = cubic Hermite paramPoly3 (normalized) between the exact lane-border points of both arms.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from lxml import etree

from tsf.compile.mapping import odr_position
from tsf.ir.scenario import SIGN_FACING_KINDS, Junction, LinearRoad, Scenario, StaticObject
from tsf.sim.odr import Geometry

JUNCTION_ID = 1000
SIGNAL_ID_BASE = 5000  # signal id of arm i = SIGNAL_ID_BASE + i
CONTROLLER_ID = 9000

# static objects (TSIR `static_objects`): <object> type/subtype per kind (OpenDRIVE 1.8 has no cone and no truck type)
OBJECT_TYPE = {"cone": ("obstacle", "cone"), "barrier": ("barrier", "none"), "parked_car": ("car", "parked"),
               "parked_truck": ("van", "truck"), "pole": ("pole", "none"), "bollard": ("pole", "bollard")}
# <signal country="DE"> StVO sign numbers (VzKat); construction signs are written as road-works signs
SIGN_CODE = {"speed_limit": "274", "stop": "206", "yield": "205", "no_entry": "267", "pedestrian_crossing": "350",
             "road_works": "123"}


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


# ----------------------------------------------------------------------------- xml helpers
def _f(v: float) -> str:
    return f"{v:.9g}"


def _sub(parent, tag, **attrs):
    return etree.SubElement(parent, tag, {k.rstrip("_"): (v if isinstance(v, str) else _f(v)) for k, v in attrs.items()})


def _lane(parent, lid: int, width: float, mark: str, color: str = "standard", pred=None, succ=None):
    ln = _sub(parent, "lane", id=str(lid), type="driving", level="false")
    if pred is not None or succ is not None:
        lk = _sub(ln, "link")
        if pred is not None:
            _sub(lk, "predecessor", id=str(pred))
        if succ is not None:
            _sub(lk, "successor", id=str(succ))
    _sub(ln, "width", sOffset=0.0, a=width, b=0.0, c=0.0, d=0.0)
    if mark != "none":
        _sub(ln, "roadMark", sOffset=0.0, type=mark, weight="standard", color=color, width=0.12,
             laneChange="none" if mark == "solid" else "both")
    return ln


def _road(root, rid: str, length: float, junction: str, kph: float | None):
    r = _sub(root, "road", name=f"road_{rid}", id=rid, length=length, junction=junction, rule="RHT")
    return r


def _road_type(r, kph: float):
    t = _sub(r, "type", s=0.0, type="town" if kph <= 60 else "motorway")
    _sub(t, "speed", max=kph, unit="km/h")


def _geometry(pv, g: Geometry):
    ge = _sub(pv, "geometry", s=g.s, x=g.x, y=g.y, hdg=g.hdg, length=g.length)
    if g.kind == "line":
        _sub(ge, "line")
    elif g.kind == "arc":
        _sub(ge, "arc", curvature=g.p["curvature"])
    elif g.kind == "spiral":
        _sub(ge, "spiral", curvStart=g.p["curvStart"], curvEnd=g.p["curvEnd"])
    else:
        _sub(ge, "paramPoly3", pRange="normalized", **{k: g.p[k] for k in
             ("aU", "bU", "cU", "dU", "aV", "bV", "cV", "dV")})


def _header(name: str):
    root = etree.Element("OpenDRIVE")
    _sub(root, "header", revMajor="1", revMinor="8", name=name, version="1.0",
         date=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"), vendor="Tanit Scenario Flywheel")
    return root


def _lanes_block(r, left: list[tuple], right: list[tuple]):
    """left/right: [(width, mark, pred, succ), ...] innermost first."""
    lanes = _sub(r, "lanes")
    sec = _sub(lanes, "laneSection", s=0.0)
    if left:
        L = _sub(sec, "left")
        for k in range(len(left), 0, -1):
            w, mark, p, s_ = left[k - 1]
            _lane(L, k, w, mark, pred=p, succ=s_)
    C = _sub(sec, "center")
    c = _sub(C, "lane", id="0", type="none", level="false")
    if left and right:
        _sub(c, "roadMark", sOffset=0.0, type="solid", weight="standard", color="yellow", width=0.15,
             laneChange="none")
    if right:
        R = _sub(sec, "right")
        for k in range(1, len(right) + 1):
            w, mark, p, s_ = right[k - 1]
            _lane(R, -k, w, mark, pred=p, succ=s_)


def _marks(n: int) -> list[str]:
    return ["broken"] * (n - 1) + ["solid"] if n else []


# ----------------------------------------------------------------------------- static objects and signs
def _lane_width(sc: Scenario, rid: str, lid: int) -> float:
    road = sc.road
    if isinstance(road, LinearRoad):
        return road.forward.lane_width if lid < 0 else road.backward.lane_width
    ap = road.approaches[int(rid)]
    return ap.inbound.lane_width if lid > 0 else ap.outbound.lane_width


def static_road_pose(sc: Scenario, o: StaticObject) -> tuple[str, int, float, float, float]:
    """(road id, lane id, road s, road t, heading relative to the road reference line) of a static object.
    Lane centres sit half a lane width per lane from the centre line; t_offset is + = left of the driving direction,
    which is the -t direction on left lanes (they drive against the reference line)."""
    rid, lid, s = odr_position(sc, o.position)
    centre = math.copysign((abs(lid) - 0.5) * _lane_width(sc, rid, lid), lid)
    t = centre + o.position.t_offset * (1.0 if lid < 0 else -1.0)
    hdg = math.radians(o.heading_deg) + (math.pi if lid > 0 else 0.0)
    return rid, lid, float(s), float(t), math.remainder(hdg, 2 * math.pi)


def _statics(r, sc: Scenario, rid: str):
    """<objects> (cones, barriers, parked vehicles, poles, bollards) and static <signals> (signs) of road `rid`."""
    objs, sigs = [], []
    for o in sc.static_objects:
        pose = static_road_pose(sc, o)
        if pose[0] == rid:
            (sigs if o.kind in SIGN_FACING_KINDS else objs).append((o, pose))
    if objs:
        ob = _sub(r, "objects")
        for o, (_, lid, s, t, hdg) in objs:
            l, w, h = o.dims()
            typ, sub = OBJECT_TYPE[o.kind]
            _sub(ob, "object", id=o.id, name=o.id, type=typ, subtype=sub, dynamic="no", orientation="none", s=s, t=t,
                 zOffset=0.0, hdg=hdg, length=l, width=w, height=h)
    if sigs:
        sg = r.find("signals")
        if sg is None:
            sg = _sub(r, "signals")
        for o, (_, lid, s, t, hdg) in sigs:
            l, w, h = o.dims()
            sign = "road_works" if o.kind == "construction_sign" else o.sign_type
            panel = min(w, h) if o.kind == "traffic_sign" else 0.6 * h
            at = dict(id=o.id, name=o.id, s=s, t=t, dynamic="no", orientation="+" if lid < 0 else "-", zOffset=h - panel,
                      country="DE", type=SIGN_CODE[sign], subtype="-1", height=panel, width=w)
            if abs(o.heading_deg) > 1e-9:
                at["hOffset"] = math.radians(o.heading_deg)
            if sign == "speed_limit":
                at.update(subtype=f"{o.sign_value:g}", value=o.sign_value, unit="km/h")
            _sub(sg, "signal", **at)


# ----------------------------------------------------------------------------- linear road
def _linear(sc: Scenario) -> etree._Element:
    road: LinearRoad = sc.road
    root = _header("tsf_linear")
    geoms, s, x, y, h = [], 0.0, 0.0, 0.0, 0.0
    for seg in road.segments:
        if seg.kind == "line":
            g = Geometry(s, x, y, h, seg.length, "line", {})
        elif seg.kind == "arc":
            g = Geometry(s, x, y, h, seg.length, "arc", {"curvature": seg.curvature_start})
        else:
            g = Geometry(s, x, y, h, seg.length, "spiral", {"curvStart": seg.curvature_start,
                                                           "curvEnd": seg.curvature_end})
        geoms.append(g)
        ex, ey, eh = g.eval(np.array([seg.length]))
        s, x, y, h = s + seg.length, float(ex[0]), float(ey[0]), float(eh[0])
    r = _road(root, "0", s, "-1", road.speed_limit_kph)
    _road_type(r, road.speed_limit_kph)
    pv = _sub(r, "planView")
    for g in geoms:
        _geometry(pv, g)
    fw, bw = road.forward, road.backward
    right = [(fw.lane_width, m, None, None) for m in _marks(fw.driving_lanes)]
    left = [(bw.lane_width, m, None, None) for m in _marks(bw.driving_lanes)] if bw else []
    if not left:  # one-way carriageway: leftmost forward lane gets a solid inner edge via centre lane mark
        pass
    _lanes_block(r, left, right)
    _statics(r, sc, "0")
    return root


# ----------------------------------------------------------------------------- junction
def _hermite(p0, h0, p1, h1) -> tuple[float, dict]:
    """Normalized cubic Hermite paramPoly3 in the local frame of (p0, h0); returns (arc length, coeffs)."""
    d = np.asarray(p1) - np.asarray(p0)
    c, s = math.cos(-h0), math.sin(-h0)
    du, dv = c * d[0] - s * d[1], s * d[0] + c * d[1]
    dh = math.remainder(h1 - h0, 2 * math.pi)
    chord = math.hypot(du, dv)
    k = chord * (1.0 + 0.27 * abs(dh) / (math.pi / 2))  # tangent scale ~ circular-arc fit
    t0u, t0v = k, 0.0
    t1u, t1v = k * math.cos(dh), k * math.sin(dh)
    coef = {"aU": 0.0, "bU": t0u, "cU": 3 * du - 2 * t0u - t1u, "dU": t0u + t1u - 2 * du,
            "aV": 0.0, "bV": t0v, "cV": 3 * dv - 2 * t0v - t1v, "dV": t0v + t1v - 2 * dv}
    p = np.linspace(0, 1, 401)
    u = coef["bU"] * p + coef["cU"] * p ** 2 + coef["dU"] * p ** 3
    v = coef["bV"] * p + coef["cV"] * p ** 2 + coef["dV"] * p ** 3
    return float(np.sum(np.hypot(np.diff(u), np.diff(v)))), coef


def _junction(sc: Scenario) -> etree._Element:
    J: Junction = sc.road
    root = _header("tsf_junction")
    max_half = max(max(a.inbound.driving_lanes * a.inbound.lane_width, a.outbound.driving_lanes * a.outbound.lane_width)
                   for a in J.approaches)
    R = J.corner_radius + max_half
    arms = []
    for i, ap in enumerate(J.approaches):
        h = math.radians(ap.heading_deg)
        arms.append((R * math.cos(h), R * math.sin(h), h))
    moves = lane_movements(sc)
    conn_ids = {}
    for n, mv in enumerate(moves):
        conn_ids[mv] = JUNCTION_ID + 1 + n
    # arm roads
    for i, ap in enumerate(J.approaches):
        x, y, h = arms[i]
        r = _road(root, str(i), ap.length, "-1", ap.speed_limit_kph)
        lk = _sub(r, "link")
        _sub(lk, "predecessor", elementType="junction", elementId=str(JUNCTION_ID))
        _road_type(r, ap.speed_limit_kph)
        pv = _sub(r, "planView")
        _geometry(pv, Geometry(0.0, x, y, h, ap.length, "line", {}))
        left = [(ap.inbound.lane_width, m, None, None) for m in _marks(ap.inbound.driving_lanes)]
        right = [(ap.outbound.lane_width, m, None, None) for m in _marks(ap.outbound.driving_lanes)]
        _lanes_block(r, left, right)
        _statics(r, sc, str(i))
        if J.signalized:
            sg = r.find("signals")
            if sg is None:
                sg = _sub(r, "signals")
            _sub(sg, "signal", s=0.5, t=ap.inbound.driving_lanes * ap.inbound.lane_width + 1.0,
                 id=str(SIGNAL_ID_BASE + i), name=f"tl_arm{i}", dynamic="yes", orientation="-", zOffset=2.5,
                 country="DE", type="1000001", subtype="-1", height=0.9, width=0.35)
    # connecting roads
    for mv, cid in conn_ids.items():
        fi, fl, ti, tl, turn = mv
        ai, aj = J.approaches[fi], J.approaches[ti]
        xi, yi, hi = arms[fi]
        xj, yj, hj = arms[ti]
        w = ai.inbound.lane_width
        # start: left border (driver's view) of inbound lane fl = arm-i border at t=(fl-1)w, heading into junction
        t0 = (fl - 1) * ai.inbound.lane_width
        p0 = (xi - math.sin(hi) * t0, yi + math.cos(hi) * t0)
        h0 = hi + math.pi
        # end: left border of outbound lane tl = arm-j border at t=-(tl-1)w, heading outward
        t1 = -(tl - 1) * aj.outbound.lane_width
        p1 = (xj - math.sin(hj) * t1, yj + math.cos(hj) * t1)
        length, coef = _hermite(p0, h0, p1, hj)
        r = _road(root, str(cid), length, str(JUNCTION_ID), ai.speed_limit_kph)
        lk = _sub(r, "link")
        _sub(lk, "predecessor", elementType="road", elementId=str(fi), contactPoint="start")
        _sub(lk, "successor", elementType="road", elementId=str(ti), contactPoint="start")
        _road_type(r, ai.speed_limit_kph)
        pv = _sub(r, "planView")
        _geometry(pv, Geometry(0.0, p0[0], p0[1], h0, length, "paramPoly3", coef))
        _lanes_block(r, [], [(w, "none", fl, -tl)])
        r.find("lanes/laneSection/center/lane").set("type", "none")
    jn = _sub(root, "junction", id=str(JUNCTION_ID), name="tsf_junction", type="default")
    for n, (mv, cid) in enumerate(conn_ids.items()):
        fi, fl, *_ = mv
        c = _sub(jn, "connection", id=str(n), incomingRoad=str(fi), connectingRoad=str(cid), contactPoint="start")
        _sub(c, "laneLink", **{"from": str(fl), "to": "-1"})
    if J.signalized:
        for i in range(len(J.approaches)):
            _sub(jn, "controller", id=str(CONTROLLER_ID + i), type="0", sequence=str(i))
            ctl = _sub(root, "controller", id=str(CONTROLLER_ID + i), name=f"ctrl_arm{i}", sequence=str(i))
            _sub(ctl, "control", signalId=str(SIGNAL_ID_BASE + i), type="0")
    # ASAM schema order: header, road*, controller*, junction*
    for ctl in root.findall("controller"):
        root.remove(ctl)
        root.insert(list(root).index(jn), ctl)
    return root


def build(sc: Scenario) -> etree._Element:
    if isinstance(sc.road, LinearRoad):
        return _linear(sc)
    if isinstance(sc.road, Junction):
        return _junction(sc)
    raise NotImplementedError("opendrive_file roads are loaded, not compiled")


def write(sc: Scenario, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if getattr(sc.road, "type", None) == "opendrive_file":
        path.write_bytes(Path(sc.road.path).read_bytes())
    else:
        etree.ElementTree(build(sc)).write(str(path), pretty_print=True, xml_declaration=True, encoding="UTF-8")
    return path
