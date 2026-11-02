"""TanitSim OpenDRIVE engine: parse .xodr (1.4-1.8) and evaluate exact road/lane geometry.

Independent from the writer used to *generate* maps, so generated maps are cross-checked
by a second implementation. Geometry follows ASAM OpenDRIVE: reference line (line, arc,
spiral via Fresnel integrals, poly3, paramPoly3), laneOffset, cubic lane widths per lane
section, road/lane linkage and junction connections, signals and controllers.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from lxml import etree
from scipy.special import fresnel

LaneKey = tuple[str, int, int]  # (road_id, lane_section_index, lane_id)


# ----------------------------------------------------------------------------- reference line
@dataclass
class Geometry:
    s: float
    x: float
    y: float
    hdg: float
    length: float
    kind: str
    p: dict

    def eval(self, ds: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Local evaluation at ds in [0, length] -> world x, y, heading."""
        k = self.kind
        if k == "line":
            u, v, th = ds, np.zeros_like(ds), np.zeros_like(ds)
        elif k == "arc":
            c = self.p["curvature"]
            if abs(c) < 1e-12:
                u, v, th = ds, np.zeros_like(ds), np.zeros_like(ds)
            else:
                th = c * ds
                u, v = np.sin(th) / c, (1 - np.cos(th)) / c
        elif k == "spiral":
            u, v, th = _spiral(ds, self.p["curvStart"], self.p["curvEnd"], self.length)
        elif k == "poly3":
            a, b, c, d = (self.p[n] for n in "abcd")
            u = _arc_param_to_u(ds, b, c, d)
            v = a + b * u + c * u**2 + d * u**3
            th = np.arctan(b + 2 * c * u + 3 * d * u**2)
        elif k == "paramPoly3":
            P = self.p
            rng = self.length if P.get("pRange", "arcLength") == "arcLength" else 1.0
            t = ds / self.length * rng
            u = P["aU"] + P["bU"] * t + P["cU"] * t**2 + P["dU"] * t**3
            v = P["aV"] + P["bV"] * t + P["cV"] * t**2 + P["dV"] * t**3
            du = P["bU"] + 2 * P["cU"] * t + 3 * P["dU"] * t**2
            dv = P["bV"] + 2 * P["cV"] * t + 3 * P["dV"] * t**2
            th = np.arctan2(dv, du)
        else:
            raise ValueError(f"unsupported geometry {k}")
        ch, sh = math.cos(self.hdg), math.sin(self.hdg)
        return self.x + ch * u - sh * v, self.y + sh * u + ch * v, self.hdg + th


def _spiral(ds, k0, k1, length):
    """Clothoid with curvature k0 -> k1 over `length`, starting at origin heading 0."""
    cdot = (k1 - k0) / length
    if abs(cdot) < 1e-12:
        if abs(k0) < 1e-12:
            return ds, np.zeros_like(ds), np.zeros_like(ds)
        th = k0 * ds
        return np.sin(th) / k0, (1 - np.cos(th)) / k0, th
    # shift to the clothoid's zero-curvature point, evaluate standard clothoid, transform back
    s0 = k0 / cdot
    a = math.sqrt(math.pi / abs(cdot))
    sgn = 1.0 if cdot > 0 else -1.0

    def std(s):
        S, C = fresnel(s / a)
        return a * C, sgn * a * S, 0.5 * cdot * s * s

    x0, y0, t0 = std(np.asarray(s0))
    x, y, th = std(ds + s0)
    dx, dy = x - x0, y - y0
    c, s = math.cos(-t0), math.sin(-t0)
    return c * dx - s * dy, s * dx + c * dy, th - t0


def _arc_param_to_u(ds, b, c, d, n=512):
    """poly3 is parametrised by u (local x); map arc length -> u numerically."""
    umax = float(ds.max()) * 1.2 + 1.0
    uu = np.linspace(0, umax, n)
    dv = b + 2 * c * uu + 3 * d * uu**2
    seg = np.sqrt(1 + dv**2)
    sl = np.concatenate([[0], np.cumsum(0.5 * (seg[1:] + seg[:-1]) * np.diff(uu))])
    return np.interp(ds, sl, uu)


# ----------------------------------------------------------------------------- data model
@dataclass
class Poly:
    s: float
    a: float
    b: float
    c: float
    d: float

    def __call__(self, ds):
        return self.a + self.b * ds + self.c * ds**2 + self.d * ds**3


def _piecewise(polys: list[Poly], s: np.ndarray, base: float = 0.0) -> np.ndarray:
    """Evaluate a list of polynomial records (sorted by s offset) at s (relative to `base`)."""
    out = np.zeros_like(s)
    if not polys:
        return out
    starts = np.array([p.s for p in polys])
    idx = np.clip(np.searchsorted(starts, s - base, side="right") - 1, 0, len(polys) - 1)
    for i, p in enumerate(polys):
        m = idx == i
        if m.any():
            out[m] = p(s[m] - base - p.s)
    return out


@dataclass
class LaneDef:
    id: int
    type: str
    widths: list[Poly]
    predecessor: int | None = None
    successor: int | None = None
    road_mark: str = "solid"


@dataclass
class LaneSection:
    s: float
    lanes: dict[int, LaneDef]
    s_end: float = 0.0


@dataclass
class Signal:
    id: str
    road: str
    s: float
    t: float
    type: str
    orientation: str
    dynamic: bool
    x: float = 0.0
    y: float = 0.0
    hdg: float = 0.0
    subtype: str = ""
    value: float | None = None
    name: str = ""
    height: float = 0.0
    width: float = 0.0
    z_offset: float = 0.0


@dataclass
class RoadObject:
    """`<object>`: a static object placed at (s, t) of a road. x, y, hdg are the world pose of its local x axis."""
    id: str
    road: str
    s: float
    t: float
    type: str
    subtype: str = ""
    name: str = ""
    hdg_rel: float = 0.0
    length: float = 0.0
    width: float = 0.0
    height: float = 0.0
    z_offset: float = 0.0
    x: float = 0.0
    y: float = 0.0
    hdg: float = 0.0


@dataclass
class Road:
    id: str
    length: float
    junction: str
    geoms: list[Geometry]
    lane_offset: list[Poly]
    sections: list[LaneSection]
    link: dict = field(default_factory=dict)
    speed_kph: float | None = None
    signals: list[Signal] = field(default_factory=list)
    objects: list[RoadObject] = field(default_factory=list)

    def ref(self, s: np.ndarray):
        s = np.clip(np.asarray(s, dtype=float), 0, self.length)
        starts = np.array([g.s for g in self.geoms])
        idx = np.clip(np.searchsorted(starts, s, side="right") - 1, 0, len(self.geoms) - 1)
        x, y, h = np.empty_like(s), np.empty_like(s), np.empty_like(s)
        for i, g in enumerate(self.geoms):
            m = idx == i
            if m.any():
                x[m], y[m], h[m] = g.eval(np.clip(s[m] - g.s, 0, g.length))
        return x, y, h

    def lane_t(self, sec_i: int, lane_id: int, s: np.ndarray, where: str = "center") -> np.ndarray:
        """Lateral coordinate of a lane's centre / inner / outer border at s (within section)."""
        sec = self.sections[sec_i]
        t = _piecewise(self.lane_offset, s)
        sign = 1 if lane_id > 0 else -1
        inner = np.zeros_like(s)
        for lid in range(1, abs(lane_id)):
            ld = sec.lanes.get(sign * lid)
            if ld:
                inner += _piecewise(ld.widths, s, sec.s)
        w = _piecewise(sec.lanes[lane_id].widths, s, sec.s)
        frac = {"inner": 0.0, "center": 0.5, "outer": 1.0}[where]
        return t + sign * (inner + frac * w)


@dataclass
class Connection:
    id: str
    incoming: str
    connecting: str
    contact: str
    lane_links: list[tuple[int, int]]


@dataclass
class JunctionDef:
    id: str
    connections: list[Connection]
    controllers: list[str] = field(default_factory=list)


@dataclass
class Lane:
    """Sampled drivable lane in driving direction (TanitSim's routing primitive)."""
    key: LaneKey
    xy: np.ndarray            # (N, 2) centre line in driving direction
    hdg: np.ndarray           # (N,)
    width: np.ndarray         # (N,)
    s_road: np.ndarray        # (N,) road s of each sample
    left_border: np.ndarray   # (N, 2) w.r.t. driving direction
    right_border: np.ndarray
    junction: str
    speed: float              # m/s
    successors: list[LaneKey] = field(default_factory=list)
    predecessors: list[LaneKey] = field(default_factory=list)
    left: LaneKey | None = None   # adjacent lane, same direction, to the left
    right: LaneKey | None = None
    cum: np.ndarray = field(default=None)  # cumulative arc length

    def __post_init__(self):
        d = np.linalg.norm(np.diff(self.xy, axis=0), axis=1)
        self.cum = np.concatenate([[0.0], np.cumsum(d)])

    @property
    def length(self) -> float:
        return float(self.cum[-1])

    def pose(self, s: float, d: float = 0.0) -> tuple[float, float, float]:
        s = min(max(s, 0.0), self.length)
        x = np.interp(s, self.cum, self.xy[:, 0])
        y = np.interp(s, self.cum, self.xy[:, 1])
        h = _interp_angle(s, self.cum, self.hdg)
        return x - math.sin(h) * d, y + math.cos(h) * d, h


def _interp_angle(s, xs, angles):
    un = np.unwrap(angles)
    return float(np.interp(s, xs, un))


# ----------------------------------------------------------------------------- map
@dataclass
class OdrMap:
    roads: dict[str, Road]
    junctions: dict[str, JunctionDef]
    controllers: dict[str, list[str]]
    lanes: dict[LaneKey, Lane]
    header: dict

    def signals(self, dynamic_only: bool = True) -> list[Signal]:
        """Traffic lights (dynamic signals) by default; `dynamic_only=False` also returns static signs."""
        return [sg for r in self.roads.values() for sg in r.signals if sg.dynamic or not dynamic_only]

    def signs(self) -> list[Signal]:
        """Static `<signal>`s (traffic signs)."""
        return [sg for r in self.roads.values() for sg in r.signals if not sg.dynamic]

    def objects(self) -> list[RoadObject]:
        return [ob for r in self.roads.values() for ob in r.objects]

    def bounds(self) -> tuple[float, float, float, float]:
        pts = np.concatenate([ln.xy for ln in self.lanes.values()])
        return float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(pts[:, 1].max())

    def nearest_lane(self, x: float, y: float, hdg: float | None = None) -> tuple[LaneKey, float, float]:
        """Map-match a world point -> (lane, s along lane, signed lateral offset)."""
        best = (None, 0.0, 0.0, 1e18)
        for key, ln in self.lanes.items():
            dd = np.hypot(ln.xy[:, 0] - x, ln.xy[:, 1] - y)
            i = int(dd.argmin())
            cost = dd[i] + (0 if hdg is None else 4.0 * abs(math.remainder(ln.hdg[i] - hdg, 2 * math.pi)))
            if cost < best[3]:
                h = ln.hdg[i]
                d = -math.sin(h) * (x - ln.xy[i, 0]) + math.cos(h) * (y - ln.xy[i, 1])
                best = (key, float(ln.cum[i]), float(d), cost)
        return best[0], best[1], best[2]


def _f(e, k, default=0.0):
    v = e.get(k)
    return float(v) if v is not None else default


def _polys(parent, tag) -> list[Poly]:
    return sorted((Poly(_f(e, "sOffset", _f(e, "s")), _f(e, "a"), _f(e, "b"), _f(e, "c"), _f(e, "d"))
                   for e in parent.findall(tag)), key=lambda p: p.s)


def load(path: str | Path, ds: float = 0.5) -> OdrMap:
    root = etree.parse(str(path)).getroot()
    hdr = root.find("header")
    header = dict(hdr.attrib) if hdr is not None else {}
    roads: dict[str, Road] = {}
    for r in root.findall("road"):
        geoms = []
        for g in r.find("planView").findall("geometry"):
            child = g[0]
            p = {k: float(v) for k, v in child.attrib.items() if k != "pRange"}
            if child.tag == "paramPoly3":
                p["pRange"] = child.get("pRange", "arcLength")
            geoms.append(Geometry(_f(g, "s"), _f(g, "x"), _f(g, "y"), _f(g, "hdg"), _f(g, "length"),
                                  child.tag, p))
        lanes_e = r.find("lanes")
        sections = []
        for sec in lanes_e.findall("laneSection"):
            lanes = {}
            for side in ("left", "right"):
                se = sec.find(side)
                if se is None:
                    continue
                for ln in se.findall("lane"):
                    link = ln.find("link")
                    pre = suc = None
                    if link is not None:
                        if link.find("predecessor") is not None:
                            pre = int(link.find("predecessor").get("id"))
                        if link.find("successor") is not None:
                            suc = int(link.find("successor").get("id"))
                    rm = ln.find("roadMark")
                    lanes[int(ln.get("id"))] = LaneDef(int(ln.get("id")), ln.get("type", "driving"),
                                                      _polys(ln, "width"), pre, suc,
                                                      rm.get("type", "solid") if rm is not None else "none")
            sections.append(LaneSection(_f(sec, "s"), lanes))
        length = _f(r, "length")
        for i, sec in enumerate(sections):
            sec.s_end = sections[i + 1].s if i + 1 < len(sections) else length
        link = {}
        le = r.find("link")
        if le is not None:
            for kind in ("predecessor", "successor"):
                x = le.find(kind)
                if x is not None:
                    link[kind] = (x.get("elementType"), x.get("elementId"), x.get("contactPoint"))
        speed = None
        te = r.find("type/speed")
        if te is not None and te.get("max") not in (None, "no limit", "undefined"):
            speed = float(te.get("max")) * (1.609344 if te.get("unit") == "mph" else 1.0)
            if te.get("unit") == "m/s":
                speed *= 3.6
        road = Road(r.get("id"), length, r.get("junction", "-1"), geoms, _polys(lanes_e, "laneOffset"),
                    sections, link, speed)
        sigs = r.find("signals")
        if sigs is not None:
            for sg in sigs.findall("signal"):
                road.signals.append(Signal(sg.get("id"), road.id, _f(sg, "s"), _f(sg, "t"), sg.get("type", ""),
                                           sg.get("orientation", "+"), sg.get("dynamic", "no") == "yes",
                                           subtype=sg.get("subtype", ""),
                                           value=float(sg.get("value")) if sg.get("value") is not None else None,
                                           name=sg.get("name", ""), height=_f(sg, "height"), width=_f(sg, "width"),
                                           z_offset=_f(sg, "zOffset")))
        objs = r.find("objects")
        if objs is not None:
            for ob in objs.findall("object"):
                road.objects.append(RoadObject(ob.get("id"), road.id, _f(ob, "s"), _f(ob, "t"), ob.get("type", "none"),
                                               ob.get("subtype", ""), ob.get("name", ""), _f(ob, "hdg"),
                                               _f(ob, "length"), _f(ob, "width"), _f(ob, "height"), _f(ob, "zOffset")))
        roads[road.id] = road
    junctions = {}
    for j in root.findall("junction"):
        conns = [Connection(c.get("id"), c.get("incomingRoad"), c.get("connectingRoad"),
                            c.get("contactPoint", "start"),
                            [(int(ll.get("from")), int(ll.get("to"))) for ll in c.findall("laneLink")])
                 for c in j.findall("connection")]
        junctions[j.get("id")] = JunctionDef(j.get("id"), conns, [c.get("id") for c in j.findall("controller")])
    controllers = {c.get("id"): [cc.get("signalId") for cc in c.findall("control")] for c in root.findall("controller")}
    m = OdrMap(roads, junctions, controllers, {}, header)
    _sample_lanes(m, ds)
    _link_lanes(m)
    for r in roads.values():
        for sg in r.signals:
            x, y, h = r.ref(np.array([sg.s]))
            n = -math.sin(h[0]), math.cos(h[0])
            sg.x, sg.y = float(x[0] + n[0] * sg.t), float(y[0] + n[1] * sg.t)
            sg.hdg = float(h[0] + (math.pi if sg.orientation == "-" else 0.0))
        for ob in r.objects:
            x, y, h = r.ref(np.array([ob.s]))
            ob.x, ob.y = float(x[0] - math.sin(h[0]) * ob.t), float(y[0] + math.cos(h[0]) * ob.t)
            ob.hdg = float(h[0] + ob.hdg_rel)
    return m


def _sample_lanes(m: OdrMap, ds: float):
    for r in m.roads.values():
        for si, sec in enumerate(r.sections):
            n = max(2, int(math.ceil((sec.s_end - sec.s) / ds)) + 1)
            s = np.linspace(sec.s, sec.s_end, n)
            x, y, h = r.ref(s)
            nx, ny = -np.sin(h), np.cos(h)
            for lid, ld in sec.lanes.items():
                if lid == 0 or ld.type not in ("driving", "entry", "exit", "onRamp", "offRamp", "bidirectional"):
                    continue
                tc = r.lane_t(si, lid, s, "center")
                ti, to = r.lane_t(si, lid, s, "inner"), r.lane_t(si, lid, s, "outer")
                w = np.abs(to - ti)
                c = np.stack([x + nx * tc, y + ny * tc], 1)
                bi = np.stack([x + nx * ti, y + ny * ti], 1)
                bo = np.stack([x + nx * to, y + ny * to], 1)
                if lid < 0:   # right lanes drive along +s (right-hand traffic)
                    xy, hd, sr, lb, rb = c, h, s, bi, bo
                else:         # left lanes drive against s
                    xy, hd, sr, lb, rb = c[::-1], h[::-1] + math.pi, s[::-1], bi[::-1], bo[::-1]
                if w.max() < 0.3:   # zero-width taper of a lane that starts/ends inside this section
                    continue
                spd = (r.speed_kph or 50.0) / 3.6
                m.lanes[(r.id, si, lid)] = Lane((r.id, si, lid), xy, np.unwrap(hd), w, sr, lb, rb, r.junction, spd)
    for (rid, si, lid), ln in m.lanes.items():
        sgn = 1 if lid > 0 else -1
        # neighbours in driving direction: for right lanes (lid<0) the left neighbour is lid+1 (toward centre)
        inner, outer = (rid, si, lid - sgn), (rid, si, lid + sgn)
        ln.left = inner if (lid - sgn) != 0 and inner in m.lanes else None
        ln.right = outer if outer in m.lanes else None


def _end_section(road: Road, contact: str) -> int:
    return 0 if contact == "start" else len(road.sections) - 1


def _link_lanes(m: OdrMap):
    def add(a: LaneKey, b: LaneKey):
        if a in m.lanes and b in m.lanes and b not in m.lanes[a].successors:
            m.lanes[a].successors.append(b)
            m.lanes[b].predecessors.append(a)

    for r in m.roads.values():
        # within road: consecutive lane sections
        for si in range(len(r.sections) - 1):
            for lid, ld in r.sections[si].lanes.items():
                if lid < 0 and ld.successor is not None:
                    add((r.id, si, lid), (r.id, si + 1, ld.successor))
            for lid, ld in r.sections[si + 1].lanes.items():
                if lid > 0 and ld.predecessor is not None:
                    add((r.id, si + 1, lid), (r.id, si, ld.predecessor))
        # road-to-road (non-junction) links
        for kind in ("successor", "predecessor"):
            if kind not in r.link or r.link[kind][0] != "road":
                continue
            _, oid, contact = r.link[kind]
            other = m.roads.get(oid)
            if other is None:
                continue
            si = len(r.sections) - 1 if kind == "successor" else 0
            osi = _end_section(other, contact or "start")
            for lid, ld in r.sections[si].lanes.items():
                olid = ld.successor if kind == "successor" else ld.predecessor
                if olid is None:
                    continue
                drives_out = (lid < 0) == (kind == "successor")
                a, b = (r.id, si, lid), (other.id, osi, olid)
                add(*(a, b) if drives_out else (b, a))
    for j in m.junctions.values():
        for c in j.connections:
            inc, con = m.roads.get(c.incoming), m.roads.get(c.connecting)
            if inc is None or con is None:
                continue
            # which end of the incoming road touches the junction
            inc_end = "end" if inc.link.get("successor", (None, None))[1] == j.id else "start"
            isi = _end_section(inc, inc_end)
            csi = _end_section(con, c.contact)
            for fl, tl in c.lane_links:
                a, b = (inc.id, isi, fl), (con.id, csi, tl)
                # incoming lane drives into the junction if it leaves the road at the touching end
                into = (fl < 0) == (inc_end == "end")
                if into:
                    add(a, b)
                else:
                    add(b, a)
            # connecting road's far end: follow its own road link to the outgoing road
            far = "successor" if c.contact == "start" else "predecessor"
            if far in con.link and con.link[far][0] == "road":
                _, oid, ocontact = con.link[far]
                out = m.roads.get(oid)
                if out is None:
                    continue
                fsi = len(con.sections) - 1 if far == "successor" else 0
                osi = _end_section(out, ocontact or "start")
                for lid, ld in con.sections[fsi].lanes.items():
                    olid = ld.successor if far == "successor" else ld.predecessor
                    if olid is None:
                        continue
                    a, b = (con.id, fsi, lid), (out.id, osi, olid)
                    drives_out = (lid < 0) == (far == "successor")
                    add(*(a, b) if drives_out else (b, a))
