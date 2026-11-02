"""TanitSim -> RDS-HQ exporter (input of NVIDIA's WSM v4 control-video renderer).

``export_clip(sim, log, out_root, clip_id)`` turns one TanitSim run (``TanitSim`` + ``SimLog``) into a
valid RDS-HQ clip: one uncompressed tar per component, WebDataset sample inside, 30 Hz pose timeline,
NVIDIA Hyperion camera rig, full map layers derived from the ``OdrMap`` and traffic-light states
derived from ``log.signals``.

Coordinate frames (all right-handed, metres)
  sim      TanitSim / OpenDRIVE inertial frame, z up, yaw CCW from +x. ``states[:, :, :2]`` is the *centre*
           of each vehicle box on the ground.
  world    RDS-HQ clip frame: origin = ego *rear-axle ground point* at the first output frame, axes =
           ego FLU axes at that frame (x forward, y left, z up). World z = 0 is the road surface.
           ``world = Rz(-yaw0) @ (sim - rear_axle0)``; the ground plane of the sim stays z = 0.
  vehicle  FLU at the rear-axle ground point. ``vehicle_to_world[0] == I`` exactly.
  camera   OpenCV (x right, y down, z forward); ``pose = camera_to_world = vehicle_to_world @ camera_to_vehicle``
           with ``camera_to_vehicle`` straight from NVIDIA's Hyperion rig asset.
  object   ``object_to_world`` at the geometric box centre (z = height / 2), local x along the length.
  cuboid   traffic lights / signs: local +x is the outward front-face normal (towards approaching traffic).

TanitSim has no pitch / roll / terrain, so all poses are planar and every map element lies at z = 0
(except poles, sign panels and signal heads).

The ego rear axle is ``centre - 0.3 * length`` along the heading. The ego box itself is *not* written to
``all_object_info`` (it carries the cameras; the renderer would suppress boxes containing a camera anyway).

Depends on numpy / scipy / shapely only. NVIDIA's ``rds_hq_io.py`` (writer + structural validator) and the
Hyperion rig asset are loaded from the Cosmos framework checkout, see :func:`wsm_dir`.
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import time
from pathlib import Path
from types import ModuleType

import numpy as np
import shapely
from scipy.interpolate import PchipInterpolator
from scipy.spatial import cKDTree
from shapely.geometry import LineString, MultiLineString, Polygon
from shapely.ops import linemerge, unary_union

OUT_FPS = 30
CHUNK_SECONDS = 9.9            # 297 frames at 30 fps, the Auto model chunk
REAR_AXLE_FRAC = 0.30          # rear axle = centre - 0.3 * length along the heading

# the 7 standard cameras of the PhysicalAI-AV / Hyperion 8 dataset rig (f-theta), in renderer order
CAMERAS_7 = (
    "camera_front_wide_120fov", "camera_front_tele_30fov", "camera_cross_left_120fov", "camera_cross_right_120fov",
    "camera_rear_left_70fov", "camera_rear_right_70fov", "camera_rear_tele_30fov",
)

_WSM_REL = "skills/scenario-to-rds-hq"
_DEFAULT_WSM = ("~/tsf/third_party/cosmos-framework/.agents/skills/cookbooks/cosmos3/simulation/auto/wsm")

# TanitSim category -> RDS-HQ object class
OBJECT_CLASS = {
    "car": "Car", "suv": "Car", "van": "Car", "truck": "Truck", "bus": "Truck",
    "pedestrian": "Pedestrian", "bicycle": "Cyclist", "motorcycle": "Cyclist",
}

# static environment objects (TSIR `static_objects`): parked vehicles keep their semantic class, cones / barriers /
# bollards are `Others` cuboids (static, so the LiDAR range-map control ray-casts them like vehicles); signs go to
# `3d_traffic_signs` with a sign_class (their posts and poles to `3d_poles`)
STATIC_OBJECT_CLASS = {"parked_car": "Car", "parked_truck": "Truck", "cone": "Others", "barrier": "Others",
                       "bollard": "Others"}
SIGN_CLASS = {
    "speed_limit": "TRAFFIC_SIGN_REGULATORY_R2_SPEED_LIMIT", "stop": "TRAFFIC_SIGN_REGULATORY_R1_STOP",
    "yield": "TRAFFIC_SIGN_REGULATORY_R1_YIELD", "no_entry": "TRAFFIC_SIGN_REGULATORY_R5_DO_NOT_ENTER",
    "pedestrian_crossing": "TRAFFIC_SIGN_WARNING_W11_PEDESTRIAN_CROSSING", "road_works": "TRAFFIC_SIGN_WARNING_W20_ROAD_WORK",
}

# traffic-light head geometry (the xodr compiler places 0.9 m x 0.35 m heads with their bottom edge 2.5 m up)
TL_BOTTOM, TL_HEIGHT, TL_WIDTH, TL_DEPTH = 2.5, 0.9, 0.35, 0.50   # depth: see-through of the renderer glyph bias beyond ~33 m
SIGN_SIZE, SIGN_CENTER_Z, SIGN_DEPTH = 0.75, 2.2, 0.10


# ============================================================================== NVIDIA resources
def wsm_dir() -> Path:
    """Directory of the framework's standalone WSM project (``TSF_WSM_DIR`` or the default checkout)."""
    cands = [os.environ["TSF_WSM_DIR"]] if os.environ.get("TSF_WSM_DIR") else []
    cands.append(_DEFAULT_WSM)
    for c in cands:
        p = Path(c).expanduser()
        if (p / _WSM_REL / "scripts" / "rds_hq_io.py").exists():
            return p
    raise FileNotFoundError(f"Cosmos WSM project not found (looked at {cands}); set TSF_WSM_DIR")


_NV_IO: ModuleType | None = None


def nvidia_io() -> ModuleType:
    """NVIDIA's ``rds_hq_io.py`` (``write_clip`` / ``validate_clip``), loaded from the framework checkout."""
    global _NV_IO
    if _NV_IO is None:
        path = wsm_dir() / _WSM_REL / "scripts" / "rds_hq_io.py"
        spec = importlib.util.spec_from_file_location("nvidia_rds_hq_io", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _NV_IO = mod
    return _NV_IO


def load_rig(rig="hyperion", cameras=None) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Camera rig -> ``{name: (ftheta_intrinsics[14], camera_to_vehicle[4,4])}``.

    ``rig``: ``"hyperion"`` (the 7 standard cameras of the NVIDIA Hyperion 8 / PhysicalAI-AV rig),
    ``"hyperion11"`` (adds the four fisheyes), a path to / dict of a rig JSON in the same schema as
    the framework's ``hyperion8.json``. ``cameras`` restricts / orders the selection.
    """
    if isinstance(rig, (str, Path)) and str(rig) in ("hyperion", "hyperion7", "hyperion11", "hyperion8"):
        raw = json.loads((wsm_dir() / _WSM_REL / "assets" / "hyperion8.json").read_text())
        names = list(raw) if str(rig) in ("hyperion11", "hyperion8") else list(CAMERAS_7)
    elif isinstance(rig, dict):
        raw, names = rig, list(rig)
    else:
        raw = json.loads(Path(rig).read_text())
        names = list(raw)
    names = list(cameras) if cameras else names
    out = {}
    for n in names:
        if n not in raw:
            raise KeyError(f"camera {n!r} not in rig ({sorted(raw)})")
        out[n] = (np.asarray(raw[n]["ftheta_intrinsics"], float), np.asarray(raw[n]["camera_to_vehicle"], float))
    return out


# ============================================================================== small geometry helpers
def _rz(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _T(R: np.ndarray, t) -> np.ndarray:
    M = np.eye(4)
    M[:3, :3], M[:3, 3] = R, t
    return M


def _offset(pts: np.ndarray, d: float) -> np.ndarray:
    """Offset a polyline laterally by d (+ = left of its direction); same rule as render2d."""
    t = np.gradient(pts, axis=0)
    n = np.stack([-t[:, 1], t[:, 0]], 1)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    return pts + n * d


def _dedupe(pts: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    keep = np.concatenate([[True], np.linalg.norm(np.diff(pts, axis=0), axis=1) > eps])
    return pts[keep]


def _cuboid(center, yaw: float, size) -> np.ndarray:
    """8 vertices in the RDS-HQ order. ``size`` = (thickness along local x, width along y, height along z);
    local +x (the front-face normal) points along ``yaw``."""
    a, b, c = (np.asarray(size, float) / 2)
    loc = np.array([[-a, -b, -c], [-a, b, -c], [a, b, -c], [a, -b, -c], [-a, -b, c], [-a, b, c], [a, b, c], [a, -b, c]])
    return loc @ _rz(yaw)[:3, :3].T + np.asarray(center, float)


def _v(a) -> list:
    return np.round(np.asarray(a, float), 4).tolist()


def _lbl(kind: str, verts, attrs: list | None = None) -> dict:
    return {"labelData": {"shape3d": {"attributes": attrs or [], kind: {"vertices": _v(verts)}}}}


def _text(name, val):
    return {"name": name, "text": val}


# ============================================================================== window selection
def _statics(sim, log) -> list:
    """Static environment bodies of a run (``SimLog.static``, else ``sim.static``)."""
    return list(getattr(log, "static", None) or getattr(sim, "static", None) or [])


def choose_t0(sim, log, duration: float = CHUNK_SECONDS, step: float = 0.25) -> tuple[float, dict]:
    """Pick the clip start with the most ego motion / interaction.

    score(t0) = mean ego speed [m/s] + 25 * total |ego yaw change| [rad] + 6 * mean #actors within 40 m
                + 10 * mean #actors within 15 m + 3 * #scripted action start/end events in the window
                + 1.5 * mean #static objects within 40 m.
    Candidates are every ``step`` s such that the whole window fits in the log. Returns ``(t0, info)``.
    """
    t = np.asarray(log.t)
    dt = float(t[1] - t[0])
    n, win = len(t), int(round(duration / dt))
    last = n - 1 - win
    if last <= 0:
        return 0.0, {"reason": "log shorter than the requested window", "score": None}
    e = log.roles.index("ego")
    S = np.asarray(log.states)
    others = [j for j in range(S.shape[1]) if j != e]
    yaw = np.unwrap(S[:, e, 2])
    dyaw = np.abs(np.diff(yaw, prepend=yaw[0]))
    dist = np.hypot(S[:, others, 0] - S[:, [e], 0], S[:, others, 1] - S[:, [e], 1]) if others else np.zeros((n, 0))
    near, close = (dist < 40).sum(1), (dist < 15).sum(1)
    stat = _statics(sim, log)
    if stat:        # a work zone / parked vehicles within 40 m of the ego count like nearby actors
        sd = np.hypot(np.array([b.x for b in stat])[None, :] - S[:, [e], 0], np.array([b.y for b in stat])[None, :] - S[:, [e], 1])
        near_s = (sd < 40).sum(1)
    else:
        near_s = np.zeros(n)
    ev_t = np.array([ev["t"] for ev in log.events if ev.get("type") in ("action_start", "action_end")])

    def cs(x):
        return np.concatenate([[0.0], np.cumsum(x)])

    c_v, c_y, c_n, c_c, c_s = cs(S[:, e, 3]), cs(dyaw), cs(near), cs(close), cs(near_s)
    idxs = np.arange(0, last + 1, max(1, int(round(step / dt))))
    best, scores = None, []
    for i in idxs:
        j = i + win
        n_ev = int(((ev_t >= t[i]) & (ev_t <= t[j])).sum()) if len(ev_t) else 0
        parts = dict(speed=(c_v[j] - c_v[i]) / win, turn=25 * (c_y[j] - c_y[i]), near=6 * (c_n[j] - c_n[i]) / win,
                     close=10 * (c_c[j] - c_c[i]) / win, events=3.0 * n_ev, static=1.5 * (c_s[j] - c_s[i]) / win)
        sc = float(sum(parts.values()))
        scores.append(sc)
        if best is None or sc > best[0] + 1e-9:
            best = (sc, float(t[i]), parts)
    return best[1], {"score": round(best[0], 2), "parts": {k: round(float(v), 2) for k, v in best[2].items()},
                     "candidates": [(round(float(t[i]), 2), round(s, 1)) for i, s in zip(idxs, scores)][::4]}


# ============================================================================== timeline / world frame
class _Resampler:
    """C1 (PCHIP) resampling of a SimLog actor track to arbitrary times; yaw is unwrapped first."""

    def __init__(self, log, j: int):
        t, S = np.asarray(log.t), np.asarray(log.states)
        self.t0, self.t1 = float(t[0]), float(t[-1])
        self.fx, self.fy = PchipInterpolator(t, S[:, j, 0]), PchipInterpolator(t, S[:, j, 1])
        self.fa = PchipInterpolator(t, np.unwrap(S[:, j, 2]))

    def __call__(self, tt: np.ndarray):
        tt = np.clip(tt, self.t0, self.t1)
        return self.fx(tt), self.fy(tt), self.fa(tt)


class WorldFrame:
    """sim -> world rebase anchored at the ego rear-axle ground point of the first output frame."""

    def __init__(self, origin_xy: np.ndarray, yaw0: float):
        self.o, self.yaw0 = np.asarray(origin_xy, float), float(yaw0)
        c, s = math.cos(yaw0), math.sin(yaw0)
        self._R = np.array([[c, s], [-s, c]])        # Rz(-yaw0)

    def xy(self, p) -> np.ndarray:
        return (np.asarray(p, float) - self.o) @ self._R.T

    def yaw(self, a):
        return np.asarray(a) - self.yaw0

    def xyz(self, p, z=0.0) -> np.ndarray:
        q = self.xy(np.asarray(p, float)[..., :2])
        zz = np.full(q.shape[:-1] + (1,), z) if np.isscalar(z) else np.asarray(z)[..., None]
        return np.concatenate([q, zz], -1)


# ============================================================================== map layers
class MapLayers:
    """Derive the eight RDS-HQ geometry layers (+ traffic-light head table) from an ``OdrMap``.

    Everything is built in the sim frame, cropped to ``radius`` m around the ego path, then rebased to world.
    Marking logic follows ``tsf.sim.render2d.MapArt`` (dashed white between same-direction lanes, double yellow
    centre line, solid white edges, stop bars at inbound lane ends, zebra crossings on junction arms).
    """

    def __init__(self, sim, log, wf: WorldFrame, ego_xy: np.ndarray, radius: float = 300.0,
                 left_arrow_heads: bool = False):
        self.sim, self.m, self.log, self.wf, self.R = sim, sim.m, log, wf, radius
        self.tree = cKDTree(ego_xy[:: max(1, len(ego_xy) // 300)])
        self.left_arrow_heads = left_arrow_heads
        self.layers: dict[str, list] = {k: [] for k in (
            "3d_lanelines", "3d_road_boundaries", "3d_crosswalks", "3d_wait_lines", "3d_poles",
            "3d_traffic_signs", "3d_road_markings", "3d_traffic_lights")}
        self.heads: list[dict] = []     # traffic-light heads: feature_id, arm, kind ("main"|"left"), idx in layer
        self.stats: dict[str, int] = {}
        lanes = list(self.m.lanes.values())
        self.urban = min((ln.speed for ln in lanes), default=13.9) * 3.6 <= 70 or bool(self.m.junctions)
        self.static = _statics(sim, log)
        self.static_report: dict = {"signs": {}, "poles": [], "objects": {}}
        self._lanelines()
        self._boundaries()
        self._arms_markings()
        self._arrows()
        self._signs()
        self._statics()
        self._signals()
        self.stats = {k: len(v) for k, v in self.layers.items()}

    # ---- cropping / emitting -------------------------------------------------------------------
    def _near(self, pts: np.ndarray, pad: float = 0.0) -> np.ndarray:
        return self.tree.query(pts[:, :2])[0] <= self.R + pad

    def _runs(self, pts: np.ndarray):
        """Contiguous sub-polylines of ``pts`` that lie within the region of interest."""
        mask = self._near(pts)
        idx = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(int), [0]])))
        for a, b in zip(idx[::2], idx[1::2]):
            seg = _dedupe(pts[a:b])
            if len(seg) >= 2:
                yield seg

    def _line(self, layer: str, pts: np.ndarray, attrs: list, z: float = 0.0):
        for seg in self._runs(np.asarray(pts, float)):
            self.layers[layer].append(_lbl("polyline3d", self.wf.xyz(seg, z), attrs))

    def _lane_attrs(self, color: str, style: str):
        return [{"name": "colors", "enumsList": {"enumsList": [color]}},
                {"name": "styles", "enumsList": {"enumsList": [style]}}]

    # ---- lane lines ------------------------------------------------------------------------------
    def _lanelines(self):
        m = self.m
        for (rid, si, lid), ln in m.lanes.items():
            if ln.junction != "-1":
                continue
            opposite = any(k[0] == rid and k[1] == si and (k[2] > 0) != (lid > 0) for k in m.lanes)
            if ln.left is not None:       # shared border with the same-direction lane on the left: drawn once
                self._line("3d_lanelines", ln.left_border, self._lane_attrs("WHITE", "LONG_DASHED_SINGLE"))
            elif opposite:
                if lid < 0:               # shared centre line of a two-way road: one double-yellow path
                    self._line("3d_lanelines", ln.left_border, self._lane_attrs("YELLOW", "SOLID_GROUP"))
            else:                          # one-way road, left edge line just inside the lane
                self._line("3d_lanelines", _offset(ln.left_border, -0.3), self._lane_attrs("WHITE", "SOLID_SINGLE"))
            if ln.right is None:
                self._line("3d_lanelines", _offset(ln.right_border, 0.3), self._lane_attrs("WHITE", "SOLID_SINGLE"))

    # ---- road boundaries ---------------------------------------------------------------------------
    def _boundaries(self):
        m = self.m
        polys, caps = [], []
        for ln in m.lanes.values():
            polys.append(Polygon(np.concatenate([ln.left_border, ln.right_border[::-1]])).buffer(0))
            if not ln.successors:
                caps.append(LineString([ln.left_border[-1], ln.right_border[-1]]))
            if not ln.predecessors:
                caps.append(LineString([ln.left_border[0], ln.right_border[0]]))
        road = unary_union(polys).buffer(1.5, join_style=1).buffer(-1.5, join_style=1)
        cap_geom = unary_union(caps) if caps else None

        def rings(geom):
            out = []
            for poly in (geom.geoms if hasattr(geom, "geoms") else [geom]):
                if poly.is_empty or not hasattr(poly, "exterior"):
                    continue
                out.append(poly.exterior)
                out += [r for r in poly.interiors if Polygon(r).area > 5.0]
            return unary_union(out)

        def emit(dist: float, btype: str):
            geom = road if dist == 0 else road.buffer(dist, join_style=2)
            lines = rings(geom)
            if cap_geom is not None:      # drop the artificial cut across lanes that end at the map edge
                lines = lines.difference(cap_geom.buffer(dist + 0.4))
            lines = linemerge(lines) if isinstance(lines, MultiLineString) else lines
            for ls in (lines.geoms if hasattr(lines, "geoms") else [lines]):
                if ls.is_empty or ls.geom_type != "LineString" or ls.length < 2.0:
                    continue
                # densify first: straight edges are 2-vertex lines and would be dropped by the vertex-wise crop
                ls = shapely.segmentize(ls, 2.0)
                self._line("3d_road_boundaries", np.asarray(ls.coords), [_text("boundary_type", btype)])

        if self.urban:
            emit(0.0, "curb")
        else:
            emit(0.0, "road_boundary")    # asphalt / shoulder seam
            emit(3.3, "barrier")          # guardrail, as drawn by render2d

    # ---- stop bars + zebra crossings ------------------------------------------------------------------
    def _arms(self):
        """(mouth point, outward heading, inward heading, left extent, right extent) of roads touching a junction."""
        out = []
        for r in self.m.roads.values():
            if r.junction != "-1":
                continue
            for kind, s in (("predecessor", 0.0), ("successor", r.length)):
                lk = r.link.get(kind)
                if not lk or lk[0] != "junction":
                    continue
                x, y, h = r.ref(np.array([s]))
                sec = 0 if s == 0.0 else len(r.sections) - 1
                lids = list(r.sections[sec].lanes)
                tl = max((float(r.lane_t(sec, i, np.array([s]), "outer")[0]) for i in lids if i > 0), default=0.0)
                tr = min((float(r.lane_t(sec, i, np.array([s]), "outer")[0]) for i in lids if i < 0), default=0.0)
                out.append((np.array([x[0], y[0]]), float(h[0]), float(h[0]) + (math.pi if s == 0.0 else 0.0), tl, tr))
        return out

    def _arms_markings(self):
        for p, h, inward, tl, tr in self._arms():
            fwd = np.array([math.cos(inward), math.sin(inward)])
            left = np.array([-math.sin(h), math.cos(h)])
            a0, a1 = p + fwd * 1.2 + left * (tr + 0.4), p + fwd * 1.2 + left * (tl - 0.3)
            poly = np.array([a0, a1, a1 + fwd * 3.6, a0 + fwd * 3.6])
            if self._near(poly).any() and Polygon(poly).is_valid:
                self.layers["3d_crosswalks"].append(_lbl("surface", self.wf.xyz(poly, 0.0)))
        explicit = [{"name": "is_implicit", "flag": False}]
        for ln in self.m.lanes.values():
            if ln.junction == "-1" and ln.successors and self.m.lanes[ln.successors[0]].junction != "-1":
                bar = np.array([ln.left_border[-1], ln.right_border[-1]])
                if self._near(bar).any():
                    self.layers["3d_wait_lines"].append(_lbl("polyline3d", self.wf.xyz(bar, 0.0), explicit))

    # ---- painted turn arrows --------------------------------------------------------------------------
    def _arrows(self):
        from tsf.sim.engine import lane_turn
        cls = {frozenset({"straight"}): "FORWARD", frozenset({"left"}): "LEFT", frozenset({"right"}): "RIGHT",
               frozenset({"left", "straight"}): "LEFT_FORWARD", frozenset({"right", "straight"}): "RIGHT_FORWARD"}
        for ln in self.m.lanes.values():
            if ln.junction != "-1" or not ln.successors or ln.length < 20:
                continue
            turns = {lane_turn(self.m.lanes[s]) for s in ln.successors if self.m.lanes[s].junction != "-1"} - {"u_turn"}
            if not turns:
                continue
            order = {"right": 0, "straight": 1, "left": 2}          # lateral order, rightmost first
            names = [cls[frozenset(turns)]] if frozenset(turns) in cls else \
                [cls[frozenset({t})] for t in sorted(turns, key=order.get)]
            lane_w = float(np.median(ln.width))
            gw = min(1.5, (lane_w - 0.8) / len(names))
            for back in (7.0, 27.0):
                if ln.length < back + 6:
                    continue
                x, y, h = ln.pose(ln.length - back)
                f, l = np.array([math.cos(h), math.sin(h)]), np.array([-math.sin(h), math.cos(h)])
                for k, nm in enumerate(names):
                    c = np.array([x, y]) + l * (k - (len(names) - 1) / 2) * (gw + 0.1)
                    quad = np.array([c + f * 2.0 - l * gw / 2, c + f * 2.0 + l * gw / 2,
                                     c - f * 2.0 + l * gw / 2, c - f * 2.0 - l * gw / 2])   # FR, FL, RL, RR
                    if self._near(quad).any():
                        self.layers["3d_road_markings"].append(
                            _lbl("surface", self.wf.xyz(quad, 0.0),
                                 [_text("road_marking_class", f"ROI_POLYGON_DIRECTION_PAINT_{nm}")]))

    # ---- speed-limit signs (as in render2d) ------------------------------------------------------------
    def _signs(self):
        explicit = {b.road for b in self.static if b.sign_type == "speed_limit"}   # roads with their own speed signs
        for r in self.m.roads.values():
            if r.junction != "-1" or r.speed_kph is None or r.id in explicit:
                continue
            for side, s in ((-1, min(40.0, r.length * 0.3)), (1, max(r.length - 40.0, r.length * 0.7))):
                lids = [k for k in r.sections[0].lanes if (k > 0) == (side > 0)]
                if not lids:
                    continue
                outer = max(lids, key=abs)
                si = 0 if s < r.sections[0].s_end else len(r.sections) - 1
                t = float(r.lane_t(si, outer, np.array([s]), "outer")[0]) + 2.4 * side
                x, y, h = r.ref(np.array([s]))
                pos = np.array([x[0] - math.sin(h[0]) * t, y[0] + math.cos(h[0]) * t])
                if not self._near(pos[None]).any():
                    continue
                normal = float(h[0]) + (math.pi if side < 0 else 0.0)    # faces the oncoming traffic
                cub = _cuboid([*self.wf.xy(pos), SIGN_CENTER_Z], float(self.wf.yaw(normal)),
                              (SIGN_DEPTH, SIGN_SIZE, SIGN_SIZE))
                self.layers["3d_traffic_signs"].append(
                    _lbl("cuboid3d", cub, [_text("sign_class", "TRAFFIC_SIGN_REGULATORY_R2_SPEED_LIMIT")]))
                self._pole(pos, SIGN_CENTER_Z + SIGN_SIZE / 2 + 0.2, behind=normal, back=SIGN_DEPTH / 2 + 0.05)

    # ---- static environment objects ---------------------------------------------------------------------
    def _statics(self):
        """Signs -> ``3d_traffic_signs`` (+ post in ``3d_poles``), poles -> ``3d_poles``. Cones, barriers, bollards and
        parked vehicles are tracked objects (see ``export_clip``), not map elements."""
        for b in self.static:
            pos = np.array([b.x, b.y])
            if not self._near(pos[None]).any():
                continue
            l, w, h = b.lwh
            if b.kind == "pole":
                q = self.wf.xy(pos)
                self.layers["3d_poles"].append(_lbl("polyline3d", [[q[0], q[1], 0.0], [q[0], q[1], h]]))
                self.static_report["poles"].append(b.id)
            elif b.is_panel:
                cls = SIGN_CLASS[b.sign_type or "road_works"]
                panel = min(w, h) if b.kind == "traffic_sign" else 0.6 * h
                depth = max(l, SIGN_DEPTH)
                cub = _cuboid([*self.wf.xy(pos), h - panel / 2], float(self.wf.yaw(b.yaw)), (depth, w, panel))
                self.layers["3d_traffic_signs"].append(_lbl("cuboid3d", cub, [_text("sign_class", cls)]))
                self._pole(pos, h - panel, behind=b.yaw, back=depth / 2 + 0.05)
                self.static_report["signs"][b.id] = cls

    def _pole(self, pos_sim: np.ndarray, top: float, behind: float, back: float):
        base = np.asarray(pos_sim) - back * np.array([math.cos(behind), math.sin(behind)])
        w = self.wf.xy(base)
        self.layers["3d_poles"].append(_lbl("polyline3d", [[w[0], w[1], 0.0], [w[0], w[1], top]]))

    # ---- traffic-light heads ---------------------------------------------------------------------------
    def _arm_of(self, sg) -> int | None:
        arms = list(self.log.signals)
        if not arms:
            return None
        if sg.road.lstrip("-").isdigit() and int(sg.road) in arms:
            return int(sg.road)
        if sg.id.isdigit() and int(sg.id) % 1000 in arms:
            return int(sg.id) % 1000
        # fallback: the arm whose inbound stop line is closest to the head
        best = (None, 1e18)
        for (rid, si, lid), ln in self.m.lanes.items():
            if rid.isdigit() and int(rid) in arms and ln.junction == "-1" and ln.successors:
                d = float(np.hypot(*(ln.xy[-1] - np.array([sg.x, sg.y]))))
                if d < best[1]:
                    best = (int(rid), d)
        return best[0]

    def _signals(self):
        for sg in self.m.signals():
            if not sg.dynamic:
                continue
            arm = self._arm_of(sg)
            pos = np.array([sg.x, sg.y])
            if arm is None or not self._near(pos[None]).any():
                continue
            normal = sg.hdg + math.pi                      # sg.hdg = direction of the traffic it governs
            n = np.array([math.cos(normal), math.sin(normal)])
            toward_lane = np.array([math.sin(normal), -math.cos(normal)])      # lateral, towards the road centre
            specs = [("main", pos)]
            if self.left_arrow_heads:
                specs.append(("left", pos + toward_lane * (TL_WIDTH + 0.15)))
            for kind, p in specs:
                fid = f"tl_{sg.id}" + ("_left" if kind == "left" else "")
                cub = _cuboid([*self.wf.xy(p), TL_BOTTOM + TL_HEIGHT / 2], float(self.wf.yaw(normal)),
                              (TL_DEPTH, TL_WIDTH, TL_HEIGHT))
                self.heads.append({"feature_id": fid, "arm": arm, "kind": kind, "signal_id": sg.id,
                                   "pos_sim": [round(float(p[0]), 3), round(float(p[1]), 3)], "index": len(self.layers["3d_traffic_lights"])})
                self.layers["3d_traffic_lights"].append(_lbl("cuboid3d", cub, [_text("feature_id", fid)]))
            self._pole(pos, TL_BOTTOM + TL_HEIGHT + 0.15, behind=normal, back=TL_DEPTH / 2 + 0.1)   # pole behind the head


# ============================================================================== traffic-light states
_STATE_MAP = {
    "red_solid": "red", "yellow_solid": "yellow", "green_solid": "green",
    "red_left_arrow": "red", "yellow_left_arrow": "yellow", "green_left_arrow": "green", "unknown": "unknown",
}


def _tl_states(sim, log, heads: list[dict], tt: np.ndarray) -> tuple[dict, dict]:
    """Per-frame state strings of every head from ``log.signals`` (nearest 20 Hz tick, no interpolation).

    Heads of a protected-left junction are split: the main head shows the straight/right state and the
    ``*_left`` head the left-arrow state, both from the ``SignalPlan`` at the tick time; their merge is checked
    against ``log.signals`` (green if any green, else yellow if any yellow, else red).
    """
    t = np.asarray(log.t)
    dt = float(t[1] - t[0])
    tick = np.clip(np.rint(tt / dt).astype(int), 0, len(t) - 1)
    plan = getattr(sim, "signals", None)
    out, mismatch = {}, 0
    for h in heads:
        arm = h["arm"]
        base = np.array(log.signals[arm])[tick]
        if h["kind"] == "main" and plan is not None and plan.enabled and any(x["kind"] == "left" and x["arm"] == arm for x in heads):
            col = [plan.state(arm, "straight", float(t[i])) for i in tick]
        elif h["kind"] == "left":
            col = [plan.state(arm, "left", float(t[i])) for i in tick]
        else:
            col = list(base)
        out[h["feature_id"]] = [f"{c}_left_arrow" if h["kind"] == "left" else f"{c}_solid" for c in col]
    for arm in {h["arm"] for h in heads}:
        parts = [[s.split("_")[0] for s in out[h["feature_id"]]] for h in heads if h["arm"] == arm]
        base = np.array(log.signals[arm])[tick]
        for k in range(len(tick)):
            cs = {p[k] for p in parts}
            merged = "green" if "green" in cs else "yellow" if "yellow" in cs else "red"
            mismatch += int(merged != base[k])
    return out, {"merge_mismatch_frames": mismatch}


# ============================================================================== export
def export_clip(sim, log, out_root, clip_id: str, t0: float | None = None, duration: float = CHUNK_SECONDS,
                fps: int = OUT_FPS, rig="hyperion", *, cameras=None, max_obj_range: float = 250.0,
                map_radius: float = 300.0, left_arrow_heads: bool | None = None, overwrite: bool = False,
                require_planar_polygons: bool = True, pad_frames: int = 0, write_vehicle_pose: bool = True) -> Path:
    """Write one RDS-HQ clip (30 Hz, ``duration`` s, Hyperion rig) from a TanitSim run and validate it.

    pad_frames: extra pose frames after the ``duration`` window (``pad_frames=1`` gives 298 poses for a 297-frame
    video: the renderer still emits 297 frames, and the LiDAR range-map renderer gets the end pose of its last
    10 Hz sweep). write_vehicle_pose: also write ``vehicle_pose/<clip>.tar`` (``<frame>.vehicle_pose.npy``), the
    pose source ``cosmos_auto.lidar`` prefers; the WSM renderer ignores it.

    Returns ``out_root`` (the RDS-HQ dataset root holding ``pose/<clip_id>.tar`` etc.). Side files, outside the
    tars, in ``out_root/tsf_export/``: ``<clip_id>.json`` (report: window, frames, transforms, class mapping,
    map counts, validator output, timings) and ``<clip_id>.vehicle_to_world.npy`` ([T,4,4], for NVIDIA's
    ``retarget_hyperion.py``).

    t0: sim time of the first frame; ``None`` = automatic (:func:`choose_t0`). The window is shortened if the log
    ends earlier. ``fps`` must divide the pose rate the renderer is told (30 for the Auto models).
    """
    tic = time.perf_counter()
    nv = nvidia_io()
    out_root = Path(out_root)
    t_log, S = np.asarray(log.t), np.asarray(log.states)
    e = log.roles.index("ego")
    info = {}
    if t0 is None:
        t0, info = choose_t0(sim, log, duration)
    t0 = float(np.clip(t0, 0.0, max(0.0, t_log[-1] - 1.0 / fps)))
    n_req = int(round(duration * fps))
    n = max(2, min(n_req + pad_frames, int(math.floor((t_log[-1] - t0) * fps + 1e-6)) + 1))
    tt = t0 + np.arange(n) / fps

    # ---- ego rear-axle trajectory and world frame
    L_ego = float(log.lwh[e, 0])
    ex, ey, eyaw = _Resampler(log, e)(tt)
    ra = np.stack([ex, ey], 1) - REAR_AXLE_FRAC * L_ego * np.stack([np.cos(eyaw), np.sin(eyaw)], 1)
    wf = WorldFrame(ra[0], eyaw[0])
    ra_w, yaw_w = wf.xy(ra), eyaw - eyaw[0]
    v2w = np.tile(np.eye(4), (n, 1, 1))
    for k in range(n):
        v2w[k, :3, :3] = _rz(yaw_w[k])[:3, :3]
        v2w[k, :2, 3] = ra_w[k]
    v2w[0] = np.eye(4)

    # ---- cameras
    rig_d = load_rig(rig, cameras)
    poses = {name: np.einsum("tij,jk->tik", v2w, c2v) for name, (_, c2v) in rig_d.items()}
    intr = {name: i for name, (i, _) in rig_d.items()}

    # ---- objects (all actors but the ego), full rate
    objects: list[dict] = [dict() for _ in range(n)]
    cls_map, moving_flags = {}, {}
    for j, aid in enumerate(log.ids):
        if j == e:
            continue
        cat = log.categories[j]
        otype = OBJECT_CLASS.get(cat, "Others")
        cls_map[aid] = f"{cat} -> {otype}"
        x, y, a = _Resampler(log, j)(tt)
        l, w, h = (float(v) for v in log.lwh[j])
        cw, aw = wf.xy(np.stack([x, y], 1)), wf.yaw(a)
        moving = bool(np.hypot(*(cw - cw[0]).T).max() > 0.5)
        moving_flags[aid] = moving
        near = np.hypot(x - ex, y - ey) <= max_obj_range
        for k in np.flatnonzero(near):
            objects[k][aid] = {"object_type": otype, "object_is_moving": moving, "object_lwh": [l, w, h],
                               "object_to_world": _T(_rz(aw[k])[:3, :3], [cw[k, 0], cw[k, 1], h / 2]).tolist()}

    # ---- static environment objects: tracked, never moving, fixed world pose
    static = _statics(sim, log)
    static_cls = {}
    for b in static:
        otype = STATIC_OBJECT_CLASS.get(b.kind)
        if otype is None:                  # signs and poles are map elements (MapLayers._statics)
            continue
        static_cls[b.id] = f"{b.kind} -> {otype}"
        l, w, h = b.lwh
        cw, aw = wf.xy(np.array([b.x, b.y])), float(wf.yaw(b.yaw))
        tf = _T(_rz(aw)[:3, :3], [cw[0], cw[1], h / 2]).tolist()
        moving_flags[b.id] = False
        for k in np.flatnonzero(np.hypot(b.x - ex, b.y - ey) <= max_obj_range):
            objects[k][b.id] = {"object_type": otype, "object_is_moving": False, "object_lwh": [l, w, h],
                                "object_to_world": tf}
    cls_map.update(static_cls)

    # ---- map + traffic lights
    protected = bool(getattr(getattr(sim, "signals", None), "enabled", False) and getattr(sim.signals, "protected", False))
    lam = protected if left_arrow_heads is None else left_arrow_heads
    ml = MapLayers(sim, log, wf, np.stack([ex, ey], 1), map_radius, left_arrow_heads=bool(lam))
    layers = {k: {"labels": v} for k, v in ml.layers.items()}
    states, sig_info = _tl_states(sim, log, ml.heads, tt)
    tl = {"state_to_legacy_render_state": {}, "traffic_light_states": {}}
    for i, h in enumerate(ml.heads):
        tl["traffic_light_states"][str(i)] = {"feature_id": h["feature_id"], "state": states[h["feature_id"]]}
        for s in states[h["feature_id"]]:
            tl["state_to_legacy_render_state"][s] = _STATE_MAP[s]
    # head index in the state table must equal its index in the geometry layer
    assert [h["index"] for h in ml.heads] == list(range(len(ml.heads)))

    # ---- write + validate with NVIDIA's helper
    if overwrite:
        for comp in list((out_root).glob(f"*/{clip_id}.tar")):
            comp.unlink()
    t_build = time.perf_counter()
    rep = nv.write_clip(out_root, clip_id, poses, intr, objects, layers, tl, "ftheta",
                        require_planar_polygons=require_planar_polygons)
    rep2 = nv.validate_clip(out_root, clip_id, "ftheta", require_planar_polygons=require_planar_polygons)
    t_done = time.perf_counter()

    side = out_root / "tsf_export"
    side.mkdir(parents=True, exist_ok=True)
    np.save(side / f"{clip_id}.vehicle_to_world.npy", v2w)
    if write_vehicle_pose:
        vp = out_root / "vehicle_pose" / f"{clip_id}.tar"
        if overwrite and vp.exists():
            vp.unlink()
        nv.write_archive(vp, clip_id, {f"{k:06d}.vehicle_pose.npy": v2w[k] for k in range(n)})
    timings = {"export_s": round(t_done - tic, 3), "build_s": round(t_build - tic, 3),
               "write_validate_s": round(t_done - t_build, 3)}
    report = {
        "clip_id": clip_id, "scenario": sim.sc.meta.id, "tanitsim_dt": float(t_log[1] - t_log[0]),
        "t0_s": round(t0, 3), "t0_auto": bool(info), "window_choice": info, "duration_s": round((n - 1) / fps, 4),
        "frames": n, "render_frames": min(n, n_req), "fps": fps, "rig": rig if isinstance(rig, str) else "custom", "cameras": list(rig_d),
        "frames_convention": "world = Rz(-yaw0)(sim - rear_axle0); z up; origin = ego rear-axle ground point at frame 0",
        "ego": {"id": log.ids[e], "length": L_ego, "rear_axle_frac": REAR_AXLE_FRAC,
                "sim_rear_axle0": [round(float(v), 4) for v in ra[0]], "sim_yaw0": round(float(eyaw[0]), 5),
                "path_len_m": round(float(np.hypot(*np.diff(ra_w, axis=0).T).sum()), 2),
                "yaw_change_deg": round(float(np.degrees(yaw_w[-1])), 2),
                "speed0_ms": round(float(np.interp(t0, t_log, S[:, e, 3])), 3)},
        "objects": {"tracks": rep["tracks"], "class_mapping": cls_map, "moving": moving_flags,
                    "observations": rep["object_class_observations"]},
        "static_objects": {"n": len(static), "as_objects": static_cls, "as_signs": ml.static_report["signs"],
                           "as_poles": ml.static_report["poles"],
                           "dropped_outside_map_radius": [b.id for b in static if b.id not in static_cls
                                                          and b.id not in ml.static_report["signs"]
                                                          and b.id not in ml.static_report["poles"]]},
        "map_labels": rep["map_labels"], "traffic_lights": ml.heads, "signal_merge_check": sig_info,
        "signal_state_observations": rep["signal_state_observations"],
        "validator": {"helper": "NVIDIA rds_hq_io.py write_clip + validate_clip", "ok": True,
                      "structural_report": rep2, "planar_polygons_required": require_planar_polygons},
        "timings": timings,
    }
    (side / f"{clip_id}.json").write_text(json.dumps(report, indent=1))
    return out_root


def export_report(out_root, clip_id: str) -> dict:
    return json.loads((Path(out_root) / "tsf_export" / f"{clip_id}.json").read_text())
