"""TanitSim 2D renderer: cartographic bird's-eye view with production-grade visual detail.

Layers
  static (rasterised once, 2x supersampled): context (city blocks + buildings + trees, or grass +
    guardrails), sidewalks/shoulders, curbs, asphalt (shapely union of all lanes, junction plates
    closed), lane markings (dashed/solid white, double-yellow centre), stop bars, zebra crossings,
    painted turn arrows, speed-limit signs, signal poles.
  dynamic (per frame, 2x): soft vehicle shadows, category sprites (body, roof, glass, head/tail
    lights, brake lights, blinking indicators), ego halo, fading trail + future path, TTC threat
    links, traffic-light heads with glowing active lamp.
  HUD (1x, PIL typography): title bar + progress, status chips, speed gauge, accel bar, event
    ticker, minimap.
"""
from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

import cv2
import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import MultiPolygon, Polygon, box
from shapely.ops import unary_union

from tsf.sim.engine import SimLog, TanitSim, lane_turn, obb_corners, obb_overlap
from tsf.sim.odr import OdrMap

SS = 2                      # supersampling factor for geometry
MAX_CANVAS = 26000
FONT_DIR = Path("/usr/share/fonts/truetype/noto")


def _hex(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def bgr(h: str) -> tuple[int, int, int]:
    r, g, b = _hex(h)
    return b, g, r


P = {  # palette (hex, RGB)
    "ground": "#11151b", "block": "#171c24", "lot": "#1b212a", "roof": ["#273040", "#2b3546", "#303b4d", "#252d3a"],
    "roof_hi": "#3a465a", "bshadow": "#07090c", "tree": "#1d3a2a", "tree_hi": "#2a5139",
    "grass": "#142019", "grass2": "#17251c", "shoulder": "#2c3037", "guardrail": "#8d96a3",
    "sidewalk": "#353b45", "curb": "#79828f", "asphalt": "#262a31", "plate": "#292d35",
    "white": "#e6eaf0", "yellow": "#f0bf4c", "stop": "#f2f4f7",
    "ego": "#2ee6d6", "amber": "#ffb020", "red": "#ff4d5e", "green": "#3ddc84",
    "glass": "#16202b", "head": "#fff3c4", "tail": "#7a1d24", "brake": "#ff3040", "ind": "#ffb020",
    "hud": "#0d1117", "hud_text": "#e8edf3", "hud_dim": "#8b96a5",
}
VEH_COLORS = {"white": "#e9edf2", "silver": "#b9c0ca", "grey": "#7c8591", "gray": "#7c8591", "black": "#30353d",
              "red": "#d64045", "blue": "#3a7bd5", "yellow": "#f2c14e", "green": "#3fb27f", "orange": "#f08a3c",
              "brown": "#8b5e3c"}


# ----------------------------------------------------------------------------- geometry helpers
def _offset(pts: np.ndarray, d: float) -> np.ndarray:
    """Offset a polyline laterally by d (+ = left of its direction)."""
    t = np.gradient(pts, axis=0)
    n = np.stack([-t[:, 1], t[:, 0]], 1)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    return pts + n * d


def _local(cx, cy, yaw, pts) -> np.ndarray:
    pts = np.asarray(pts, dtype=float)
    c, s = math.cos(yaw), math.sin(yaw)
    return np.stack([cx + c * pts[:, 0] - s * pts[:, 1], cy + s * pts[:, 0] + c * pts[:, 1]], 1)


@lru_cache(maxsize=256)
def _rrect_local(l: float, w: float, r: float, n: int = 6) -> np.ndarray:
    r = min(r, l / 2, w / 2)
    pts = []
    for cx, cy, a0 in ((l / 2 - r, w / 2 - r, 0), (-l / 2 + r, w / 2 - r, 90), (-l / 2 + r, -w / 2 + r, 180),
                       (l / 2 - r, -w / 2 + r, 270)):
        for k in range(n + 1):
            a = math.radians(a0 + 90 * k / n)
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return np.array(pts)


def _lane_poly(ln) -> Polygon:
    return Polygon(np.concatenate([ln.left_border, ln.right_border[::-1]])).buffer(0)


def _geoms(g):
    if g.is_empty:
        return []
    return list(g.geoms) if isinstance(g, MultiPolygon) or hasattr(g, "geoms") else [g]


# ----------------------------------------------------------------------------- static map art
class MapArt:
    def __init__(self, m: OdrMap, ppm: float, seed: int = 7, region=None):
        self.m, self.ppm, self.rng = m, ppm * SS, np.random.default_rng(seed)
        lanes = list(m.lanes.values())
        self.urban = min((ln.speed for ln in lanes), default=13.9) * 3.6 <= 70 or bool(m.junctions)
        road = unary_union([_lane_poly(ln) for ln in lanes])
        self.road = road.buffer(1.5, join_style=1).buffer(-1.5, join_style=1)   # close junction notches
        x0, y0, x1, y1 = self.road.bounds
        pad = 70.0
        if region is not None:
            x0, y0, x1, y1 = max(x0, region[0]), max(y0, region[1]), min(x1, region[2]), min(y1, region[3])
        self.bounds = (x0 - pad, y0 - pad, x1 + pad, y1 + pad)
        bx0, by0, bx1, by1 = self.bounds
        self.ppm = min(self.ppm, MAX_CANVAS / (bx1 - bx0), MAX_CANVAS / (by1 - by0))
        self.W, self.H = int((bx1 - bx0) * self.ppm), int((by1 - by0) * self.ppm)
        self.img = np.full((self.H, self.W, 3), bgr(P["ground"]), np.uint8)
        self._draw()

    # world -> canvas pixel (x right, y down)
    def px(self, xy) -> np.ndarray:
        xy = np.asarray(xy, dtype=float)
        return np.stack([(xy[..., 0] - self.bounds[0]) * self.ppm, (self.bounds[3] - xy[..., 1]) * self.ppm], -1)

    def fill(self, img, pts, color, off=(0, 0)):
        p = ((self.px(pts) - np.asarray(off)) * 16).astype(np.int32)
        cv2.fillPoly(img, [p], color, cv2.LINE_AA, shift=4)

    def line(self, img, pts, color, width_m, off=(0, 0), closed=False):
        p = ((self.px(pts) - np.asarray(off)) * 16).astype(np.int32)
        cv2.polylines(img, [p], closed, color, max(1, int(round(width_m * self.ppm))), cv2.LINE_AA, shift=4)

    def fill_geom(self, img, g, color, hole_color):
        for poly in _geoms(g):
            if poly.is_empty or not hasattr(poly, "exterior"):
                continue
            self.fill(img, np.asarray(poly.exterior.coords), color)
            for hole in poly.interiors:
                self.fill(img, np.asarray(hole.coords), hole_color)

    def dashed(self, img, pts, color, width_m, dash, gap):
        d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        cum = np.concatenate([[0], np.cumsum(d)])
        s = gap * 0.5
        while s < cum[-1]:
            a, b = s, min(s + dash, cum[-1])
            seg = np.stack([np.interp(np.linspace(a, b, 5), cum, pts[:, i]) for i in (0, 1)], 1)
            self.line(img, seg, color, width_m)
            s += dash + gap

    def _shadow(self, mask_geom, dx=1.2, dy=-1.2, blur_m=2.5, strength=0.55):
        mask = np.zeros((self.H, self.W), np.uint8)
        for poly in _geoms(mask_geom):
            if hasattr(poly, "exterior"):
                pts = np.asarray(poly.exterior.coords) + (dx, dy)
                self.fill(mask, pts, 255)
        k = int(blur_m * self.ppm) | 1
        mask = cv2.GaussianBlur(mask, (k, k), 0).astype(np.float32) / 255.0 * strength
        self.img[:] = (self.img.astype(np.float32) * (1 - mask[..., None])).astype(np.uint8)

    def _draw(self):
        img, rng = self.img, self.rng
        box_all = box(*self.bounds)
        if self.urban:
            walk = self.road.buffer(3.2, join_style=1)
            blocks = box_all.difference(walk.buffer(0.6))
            self.fill_geom(img, blocks, bgr(P["block"]), bgr(P["ground"]))
            buildings, trees = [], []
            for blk in _geoms(blocks):
                if blk.area < 150:
                    continue
                inner = blk.buffer(-4.0)
                if inner.is_empty:
                    continue
                mrr = inner.minimum_rotated_rectangle
                c = np.asarray(mrr.exterior.coords)[:4]
                e0 = c[1] - c[0]
                ang = math.atan2(e0[1], e0[0])
                cx, cy = inner.centroid.x, inner.centroid.y
                span = max(np.linalg.norm(c[1] - c[0]), np.linalg.norm(c[2] - c[1])) + 40
                step = 26.0
                for u in np.arange(-span / 2, span / 2, step):
                    for v in np.arange(-span / 2, span / 2, step):
                        lw = rng.uniform(13, 22, 2)
                        lot = Polygon(_local(cx, cy, ang, _rrect_local(float(lw[0]), float(lw[1]), 0.8, 2)
                                             + (u + rng.uniform(-2, 2), v + rng.uniform(-2, 2))))
                        if inner.contains(lot):
                            buildings.append(lot)
                        elif inner.contains(lot.centroid) and rng.random() < 0.6:
                            trees.append((lot.centroid.x, lot.centroid.y, rng.uniform(2.0, 3.4)))
            if buildings:
                self._shadow(unary_union(buildings), dx=2.5, dy=-2.5, blur_m=3.0, strength=0.6)
            for b in buildings:
                col = bgr(P["roof"][int(rng.integers(len(P["roof"])))])
                pts = np.asarray(b.exterior.coords)
                self.fill(img, pts, col)
                self.line(img, pts, bgr(P["roof_hi"]), 0.25, closed=True)
                inset = b.buffer(-2.2)
                if not inset.is_empty and hasattr(inset, "exterior") and rng.random() < 0.5:
                    self.line(img, np.asarray(inset.exterior.coords), bgr(P["block"]), 0.18, closed=True)
            for (x, y, r) in trees:
                circ = np.array([(x + r * math.cos(a), y + r * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 18)])
                self.fill(img, circ + (0.8, -0.8), bgr(P["bshadow"]))
                self.fill(img, circ, bgr(P["tree"]))
                self.fill(img, (circ - (x, y)) * 0.55 + (x - 0.3 * r, y + 0.3 * r), bgr(P["tree_hi"]))
            self.fill_geom(img, walk, bgr(P["sidewalk"]), bgr(P["ground"]))
            for poly in _geoms(walk):
                if hasattr(poly, "exterior"):
                    self.line(img, np.asarray(poly.exterior.coords), bgr(P["block"]), 0.3, closed=True)
        else:
            self.fill(img, np.asarray(box_all.exterior.coords), bgr(P["grass"]))
            x0, y0, x1, y1 = self.bounds
            for yy in np.arange(y0, y1, 6.0):  # mowing stripes
                if int(yy / 6) % 2:
                    self.fill(img, np.array([(x0, yy), (x1, yy), (x1, yy + 3), (x0, yy + 3)]), bgr(P["grass2"]))
            shoulder = self.road.buffer(2.4, join_style=2)
            self.fill_geom(img, shoulder, bgr(P["shoulder"]), bgr(P["grass"]))
            for poly in _geoms(self.road.buffer(3.3, join_style=2)):
                if hasattr(poly, "exterior"):
                    ring = np.asarray(poly.exterior.coords)
                    self.line(img, ring, bgr(P["guardrail"]), 0.22, closed=True)
            edge = self.road.buffer(9.0)
            for _ in range(int((x1 - x0) / 9)):
                x, y = rng.uniform(x0, x1), rng.uniform(y0, y1)
                if edge.contains(Polygon([(x - 1, y - 1), (x + 1, y - 1), (x + 1, y + 1)])):
                    continue
                r = rng.uniform(2.0, 3.6)
                circ = np.array([(x + r * math.cos(a), y + r * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 16)])
                self.fill(img, circ + (0.9, -0.9), bgr(P["bshadow"]))
                self.fill(img, circ, bgr(P["tree"]))
                self.fill(img, (circ - (x, y)) * 0.5 + (x - 0.3 * r, y + 0.3 * r), bgr(P["tree_hi"]))
        self._shadow(self.road, dx=0.0, dy=0.0, blur_m=1.6, strength=0.35)
        self.fill_geom(img, self.road, bgr(P["asphalt"]), bgr(P["sidewalk"]))
        for ln in self.m.lanes.values():
            if ln.junction != "-1":
                self.fill(img, np.concatenate([ln.left_border, ln.right_border[::-1]]), bgr(P["plate"]))
        for poly in _geoms(self.road):
            if hasattr(poly, "exterior"):
                self.line(img, np.asarray(poly.exterior.coords), bgr(P["curb"]), 0.28, closed=True)
        self._markings()
        self._crosswalks()
        self._arrows()
        self._signs()

    # ---------------------------------------------------------------- road markings
    def _markings(self):
        img = self.img
        white, yellow = bgr(P["white"]), bgr(P["yellow"])
        for (rid, si, lid), ln in self.m.lanes.items():
            if ln.junction != "-1":
                continue
            highway = ln.speed * 3.6 > 75
            dash, gap = (6.0, 12.0) if highway else (3.0, 6.0)
            opposite = any(k[0] == rid and k[1] == si and (k[2] > 0) != (lid > 0) for k in self.m.lanes)
            if ln.left is not None:
                self.dashed(img, ln.left_border, white, 0.14, dash, gap)
            elif opposite:
                if lid < 0:  # draw the shared centre line once (double yellow)
                    self.line(img, _offset(ln.left_border, 0.13), yellow, 0.12)
                    self.line(img, _offset(ln.left_border, -0.13), yellow, 0.12)
            else:
                self.line(img, _offset(ln.left_border, -0.3), white, 0.15)
            if ln.right is None:
                self.line(img, _offset(ln.right_border, 0.3), white, 0.15)
            if ln.successors and self.m.lanes[ln.successors[0]].junction != "-1":
                a, b = ln.left_border[-1], ln.right_border[-1]
                h = ln.hdg[-1]
                back = np.array([math.cos(h), math.sin(h)]) * 0.55
                self.fill(img, np.array([a - back, b - back, b, a]), bgr(P["stop"]))

    def _arms(self):
        """(mouth point, inward heading, half-width left, half-width right) for roads touching a junction."""
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
                lids = [k for k in r.sections[sec].lanes]
                tl = max((float(r.lane_t(sec, i, np.array([s]), "outer")[0]) for i in lids if i > 0), default=0.0)
                tr = min((float(r.lane_t(sec, i, np.array([s]), "outer")[0]) for i in lids if i < 0), default=0.0)
                inward = float(h[0]) + (math.pi if s == 0.0 else 0.0)
                out.append((np.array([x[0], y[0]]), float(h[0]), inward, tl, tr))
        return out

    def _crosswalks(self):
        img = self.img
        for p, h, inward, tl, tr in self._arms():
            fwd = np.array([math.cos(inward), math.sin(inward)])
            left = np.array([-math.sin(h), math.cos(h)])
            for t in np.arange(tr + 0.4, tl - 0.3, 1.1):
                a = p + fwd * 1.2 + left * t
                q = [a, a + left * 0.55, a + left * 0.55 + fwd * 3.6, a + fwd * 3.6]
                self.fill(img, np.array(q), bgr(P["white"]))

    def _arrows(self):
        img = self.img
        col = bgr(P["white"])
        for k, ln in self.m.lanes.items():
            if ln.junction != "-1" or not ln.successors:
                continue
            turns = {lane_turn(self.m.lanes[s]) for s in ln.successors if self.m.lanes[s].junction != "-1"}
            if not turns or ln.length < 20:
                continue
            for back in (7.0, 27.0):
                if ln.length < back + 6:
                    continue
                s = ln.length - back
                x, y, h = ln.pose(s)
                for t in sorted(turns):
                    self._arrow(img, x, y, h, t, col)

    def _arrow(self, img, x, y, h, turn, col):
        """Road-marking arrow glyphs (~5 m long), local x = driving direction, y = left."""
        self.line(img, _local(x, y, h, np.array([(-3.0, 0), (0.2, 0)])), col, 0.24)
        if turn == "straight":
            self.line(img, _local(x, y, h, np.array([(0.2, 0), (1.2, 0)])), col, 0.24)
            head = np.array([(2.6, 0), (1.1, 0.62), (1.1, -0.62)])
        else:
            sgn = 1 if turn == "left" else -1
            bend = np.array([(0.2 + 0.9 * math.sin(a), sgn * 0.9 * (1 - math.cos(a))) for a in np.linspace(0, 1.2, 8)])
            self.line(img, _local(x, y, h, bend), col, 0.24)
            ex, ey = bend[-1]
            d = np.array([math.cos(1.2), sgn * math.sin(1.2)])
            n = np.array([-d[1], d[0]])
            tip = np.array([ex, ey]) + d * 1.3
            head = np.array([tip, (ex, ey) + n * 0.6 - d * 0.1, (ex, ey) - n * 0.6 - d * 0.1])
        self.fill(img, _local(x, y, h, head), col)

    def _signs(self):
        img = self.img
        self.sign_spots = []
        for r in self.m.roads.values():
            if r.junction != "-1" or r.speed_kph is None:
                continue
            for lid_sign, s in ((-1, min(40.0, r.length * 0.3)), (1, max(r.length - 40.0, r.length * 0.7))):
                lids = [k for k in r.sections[0].lanes if (k > 0) == (lid_sign > 0)]
                if not lids:
                    continue
                outer = max(lids, key=abs)
                si = 0 if s < r.sections[0].s_end else len(r.sections) - 1
                t = float(r.lane_t(si, outer, np.array([s]), "outer")[0]) + 2.4 * (1 if lid_sign > 0 else -1)
                x, y, h = r.ref(np.array([s]))
                cx, cy = x[0] - math.sin(h[0]) * t, y[0] + math.cos(h[0]) * t
                rad = 1.25
                circ = np.array([(cx + rad * math.cos(a), cy + rad * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 28)])
                self.fill(img, circ + (0.25, -0.25), bgr(P["bshadow"]))
                self.fill(img, circ, (60, 60, 220))
                self.fill(img, (circ - (cx, cy)) * 0.78 + (cx, cy), (245, 245, 245))
                c = self.px([cx, cy]).astype(int)
                txt = str(int(round(r.speed_kph)))
                fs = rad * self.ppm / 26.0
                (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_DUPLEX, fs, max(1, int(fs * 1.6)))
                cv2.putText(img, txt, (c[0] - tw // 2, c[1] + th // 2), cv2.FONT_HERSHEY_DUPLEX, fs, (20, 20, 20),
                            max(1, int(fs * 1.6)), cv2.LINE_AA)


# ----------------------------------------------------------------------------- renderer
class Renderer:
    def __init__(self, sim: TanitSim, log: SimLog, size=(1280, 720), ppm: float | None = None, title: str = ""):
        self.sim, self.log, self.W, self.H = sim, log, size[0], size[1]
        self.title = title or sim.sc.meta.title
        vmax = float(log.states[:, :, 3].max())
        self.ppm = ppm or float(np.clip(1280 / (vmax * 4 + 40), 4.5, 12.0))
        S = log.states
        xs, ys = S[:, :, 0], S[:, :, 1]
        view = (self.W / self.ppm, self.H / self.ppm)
        region = (xs.min() - view[0], ys.min() - view[1], xs.max() + view[0], ys.max() + view[1])
        self.art = MapArt(sim.m, self.ppm, seed=abs(hash(sim.sc.meta.id)) % 2**31, region=region)
        self.colors = {}
        for a in sim.sc.actors:
            self.colors[a.id] = VEH_COLORS.get(a.color.lower(), "#9aa3ad")
        self.e = log.roles.index("ego")
        self._indicators()
        self._fonts()
        self._minimap_init()

    # ---------------------------------------------------------------- precomputation
    def _indicators(self):
        """Per actor per tick: -1 right, +1 left, 0 off (lateral drift or upcoming junction turn)."""
        log, m = self.log, self.sim.m
        n, N = len(log.t), len(log.ids)
        ind = np.zeros((n, N), np.int8)
        dt = log.t[1] - log.t[0] if n > 1 else 0.05
        look = int(3.5 / dt)
        for j in range(N):
            d = log.states[:, j, 5]
            dd = np.gradient(d, dt) if n > 2 else np.zeros(n)
            ind[:, j] = np.where(dd > 0.12, 1, np.where(dd < -0.12, -1, 0))
            seq = [row[j] for row in log.lanes]
            for i in range(n):
                if ind[i, j]:
                    continue
                cur = seq[i]
                if cur is None or m.lanes[cur].junction != "-1":
                    if cur is not None and m.lanes[cur].junction != "-1":
                        t = lane_turn(m.lanes[cur])
                        ind[i, j] = 1 if t == "left" else -1 if t == "right" else 0
                    continue
                for k in range(i + 1, min(n, i + look)):
                    nk = seq[k]
                    if nk is not None and nk != cur and m.lanes[nk].junction != "-1":
                        t = lane_turn(m.lanes[nk])
                        ind[i, j] = 1 if t == "left" else -1 if t == "right" else 0
                        break
        self.ind = ind

    def _fonts(self):
        def f(name, size):
            try:
                return ImageFont.truetype(str(FONT_DIR / name), size)
            except OSError:
                return ImageFont.load_default()
        self.f_small, self.f_med = f("NotoSans-Regular.ttf", 13), f("NotoSans-Regular.ttf", 15)
        self.f_bold, self.f_big = f("NotoSans-Bold.ttf", 16), f("NotoSans-Bold.ttf", 30)
        self.f_title = f("NotoSans-Bold.ttf", 17)

    def _minimap_init(self, size=(230, 150)):
        x0, y0, x1, y1 = self.sim.m.bounds()
        s = min((size[0] - 16) / max(x1 - x0, 1), (size[1] - 16) / max(y1 - y0, 1))
        self.mm_tf = (x0, y1, s, size)
        img = Image.new("RGBA", size, (13, 17, 23, 215))
        d = ImageDraw.Draw(img)
        for ln in self.sim.m.lanes.values():
            pts = [((x - x0) * s + 8, (y1 - y) * s + 8) for x, y in ln.xy[::4]]
            if len(pts) > 1:
                d.line(pts, fill=(96, 106, 120, 255), width=2)
        self.mm_base = img

    # ---------------------------------------------------------------- per-frame drawing (2x)
    def _view(self, i):
        S = self.log.states
        cx, cy = S[i, self.e, 0], S[i, self.e, 1]
        c = self.art.px([cx, cy])
        return (int(c[0] - self.W * SS / 2), int(c[1] - self.H * SS / 2))

    def _crop(self, off):
        Wp, Hp = self.W * SS, self.H * SS
        img = np.full((Hp, Wp, 3), bgr(P["ground"]), np.uint8)
        H, W = self.art.img.shape[:2]
        xa, ya = max(0, off[0]), max(0, off[1])
        xb, yb = min(W, off[0] + Wp), min(H, off[1] + Hp)
        if xb > xa and yb > ya:
            img[ya - off[1]:yb - off[1], xa - off[0]:xb - off[0]] = self.art.img[ya:yb, xa:xb]
        return img

    def _vehicle(self, img, j, x, y, yaw, l, w, cat, color, acc, ind_state, blink, off):
        A = self.art
        body = _rrect_local(l, w, 0.12 * w if cat in ("truck", "bus") else 0.32 * w)
        A.fill(img, _local(x, y, yaw, body), bgr(color), off)
        dark = tuple(int(c * 0.55) for c in bgr(color))
        A.line(img, _local(x, y, yaw, body), dark, 0.08, off, closed=True)
        glass = bgr(P["glass"])
        if cat in ("truck",):
            cab = 2.3
            A.fill(img, _local(x, y, yaw, np.array([(l / 2 - 0.25, w / 2 - 0.2), (l / 2 - 0.25, -w / 2 + 0.2),
                                                    (l / 2 - 0.9, -w / 2 + 0.25), (l / 2 - 0.9, w / 2 - 0.25)])), glass, off)
            A.line(img, _local(x, y, yaw, np.array([(l / 2 - cab, w / 2), (l / 2 - cab, -w / 2)])), dark, 0.25, off)
            A.fill(img, _local(x, y, yaw, _rrect_local(l - cab - 0.6, w - 0.5, 0.1) - (cab / 2 + 0.1, 0)),
                   tuple(min(255, int(c * 1.08)) for c in bgr(color)), off)
        elif cat == "bus":
            for side in (1, -1):
                A.fill(img, _local(x, y, yaw, np.array([(l / 2 - 1.4, side * (w / 2 - 0.08)), (-l / 2 + 1.0, side * (w / 2 - 0.08)),
                                                        (-l / 2 + 1.0, side * (w / 2 - 0.42)), (l / 2 - 1.4, side * (w / 2 - 0.42))])),
                       glass, off)
            A.fill(img, _local(x, y, yaw, np.array([(l / 2 - 0.15, w / 2 - 0.15), (l / 2 - 0.15, -w / 2 + 0.15),
                                                    (l / 2 - 0.9, -w / 2 + 0.2), (l / 2 - 0.9, w / 2 - 0.2)])), glass, off)
            for k in range(3):
                A.fill(img, _local(x, y, yaw, _rrect_local(1.2, 0.9, 0.1) + (l / 2 - 3.5 - k * 3.2, 0)), dark, off)
        elif cat in ("motorcycle", "bicycle"):
            A.fill(img, _local(x, y, yaw, _rrect_local(0.6, 0.6, 0.3) + (-0.1, 0)), (40, 40, 40), off)
        else:  # car / suv / van
            roof_l = 0.42 * l if cat != "van" else 0.62 * l
            A.fill(img, _local(x, y, yaw, np.array([(roof_l / 2 + 0.95, w / 2 - 0.22), (roof_l / 2 + 0.95, -w / 2 + 0.22),
                                                    (roof_l / 2 + 0.05, -w / 2 + 0.3), (roof_l / 2 + 0.05, w / 2 - 0.3)]) + (0.15, 0)),
                   glass, off)
            A.fill(img, _local(x, y, yaw, np.array([(-roof_l / 2 - 0.05, w / 2 - 0.3), (-roof_l / 2 - 0.05, -w / 2 + 0.3),
                                                    (-roof_l / 2 - 0.6, -w / 2 + 0.25), (-roof_l / 2 - 0.6, w / 2 - 0.25)]) + (0.15, 0)),
                   glass, off)
            roof = tuple(int(min(255, c * 0.9 + 12)) for c in bgr(color))
            A.fill(img, _local(x, y, yaw, _rrect_local(roof_l, w - 0.6, 0.25) + (0.15, 0)), roof, off)
        # lights
        hl, tl = bgr(P["head"]), bgr(P["brake"] if acc < -0.8 else P["tail"])
        for side in (1, -1):
            A.fill(img, _local(x, y, yaw, _rrect_local(0.22, 0.42, 0.08) + (l / 2 - 0.12, side * (w / 2 - 0.3))), hl, off)
            A.fill(img, _local(x, y, yaw, _rrect_local(0.2, 0.4, 0.08) + (-l / 2 + 0.11, side * (w / 2 - 0.3))), tl, off)
        if acc < -0.8:
            ov = img.copy()
            for side in (1, -1):
                c = A.px(_local(x, y, yaw, np.array([(-l / 2 - 0.05, side * (w / 2 - 0.3))])))[0] - off
                cv2.circle(ov, tuple(c.astype(int)), max(2, int(0.38 * A.ppm)), bgr(P["brake"]), -1, cv2.LINE_AA)
            cv2.addWeighted(ov, 0.45, img, 0.55, 0, img)
        if ind_state and blink:
            for fx in (l / 2 - 0.15, -l / 2 + 0.15):
                c = A.px(_local(x, y, yaw, np.array([(fx, ind_state * (w / 2 + 0.05))])))[0] - off
                cv2.circle(img, tuple(c.astype(int)), max(2, int(0.32 * A.ppm)), bgr(P["ind"]), -1, cv2.LINE_AA)

    def _pedestrian(self, img, x, y, yaw, color, off):
        A = self.art
        A.fill(img, _local(x, y, yaw, _rrect_local(0.42, 0.62, 0.2)), bgr(color), off)
        c = A.px([x, y]) - off
        cv2.circle(img, tuple(c.astype(int)), max(2, int(0.17 * A.ppm)), (60, 60, 60), -1, cv2.LINE_AA)

    def _signals(self, img, i, off):
        A, log = self.art, self.log
        for arm, states in log.signals.items():
            st = states[i]
            for sg in [sg for sg in self.sim.m.signals() if sg.road == str(arm)]:
                h = sg.hdg
                housing = _local(sg.x, sg.y, h, _rrect_local(2.6, 1.0, 0.25))
                A.fill(img, housing + (0.3, -0.3), bgr(P["bshadow"]), off)
                A.fill(img, housing, (28, 26, 24), off)
                A.line(img, housing, (90, 90, 90), 0.08, off, closed=True)
                for k, name in enumerate(("red", "yellow", "green")):
                    c = A.px(_local(sg.x, sg.y, h, np.array([(0.8 - 0.8 * k, 0.0)])))[0] - off
                    on = name == st
                    col = bgr(P["red"] if name == "red" else P["amber"] if name == "yellow" else P["green"])
                    if on:
                        for rr, a in ((1.6, 0.18), (1.1, 0.3)):
                            ov = img.copy()
                            cv2.circle(ov, tuple(c.astype(int)), int(rr * A.ppm), col, -1, cv2.LINE_AA)
                            cv2.addWeighted(ov, a, img, 1 - a, 0, img)
                    cv2.circle(img, tuple(c.astype(int)), max(2, int(0.3 * A.ppm)),
                               col if on else tuple(int(v * 0.25) for v in col), -1, cv2.LINE_AA)

    def _ttc_links(self, img, i, off):
        log, A, e = self.log, self.art, self.e
        S = log.states
        ex, ey, eyaw, ev = S[i, e, :4]
        links = []
        for j in range(len(log.ids)):
            if j == e or math.hypot(S[i, j, 0] - ex, S[i, j, 1] - ey) > 70:
                continue
            va = np.array([math.cos(eyaw), math.sin(eyaw)]) * ev
            vb = np.array([math.cos(S[i, j, 2]), math.sin(S[i, j, 2])]) * S[i, j, 3]
            ttc = None
            for tau in np.arange(0.2, 4.01, 0.2):
                Aa = obb_corners(ex + va[0] * tau, ey + va[1] * tau, eyaw, log.lwh[e, 0], log.lwh[e, 1])
                Bb = obb_corners(S[i, j, 0] + vb[0] * tau, S[i, j, 1] + vb[1] * tau, S[i, j, 2], log.lwh[j, 0],
                                 log.lwh[j, 1])
                if obb_overlap(Aa, Bb):
                    ttc = float(tau)
                    break
            if ttc is not None:
                col = bgr(P["red"] if ttc < 1.5 else P["amber"])
                A.line(img, np.array([[ex, ey], S[i, j, :2]]), col, 0.18, off)
                links.append((j, ttc))
        return links

    def frame(self, i: int) -> np.ndarray:
        log, A, e = self.log, self.art, self.e
        S = log.states
        off = self._view(i)
        img = self._crop(off)
        # shadows
        mask = np.zeros(img.shape[:2], np.uint8)
        for j in range(len(log.ids)):
            x, y, yaw = S[i, j, :3]
            l, w = log.lwh[j, :2]
            A.fill(mask, _local(x + 0.45, y - 0.45, yaw, _rrect_local(l, w, 0.3 * w)), 255, off)
        k = int(1.2 * A.ppm) | 1
        mk = cv2.GaussianBlur(mask, (k, k), 0).astype(np.float32) / 255 * 0.55
        img = (img.astype(np.float32) * (1 - mk[..., None])).astype(np.uint8)
        # ego trail (fading) and future path (dots)
        dt = log.t[1] - log.t[0] if len(log.t) > 1 else 0.05
        n_tr = int(6 / dt)
        a0 = max(0, i - n_tr)
        ego_c = np.array(bgr(P["ego"]), float)
        bg = np.array(bgr(P["asphalt"]), float)
        for q in range(a0, i, 2):
            f = (q - a0) / max(1, i - a0)
            col = tuple(int(v) for v in bg + (ego_c - bg) * (0.15 + 0.85 * f))
            A.line(img, S[q:q + 3, e, :2], col, 0.35 + 0.35 * f, off)
        for q in range(i + int(0.5 / dt), min(len(log.t), i + int(4 / dt)), max(1, int(0.25 / dt))):
            c = A.px(S[q, e, :2]) - off
            cv2.circle(img, tuple(c.astype(int)), max(2, int(0.16 * A.ppm)), tuple(int(v) for v in ego_c * 0.85), -1,
                       cv2.LINE_AA)
        links = self._ttc_links(img, i, off)
        self._signals(img, i, off)
        blink = int(log.t[i] * 3) % 2 == 0
        # ego halo
        x, y, yaw = S[i, e, :3]
        halo = np.zeros(img.shape[:2], np.uint8)
        A.fill(halo, _local(x, y, yaw, _rrect_local(log.lwh[e, 0] + 1.6, log.lwh[e, 1] + 1.6, 0.9)), 255, off)
        kh = int(1.4 * A.ppm) | 1
        hm = cv2.GaussianBlur(halo, (kh, kh), 0).astype(np.float32) / 255 * 0.65
        img = (img.astype(np.float32) * (1 - hm[..., None]) + ego_c[None, None, :] * hm[..., None]).astype(np.uint8)
        for j, aid in enumerate(log.ids):
            x, y, yaw, v, acc = S[i, j, :5]
            l, w = log.lwh[j, :2]
            cat = log.categories[j]
            if cat == "pedestrian":
                self._pedestrian(img, x, y, yaw, self.colors.get(aid, "#9aa3ad"), off)
            else:
                self._vehicle(img, j, x, y, yaw, l, w, cat, self.colors.get(aid, "#9aa3ad"), acc,
                              int(self.ind[i, j]), blink, off)
        small = cv2.resize(img, (self.W, self.H), interpolation=cv2.INTER_AREA)
        return self._hud(small, i, off, links)

    # ---------------------------------------------------------------- HUD (1x, PIL)
    def _hud(self, img_bgr, i, off, links) -> np.ndarray:
        log, e, A = self.log, self.e, self.art
        S = log.states
        base = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")
        ov = Image.new("RGBA", base.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(ov)
        W, H = self.W, self.H
        txt, dim, ego = _hex(P["hud_text"]), _hex(P["hud_dim"]), _hex(P["ego"])
        # labels for actors in view
        for j, aid in enumerate(log.ids):
            c = (A.px(S[i, j, :2]) - off) / SS
            if not (20 < c[0] < W - 20 and 50 < c[1] < H - 20):
                continue
            label = "EGO" if j == e else aid
            tw = d.textlength(label, font=self.f_small)
            x0, y0 = c[0] + 10, c[1] - 26
            fill = (*ego, 235) if j == e else (13, 17, 23, 190)
            d.rounded_rectangle([x0, y0, x0 + tw + 12, y0 + 18], 5, fill=fill)
            d.text((x0 + 6, y0 + 1), label, font=self.f_small, fill=(10, 14, 18, 255) if j == e else (*txt, 255))
        for j, ttc in links:
            c = (A.px(S[i, j, :2]) - off) / SS
            col = _hex(P["red"] if ttc < 1.5 else P["amber"])
            d.rounded_rectangle([c[0] - 34, c[1] + 12, c[0] + 34, c[1] + 30], 5, fill=(*col, 230))
            d.text((c[0] - 28, c[1] + 13), f"TTC {ttc:.1f}s", font=self.f_small, fill=(12, 12, 12, 255))
        # top bar
        d.rectangle([0, 0, W, 44], fill=(*_hex(P["hud"]), 235))
        d.ellipse([14, 12, 34, 32], outline=(*ego, 255), width=3)
        d.ellipse([20, 18, 28, 26], fill=(*ego, 255))
        d.text((44, 11), "TANIT SIM", font=self.f_title, fill=(*txt, 255))
        d.text((150, 13), self.title if len(self.title) <= 80 else self.title[:77] + "...", font=self.f_med, fill=(*dim, 255))
        d.text((W - 118, 12), f"t {log.t[i]:6.2f} s", font=self.f_bold, fill=(*ego, 255))
        prog = log.t[i] / max(log.t[-1], 1e-6)
        d.rectangle([0, 44, W, 47], fill=(30, 36, 44, 255))
        d.rectangle([0, 44, int(W * prog), 47], fill=(*ego, 255))
        # chips
        env = self.sim.sc.environment
        chips = [(f"{env.time_of_day} · {env.weather} · {env.road_surface}", dim)]
        mt = log.metrics.get("min_ttc")
        if log.metrics.get("collision"):
            chips.append(("COLLISION", _hex(P["red"])))
        else:
            chips.append((f"min TTC {mt if mt is not None else 'none'}{' s' if mt is not None else ''}", _hex(P["green"]) if (mt or 9) > 2.5 else _hex(P["amber"])))
        arm = None
        lane = log.lanes[i][e]
        if lane and lane[0].isdigit() and int(lane[0]) in log.signals:
            arm = int(lane[0])
        if arm is not None:
            st = log.signals[arm][i]
            chips.append((f"signal {st}", _hex({"green": P["green"], "yellow": P["amber"], "red": P["red"]}[st])))
        x = 14
        for label, col in chips:
            tw = d.textlength(label, font=self.f_small)
            d.rounded_rectangle([x, 56, x + tw + 22, 78], 11, fill=(13, 17, 23, 200), outline=(*col, 255), width=1)
            d.ellipse([x + 8, 63, x + 14, 69], fill=(*col, 255))
            d.text((x + 18, 58), label, font=self.f_small, fill=(*txt, 255))
            x += tw + 32
        # events
        fired = [ev for ev in log.events if ev["type"] == "action_start" and ev["t"] <= log.t[i] + 1e-9][-4:]
        for k, ev in enumerate(reversed(fired)):
            y = H - 34 - 26 * k
            alpha = 230 - 45 * k
            label = f"{ev['t']:5.1f}s   {ev['actor']}  ›  {ev['id']}"
            tw = d.textlength(label, font=self.f_small)
            d.rounded_rectangle([14, y, 14 + tw + 18, y + 21], 6, fill=(13, 17, 23, alpha))
            d.text((23, y + 2), label, font=self.f_small, fill=(*txt, alpha))
        # speed gauge
        v = S[i, e, 3] * 3.6
        vmax = max(60.0, float(np.ceil(S[:, e, 3].max() * 3.6 * 1.2 / 20) * 20))
        cx, cy, r = W - 92, H - 92, 62
        d.ellipse([cx - r - 12, cy - r - 12, cx + r + 12, cy + r + 12], fill=(13, 17, 23, 215))
        d.arc([cx - r, cy - r, cx + r, cy + r], 135, 405, fill=(45, 52, 62, 255), width=8)
        d.arc([cx - r, cy - r, cx + r, cy + r], 135, 135 + 270 * min(v / vmax, 1.0), fill=(*ego, 255), width=8)
        s = f"{v:.0f}"
        tw = d.textlength(s, font=self.f_big)
        d.text((cx - tw / 2, cy - 24), s, font=self.f_big, fill=(*txt, 255))
        d.text((cx - 17, cy + 10), "km/h", font=self.f_small, fill=(*dim, 255))
        acc = float(S[i, e, 4])
        bw = 110
        d.rounded_rectangle([cx - bw / 2, cy + 40, cx + bw / 2, cy + 46], 3, fill=(45, 52, 62, 255))
        frac = max(-1.0, min(1.0, acc / (6.0 if acc < 0 else 3.0)))
        col = _hex(P["red"] if acc < -0.5 else P["green"] if acc > 0.3 else P["hud_dim"])
        x1 = cx + frac * bw / 2
        d.rounded_rectangle([min(cx, x1), cy + 40, max(cx, x1), cy + 46], 3, fill=(*col, 255))
        # minimap
        x0m, y1m, sm, (mw, mh) = self.mm_tf
        mm = self.mm_base.copy()
        md = ImageDraw.Draw(mm)
        for j in range(len(log.ids)):
            px = ((S[i, j, 0] - x0m) * sm + 8, (y1m - S[i, j, 1]) * sm + 8)
            rr = 4 if j == e else 2.5
            md.ellipse([px[0] - rr, px[1] - rr, px[0] + rr, px[1] + rr], fill=(*ego, 255) if j == e else (200, 205, 212, 255))
        vw, vh = W / self.ppm * sm, H / self.ppm * sm
        ex, ey = (S[i, e, 0] - x0m) * sm + 8, (y1m - S[i, e, 1]) * sm + 8
        md.rectangle([ex - vw / 2, ey - vh / 2, ex + vw / 2, ey + vh / 2], outline=(*ego, 150), width=1)
        ov.paste(mm, (W - mw - 14, 56), mm)
        d.rounded_rectangle([W - mw - 14, 56, W - 14, 56 + mh], 8, outline=(60, 68, 80, 255), width=1)
        out = Image.alpha_composite(base, ov).convert("RGB")
        return np.asarray(out)


def render_video(sim: TanitSim, log: SimLog, out: str | Path, fps: int = 20, title: str = "",
                 size=(1280, 720), follow: bool = True) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    r = Renderer(sim, log, size, title=title)
    dt = log.t[1] - log.t[0] if len(log.t) > 1 else 0.05
    stride = max(1, int(round(1.0 / (fps * dt))))
    frames = [r.frame(i) for i in range(0, len(log.t), stride)]
    iio.imwrite(out, np.stack(frames), fps=fps, codec="libx264", quality=8, macro_block_size=8)
    return out


def render_png(sim: TanitSim, log: SimLog, out: str | Path, i: int | None = None, title: str = "",
               size=(1280, 720)) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    r = Renderer(sim, log, size, title=title)
    rgb = r.frame(len(log.t) // 2 if i is None else i)
    cv2.imwrite(str(out), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return out
