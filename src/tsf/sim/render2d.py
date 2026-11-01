"""TanitSim 2D bird's-eye renderer (OpenCV) -> MP4 / PNG.

Static map layer is rasterised once on a large canvas at the chosen scale; each frame crops an
ego-centred window (chase view) and overlays signals, actors, ego trail, HUD and a minimap.
"""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import imageio.v3 as iio
import numpy as np

from tsf.sim.engine import SimLog, TanitSim, obb_corners
from tsf.sim.odr import OdrMap

BG, ASPHALT, JUNCTION = (20, 23, 28), (54, 50, 46), (62, 58, 54)
WHITE, YELLOW = (235, 235, 235), (40, 200, 240)
CAT = {"car": (200, 160, 90), "suv": (180, 140, 100), "van": (150, 170, 110), "truck": (90, 140, 210),
       "bus": (60, 190, 230), "motorcycle": (200, 120, 200), "bicycle": (120, 220, 160),
       "pedestrian": (130, 230, 250)}
EGO = (255, 200, 40)
SIG = {"green": (90, 220, 90), "yellow": (40, 200, 240), "red": (70, 70, 235)}
MAX_CANVAS = 24000


class Bev:
    def __init__(self, m: OdrMap, size=(1280, 720), ppm: float | None = None, follow: bool = True):
        self.m, self.W, self.H, self.follow = m, size[0], size[1], follow
        x0, y0, x1, y1 = m.bounds()
        if ppm is None:
            ppm = min(self.W / (x1 - x0 + 20), self.H / (y1 - y0 + 20)) if not follow else 8.0
        ppm = min(ppm, MAX_CANVAS / max(x1 - x0 + 2 * self.W / ppm, 1), MAX_CANVAS / max(y1 - y0 + 2 * self.H / ppm, 1))
        self.ppm = ppm
        mx, my = self.W / ppm, self.H / ppm
        self.ox, self.oy = x0 - mx, y1 + my             # world coords of canvas top-left
        cw, ch = int((x1 - x0 + 2 * mx) * ppm), int((y1 - y0 + 2 * my) * ppm)
        self.base = np.full((ch, cw, 3), BG, np.uint8)
        self._draw_static(self.base)
        self.mini = self._minimap(x0, y0, x1, y1)

    # world -> canvas pixels
    def px(self, xy) -> np.ndarray:
        xy = np.asarray(xy, dtype=float)
        return np.stack([(xy[..., 0] - self.ox) * self.ppm, (self.oy - xy[..., 1]) * self.ppm], -1)

    def _poly(self, img, pts, color, closed=True, thick=-1, off=(0, 0)):
        p = ((self.px(pts) - np.asarray(off)) * 16).astype(np.int32)
        if thick < 0:
            cv2.fillPoly(img, [p], color, cv2.LINE_AA, shift=4)
        else:
            cv2.polylines(img, [p], closed, color, thick, cv2.LINE_AA, shift=4)

    def _dashed(self, img, pts, color, thick, dash=3.0, gap=6.0):
        d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        cum = np.concatenate([[0], np.cumsum(d)])
        s = 0.0
        while s < cum[-1]:
            a, b = s, min(s + dash, cum[-1])
            seg = np.stack([np.interp(np.linspace(a, b, 4), cum, pts[:, i]) for i in (0, 1)], 1)
            self._poly(img, seg, color, closed=False, thick=thick)
            s += dash + gap

    def _draw_static(self, img):
        t = max(1, int(round(self.ppm * 0.15)))
        for ln in self.m.lanes.values():
            self._poly(img, np.concatenate([ln.left_border, ln.right_border[::-1]]),
                       JUNCTION if ln.junction != "-1" else ASPHALT)
        for ln in self.m.lanes.values():
            if ln.junction != "-1":
                continue
            self._poly(img, ln.right_border, WHITE, closed=False, thick=t)
            if ln.left is None:
                self._poly(img, ln.left_border, YELLOW, closed=False, thick=t)
            else:
                self._dashed(img, ln.left_border, WHITE, t)
            if ln.successors and self.m.lanes[ln.successors[0]].junction != "-1":
                self._poly(img, np.stack([ln.left_border[-1], ln.right_border[-1]]), WHITE, closed=False,
                           thick=max(2, int(self.ppm * 0.4)))

    def _minimap(self, x0, y0, x1, y1, size=220):
        s = size / max(x1 - x0, y1 - y0, 1)
        w, h = max(40, int((x1 - x0) * s)), max(20, int((y1 - y0) * s))
        img = np.full((h + 8, w + 8, 3), (14, 16, 20), np.uint8)
        for ln in self.m.lanes.values():
            p = np.stack([(ln.xy[:, 0] - x0) * s + 4, (y1 - ln.xy[:, 1]) * s + 4], 1).astype(np.int32)
            cv2.polylines(img, [p], False, (110, 110, 110), 1, cv2.LINE_AA)
        self._mini_tf = (x0, y1, s)
        return img

    def frame(self, log: SimLog, i: int, title: str = "", trail: int = 160) -> np.ndarray:
        S = log.states
        e = log.roles.index("ego")
        if self.follow:
            cx, cy = S[i, e, 0], S[i, e, 1]
        else:
            x0, y0, x1, y1 = self.m.bounds()
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        c = self.px([cx, cy])
        off = (int(c[0] - self.W / 2), int(c[1] - self.H / 2))
        ch, cw = self.base.shape[:2]
        x_a, y_a = max(0, off[0]), max(0, off[1])
        img = np.full((self.H, self.W, 3), BG, np.uint8)
        crop = self.base[y_a:min(ch, off[1] + self.H), x_a:min(cw, off[0] + self.W)]
        img[y_a - off[1]:y_a - off[1] + crop.shape[0], x_a - off[0]:x_a - off[0] + crop.shape[1]] = crop
        r = max(4, int(self.ppm * 0.9))
        for arm, states in log.signals.items():
            for sg in [sg for sg in self.m.signals() if sg.road == str(arm)]:
                p = (self.px([sg.x, sg.y]) - off).astype(int)
                cv2.circle(img, tuple(p), r, SIG[states[i]], -1, cv2.LINE_AA)
                cv2.circle(img, tuple(p), r, (15, 15, 15), 1, cv2.LINE_AA)
        a = max(0, i - trail)
        if i - a > 1:
            self._poly(img, S[a:i + 1, e, :2], EGO, closed=False, thick=max(1, int(self.ppm * 0.25)), off=off)
        for j, aid in enumerate(log.ids):
            x, y, yaw = S[i, j, :3]
            l, w = log.lwh[j, :2]
            col = EGO if j == e else CAT.get(log.categories[j], (200, 200, 200))
            box = obb_corners(x, y, yaw, l, w)
            self._poly(img, box, col, off=off)
            self._poly(img, box, (10, 10, 10), thick=1, off=off)
            nose = obb_corners(x + math.cos(yaw) * l * 0.38, y + math.sin(yaw) * l * 0.38, yaw, l * 0.18, w * 0.8)
            self._poly(img, nose, (240, 240, 240) if j != e else (40, 40, 40), off=off)
            p = (self.px([x, y]) - off).astype(int)
            if self.ppm > 2.5 and 0 <= p[0] < self.W and 0 <= p[1] < self.H:
                cv2.putText(img, aid, (p[0] + 8, p[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (225, 225, 225), 1,
                            cv2.LINE_AA)
        self._overlay_minimap(img, log, i)
        self._hud(img, log, i, title)
        return img

    def _overlay_minimap(self, img, log, i):
        mm = self.mini.copy()
        x0, y1, s = self._mini_tf
        for j in range(len(log.ids)):
            p = (int((log.states[i, j, 0] - x0) * s + 4), int((y1 - log.states[i, j, 1]) * s + 4))
            ego = log.roles[j] == "ego"
            cv2.circle(mm, p, 3 if ego else 2, EGO if ego else (200, 200, 200), -1, cv2.LINE_AA)
        h, w = mm.shape[:2]
        if h < self.H - 60 and w < self.W - 20:
            y, x = 46, self.W - w - 12
            img[y:y + h, x:x + w] = cv2.addWeighted(img[y:y + h, x:x + w], 0.25, mm, 0.75, 0)
            cv2.rectangle(img, (x, y), (x + w, y + h), (70, 70, 70), 1)

    def _hud(self, img, log: SimLog, i: int, title: str):
        e = log.roles.index("ego")
        v = log.states[i, e, 3] * 3.6
        acc = log.states[i, e, 4]
        cv2.rectangle(img, (0, 0), (self.W, 34), (12, 14, 18), -1)
        cv2.putText(img, f"TanitSim  {title}", (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1, cv2.LINE_AA)
        txt = f"t {log.t[i]:5.2f}s   ego {v:5.1f} km/h   a {acc:+4.1f} m/s2"
        cv2.putText(img, txt, (self.W - 430, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, EGO, 1, cv2.LINE_AA)
        fired = [ev for ev in log.events if ev["type"] == "action_start" and ev["t"] <= log.t[i] + 1e-9]
        for k, ev in enumerate(fired[-4:]):
            cv2.putText(img, f"{ev['t']:5.1f}s  {ev['actor']}: {ev['id']}", (12, self.H - 40 - 18 * k),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 190, 210), 1, cv2.LINE_AA)
        mt = log.metrics.get("min_ttc")
        badge = "COLLISION" if log.metrics.get("collision") else f"min TTC {mt if mt is not None else 'inf'} s"
        col = (60, 60, 230) if log.metrics.get("collision") else (120, 220, 120)
        cv2.putText(img, badge, (12, self.H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1, cv2.LINE_AA)


def _auto_ppm(log: SimLog) -> float:
    vmax = float(log.states[:, :, 3].max())
    return float(np.clip(1280 / (vmax * 6 + 60), 3.5, 12.0))


def render_video(sim: TanitSim, log: SimLog, out: str | Path, fps: int = 20, title: str = "",
                 size=(1280, 720), follow: bool = True) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    bev = Bev(sim.m, size, ppm=_auto_ppm(log) if follow else None, follow=follow)
    stride = max(1, int(round(1.0 / (fps * (log.t[1] - log.t[0])))) if len(log.t) > 1 else 1)
    frames = [cv2.cvtColor(bev.frame(log, i, title), cv2.COLOR_BGR2RGB) for i in range(0, len(log.t), stride)]
    iio.imwrite(out, np.stack(frames), fps=fps, codec="libx264", quality=8, macro_block_size=8)
    return out


def render_png(sim: TanitSim, log: SimLog, out: str | Path, i: int | None = None, title: str = "",
               follow: bool = True) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    bev = Bev(sim.m, ppm=_auto_ppm(log) if follow else None, follow=follow)
    cv2.imwrite(str(out), bev.frame(log, len(log.t) // 2 if i is None else i, title))
    return out
