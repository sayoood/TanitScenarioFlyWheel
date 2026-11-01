"""TanitSim 2D bird's-eye renderer (OpenCV) -> MP4 / PNG frames."""
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


class Bev:
    def __init__(self, m: OdrMap, size=(1280, 720), ppm: float | None = None, margin=10.0):
        self.m, self.W, self.H = m, size[0], size[1]
        x0, y0, x1, y1 = m.bounds()
        self.cx, self.cy = (x0 + x1) / 2, (y0 + y1) / 2
        self.ppm = ppm or min(self.W / (x1 - x0 + 2 * margin), self.H / (y1 - y0 + 2 * margin))
        self.base = self._static()

    def px(self, xy) -> np.ndarray:
        xy = np.asarray(xy, dtype=float)
        u = (xy[..., 0] - self.cx) * self.ppm + self.W / 2
        v = -(xy[..., 1] - self.cy) * self.ppm + self.H / 2
        return np.stack([u, v], -1)

    def _poly(self, img, pts, color, closed=True, thick=-1):
        p = (self.px(pts) * 16).astype(np.int32)
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

    def _static(self) -> np.ndarray:
        img = np.full((self.H, self.W, 3), BG, np.uint8)
        t = max(1, int(self.ppm * 0.15))
        for ln in self.m.lanes.values():
            poly = np.concatenate([ln.left_border, ln.right_border[::-1]])
            self._poly(img, poly, JUNCTION if ln.junction != "-1" else ASPHALT)
        for ln in self.m.lanes.values():
            if ln.junction != "-1":
                continue
            # outer border solid; inner border dashed if a same-direction neighbour exists, else centre line
            self._poly(img, ln.right_border, WHITE, closed=False, thick=t)
            if ln.left is None:
                self._poly(img, ln.left_border, YELLOW, closed=False, thick=t)
            else:
                self._dashed(img, ln.left_border, WHITE, t)
            if ln.successors and self.m.lanes[ln.successors[0]].junction != "-1":
                self._poly(img, np.stack([ln.left_border[-1], ln.right_border[-1]]), WHITE, closed=False,
                           thick=max(2, int(self.ppm * 0.4)))
        return img

    def frame(self, log: SimLog, i: int, title: str = "", trail: int = 120) -> np.ndarray:
        img = self.base.copy()
        S = log.states
        e = log.roles.index("ego")
        # signal heads per arm at the stop line
        for arm, states in log.signals.items():
            sgs = [sg for sg in self.m.signals() if sg.road == str(arm)]
            for sg in sgs:
                c = self.px([sg.x, sg.y]).astype(int)
                cv2.circle(img, tuple(c), max(4, int(self.ppm * 0.9)), SIG[states[i]], -1, cv2.LINE_AA)
                cv2.circle(img, tuple(c), max(4, int(self.ppm * 0.9)), (15, 15, 15), 1, cv2.LINE_AA)
        # ego trail
        a = max(0, i - trail)
        if i - a > 1:
            self._poly(img, S[a:i + 1, e, :2], EGO, closed=False, thick=max(1, int(self.ppm * 0.25)))
        for j, aid in enumerate(log.ids):
            x, y, yaw = S[i, j, :3]
            l, w = log.lwh[j, :2]
            col = EGO if j == e else CAT.get(log.categories[j], (200, 200, 200))
            box = obb_corners(x, y, yaw, l, w)
            self._poly(img, box, col)
            self._poly(img, box, (10, 10, 10), thick=1)
            nose = obb_corners(x + math.cos(yaw) * l * 0.38, y + math.sin(yaw) * l * 0.38, yaw, l * 0.18, w * 0.8)
            self._poly(img, nose, (240, 240, 240) if j != e else (40, 40, 40))
            c = self.px([x, y]).astype(int)
            if self.ppm > 2.5:
                cv2.putText(img, aid, (c[0] + 8, c[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (225, 225, 225), 1,
                            cv2.LINE_AA)
        self._hud(img, log, i, title)
        return img

    def _hud(self, img, log: SimLog, i: int, title: str):
        e = log.roles.index("ego")
        v = log.states[i, e, 3] * 3.6
        acc = log.states[i, e, 4]
        cv2.rectangle(img, (0, 0), (self.W, 34), (12, 14, 18), -1)
        cv2.putText(img, f"TanitSim  {title}", (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1, cv2.LINE_AA)
        txt = f"t {log.t[i]:5.2f}s   ego {v:5.1f} km/h   a {acc:+4.1f} m/s2"
        cv2.putText(img, txt, (self.W - 430, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, EGO, 1, cv2.LINE_AA)
        mt = log.metrics.get("min_ttc")
        badge = "COLLISION" if log.metrics.get("collision") else f"min TTC {mt if mt is not None else 'inf'} s"
        col = (60, 60, 230) if log.metrics.get("collision") else (120, 220, 120)
        cv2.putText(img, badge, (12, self.H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1, cv2.LINE_AA)


def render_video(sim: TanitSim, log: SimLog, out: str | Path, fps: int = 20, title: str = "",
                 size=(1280, 720)) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    bev = Bev(sim.m, size)
    stride = max(1, int(round(1.0 / (fps * log.t[1] if len(log.t) > 1 else 1))))
    frames = [cv2.cvtColor(bev.frame(log, i, title), cv2.COLOR_BGR2RGB) for i in range(0, len(log.t), stride)]
    iio.imwrite(out, np.stack(frames), fps=fps, codec="libx264", quality=8, macro_block_size=8)
    return out


def render_png(sim: TanitSim, log: SimLog, out: str | Path, i: int | None = None, title: str = "") -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    bev = Bev(sim.m)
    cv2.imwrite(str(out), bev.frame(log, len(log.t) // 2 if i is None else i, title))
    return out
