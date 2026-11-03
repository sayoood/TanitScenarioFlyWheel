"""Drift guard for chained Sim-Auto generation (T1 closed loop, long sensor clips).

Every chained request is conditioned (I2V) on one frame of the previous output, so blur and colour errors compound:
in the 28 s T1 run on the UC1 left turn the front camera's sharpness (variance of the Laplacian) fell from 90 to 13
by t = 13 s and its mean saturation climbed from 70 to 105. Per camera the guard keeps the first conditioning frame as
the reference and returns every later one

  1. colour-corrected: Lab a/b mean and spread moved back to the reference; L contrast and mean pulled back by a
     limited step per chunk, so a slow real lighting change survives but accumulated haze does not,
  2. re-sharpened: an unsharp mask, raised in small steps, just strong enough to bring the sharpness back toward the
     reference without overshooting it (capped, since part of a sharpness drop is content: an empty road has fewer
     edges than a junction),

and asks for a re-anchor (one request without I2V, generated from the WSM controls alone) when a frame is still far
below the reference after that, or when a veil covers it, at most every ``min_gap`` chunks.

The veil is the chain's other failure: a brown smear or lens blobs that grow from chunk to chunk over part of the image
while the sky and poles stay sharp, so the whole-frame sharpness and saturation above do not move. In the 40 s 720p run
a smear in the lower centre made the front camera nearly opaque from 16 to 28 s (whole-frame sharpness 69-78 against a
reference of 80 throughout). :func:`veil` compares the local contrast of tiles between the sky and the hood with the
reference: the median tile ratio fell to 0.74 by 10 s and to 0.35 when opaque, against 0.83 or more in clean runs.
A vehicle close to the camera lowers the same ratio: its flat side has less local contrast than the road and verge it
covers. In the 40 s protected-left run three of the four re-anchors fired on that (vehicles over 19-80 % of the band,
veil 0.58-0.73; the fourth after the 90° turn, when the view no longer resembles chunk 0), and each redrew every
vehicle and the scenery. Given the sim's projected vehicle boxes, :func:`veil` only compares the tiles no vehicle
covers in either frame, and judges nothing when fewer than half of them are left. Replayed on that run, the masked
score still fell to 0.68-0.72 on the cross-right camera with no smear in view: a side camera's view changes completely
along the drive, so chunk 0 is no reference for it. ``veil_cams`` limits the veil re-anchor to the cameras whose view
stays alike (the front ones); on the smeared left-turn run the masked front-only score still re-anchors at 10.7 s.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def sharpness(bgr: np.ndarray) -> float:
    return float(cv2.Laplacian(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


def saturation(bgr: np.ndarray) -> float:
    return float(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[..., 1].mean())


def veil_tiles(bgr: np.ndarray, rows: int = 4, cols: int = 8, top: float = 0.3, bottom: float = 0.78) -> np.ndarray:
    """Local contrast (std of grey) of a rows x cols grid over the band between the sky and the ego's hood, at 832x480."""
    g = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (832, 480)).astype(np.float32)
    band = g[int(top * 480):int(bottom * 480)]
    th, tw = band.shape[0] // rows, 832 // cols
    return np.array([[band[r * th:(r + 1) * th, c * tw:(c + 1) * tw].std() for c in range(cols)] for r in range(rows)])


def box_cover(boxes: list, size, rows: int = 4, cols: int = 8, top: float = 0.3, bottom: float = 0.78) -> np.ndarray:
    """Share of each :func:`veil_tiles` tile covered by the boxes ``[u0, v0, u1, v1]`` given in pixels of ``size`` (w, h)."""
    m = np.zeros((480, 832), np.float32)
    sx, sy = 832 / size[0], 480 / size[1]
    for u0, v0, u1, v1 in boxes:
        m[int(v0 * sy):int(np.ceil(v1 * sy)), int(u0 * sx):int(np.ceil(u1 * sx))] = 1.0
    band = m[int(top * 480):int(bottom * 480)]
    th, tw = band.shape[0] // rows, 832 // cols
    return np.array([[band[r * th:(r + 1) * th, c * tw:(c + 1) * tw].mean() for c in range(cols)] for r in range(rows)])


def veil(bgr: np.ndarray, ref_tiles: np.ndarray, valid: np.ndarray | None = None, min_tiles: int = 16) -> float:
    """Median ratio of the tiles' local contrast to the reference's: about 1 for a clean frame, low under a veil.
    ``valid``: the tiles to compare (no vehicle on them); with fewer than ``min_tiles`` left nothing is judged (1.0)."""
    r = veil_tiles(bgr) / np.maximum(ref_tiles, 1e-3)
    if valid is not None:
        r = r[valid]
        if r.size < min_tiles:
            return 1.0
    return float(np.median(r))


def _lab_stats(bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float64)
    return np.stack([lab.mean(0), lab.std(0) + 1e-6])          # [2, 3]: mean, std of L, a, b


def match_colour(bgr: np.ndarray, ref: np.ndarray, max_contrast_step: float = 0.15, max_light_step: float = 4.0) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float64)
    (mu, sd), (mu_r, sd_r) = _lab_stats(bgr), ref
    for ch in (1, 2):                                          # chroma: back to the reference
        lab[..., ch] = (lab[..., ch] - mu[ch]) * (sd_r[ch] / sd[ch]) + mu_r[ch]
    # luminance: contrast and mean pulled back by a limited step per chunk, so a slow real lighting change survives but
    # the haze a chain builds up (brighter, flatter, hence less saturated: 53 -> 46 in 12 s with chroma alone) does not
    k = np.clip(sd_r[0] / sd[0], 1 - max_contrast_step, 1 + max_contrast_step)
    shift = np.clip(mu_r[0] - mu[0], -max_light_step, max_light_step)
    lab[..., 0] = (lab[..., 0] - mu[0]) * k + mu[0] + shift
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def unsharp(bgr: np.ndarray, amount: float, sigma: float = 1.2) -> np.ndarray:
    if amount <= 0:
        return bgr
    blur = cv2.GaussianBlur(bgr, (0, 0), sigma)
    return cv2.addWeighted(bgr, 1 + amount, blur, -amount, 0)


class DriftGuard:
    def __init__(self, target: float = 0.85, max_amount: float = 1.0, reanchor_below: float = 0.3, min_gap: int = 8,
                 step: float = 0.1, veil_below: float = 0.75, veil_cams: tuple[str, ...] | None = None):
        self.target, self.max_amount, self.reanchor_below, self.min_gap = target, max_amount, reanchor_below, min_gap
        self.step, self.veil_below, self.veil_cams = step, veil_below, veil_cams    # veil_cams None: every camera
        self.ref: dict[str, dict] = {}
        self.k = 0
        self.last_anchor = 0

    def reference(self, frames: dict[str, str | Path], boxes: dict | None = None) -> None:
        """Record the reference frames (the first chunk's last frames) without correcting anything. ``boxes``: the
        sim's vehicles in them, ``{camera: {"size": [w, h], "boxes": [[u0, v0, u1, v1], ...]}}``."""
        for cam, p in frames.items():
            img = cv2.imread(str(p))
            self.ref[cam] = {"lab": _lab_stats(img), "sharp": sharpness(img), "sat": saturation(img),
                             "tiles": veil_tiles(img), "cover": self._cover(boxes, cam)}

    @staticmethod
    def _cover(boxes: dict | None, cam: str) -> np.ndarray:
        b = (boxes or {}).get(cam)
        return box_cover(b["boxes"], b["size"]) if b else np.zeros((4, 8))

    def __call__(self, frames: dict[str, str | Path], out_dir: str | Path,
                 boxes: dict | None = None) -> tuple[dict[str, Path] | None, dict]:
        """Corrected conditioning PNGs for the next request (``None`` = re-anchor: generate it without I2V) and the
        per-camera stats {sharp, sharp_ref, sat, sat_ref, amount, veil, veil_tiles}; the first call only records the
        references. ``boxes`` (as in :meth:`reference`): the sim's vehicles in these frames, kept out of the veil."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.k += 1
        out, stats, low = {}, {}, []
        for cam, p in frames.items():
            if cam not in self.ref:
                self.reference({cam: p}, boxes)
                out[cam], stats[cam] = Path(p), {"sharp": round(self.ref[cam]["sharp"], 1), "reference": True}
                continue
            img = cv2.imread(str(p))
            ref = self.ref[cam]
            free = (self._cover(boxes, cam) < 0.25) & (ref["cover"] < 0.25)
            v = veil(img, ref["tiles"], free)            # on the generated frame: no correction lifts a veil
            img = match_colour(img, ref["lab"])
            goal, amount, s = self.target * ref["sharp"], 0.0, sharpness(img)
            while s < goal and amount < self.max_amount:
                a = round(amount + self.step, 2)
                sa = sharpness(unsharp(img, a))
                if sa - ref["sharp"] > goal - s:         # this step overshoots the reference by more than it closes the gap
                    break                                # (0.25 steps landed 16 % above it, a halo that compounds per chunk)
                amount, s = a, sa
            img = unsharp(img, amount)
            q = out_dir / f"cond_{cam}.png"
            cv2.imwrite(str(q), img)
            out[cam] = q
            stats[cam] = {"sharp": round(s, 1), "sharp_ref": round(ref["sharp"], 1), "sat": round(saturation(img), 1),
                          "sat_ref": round(ref["sat"], 1), "amount": amount, "veil": round(v, 2),
                          "veil_tiles": int(free.sum())}
            if s < self.reanchor_below * ref["sharp"] or (
                    v < self.veil_below and (self.veil_cams is None or cam in self.veil_cams)):
                low.append(cam)
        if low and self.k - self.last_anchor >= self.min_gap:
            self.last_anchor = self.k
            return None, {**stats, "reanchor": low}
        return out, stats
