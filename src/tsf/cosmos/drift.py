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
below the reference after that, at most every ``min_gap`` chunks.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def sharpness(bgr: np.ndarray) -> float:
    return float(cv2.Laplacian(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


def saturation(bgr: np.ndarray) -> float:
    return float(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[..., 1].mean())


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
                 step: float = 0.1):
        self.target, self.max_amount, self.reanchor_below, self.min_gap = target, max_amount, reanchor_below, min_gap
        self.step = step
        self.ref: dict[str, dict] = {}
        self.k = 0
        self.last_anchor = 0

    def reference(self, frames: dict[str, str | Path]) -> None:
        """Record the reference frames (the first chunk's last frames) without correcting anything."""
        for cam, p in frames.items():
            img = cv2.imread(str(p))
            self.ref[cam] = {"lab": _lab_stats(img), "sharp": sharpness(img), "sat": saturation(img)}

    def __call__(self, frames: dict[str, str | Path], out_dir: str | Path) -> tuple[dict[str, Path] | None, dict]:
        """Corrected conditioning PNGs for the next request (``None`` = re-anchor: generate it without I2V) and the
        per-camera stats {sharp, sharp_ref, sat, sat_ref, amount}; the first call only records the references."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.k += 1
        out, stats, low = {}, {}, []
        for cam, p in frames.items():
            if cam not in self.ref:
                self.reference({cam: p})
                out[cam], stats[cam] = Path(p), {"sharp": round(self.ref[cam]["sharp"], 1), "reference": True}
                continue
            img = cv2.imread(str(p))
            ref = self.ref[cam]
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
                          "sat_ref": round(ref["sat"], 1), "amount": amount}
            if s < self.reanchor_below * ref["sharp"]:
                low.append(cam)
        if low and self.k - self.last_anchor >= self.min_gap:
            self.last_anchor = self.k
            return None, {**stats, "reanchor": low}
        return out, stats
