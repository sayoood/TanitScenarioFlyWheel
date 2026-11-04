"""Drift guard for chained Sim-Auto generation (T1 closed loop, long sensor clips).

Every chained request is conditioned (I2V) on one frame of the previous output, so blur and colour errors compound:
in the 28 s T1 run on the UC1 left turn the front camera's sharpness (variance of the Laplacian) fell from 90 to 13
by t = 13 s and its mean saturation climbed from 70 to 105. Per camera the guard keeps the first conditioning frame as
the reference and returns every later one

  1. colour-corrected: Lab a/b mean and spread moved back to the reference; L contrast and mean pulled back by a
     limited step per chunk, so a slow real lighting change survives but accumulated haze does not (a re-anchor's
     frames all the way at once, see below),
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

The third failure keeps the veil score up: the road turns into smooth shading. In the 40 s protected-left rerun with
the masked front-only veil the road lost its texture and markings from about 12 s, the clouds became repeated blobs and
a haze lay over every camera from 16 s, while the veil stayed at 0.83-1.1 (a smooth gradient keeps a tile's std) and
the sharpness at 73-84 (poles and signs keep the Laplacian); it re-anchored only at 28 s, in the middle of the turn.
:func:`detail` compares the tiles' fine detail (grey minus its blur): there it fell to 0.58 by 12 s, on the smeared
left-turn run to 0.57 by 11 s, against 0.8-1.0 in the first 8 s of both. A re-anchor moves the structural references
(veil and detail tiles, vehicle cover) to the fresh frame, since the view by then (a junction, a turn) no longer
resembles chunk 0; colour and sharpness keep chunk 0, so the run keeps its look.

Detail missed the haze once the ego reached the junction: in the next rerun crosswalk lines, streaks and the tree line
held it at 1.0-1.19 from 24 to 30 s while the road under them had gone to grey shading again. :func:`markings` asks
the control instead: on the thin white and yellow lines the WSM control draws in the near road band (vehicle boxes
left out), how far does the generated frame's local brightness peak rise above its surroundings? Replayed on three
40 s runs it starts at 41-44 (27 on the smeared run), falls to 17-20 by 13 s, 7 on the haze the veil missed, 1.4 when
opaque, and is back at 34-41 after every good re-anchor; on the protected-left rerun it was at 0.52 of its re-anchored
value by 23.5 s, as the junction haze set in. Far lines are thin and Cosmos places them loosely, so the band starts
at 0.55 of the height and fewer than ``marks_px`` line pixels judge nothing. The reference is the first judged frame
after each re-anchor.

The colour correction itself recoloured vehicles: the a/b chroma of the whole frame went back to chunk 0's, so a red
car filling a third of a cross camera after a re-anchor came back brown one chunk later. With the sim's boxes the
statistics come from the pixels outside them; the transform still applies to the whole frame.

A re-anchor's exposure is the generator's own, since nothing conditions it: in the next 40 s protected-left rerun the
six redraws came out with an L spread (outside the vehicles) of 25-46 against 21-25 along the chain, the one after the
oncoming truck passed the ego at 34.7 s as dark ground under a bright sky on every camera, and the limited steps above
took 4-8 chunks (up to 4 s) to bring each back. A redraw has broken continuity already, so its frames are moved all
the way to the reference in one step, unless vehicles cover more than 30 % of the view: then what is left outside
them need not stand for the scene (the van filling a cross camera left blown-out sky, L 166, and matching that turned
the frame dark), and the limited step applies.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def sharpness(bgr: np.ndarray) -> float:
    return float(cv2.Laplacian(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


def saturation(bgr: np.ndarray) -> float:
    return float(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[..., 1].mean())


def _grid(img: np.ndarray, stat, rows: int = 4, cols: int = 8, top: float = 0.3, bottom: float = 0.78) -> np.ndarray:
    """``stat`` of each tile of a rows x cols grid over the band between the sky and the ego's hood of a 480x832 image."""
    band = img[int(top * 480):int(bottom * 480)]
    th, tw = band.shape[0] // rows, 832 // cols
    return np.array([[stat(band[r * th:(r + 1) * th, c * tw:(c + 1) * tw]) for c in range(cols)] for r in range(rows)])


def veil_tiles(bgr: np.ndarray) -> np.ndarray:
    """Local contrast (std of grey) of the tiles, at 832x480."""
    return _grid(cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (832, 480)).astype(np.float32), np.std)


def detail_tiles(bgr: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """Fine detail (std of grey minus its blur) of the tiles, at 832x480: road texture, markings, edges; not shading."""
    g = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (832, 480), interpolation=cv2.INTER_AREA).astype(np.float32)
    return _grid(g - cv2.GaussianBlur(g, (0, 0), sigma), np.std)


def box_mask(boxes: list, size, shape: tuple[int, int] = (480, 832), pad: int = 0) -> np.ndarray:
    """Pixels of an image of ``shape`` (h, w) inside the boxes ``[u0, v0, u1, v1]`` given in pixels of ``size`` (w, h),
    each grown by ``pad`` pixels."""
    m = np.zeros(shape, bool)
    sx, sy = shape[1] / size[0], shape[0] / size[1]
    for u0, v0, u1, v1 in boxes:
        m[max(int(v0 * sy) - pad, 0):int(np.ceil(v1 * sy)) + pad, max(int(u0 * sx) - pad, 0):int(np.ceil(u1 * sx)) + pad] = True
    return m


def box_cover(boxes: list, size) -> np.ndarray:
    """Share of each tile covered by the boxes ``[u0, v0, u1, v1]`` given in pixels of ``size`` (w, h)."""
    return _grid(box_mask(boxes, size).astype(np.float32), np.mean)


def marking_mask(control: np.ndarray, boxes: list | None = None, size=None, top: float = 0.55,
                 bottom: float = 0.82) -> np.ndarray:
    """The thin white and yellow lines (lane lines, stop lines; not filled areas) of a WSM control frame between
    ``top`` and ``bottom`` of the height, at 832x480, away from the vehicle boxes (whose edges the control draws light)."""
    b, g, r = np.moveaxis(cv2.resize(control, (832, 480), interpolation=cv2.INTER_AREA).astype(int), -1, 0)
    m = (((b > 170) & (g > 170) & (r > 170) & (np.abs(b - r) < 40)) | ((r > 170) & (g > 170) & (b < 120))).astype(np.uint8)
    m &= 1 - cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((11, 11), np.uint8))
    m = m.astype(bool)
    if boxes:
        m &= ~box_mask(boxes, size, pad=6)
    m[:int(top * 480)] = False
    m[int(bottom * 480):] = False
    return m


def markings(bgr: np.ndarray, mask: np.ndarray) -> float:
    """Median height of the generated frame's local brightness peak (7x7 maximum) over its surroundings (blur, sigma 8)
    on the control's marking pixels ``mask`` (:func:`marking_mask`): high where the lines are drawn, low under haze."""
    g = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (832, 480), interpolation=cv2.INTER_AREA).astype(np.float32)
    peak = cv2.dilate(g, np.ones((7, 7), np.uint8)) - cv2.GaussianBlur(g, (0, 0), 8)
    return float(np.median(peak[mask]))


def _ratio(tiles: np.ndarray, ref_tiles: np.ndarray, valid: np.ndarray | None, min_tiles: int) -> float:
    r = tiles / np.maximum(ref_tiles, 1e-3)
    if valid is not None:
        r = r[valid]
        if r.size < min_tiles:
            return 1.0
    return float(np.median(r))


def veil(bgr: np.ndarray, ref_tiles: np.ndarray, valid: np.ndarray | None = None, min_tiles: int = 16) -> float:
    """Median ratio of the tiles' local contrast to the reference's: about 1 for a clean frame, low under a veil.
    ``valid``: the tiles to compare (no vehicle on them); with fewer than ``min_tiles`` left nothing is judged (1.0)."""
    return _ratio(veil_tiles(bgr), ref_tiles, valid, min_tiles)


def detail(bgr: np.ndarray, ref_tiles: np.ndarray, valid: np.ndarray | None = None, min_tiles: int = 16) -> float:
    """As :func:`veil` for :func:`detail_tiles`: low when the road has turned into smooth shading."""
    return _ratio(detail_tiles(bgr), ref_tiles, valid, min_tiles)


def _lab_stats(bgr: np.ndarray, keep: np.ndarray | None = None) -> np.ndarray:
    """Mean and std of L, a, b ([2, 3]) over the pixels ``keep`` (the whole frame when that is under 5 % of it)."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float64)
    lab = lab[keep] if keep is not None and keep.mean() >= 0.05 else lab.reshape(-1, 3)
    return np.stack([lab.mean(0), lab.std(0) + 1e-6])


def match_colour(bgr: np.ndarray, ref: np.ndarray, max_contrast_step: float = 0.15, max_light_step: float = 4.0,
                 keep: np.ndarray | None = None) -> np.ndarray:
    """``bgr`` with the Lab statistics of its pixels ``keep`` (all: None) moved toward ``ref`` (:func:`_lab_stats`)."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float64)
    (mu, sd), (mu_r, sd_r) = _lab_stats(bgr, keep), ref
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
                 step: float = 0.1, veil_below: float = 0.75, veil_cams: tuple[str, ...] | None = None,
                 detail_below: float = 0.0, marks_below: float = 0.0, marks_px: int = 800):
        self.target, self.max_amount, self.reanchor_below, self.min_gap = target, max_amount, reanchor_below, min_gap
        self.step, self.veil_below, self.detail_below = step, veil_below, detail_below      # 0: that check is off
        self.marks_below, self.marks_px = marks_below, marks_px                              # 0: off
        self.veil_cams = veil_cams                  # the cameras whose structure checks may re-anchor; None: every one
        self.ref: dict[str, dict] = {}
        self.k = 0
        self.last_anchor = 0
        self.rebase = False                         # the next frames are a re-anchor's: their structure is the new ref

    def reference(self, frames: dict[str, str | Path], boxes: dict | None = None,
                  controls: dict[str, str | Path] | None = None) -> None:
        """Record the reference frames (the first chunk's last frames) without correcting anything. ``boxes``: the
        sim's vehicles in them, ``{camera: {"size": [w, h], "boxes": [[u0, v0, u1, v1], ...]}}``; ``controls``: the
        WSM control frames at the same instant, for :func:`markings`."""
        for cam, p in frames.items():
            img = cv2.imread(str(p))
            self.ref[cam] = {"lab": _lab_stats(img, ~self._mask(boxes, cam, img.shape[:2])), "sharp": sharpness(img),
                             "sat": saturation(img), "tiles": veil_tiles(img), "detail": detail_tiles(img),
                             "cover": self._cover(boxes, cam), "marks": self._marks(img, boxes, controls, cam)}

    @staticmethod
    def _cover(boxes: dict | None, cam: str) -> np.ndarray:
        b = (boxes or {}).get(cam)
        return box_cover(b["boxes"], b["size"]) if b else np.zeros((4, 8))

    @staticmethod
    def _mask(boxes: dict | None, cam: str, shape: tuple[int, int]) -> np.ndarray:
        b = (boxes or {}).get(cam)
        return box_mask(b["boxes"], b["size"], shape) if b else np.zeros(shape, bool)

    def _marks(self, img: np.ndarray, boxes: dict | None, controls: dict | None, cam: str) -> float | None:
        """:func:`markings` of ``img`` on ``cam``'s control frame; None without one or with too few line pixels."""
        c = cv2.imread(str(controls[cam])) if controls and cam in controls else None
        if c is None:
            return None
        b = (boxes or {}).get(cam) or {}
        m = marking_mask(c, b.get("boxes"), b.get("size"))
        return markings(img, m) if m.sum() >= self.marks_px else None

    def __call__(self, frames: dict[str, str | Path], out_dir: str | Path, boxes: dict | None = None,
                 controls: dict[str, str | Path] | None = None) -> tuple[dict[str, Path] | None, dict]:
        """Corrected conditioning PNGs for the next request (``None`` = re-anchor: generate it without I2V) and the
        per-camera stats {sharp, sharp_ref, sat, sat_ref, amount, veil, detail, marks, veil_tiles}; the first call only
        records the references. ``boxes``, ``controls`` (as in :meth:`reference`): the sim's vehicles in these frames,
        kept out of the veil and the colour statistics, and the control frames whose lines :func:`markings` checks."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.k += 1
        rebase, self.rebase = self.rebase, False
        out, stats, low = {}, {}, []
        for cam, p in frames.items():
            if cam not in self.ref:
                self.reference({cam: p}, boxes, controls)
                out[cam], stats[cam] = Path(p), {"sharp": round(self.ref[cam]["sharp"], 1), "reference": True}
                continue
            img = cv2.imread(str(p))
            ref = self.ref[cam]
            if rebase:
                ref.update(tiles=veil_tiles(img), detail=detail_tiles(img), cover=self._cover(boxes, cam), marks=None)
            free = (self._cover(boxes, cam) < 0.25) & (ref["cover"] < 0.25)
            v = veil(img, ref["tiles"], free)            # on the generated frame: no correction lifts a veil
            d = detail(img, ref["detail"], free)
            mk, mk_ratio = self._marks(img, boxes, controls, cam), 1.0
            if mk is not None and ref["marks"] is None:  # the first judged frame since the last (re-)anchor
                ref["marks"] = mk
            elif mk is not None:
                mk_ratio = mk / max(ref["marks"], 1e-3)
            keep = ~self._mask(boxes, cam, img.shape[:2])
            steps = {"max_contrast_step": np.inf, "max_light_step": np.inf} if rebase and keep.mean() >= 0.7 else {}
            img = match_colour(img, ref["lab"], keep=keep, **steps)
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
                          "detail": round(d, 2), "marks": round(mk_ratio, 2), "veil_tiles": int(free.sum())}
            trusted = self.veil_cams is None or cam in self.veil_cams
            if s < self.reanchor_below * ref["sharp"] or trusted and (v < self.veil_below or d < self.detail_below
                                                                      or mk_ratio < self.marks_below):
                low.append(cam)
        if low and self.k - self.last_anchor >= self.min_gap:
            self.last_anchor, self.rebase = self.k, True
            return None, {**stats, "reanchor": low}
        return out, stats
