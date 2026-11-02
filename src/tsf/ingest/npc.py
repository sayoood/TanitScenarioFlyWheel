"""Dense metric depth + ViPE instance masks + camera poses -> tracks of other road users (UC2), numpy/scipy only.

Per frame and instance (GroundingDINO + SAM + AOT; ids are stable within one ViPE window) the ground-contact
point is the bottom band of the mask. Its camera-frame depth is the fusion of (a) the median metric depth in
that band (MoGe-2, a visible surface of the vehicle) and (b) the intersection of the pixel ray with the road
plane fitted to the same frame's depth (accurate near the camera, hurt by occlusion and slopes). The frame-wise
road plane also stabilises the depth scale: MoGe-2 depth flickers by a few percent from frame to frame, the
camera height must not (median over the clip, then median-filtered over time).

Conventions (same as ``ego_track``): ViPE poses are camera-to-world with OpenCV axes; the planar frame has x
forward at t0 and y left. A measurement is first expressed relative to the camera in the gravity-aligned body
frame (forward, left) and then attached to the ``ego`` track that is passed in (e.g. the despiked one), so NPC
speeds are consistent with the ego speed. ``lat`` / ``gap`` / ``lane`` are Frenet coordinates along the ego's
own driven path (extended straight beyond both ends): ``lat`` > 0 is left, lane 0 is the ego lane, +1 the lane
to its left (``lane_w`` = 3.5 m). Camera position stands for the ego position.

Licences (Hugging Face model cards, checked): MoGe-2 ViT-L MIT, GroundingDINO and SAM weights Apache-2.0, ViPE
Apache-2.0. ViPE's instance masking vendors Segment-and-Track-Anything / AOT code under AGPL-3.0, so keep it internal
(thor). Not used, because CC BY-NC 4.0 (research only): UniDepth, Video-Depth-Anything-Large; the permissive
alternative for temporally consistent depth is Video-Depth-Anything-Small (Apache-2.0).
"""
from __future__ import annotations

import json
import math
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

from tsf.ingest.vipe import _up_axis

LANE_W = 3.5
_PHRASE = (("motorcycle", "motorcycle"), ("motorbike", "motorcycle"), ("bicycle", "bicycle"), ("bike", "bicycle"),
           ("truck", "truck"), ("bus", "bus"), ("person", "person"), ("pedestrian", "person"), ("car", "car"),
           ("vehicle", "car"))


def category(phrase: str) -> str | None:
    """Detector phrase -> car / truck / bus / motorcycle / bicycle / person (None for sky, animals, ...)."""
    p = phrase.lower()
    return next((c for k, c in _PHRASE if k in p), None)


@dataclass(frozen=True)
class NpcConfig:
    lane_w: float = LANE_W
    min_area_frac: float = 2.5e-4                  # smallest mask used, as a fraction of the image
    band_frac: float = 0.12                        # bottom band of the mask used for the contact estimate
    depth_range: tuple[float, float] = (2.0, 90.0)
    ground_range: tuple[float, float] = (4.0, 40.0)    # depth range of the pixels the road plane is fitted to
    max_gap_s: float = 1.0                         # a track breaks where it was not seen for longer
    min_dur_s: float = 1.0
    min_obs: int = 6
    smooth_s: float = 0.6                          # sigma of the local-linear trajectory fit
    max_speed: float = 45.0                        # m/s, faster is a tracking error
    link_dist: float = 3.0                         # m, window-overlap association gate
    moving_speed: float = 1.0                      # m/s
    lane_slack: float = 0.3                        # m beyond the lane edge before a track counts as changing lane ...
    lane_slope: float = 0.012                      # ... plus this much per metre of range (lateral noise grows with it)
    lane_dwell_s: float = 1.5                      # an excursion into another lane shorter than this is noise


# ----------------------------------------------------------------------------------------------- windows
def _window(path: Path, poses: np.ndarray, t: np.ndarray) -> dict:
    z = np.load(path)
    return {"depth": z["depth"], "instance": z["instance"], "K": np.asarray(z["K"], float),
            "phrases": {int(k): str(v) for k, v in json.loads(str(z["phrases"])).items()}, "poses": poses, "t": t}


def load_windows(src) -> Iterator[dict]:
    """Dense windows ``{depth (n,H,W), instance (n,H,W) uint8, K [fx fy cx cy], phrases {id: str}, poses (n,4,4),
    t (n,)}`` from a ``run_vipe(dense=True)`` result, its output directory, or window dict(s) (returned as is)."""
    if isinstance(src, dict) and "depth" in src:
        yield src
    elif isinstance(src, dict):
        for w in src["windows"]:
            if w.get("dense") is None:
                raise ValueError("run_vipe(dense=True) is needed for npc_tracks")
            yield _window(Path(w["dense"]), w["poses"], w["t"])
    elif isinstance(src, (str, Path)):
        for d in sorted(Path(src).glob("w*/dense.npz")):
            c = np.load(d.parent / "chained.npz")
            yield _window(d, c["poses"], c["t"])
    else:
        yield from src


def window_times(src) -> list[np.ndarray]:
    """Frame times of every window of a run (to split overlaps between windows)."""
    if isinstance(src, dict) and "depth" in src:
        return [np.asarray(src["t"])]
    if isinstance(src, dict):
        return [np.asarray(w["t"]) for w in src["windows"]]
    if isinstance(src, (str, Path)):
        return [np.load(d.parent / "chained.npz")["t"] for d in sorted(Path(src).glob("w*/dense.npz"))]
    return [np.asarray(w["t"]) for w in src]


def _run_arrays(src, ego_t: np.ndarray | None):
    """(merged poses, merged t) of a run, for the gravity direction and the default ego time axis."""
    if isinstance(src, dict) and "poses" in src:
        return np.asarray(src["poses"]), np.asarray(src["t"])
    if isinstance(src, (str, Path)):
        z = np.load(Path(src) / "track.npz")
        return z["poses"], z["t"]
    wins = [src] if isinstance(src, dict) else list(src)
    tt = np.concatenate([w["t"] for w in wins])
    pp = np.concatenate([w["poses"] for w in wins])
    _, keep = np.unique(np.round(tt, 4), return_index=True)
    return pp[keep], tt[keep]


# ----------------------------------------------------------------------------------------------- road plane
def ground_plane(pts: np.ndarray, down: np.ndarray, rng: np.random.Generator, iters: int = 160,
                 min_inliers: int = 150, cos_min: float = 0.93) -> tuple[np.ndarray, float, float] | None:
    """RANSAC road plane ``n.x = h`` through camera-frame points (N,3): n unit, pointing down within
    acos(cos_min) of ``down``. Returns (n, h, inlier ratio) or None."""
    if len(pts) < 300:
        return None
    idx = rng.integers(0, len(pts), (iters, 3))
    p0 = pts[idx[:, 0]]
    n = np.cross(pts[idx[:, 1]] - p0, pts[idx[:, 2]] - p0)
    nn = np.linalg.norm(n, axis=1)
    keep = nn > 1e-6
    n, p0 = n[keep] / nn[keep, None], p0[keep]
    n = np.where((n @ down)[:, None] < 0, -n, n)
    keep = n @ down > cos_min
    if not keep.any():
        return None
    n, p0 = n[keep], p0[keep]
    h = np.einsum("ij,ij->i", n, p0)
    inl = np.abs(pts @ n.T - h[None]) < (0.05 + 0.006 * pts[:, 2])[:, None]
    cnt = inl.sum(0)
    k = int(cnt.argmax())
    if cnt[k] < min_inliers:
        return None
    q = pts[inl[:, k]]
    c0 = q.mean(0)
    nf = np.linalg.svd(q - c0, full_matrices=False)[2][-1]
    nf = nf if nf @ down > 0 else -nf
    if nf @ down < cos_min:
        nf = n[k]
    return nf, float(nf @ c0), float(cnt[k] / len(pts))


def frame_planes(win: dict, up: np.ndarray, cfg: NpcConfig, seed: int = 0, stride: int = 5) -> list:
    """Per frame road plane (n, h, ratio) or None, from depth pixels outside every instance mask."""
    rng = np.random.default_rng(seed)
    fx, fy, cx, cy = win["K"]
    depth, inst = win["depth"], win["instance"]
    H, W = depth.shape[1:]
    v, u = np.mgrid[0:H:stride, 0:W:stride]
    out = []
    for i in range(len(depth)):
        z = depth[i][v, u].astype(np.float64)
        ok = (z > cfg.ground_range[0]) & (z < cfg.ground_range[1])
        ok &= ~ndimage.binary_dilation(inst[i] > 0, iterations=3)[v, u]
        pts = np.stack([(u[ok] - cx) / fx * z[ok], (v[ok] - cy) / fy * z[ok], z[ok]], 1)
        down = win["poses"][i][:3, :3].T @ -up
        pts = pts[pts @ down > 0.3]
        if len(pts) > 6000:
            pts = pts[rng.choice(len(pts), 6000, replace=False)]
        out.append(ground_plane(pts, down, rng))
    return out


def stabilize(planes: list, tol: float = 0.25) -> tuple[np.ndarray, float]:
    """Depth scale per frame that makes the fitted camera height constant: (scale (n,), reference height).

    The reference is the median over the clip; frames without a plane or whose height is off by more than
    ``tol`` from the local median (a plane fitted to something else) take the scale of their neighbours."""
    h = np.array([p[1] if p else np.nan for p in planes])
    good = np.isfinite(h)
    if good.sum() < max(5, 0.15 * len(h)):
        return np.ones(len(h)), float("nan")
    h_ref = float(np.median(h[good]))
    fill = np.interp(np.arange(len(h)), np.flatnonzero(good), h[good])
    good &= np.abs(h / ndimage.median_filter(fill, size=9, mode="nearest") - 1) < tol
    s = np.interp(np.arange(len(h)), np.flatnonzero(good), (h_ref / h)[good])
    return np.clip(s, 0.8, 1.25), h_ref


# ----------------------------------------------------------------------------------------------- detections
def _split_columns(z: np.ndarray, ok: np.ndarray, min_w: int = 6) -> list[tuple[int, int]]:
    """Split a mask at depth steps of its per-column contact depth: one contact segment per vehicle (a SAM/AOT
    instance often covers several queued vehicles). z (w,) with NaN where the column has no depth."""
    cols = np.flatnonzero(ok)
    if len(cols) == 0:
        return []
    zz = ndimage.median_filter(np.interp(np.arange(len(z)), cols, z[cols]), size=5, mode="nearest")
    cut = [c for c in range(1, len(z)) if ok[c] and ok[c - 1] and abs(zz[c] - zz[c - 1]) > max(0.07 * zz[c], 0.7)]
    edges = [0, *cut, len(z)]
    out = []
    for a, b in pairwise(edges):
        c = np.flatnonzero(ok[a:b]) + a
        if len(c) >= min_w:
            out.append((int(c[0]), int(c[-1]) + 1))
    return out


def detections(win: dict, planes: list, scale: np.ndarray, h_ref: float, cfg: NpcConfig,
               global_scale: float = 1.0) -> list[dict]:
    """Ground-contact detections of one dense window: per frame, per instance mask and per depth-separated
    segment of it. Each is a dict with ``frame``, ``label``, ``cat``, ``X`` (camera frame, m), ``z`` (depth), ``area``,
    ``q`` (quality 0..1), ``w3`` / ``h3`` (extent in m at that depth), ``airborne`` (bottom edge above the road)."""
    fx, fy, cx, cy = win["K"]
    depth, inst = win["depth"], win["instance"]
    H, W = depth.shape[1:]
    cats = {k: category(v) for k, v in win["phrases"].items()}
    min_area = cfg.min_area_frac * H * W
    out: list[dict] = []
    for i in range(len(depth)):
        s = scale[i] * global_scale
        plane = planes[i]
        for lab0, sl in enumerate(ndimage.find_objects(inst[i])):
            lab = lab0 + 1
            if sl is None or cats.get(lab) is None:
                continue
            m = inst[i][sl] == lab
            if m.sum() < min_area:
                continue
            d = depth[i][sl].astype(np.float64)
            rank = np.cumsum(m[::-1], axis=0)[::-1]                 # 1 = lowest mask pixel of its column
            low = m & (rank <= 5) & (d > 0)
            n_low = low.sum(0)
            zc = np.where(n_low > 0, (d * low).sum(0) / np.maximum(n_low, 1), np.nan)
            vb = m.shape[0] - 1 - np.argmax(m[::-1], axis=0)
            vt = np.argmax(m, axis=0)
            for c0, c1 in _split_columns(zc, n_low > 0):
                seg = np.arange(c0, c1)
                seg = seg[m[:, seg].any(0)]
                area = int(m[:, c0:c1].sum())
                if len(seg) < 6 or area < min_area / 2:
                    continue
                z_d = float(np.nanmedian(zc[seg])) * s
                u_c = sl[1].start + float(np.median(seg)) + 0.5
                v_b = sl[0].start + float(np.percentile(vb[seg], 85)) + 1.0
                ray = np.array([(u_c - cx) / fx, (v_b - cy) / fy, 1.0])
                trunc_b = sl[0].start + vb[seg].max() >= H - 2
                z_g = np.nan
                if plane is not None and np.isfinite(h_ref):
                    den = float(plane[0] @ ray)
                    z_g = h_ref * global_scale / den if den > 0.04 else np.nan
                z_g_ok = np.isfinite(z_g) and 0 < z_g < 150 and not trunc_b
                airborne = False
                if z_g_ok:
                    sd, sg = 0.06 * z_d + 0.3, 2.0 * z_g ** 2 / (fy * h_ref * global_scale)
                    if abs(z_d - z_g) < 3 * math.hypot(sd, sg):
                        z = (z_d / sd ** 2 + z_g / sg ** 2) / (1 / sd ** 2 + 1 / sg ** 2)
                    else:
                        z, airborne = z_d, z_g > 1.35 * z_d
                else:
                    z = z_d
                if not cfg.depth_range[0] < z < cfg.depth_range[1]:
                    continue
                w_px = len(seg)
                h_px = float(vb[seg].max() - vt[seg].min() + 1)
                q = min(1.0, area / (0.004 * H * W)) * (0.5 if trunc_b else 1.0) * (0.6 if airborne else 1.0)
                w3, h3 = w_px * z / fx, h_px * z / fy
                cat = cats[lab]
                if cat in ("motorcycle", "bicycle") and w3 > 1.4 and h3 < 2.4:
                    cat = "car"                        # the detector's 'motorcycle' fires on distant cars
                out.append({"frame": i, "label": lab, "cat": cat, "X": z * ray, "z": z, "area": area, "q": q,
                            "w3": w3, "h3": h3, "airborne": airborne})
    return out


def body_frame(R: np.ndarray, up: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Gravity-aligned forward / left unit vectors (world frame) of camera-to-world rotations R (...,3,3)."""
    f = R[..., :, 2] - (R[..., :, 2] @ up)[..., None] * up
    f /= np.linalg.norm(f, axis=-1, keepdims=True)
    return f, np.cross(up, f)


def _owned(t: np.ndarray, ts: list[np.ndarray], k: int) -> np.ndarray:
    """Frames of window k that it owns: overlaps are split at their midpoint."""
    lo = -np.inf if k == 0 else 0.5 * (ts[k][0] + ts[k - 1][-1])
    hi = np.inf if k + 1 == len(ts) else 0.5 * (ts[k + 1][0] + ts[k][-1])
    return (t >= lo) & (t < hi)


def _relative(win_idx: int, dets: list[dict], win: dict, up: np.ndarray, ego_t: np.ndarray, half_dt: float,
              keep: np.ndarray) -> list[dict]:
    """Attach the ego frame index and the gravity-aligned (forward, left) offset from the camera to detections."""
    R = win["poses"][:, :3, :3]
    f, left = body_frame(R, up)
    out = []
    for d in dets:
        i = d["frame"]
        if not keep[i]:
            continue
        tt = float(win["t"][i])
        ie = int(np.argmin(np.abs(ego_t - tt)))
        if abs(ego_t[ie] - tt) > half_dt:
            continue
        rel = R[i] @ d["X"]
        out.append({**d, "win": win_idx, "t": tt, "ie": ie, "a": float(rel @ f[i]), "b": float(rel @ left[i])})
    return out


# ----------------------------------------------------------------------------------------------- tracking
def _anchor(ego: np.ndarray, ie: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    c, s = np.cos(ego[ie, 2]), np.sin(ego[ie, 2])
    return np.stack([ego[ie, 0] + a * c - b * s, ego[ie, 1] + a * s + b * c], 1)


FAMILY_OF = {"car": "veh", "truck": "veh", "bus": "veh", "motorcycle": "two", "bicycle": "two", "person": "ped"}


def associate(dets: list[dict], ego: np.ndarray, cfg: NpcConfig) -> list[list[dict]]:
    """Frame-to-frame tracking-by-detection in the planar frame: constant-velocity prediction, Hungarian
    assignment with a range-dependent gate, same AOT id preferred. Returns the detections of every track."""
    by_frame: dict[int, list[dict]] = {}
    for d in dets:
        by_frame.setdefault(d["ie"], []).append(d)
    active: list[list[dict]] = []
    done: list[list[dict]] = []
    pos = {id(d): None for d in dets}
    for ie in sorted(by_frame):
        cur = by_frame[ie]
        P = _anchor(ego, np.full(len(cur), ie), np.array([d["a"] for d in cur]), np.array([d["b"] for d in cur]))
        for d, p in zip(cur, P):
            pos[id(d)] = p
        # one object seen twice (overlapping masks): keep the larger detection
        order = sorted(range(len(cur)), key=lambda j: -cur[j]["area"])
        keep: list[int] = []
        for j in order:
            if all(FAMILY_OF[cur[j]["cat"]] != FAMILY_OF[cur[k]["cat"]] or np.linalg.norm(P[j] - P[k]) > 1.0
                   for k in keep):
                keep.append(j)
        cur, P = [cur[j] for j in keep], P[keep]
        t_now = cur[0]["t"] if cur else 0.0
        for tr in list(active):
            if t_now - tr[-1]["t"] > cfg.max_gap_s:
                active.remove(tr)
                done.append(tr)
        pred = []
        for tr in active:
            last = pos[id(tr[-1])]
            pts = tr[-8:]
            if len(pts) >= 3 and pts[-1]["t"] - pts[0]["t"] > 0.4:
                tt = np.array([x["t"] for x in pts])
                v = np.polyfit(tt - tt[-1], np.array([pos[id(x)] for x in pts]), 1)[0]
                v = v * min(1.0, cfg.max_speed / max(float(np.linalg.norm(v)), 1e-6))
            else:
                v = np.zeros(2)
            pred.append(last + v * (t_now - tr[-1]["t"]))
        cost = np.full((len(active), len(cur)), 1e3)
        for i, tr in enumerate(active):
            for j, d in enumerate(cur):
                if FAMILY_OF[tr[-1]["cat"]] != FAMILY_OF[d["cat"]]:
                    continue
                gate = 1.5 + 0.07 * d["z"] + 1.5 * (t_now - tr[-1]["t"])
                dist = float(np.linalg.norm(P[j] - pred[i]))
                if dist < gate:
                    same = (tr[-1]["win"], tr[-1]["label"]) == (d["win"], d["label"])
                    cost[i, j] = dist / gate * (0.7 if same else 1.0)
        used = set()
        if len(active) and len(cur):
            for i, j in zip(*linear_sum_assignment(cost)):
                if cost[i, j] < 1.0:
                    active[i].append(cur[j])
                    used.add(j)
        for j, d in enumerate(cur):
            if j not in used:
                active.append([d])
    return done + active


def _merged(tr: list[dict]) -> dict:
    keys = ("t", "ie", "a", "b", "z", "q", "area", "w3", "h3", "airborne")
    m = {k: np.array([d[k] for d in tr]) for k in keys}
    cats = [d["cat"] for d in tr]
    w = np.array([d["area"] for d in tr], float)
    best = max(set(cats), key=lambda c: w[[x == c for x in cats]].sum())
    if best in ("truck", "bus") and float(np.median(m["h3"])) < 2.2:
        best = "car"                                           # a queued car inside a 'truck' mask
    m["category"] = best
    m["phrase"] = best
    return m


# ----------------------------------------------------------------------------------------------- trajectories
def _fit(t: np.ndarray, p: np.ndarray, w: np.ndarray, te: np.ndarray, sigma: float):
    """Weighted local-linear fit of p (n,2) at times te: positions (m,2) and velocities (m,2)."""
    dt = t[None, :] - te[:, None]
    W = np.exp(-0.5 * (dt / sigma) ** 2) * w[None, :]
    s0, s1, s2 = W.sum(1), (W * dt).sum(1), (W * dt ** 2).sum(1) + 1e-6
    det = s0 * s2 - s1 ** 2 + 1e-9
    wy, wty = W @ p, (W * dt) @ p
    pos = (s2[:, None] * wy - s1[:, None] * wty) / det[:, None]
    vel = (s0[:, None] * wty - s1[:, None] * wy) / det[:, None]
    return pos, vel


def lane_index(lat: np.ndarray, rng: np.ndarray, t: np.ndarray, cfg: NpcConfig) -> np.ndarray:
    """Lane number (0 = ego lane, + left) with hysteresis: the lane changes only once the lateral position is beyond
    the lane edge by ``lane_slack + lane_slope * range`` and the new lane is held for ``lane_dwell_s``."""
    w = cfg.lane_w
    cur, out = int(np.rint(lat[0] / w)), np.zeros(len(lat), int)
    for i, (d, r) in enumerate(zip(lat, rng)):
        if abs(d - cur * w) > w / 2 + cfg.lane_slack + cfg.lane_slope * r:
            cur = int(np.rint(d / w))
        out[i] = cur
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 1.0
    for _ in range(3):
        cuts = np.flatnonzero(np.diff(out) != 0) + 1
        runs = list(zip(np.r_[0, cuts], np.r_[cuts, len(out)]))
        short = [k for k in range(1, len(runs) - 1) if (runs[k][1] - runs[k][0]) * dt < cfg.lane_dwell_s]
        if not short:
            break
        k = short[0]
        out[runs[k][0]:runs[k][1]] = out[runs[k][0] - 1]
    return out


def _ego_path(ego: np.ndarray, ext: float = 200.0, step: float = 2.0):
    """Ego path vertices (extended straight at both ends): xy, heading, arc length, index of the first ego vertex."""
    xy, yaw = ego[:, :2], ego[:, 2]
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(xy, axis=0).T))])
    k = np.arange(ext, 0, -step)[:, None]
    back = xy[0] - k * [math.cos(yaw[0]), math.sin(yaw[0])]
    fwd = xy[-1] + k[::-1] * [math.cos(yaw[-1]), math.sin(yaw[-1])]
    return (np.vstack([back, xy, fwd]), np.concatenate([np.full(len(k), yaw[0]), yaw, np.full(len(k), yaw[-1])]),
            np.concatenate([s[0] - k[:, 0], s, s[-1] + k[::-1, 0]]), len(k))


def frenet(path, pts: np.ndarray) -> np.ndarray:
    """(s, d) of pts (m,2) along the path from ``_ego_path`` (d > 0 left)."""
    xy, hdg, s, _ = path
    i = ((pts[:, None, :] - xy[None]) ** 2).sum(-1).argmin(1)
    dv = pts - xy[i]
    return np.stack([s[i] + dv[:, 0] * np.cos(hdg[i]) + dv[:, 1] * np.sin(hdg[i]),
                     -dv[:, 0] * np.sin(hdg[i]) + dv[:, 1] * np.cos(hdg[i])], 1)


def _stitch(ms: list[dict], ego: np.ndarray, half_dt: float) -> list[dict]:
    """Join measurement sets of one object that SAM re-detected under a new id after AOT lost it (same
    category, not simultaneous, short gap, same place)."""
    ms = sorted(ms, key=lambda m: m["t"][0])
    i = 0
    while i < len(ms):
        a = ms[i]
        pa = _anchor(ego, a["ie"][-5:], a["a"][-5:], a["b"][-5:]).mean(0)
        for j in range(i + 1, len(ms)):
            b = ms[j]
            dt = b["t"][0] - a["t"][-1]
            if b["category"] != a["category"] or not -half_dt <= dt <= 1.2:
                continue
            pb = _anchor(ego, b["ie"][:5], b["a"][:5], b["b"][:5]).mean(0)
            if np.linalg.norm(pa - pb) < 2.0 + 0.06 * float(a["z"][-5:].mean()) + 2.0 * max(dt, 0.0):
                keys = ("t", "ie", "a", "b", "z", "q", "area", "w3", "h3", "airborne")
                ms[i] = {**a, **{k: np.concatenate([a[k], b[k]]) for k in keys}}
                del ms[j]
                break
        else:
            i += 1
    return ms


def _segments(t: np.ndarray, max_gap: float) -> list[np.ndarray]:
    cut = np.flatnonzero(np.diff(t) > max_gap) + 1
    return np.split(np.arange(len(t)), cut)


def _events(tr: dict, cfg: NpcConfig) -> list[dict]:
    t, lane, gap, v = tr["t"], tr["lane"], tr["gap"], tr["speed"]
    out = []
    k = max(3, round(0.5 / max(float(np.median(np.diff(t))), 1e-3))) | 1 if len(t) > 3 else 1
    lm = ndimage.median_filter(lane, size=k, mode="nearest")
    for i in np.flatnonzero(np.diff(lm) != 0):
        a, b = int(lm[i]), int(lm[i + 1])
        if gap[i + 1] < -3:
            continue
        kind = "cut_in" if b == 0 and a != 0 else "leaves_ego_lane" if a == 0 else "lane_change"
        j = i + 1
        if kind == "cut_in":
            out.append({"type": kind, "t": round(float(t[j]), 2), "from_lane": a, "gap_m": round(float(gap[j]), 1)})
        else:
            out.append({"type": kind, "t": round(float(t[j]), 2), "from_lane": a, "to_lane": b})
    vs = ndimage.uniform_filter1d(v, size=k, mode="nearest")
    if len(vs) > 3 and vs.max() > 2.0:
        i_hi = int(np.argmax(vs))
        if vs[i_hi:].min() < vs[i_hi] - 3.0 and (vs[i_hi] - vs[i_hi:].min()) > 0.4 * vs[i_hi]:
            i_lo = i_hi + int(np.argmin(vs[i_hi:]))
            out.append({"type": "stops" if vs[i_lo] < 0.8 else "brakes", "t": round(float(t[i_lo]), 2),
                        "from_kmh": round(float(vs[i_hi]) * 3.6), "to_kmh": round(float(vs[i_lo]) * 3.6)})
    return out


def _relation(tr: dict, ego: np.ndarray, ego_t: np.ndarray, cfg: NpcConfig) -> str:
    i = 0
    gap, lat, lane = tr["gap"][i], tr["lat"][i], tr["lane"][i]
    moving = tr["moving"]
    if moving:
        ie = int(np.argmin(np.abs(ego_t - tr["t"][i])))
        dpsi = (tr["heading"][i] - ego[ie, 2] + math.pi) % (2 * math.pi) - math.pi
        if abs(dpsi) > math.radians(135):
            return "oncoming"
        if math.radians(45) < abs(dpsi) <= math.radians(135):
            return "crossing_from_left" if dpsi < 0 else "crossing_from_right"
    if gap < -2.0:
        return "behind"
    if abs(gap) <= 2.0:
        return "adjacent_left" if lat > 0 else "adjacent_right"
    if not moving and lane != 0:
        return "parked"
    return "ahead_same_lane" if lane == 0 else "ahead_left_lane" if lane > 0 else "ahead_right_lane"


def _build(m: dict, ego: np.ndarray, ego_t: np.ndarray, path, cfg: NpcConfig, half_dt: float,
           drops: dict | None = None) -> list[dict]:
    out = []
    drops = {} if drops is None else drops

    def drop(why: str) -> None:
        drops[why] = drops.get(why, 0) + 1

    for seg in _segments(m["t"], cfg.max_gap_s):
        if len(seg) < cfg.min_obs or m["t"][seg[-1]] - m["t"][seg[0]] < cfg.min_dur_s:
            drop("short")
            continue
        s = {k: v[seg] for k, v in m.items() if isinstance(v, np.ndarray)}
        P = _anchor(ego, s["ie"], s["a"], s["b"])
        med = np.column_stack([ndimage.median_filter(P[:, 0], size=9, mode="nearest"),
                               ndimage.median_filter(P[:, 1], size=9, mode="nearest")])
        good = np.linalg.norm(P - med, axis=1) < 0.8 + 0.08 * s["z"]
        if good.mean() < 0.65 or good.sum() < cfg.min_obs:
            drop("unstable")
            continue
        t0, t1 = s["t"][good][0], s["t"][good][-1]
        sel_e = (ego_t >= t0 - half_dt) & (ego_t <= t1 + half_dt)
        te = ego_t[sel_e]
        pos, vel = _fit(s["t"][good], P[good], s["q"][good] + 0.05, te, cfg.smooth_s)
        fit_obs, _ = _fit(s["t"][good], P[good], s["q"][good] + 0.05, s["t"][good], cfg.smooth_s)
        rms = float(np.sqrt(np.mean(np.sum((fit_obs - P[good]) ** 2, axis=1))))
        speed = np.hypot(vel[:, 0], vel[:, 1])
        ie = np.flatnonzero(sel_e)
        if rms > 0.8 + 0.05 * float(np.median(s["z"])) or speed.max() > cfg.max_speed:
            drop("unstable")
            continue
        cat = m["category"]
        h3, w3 = float(np.median(s["h3"][good])), float(np.median(s["w3"][good]))
        big = cat in ("truck", "bus")
        if (h3 > (5.5 if big else 4.8) or w3 > 14 or s["airborne"][good].mean() > 0.5
                or (cat == "person" and not 0.4 < h3 < 2.6) or float(np.median(s["z"][good])) < 2.0):
            drop("background_or_implausible_size")
            continue
        sf = frenet(path, pos)
        sf_ego = path[2][ie + path[3]]
        gap, lat = sf[:, 0] - sf_ego, sf[:, 1]
        lane = lane_index(lat, np.hypot(pos[:, 0] - ego[ie, 0], pos[:, 1] - ego[ie, 1]), te, cfg)
        head = np.arctan2(vel[:, 1], vel[:, 0])
        slow = speed < cfg.moving_speed
        moving = bool(np.median(speed) >= cfg.moving_speed)
        if (~slow).any():
            idx = np.flatnonzero(~slow)
            head = head[idx[np.abs(np.arange(len(head))[:, None] - idx[None]).argmin(1)]]
        else:
            head = ego[ie, 2].copy()
        obs = np.isin(ie, s["ie"][good])
        cover = float(obs.mean())
        size_q = float(np.clip(np.median(s["area"][good]) / 1500.0, 0.2, 1.0))
        conf = float(np.clip(math.sqrt(cover) * math.exp(-rms / (0.5 + 0.03 * np.median(s["z"]))) * (0.5 + 0.5 * size_q)
                             * (0.6 + 0.4 * min(1.0, (t1 - t0) / 4.0)) * (0.5 + 0.5 * good.mean()), 0, 1))
        tr = {"category": cat, "phrase": m["phrase"], "confidence": round(conf, 3), "t": te.copy(),
              "x": pos[:, 0], "y": pos[:, 1], "speed": speed, "heading": head, "lat": lat, "gap": gap,
              "s_path": sf[:, 0], "lane": lane,
              "range": np.hypot(pos[:, 0] - ego[ie, 0], pos[:, 1] - ego[ie, 1]), "observed": obs,
              "t0": float(t0), "t1": float(t1), "n_obs": int(good.sum()), "moving": moving,
              "size_m": (round(w3, 2), round(h3, 2))}
        tr["relation"] = _relation(tr, ego, ego_t, cfg)
        tr["events"] = _events(tr, cfg)
        out.append(tr)
    return out


def npc_tracks(run, ego: np.ndarray, t: np.ndarray | None = None, *, scale: float = 1.0,
               cfg: NpcConfig | None = None, drops: dict | None = None) -> list[dict]:
    """Tracks of other road users from a dense ViPE run.

    ``run``: ``run_vipe(..., dense=True)`` result, its output directory, or dense window dict(s) (see
    ``load_windows``). ``ego``: ``ego_track`` result (N,4) [x, y, yaw, v], optionally despiked, at times
    ``t`` (N,) (default: the run's). ``scale`` multiplies all distances (as ``ego_track(scale=)``).
    Returns tracks sorted by first appearance, each a dict with arrays over its frames (``t``, planar ``x``,
    ``y``, ``speed`` m/s, ``heading`` rad, ``lat`` / ``gap`` / ``lane`` relative to the ego path, ``range`` to
    the camera, ``observed``) and ``id``, ``category``, ``confidence`` in 0..1 (heuristic: coverage, fit
    residual, mask size, duration), ``t0`` / ``t1``, ``moving``, ``relation`` (at first sight), ``events``
    (cut_in, leaves_ego_lane, lane_change, brakes, stops). Candidate tracks that are dropped (short, unstable,
    sky and other non-road categories, background-sized) are counted by reason in ``drops`` if given.
    """
    cfg = cfg or NpcConfig()
    poses, run_t = _run_arrays(run, t)
    ego = np.asarray(ego, float)
    ego_t = np.asarray(run_t if t is None else t, float)
    half_dt = 0.6 * float(np.median(np.diff(ego_t)))
    up = _up_axis(poses[:, :3, :3], poses[:, :3, 3] * scale)
    if not isinstance(run, (dict, str, Path)):
        run = list(run)
    wts = window_times(run)
    dets = []
    for k, win in enumerate(load_windows(run)):
        planes = frame_planes(win, up, cfg)
        sc, h_ref = stabilize(planes)
        raw = detections(win, planes, sc, h_ref, cfg, global_scale=scale)
        dets += _relative(k, raw, win, up, ego_t, half_dt, _owned(win["t"], wts, k))
        del win, planes, raw
    path = _ego_path(ego)
    tracks = []
    for m in _stitch([_merged(tr) for tr in associate(dets, ego, cfg)], ego, half_dt):
        tracks += _build(m, ego, ego_t, path, cfg, half_dt, drops)
    tracks.sort(key=lambda x: x["t0"])
    for i, tr in enumerate(tracks):
        tr["id"] = i
    return tracks


def npc_summary(tracks: list[dict], ndigits: int = 1) -> list[dict]:
    """JSON-able digest of ``npc_tracks``: first/last sight, gap and lane when first seen, ``gap_start_m`` (the gap
    at t = 0 along the ego path: the ego's travel until first sight added, the road user's own travel at its
    first-seen speed subtracted), speed, mean speed, events and the speed profile (0.5 s steps)."""
    out = []
    for tr in tracks:
        sp = tr["speed"] * 3.6
        tt = np.arange(tr["t"][0], tr["t"][-1] + 1e-6, 0.5)
        out.append({"id": tr["id"], "category": tr["category"], "relation": tr["relation"],
                    "confidence": tr["confidence"], "t0": round(tr["t0"], 1), "t1": round(tr["t1"], 1),
                    "gap0_m": round(float(tr["gap"][0]), ndigits), "lat0_m": round(float(tr["lat"][0]), ndigits),
                    "gap_start_m": round(float(tr["s_path"][0] - sp[:5].mean() / 3.6 * tr["t"][0]), ndigits),
                    "lane0": int(tr["lane"][0]), "speed0_kmh": round(float(sp[:5].mean()), ndigits),
                    "speed_mean_kmh": round(float(sp.mean()), ndigits), "moving": tr["moving"],
                    "events": tr["events"], "profile": {"t": [round(float(x), 2) for x in tt],
                                                        "v_kmh": [round(float(x), 1)
                                                                  for x in np.interp(tt, tr["t"], sp)]}})
    return out


def save_tracks(tracks: list[dict], path: str | Path) -> None:
    """Write tracks as JSON (arrays as lists, 3 decimals)."""
    def enc(x):
        if isinstance(x, np.ndarray):
            return np.round(x, 3).tolist() if x.dtype.kind == "f" else x.tolist()
        if isinstance(x, (np.integer, np.floating, np.bool_)):
            return x.item()
        raise TypeError(type(x))
    Path(path).write_text(json.dumps(tracks, default=enc))
