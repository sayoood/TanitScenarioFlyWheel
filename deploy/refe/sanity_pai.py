#!/usr/bin/env python3
"""Sanity run of REFe SFT-4 on REAL PhysicalAI-AV frames (one clip, ~10 timesteps) through the f-theta -> NAVSIM adapter.

Runs in the torch container on thor3 (numpy, OpenCV, torch). Prepare the clip first with ``prep_pai_clip.py``.
For each timestep t it
  1. decodes the 6 Hyperion camera frames nearest to t, reprojects them to CAM_F0/L0/R0/B0 (960x512) with ``RigAdapter`` using the
     CLIP'S OWN calibration (rig.json),
  2. builds the ego vector from the logged egomotion and the 2-point route goal from the logged ego path (the dataset has no HD map: the
     route is the path the ego actually drove, arc length S/2 and S ahead, S = max(v, 5) * 12 m, clamped at the end of the clip),
  3. runs the model, takes the selected proposal and resamples it to 8 points @ 0.5 s,
  4. scores ADE / FDE against the logged future ego motion over the same 4 s (rear-axle xy in the ego frame at t),
and reports latency per step and GPU memory. Baselines (constant velocity, constant curvature) use the same ground truth.
Adapter variants (rotation-only reprojection, undistorted virtual cameras) run on the same steps to show the sensitivity.

Nothing derived from the frames leaves the machine: only aggregate numbers and the model's trajectories are written (``--out`` json).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))
from tsf.closedloop import rig_adapter as ra          # noqa: E402
from tsf.closedloop import refe as rf                 # noqa: E402

WHEELBASE = 3.0       # assumption for tire_steering_angle = atan(L * curvature); the dataset carries no steering angle


def frame_index(ts: np.ndarray, t: float) -> int:
    return int(np.argmin(np.abs(ts - t)))


def load_frame(cd: Path, cam: str, idx: int) -> np.ndarray:
    """RGB uint8 frame ``idx`` of a camera, pre-decoded by prep_pai_clip.py (host ffmpeg; the container's OpenCV has no HEVC)."""
    im = cv2.imread(str(cd / "frames" / cam / f"{idx:05d}.png"), cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(f"frame {idx} of {cam}: run prep_pai_clip.py with the same --t-start/--n/--step")
    return np.ascontiguousarray(im[..., ::-1])


class Ego:
    """Logged egomotion with the lookups the sanity run needs (all in the clip's local frame)."""

    def __init__(self, z):
        self.t, self.x, self.y, self.yaw = z["t"], z["x"], z["y"], z["yaw"]
        self.z = z

    def at(self, t: float) -> dict:
        f = lambda k: float(np.interp(t, self.t, self.z[k]))        # noqa: E731
        return dict(x=f("x"), y=f("y"), yaw=f("yaw"), vx=f("vx"), vy=f("vy"), ax=f("ax"), ay=f("ay"), curv=f("curvature"))

    def to_frame(self, t0: float, xs, ys):
        s = self.at(t0)
        c, sn = math.cos(-s["yaw"]), math.sin(-s["yaw"])
        dx, dy = np.asarray(xs) - s["x"], np.asarray(ys) - s["y"]
        return c * dx - sn * dy, sn * dx + c * dy

    def ego_vec(self, t: float) -> np.ndarray:
        s = self.at(t)
        speed = math.hypot(s["vx"], s["vy"])
        yaw_rate = s["vx"] * s["curv"]
        return np.array([s["vx"], s["vy"], s["ax"], s["ay"], yaw_rate, math.atan(WHEELBASE * s["curv"]), speed], np.float32)

    def goal(self, t: float) -> np.ndarray:
        """2 points on the logged path at arc length S/2, S ahead of the ego (clamped at the end of the clip)."""
        s = self.at(t)
        speed = max(math.hypot(s["vx"], s["vy"]), 5.0)
        S = speed * 12.0
        m = self.t >= t
        px, py = np.concatenate([[s["x"]], self.x[m]]), np.concatenate([[s["y"]], self.y[m]])
        cum = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(px), np.diff(py)))])
        pts = []
        for j in (1, 2):
            L = min(S * j / 2, cum[-1])
            pts.append((np.interp(L, cum, px), np.interp(L, cum, py)))
        gx, gy = self.to_frame(t, [p[0] for p in pts], [p[1] for p in pts])
        return np.array([gx[0], gy[0], gx[1], gy[1]], np.float32), float(cum[-1])

    def future(self, t: float, n: int = 8, dt: float = 0.5) -> np.ndarray:
        tt = t + dt * np.arange(1, n + 1)
        fx, fy = np.interp(tt, self.t, self.x), np.interp(tt, self.t, self.y)
        a, b = self.to_frame(t, fx, fy)
        return np.stack([a, b], 1)


class LoggedPath:
    """The logged ego path as a TanitSim-style ``Path`` (``pose(p)`` by arc length, world frame) = the route of the sanity run."""

    def __init__(self, ego: Ego, t_from: float):
        m = ego.t >= t_from - 1.0
        self.x, self.y = ego.x[m], ego.y[m]
        self.cum = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(self.x), np.diff(self.y)))])
        self.length = float(self.cum[-1])

    def pose(self, p: float, d: float = 0.0):
        p = min(max(p, 0.0), self.length)
        return float(np.interp(p, self.cum, self.x)), float(np.interp(p, self.cum, self.y)), 0.0

    def progress(self, x: float, y: float) -> float:
        return float(self.cum[int(np.argmin(np.hypot(self.x - x, self.y - y)))])


def http_policy_run(ego: Ego, times, frames, adapter, url: str) -> dict:
    """The policy object a closed loop would use, fed from the logged state: ADE/FDE of its 8-point output + latency split."""
    from types import SimpleNamespace
    pol = rf.RefePolicy(server_url=url, adapter=adapter, replan_every_s=0.0)
    rows = []
    for t in times:
        st = ego.at(t)
        path = LoggedPath(ego, t)
        e = SimpleNamespace(v=math.hypot(st["vx"], st["vy"]), a=st["ax"], steer=math.atan(WHEELBASE * st["curv"]), yaw=st["yaw"],
                            x=st["x"], y=st["y"], path=path, p=path.progress(st["x"], st["y"]))
        cmd = pol(SimpleNamespace(t=t), e, frames[t])
        assert len(cmd["traj"]) == 8 and cmd["dt"] == 0.5
        ade, fde = ade_fde(np.array(cmd["traj"]), ego.future(t))
        rows.append({"t": t, "ade": ade, "fde": fde, **pol.last["latency_ms"], "k": pol.last["k"]})
    keys = ("roundtrip", "forward", "adapter")
    return {"steps": len(rows), "ADE_mean": float(np.mean([r["ade"] for r in rows])), "FDE_mean": float(np.mean([r["fde"] for r in rows])),
            **{f"{k}_ms_median": float(np.median([r[k] for r in rows])) for k in keys}, "rows": rows}


def ade_fde(pred: np.ndarray, gt: np.ndarray):
    d = np.linalg.norm(np.asarray(pred) - gt, axis=1)
    return float(d.mean()), float(d[-1])


def baselines(ego: dict, n=8, dt=0.5):
    v, w = math.hypot(ego["vx"], ego["vy"]), ego["vx"] * ego["curv"]
    cv_ = np.array([[v * dt * k, 0.0] for k in range(1, n + 1)])
    th = w * dt * np.arange(1, n + 1)
    cc = np.stack([np.where(abs(w) > 1e-4, np.sin(th) * v / (w if abs(w) > 1e-4 else 1), v * dt * np.arange(1, n + 1)),
                   np.where(abs(w) > 1e-4, (1 - np.cos(th)) * v / (w if abs(w) > 1e-4 else 1), 0.0)], 1)
    return cv_, cc


TGRID = np.concatenate([[0.0], 0.2 * np.arange(1, 21)])


def to8(pp: np.ndarray) -> np.ndarray:
    """[20, 2] model xy @ 0.2 s -> [8, 2] @ 0.5 s (same linear resampling as RefePolicy)."""
    full = np.vstack([[0.0, 0.0], pp])
    return np.stack([np.interp(0.5 * np.arange(1, 9), TGRID, full[:, c]) for c in (0, 1)], 1)


def ncc(a: np.ndarray, b: np.ndarray, m: np.ndarray) -> float:
    ga = cv2.GaussianBlur(cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float32), (0, 0), 3)[m]
    gb = cv2.GaussianBlur(cv2.cvtColor(b, cv2.COLOR_RGB2GRAY).astype(np.float32), (0, 0), 3)[m]
    ga, gb = ga - ga.mean(), gb - gb.mean()
    return float((ga * gb).sum() / (np.sqrt((ga ** 2).sum() * (gb ** 2).sum()) + 1e-9))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip-dir", required=True)
    ap.add_argument("--clip", default="bcc52cd1-9507-425e-a5a6-5f8a795ed525")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--refe-src", required=True)
    ap.add_argument("--t-start", type=float, default=2.0)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--step", type=float, default=1.0)
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--variants", action="store_true", help="also run rotation-only and undistorted adapter variants")
    ap.add_argument("--server-url", default=None, help="also send the first steps through RefePolicy -> this HTTP server")
    ap.add_argument("--http-only", action="store_true", help="no in-process engine: only the RefePolicy -> HTTP server path (needs --server-url)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import torch
    from refe_server import RefeEngine

    cd = Path(a.clip_dir)
    rig = ra.load_rig(cd / "rig.json")
    ego = Ego(np.load(cd / "ego.npz"))
    times = [a.t_start + a.step * k for k in range(a.n)]
    assert times[-1] + 4.0 < ego.t[-1], "future window leaves the clip"

    # ---- decode the frames of every step, all cameras
    frames, ts_all = {t: {} for t in times}, {}
    t0 = time.time()
    for cam in rig:
        ts = np.load(cd / f"ts_{cam}.npy")
        for t in times:
            frames[t][cam] = load_frame(cd, cam, frame_index(ts, t))
        ts_all[cam] = float(np.abs(ts[[frame_index(ts, t) for t in times]] - np.array(times)).max())
    print(f"loaded {len(times)} steps x {len(rig)} cams in {time.time() - t0:.1f}s; max |frame - t| per cam (s): "
          f"{ {k[7:16]: round(v, 3) for k, v in ts_all.items()} }", flush=True)

    # ---- adapters
    adapters = {"main": ra.RigAdapter(rig)}
    if a.variants:
        adapters["rot_only"] = ra.RigAdapter(rig, ground_plane=False)
        adapters["no_distort"] = ra.RigAdapter(rig, distort=False)
    fwh = {n: (1920, 1080) for n in rig}
    cov = adapters["main"].coverage(fwh)
    print("adapter coverage:", {k: round(v, 3) for k, v in cov.items()}, flush=True)

    if a.http_only:
        res = http_policy_run(ego, times, frames, adapters["main"], a.server_url)
        for r in res["rows"]:
            print(f"t={r['t']:5.1f} ADE {r['ade']:.2f} FDE {r['fde']:.2f} k{r['k']} roundtrip {r['roundtrip']:.0f} ms (forward {r['forward']:.0f}, adapter {r['adapter']:.0f})")
        print("HTTP_POLICY", {k: v for k, v in res.items() if k != "rows"}, flush=True)
        Path(a.out).write_text(json.dumps(res, indent=1))
        return 0

    # ---- model
    mem0 = torch.cuda.memory_allocated() / 2 ** 20 if torch.cuda.is_available() else 0
    eng = RefeEngine(a.ckpt, a.refe_src, dtype=a.dtype)
    print("engine:", eng.info(), flush=True)
    rows = []
    # warm-up (cuDNN / kernel autotune / pos3d cache) on the first step, not counted
    img0 = adapters["main"].stack(frames[times[0]])
    eng.infer(img0, ego.ego_vec(times[0]), ego.goal(times[0])[0])
    torch.cuda.reset_peak_memory_stats()
    for t in times:
        st = ego.at(t)
        e7, (g4, route_len) = ego.ego_vec(t), ego.goal(t)
        gt = ego.future(t)
        row = {"t": t, "speed": float(e7[6]), "route_len_m": route_len, "goal": g4.tolist(), "gt_end": gt[-1].tolist()}
        for name, ad in adapters.items():
            ta = time.perf_counter()
            img = ad.stack(frames[t])
            t_ad = (time.perf_counter() - ta) * 1e3
            r = eng.infer(img, e7, g4, return_all=(name == "main"))
            pred = np.array(rf.resample_traj(np.asarray(r["traj"]), 0.5, 8))
            ade, fde = ade_fde(pred, gt)
            row[name] = {"ade": ade, "fde": fde, "k": r["k"], "agg": r["agg"], "forward_ms": r["latency_ms"]["forward"],
                         "adapter_ms": t_ad, "pred_end": pred[-1].tolist(), "score_sigmoid": r["score_sigmoid"]}
            if name == "main":                                                 # oracle: the best of the 64 proposals (selection headroom)
                props = np.asarray(r["proposals"])[:, :, :2]                  # [64, 20, 2]
                row["oracle_best_of_64_ade"] = float(min(np.linalg.norm(to8(pp) - gt, axis=1).mean() for pp in props))
        cvb, ccb = baselines(st)
        row["const_vel"] = dict(zip(("ade", "fde"), ade_fde(cvb, gt)))
        row["const_curv"] = dict(zip(("ade", "fde"), ade_fde(ccb, gt)))
        rows.append(row)
        print(f"t={t:5.1f}s v={e7[6]:5.2f} gt_end=({gt[-1][0]:6.1f},{gt[-1][1]:6.1f}) | " + " | ".join(
            f"{n} ADE {row[n]['ade']:.2f} FDE {row[n]['fde']:.2f} k{row[n]['k']}" for n in adapters)
            + f" | CV {row['const_vel']['ade']:.2f}/{row['const_vel']['fde']:.2f} CC {row['const_curv']['ade']:.2f}/{row['const_curv']['fde']:.2f}"
            + f" | fwd {row['main']['forward_ms']:.0f} ms adapt {row['main']['adapter_ms']:.0f} ms", flush=True)

    # ---- aggregates
    agg = {}
    for n in list(adapters) + ["const_vel", "const_curv"]:
        agg[n] = {"ADE_mean": float(np.mean([r[n]["ade"] for r in rows])), "FDE_mean": float(np.mean([r[n]["fde"] for r in rows])),
                  "ADE_median": float(np.median([r[n]["ade"] for r in rows])), "FDE_median": float(np.median([r[n]["fde"] for r in rows]))}
    agg["oracle_best_of_64_ADE_mean"] = float(np.mean([r["oracle_best_of_64_ade"] for r in rows]))
    lat = np.array([r["main"]["forward_ms"] for r in rows])
    agg["latency_ms"] = {"forward_median": float(np.median(lat)), "forward_min": float(lat.min()), "forward_max": float(lat.max()),
                         "adapter_median": float(np.median([r["main"]["adapter_ms"] for r in rows]))}
    agg["gpu_mb"] = {"weights_alloc": eng.info()["gpu_alloc_mb"], "peak_alloc": torch.cuda.max_memory_allocated() / 2 ** 20,
                     "reserved": torch.cuda.memory_reserved() / 2 ** 20, "baseline_before_load": mem0}

    # ---- adapter self-consistency on real frames: L0/R0 reprojected from the cross camera vs from the front-wide camera, compared
    # where both see the same pixel. Above the horizon (rows < 250) the content is far, so parallax is nil and the f-theta models
    # must agree; below it (rows >= 290, road) the ground-plane reprojection must beat rotation-only if the geometry is right.
    cons = {}
    rows_idx = np.arange(512)[:, None] * np.ones((1, 960), int)
    for an in [n for n in ("main", "rot_only") if n in adapters]:
        ad = adapters[an]
        for v, srcs in (("CAM_L0", ("camera_cross_left_120fov", "camera_front_wide_120fov")),
                        ("CAM_R0", ("camera_cross_right_120fov", "camera_front_wide_120fov"))):
            la, lb = ad._build(v, srcs[0], (1920, 1080)), ad._build(v, srcs[1], (1920, 1080))
            m = (la.w > 40) & (lb.w > 40)
            vals = {"far(rows<250)": [], "road(rows>=290)": []}
            shifts = []
            for t in times:
                ia = ad.virtual_frame(v, {srcs[0]: frames[t][srcs[0]]})
                ib = cv2.remap(frames[t][srcs[1]], lb.mapx, lb.mapy, cv2.INTER_LINEAR)
                for name, mm in (("far(rows<250)", m & (rows_idx < 250)), ("road(rows>=290)", m & (rows_idx >= 290))):
                    if mm.sum() > 3000:
                        vals[name].append(ncc(ia, ib, mm))
                # global residual shift (px at 960x512) of the far field between the two reprojections: ~0 if the f-theta models,
                # extrinsics and the pixel-centre convention are mutually consistent
                far = (m & (rows_idx < 250)).astype(np.float32)
                ga = cv2.GaussianBlur(cv2.cvtColor(ia, cv2.COLOR_RGB2GRAY).astype(np.float32), (0, 0), 2) * far
                gb = cv2.GaussianBlur(cv2.cvtColor(ib, cv2.COLOR_RGB2GRAY).astype(np.float32), (0, 0), 2) * far
                (dx, dy), resp = cv2.phaseCorrelate(ga, gb, cv2.createHanningWindow((960, 512), cv2.CV_32F))
                shifts.append((dx, dy, resp))
            sh = np.array(shifts)
            cons[f"{an}:{v}"] = {"overlap_px": int(m.sum()), **{k: (float(np.mean(x)) if x else None) for k, x in vals.items()},
                                 "far_shift_px_median_dx_dy": [float(np.median(sh[:, 0])), float(np.median(sh[:, 1]))],
                                 "far_shift_px_mean_abs_dx_dy": [float(np.abs(sh[:, 0]).mean()), float(np.abs(sh[:, 1]).mean())]}
    agg["adapter_selfconsistency_real"] = cons
    agg["adapter_coverage"] = cov
    agg["frame_time_offset_s"] = ts_all

    if a.server_url:
        agg["http_policy"] = {k: v for k, v in http_policy_run(ego, times, frames, adapters["main"], a.server_url).items() if k != "rows"}
        print("http policy:", agg["http_policy"], flush=True)

    Path(a.out).write_text(json.dumps({"clip": a.clip, "times": times, "rows": rows, "aggregate": agg}, indent=1))
    print("AGGREGATE", json.dumps(agg, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
