"""PhysicalAI-AV validation of ViPE ego motion, dense depth and NPC tracks (thor3 only; the dataset licence keeps
inputs and every derivative on thor, only aggregate numbers leave).

    prep  <clip_id>...   front_wide f-theta mp4 -> virtual pinhole mp4 (+ ground truth in the camera frame)
    eval  <clip_id>...   compare ``pai/<clip>/run`` (run_vipe(dense=True) output) with the ground truth

Virtual camera: 854x480, hfov 90 deg, pitched up ``PITCH_UP`` deg from the real optical axis (the front_wide
principal point sits at row 758 of 1080, so only 20 deg are visible below the axis).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

PAI = Path(os.environ.get("TSF_PAI_AV", Path.home() / "data" / "pai_av"))
WORK = Path(os.environ.get("TSF_VIPE_PAI", Path.home() / "tsf" / "dev" / "vipe" / "pai"))
CAM = "camera_front_wide_120fov"
CHUNK = "chunk_2800"
OUT_W, OUT_H, HFOV, PITCH_UP = 854, 480, 90.0, 12.0


def _rx(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rig_camera(clip_id: str):
    """(FThetaCamera, R_vc, t_vc): real front_wide camera and its pose in the vehicle (rig) frame."""
    from scipy.spatial.transform import Rotation

    from tsf.closedloop.rig_adapter import FThetaCamera
    ci = pd.read_parquet(PAI / "calibration" / "camera_intrinsics" / f"camera_intrinsics.{CHUNK}.parquet")
    ex = pd.read_parquet(PAI / "calibration" / "sensor_extrinsics.offline" /
                         f"sensor_extrinsics.offline.{CHUNK}.parquet")
    r, e = ci.loc[(clip_id, CAM)], ex.loc[(clip_id, CAM)]
    cam = FThetaCamera.from_pai_av(int(r.width), int(r.height), r.cx, r.cy, [r[f"bw_poly_{i}"] for i in range(5)])
    return cam, Rotation.from_quat([e.qx, e.qy, e.qz, e.qw]).as_matrix(), np.array([e.x, e.y, e.z])


def virtual_rays(w: int = OUT_W, h: int = OUT_H, ss: int = 2) -> tuple[np.ndarray, float]:
    f = w / 2 / math.tan(math.radians(HFOV) / 2)
    u, v = (np.arange(n * ss) / ss + 0.5 / ss for n in (w, h))
    uu, vv = np.meshgrid(u, v)
    d = np.stack([(uu - w / 2) / f, (vv - h / 2) / f, np.ones_like(uu)], -1) @ _rx(math.radians(PITCH_UP)).T
    return d.reshape(-1, 3), f


def prep(clip_id: str) -> Path:
    import cv2
    out = WORK / clip_id
    out.mkdir(parents=True, exist_ok=True)
    cam, R_vc, t_vc = rig_camera(clip_id)
    rays, f = virtual_rays()
    px, ok = cam.ray2pixel(rays)
    ss = 2
    mapx = px[:, 0].reshape(OUT_H * ss, OUT_W * ss).astype(np.float32)
    mapy = px[:, 1].reshape(OUT_H * ss, OUT_W * ss).astype(np.float32)
    inside = ok.reshape(mapx.shape) & (mapx >= 0) & (mapy >= 0) & (mapx <= cam.width - 1) & (mapy <= cam.height - 1)
    src = out / "front_wide.mp4"
    with zipfile.ZipFile(PAI / "camera" / CAM / f"{CAM}.{CHUNK}.zip") as z:
        z.extract(f"{clip_id}.{CAM}.mp4", out)
        (out / f"{clip_id}.{CAM}.mp4").replace(src)
        (out / "timestamps.parquet").write_bytes(z.read(f"{clip_id}.{CAM}.timestamps.parquet"))
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS)
    dst = out / "pinhole.mp4"
    import imageio_ffmpeg
    ff = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo",
                           "-pix_fmt", "bgr24",
                           "-s", f"{OUT_W}x{OUT_H}", "-r", f"{fps:g}", "-i", "-", "-an", "-c:v", "libx264",
                           "-preset", "veryfast", "-crf", "12", "-pix_fmt", "yuv420p", str(dst)],
                          stdin=subprocess.PIPE)
    n = 0
    while True:
        ok_, bgr = cap.read()
        if not ok_:
            break
        big = cv2.remap(bgr, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        ff.stdin.write(cv2.resize(big, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA).tobytes())
        n += 1
    cap.release()
    ff.stdin.close()
    ff.wait()
    src.unlink()
    R_virt = R_vc @ _rx(math.radians(PITCH_UP))
    meta = {"clip": clip_id, "frames": n, "fps": fps, "f_px": f, "cx": OUT_W / 2, "cy": OUT_H / 2,
            "valid_fraction": float(inside.mean()), "pitch_up_deg": PITCH_UP, "hfov_deg": HFOV,
            "cam_in_vehicle_xyz": t_vc.tolist(), "real_axis_in_vehicle": R_vc[:, 2].tolist(),
            "virtual_axis_in_vehicle": R_virt[:, 2].tolist()}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    return dst


# ----------------------------------------------------------------------------------------------- ground truth
VEHICLES = {"automobile", "heavy_truck", "bus", "van", "trailer", "motorcycle", "bicycle", "rider", "person"}
BINS = [(4, 10), (10, 20), (20, 30), (30, 40)]
SAMPLE_KEYS = ("e_range", "e_cross", "rel", "dv", "d_foot", "d_cen")


def rot2(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


def bin_of(r: float) -> str:
    return next(f"{lo}-{hi}" for lo, hi in BINS if lo <= r <= hi)


def smooth_xy(xy: np.ndarray, sigma: float = 4.0) -> np.ndarray:
    from scipy.ndimage import gaussian_filter1d
    return gaussian_filter1d(xy, sigma, axis=0, mode="nearest")


class Gt:
    """PAI-AV ground truth of one clip in the planar camera frame of frame 0 (x forward, y left), at the ViPE
    frame times: ``cam`` (x, y, psi, v) of the camera, ``objs`` {id: dict(cls, ok, xy, yaw, lwh, speed)}."""

    def __init__(self, clip_id: str, n_frames: int, fps: float = 15.0):
        from tsf.ingest.pai_av import _yaw
        self.clip, self.meta = clip_id, json.loads((WORK / clip_id / "meta.json").read_text())
        ts = pd.read_parquet(WORK / clip_id / "timestamps.parquet")["timestamp"].to_numpy() / 1e6
        self.tau = ts[0] + np.arange(n_frames) / fps
        e = pd.read_parquet(PAI / "x" / "ego" / f"{clip_id}.egomotion.parquet").sort_values("timestamp")
        self.te = e.timestamp.to_numpy() / 1e6
        self.ex, self.ey = e.x.to_numpy(), e.y.to_numpy()
        self.eyaw = np.unwrap(_yaw(e.qx, e.qy, e.qz, e.qw).to_numpy())
        self.ev = np.hypot(e.vx, e.vy).to_numpy()
        self.quat = e[["qx", "qy", "qz", "qw"]].to_numpy()
        cx, cy, self.cam_h = self.meta["cam_in_vehicle_xyz"]
        ax = self.meta["virtual_axis_in_vehicle"]
        self.cam_off, self.yaw_off, self.pitch = np.array([cx, cy]), math.atan2(ax[1], ax[0]), math.asin(ax[2])
        p0, psi0 = self._cam(self.tau[:1])
        self.o, self.psi0 = p0[0], psi0[0]
        p, psi = self._cam(self.tau)
        self.cam = np.column_stack([self.plane(p), psi - self.psi0, np.interp(self.tau, self.te, self.ev)])
        self.objs = self._objects(_yaw)

    def _cam(self, tt):
        x, y, yaw = (np.interp(tt, self.te, a) for a in (self.ex, self.ey, self.eyaw))
        c, s = np.cos(yaw), np.sin(yaw)
        return np.column_stack([x + c * self.cam_off[0] - s * self.cam_off[1],
                                y + s * self.cam_off[0] + c * self.cam_off[1]]), yaw + self.yaw_off

    def plane(self, p):
        return (np.atleast_2d(p) - self.o) @ rot2(self.psi0)

    def _objects(self, _yaw) -> dict:
        o = pd.read_parquet(PAI / "x" / "obs" / f"{self.clip}.obstacle.offline.parquet")
        T = o.timestamp_us.to_numpy() / 1e6
        x, y, yaw = (np.interp(T, self.te, a) for a in (self.ex, self.ey, self.eyaw))
        c, s = np.cos(yaw), np.sin(yaw)
        o = o.assign(T=T, X=x + c * o.center_x - s * o.center_y, Y=y + s * o.center_x + c * o.center_y,
                     YAW=yaw + _yaw(o.orientation_x, o.orientation_y, o.orientation_z, o.orientation_w))
        out = {}
        for tid, g in o.sort_values("T").groupby("track_id"):
            cls = str(g.label_class.mode().iloc[0]).lower()
            if len(g) < 5 or cls not in VEHICLES:
                continue
            t, xy = g["T"].to_numpy(), self.plane(g[["X", "Y"]].to_numpy())
            yw = np.unwrap(g.YAW.to_numpy()) - self.psi0
            idx = np.searchsorted(t, self.tau)
            gap = t[np.clip(idx, 1, len(t) - 1)] - t[np.clip(idx - 1, 0, len(t) - 2)]
            ok = (self.tau >= t[0]) & (self.tau <= t[-1]) & (gap < 0.6)
            if ok.sum() < 5:
                continue
            xyi = np.column_stack([np.interp(self.tau, t, xy[:, 0]), np.interp(self.tau, t, xy[:, 1])])
            sp = np.hypot(*np.gradient(smooth_xy(xyi), self.tau, axis=0).T)
            ob = {"cls": cls, "ok": ok, "xy": xyi, "yaw": np.interp(self.tau, t, yw),
                  "lwh": (float(g.size_x.median()), float(g.size_y.median()), float(g.size_z.median())), "speed": sp}
            out[str(tid)] = self._derive(ob)
        return out

    def _derive(self, ob: dict) -> dict:
        """Per frame: box centre, box axes angle, nearest footprint point to the camera, its range, eligibility."""
        d = ob["xy"] - self.cam[:, :2]
        c, s = np.cos(self.cam[:, 2]), np.sin(self.cam[:, 2])
        ob["c"] = np.column_stack([d[:, 0] * c + d[:, 1] * s, -d[:, 0] * s + d[:, 1] * c])
        ob["th"] = ob["yaw"] - self.cam[:, 2]
        ob["h"] = np.array(ob["lwh"][:2]) / 2
        u = np.column_stack([np.cos(ob["th"]), np.sin(ob["th"])])
        v = np.column_stack([-u[:, 1], u[:, 0]])
        q = np.column_stack([(-ob["c"] * u).sum(1), (-ob["c"] * v).sum(1)])
        qc = np.clip(q, -ob["h"], ob["h"])
        ob["near"] = ob["c"] + qc[:, :1] * u + qc[:, 1:] * v
        ob["range"] = np.hypot(*ob["near"].T)
        bearing = np.degrees(np.arctan2(ob["near"][:, 1], ob["near"][:, 0]))
        ob["elig"] = (ob["ok"] & (ob["range"] >= 4.0) & (ob["range"] <= 40.0) & (ob["near"][:, 0] > 0)
                      & (np.abs(bearing) <= 40.0))
        return ob

    def rect_dist(self, tid: str, ks: np.ndarray, pts: np.ndarray) -> np.ndarray:
        """Distance of points (n,2) in the camera-horizontal frame of frames ks to the GT footprint of ``tid``."""
        ob = self.objs[tid]
        th = ob["th"][ks]
        u = np.column_stack([np.cos(th), np.sin(th)])
        v = np.column_stack([-u[:, 1], u[:, 0]])
        dp = pts - ob["c"][ks]
        q = np.column_stack([(dp * u).sum(1), (dp * v).sum(1)])
        return np.hypot(*(q - np.clip(q, -ob["h"], ob["h"])).T)


# ----------------------------------------------------------------------------------------------- metrics
def umeyama2d(src: np.ndarray, dst: np.ndarray):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    a, b = src - mu_s, dst - mu_d
    u, d, vt = np.linalg.svd(b.T @ a / len(a))
    sgn = np.sign(np.linalg.det(u @ vt))
    r = u @ np.diag([1, sgn]) @ vt
    sc = (d * [1, sgn]).sum() / (a ** 2).sum() * len(a)
    return sc, r, mu_d - sc * r @ mu_s


def path_len(a: np.ndarray) -> float:
    return float(np.hypot(*np.diff(a, axis=0).T).sum())


def ego_metrics(gt: Gt, ego: np.ndarray) -> dict:
    n = len(ego)
    xy, yaw, v = ego[:, :2], ego[:, 2], ego[:, 3]
    g = gt.cam[:n]
    err = np.linalg.norm(xy - g[:, :2], axis=1)
    yerr = np.degrees((yaw - g[:, 2] + np.pi) % (2 * np.pi) - np.pi)
    sc, r, tt = umeyama2d(xy, g[:, :2])
    ate_sim = np.linalg.norm(sc * xy @ r.T + tt - g[:, :2], axis=1)
    return {"speed_mae_kmh": float(np.abs(v - g[:, 3]).mean() * 3.6),
            "speed_bias_kmh": float((v - g[:, 3]).mean() * 3.6), "gt_speed_mean_kmh": float(g[:, 3].mean() * 3.6),
            "heading_err_end_deg": float(yerr[-1]), "heading_err_mean_abs_deg": float(np.abs(yerr).mean()),
            "gt_heading_change_deg": float(np.degrees(g[-1, 2] - g[0, 2])),
            "ate_start_rmse_m": float(np.sqrt((err ** 2).mean())), "final_err_m": float(err[-1]),
            "path_gt_m": path_len(g[:, :2]), "path_ratio": path_len(xy) / max(path_len(g[:, :2]), 1e-6),
            "ate_sim2_rmse_m": float(np.sqrt((ate_sim ** 2).mean())), "sim2_scale": float(sc)}


def pred_rel(tr: dict, ego: np.ndarray):
    """Frame indices and predicted contact points in the camera-horizontal frame of the (ViPE) ego."""
    k = np.round(tr["t"] * 15.0).astype(int)
    e = ego[k]
    d = np.column_stack([tr["x"], tr["y"]]) - e[:, :2]
    c, s = np.cos(e[:, 2]), np.sin(e[:, 2])
    return k, np.column_stack([d[:, 0] * c + d[:, 1] * s, -d[:, 0] * s + d[:, 1] * c])


def match_tracks(gt: Gt, tracks: list[dict], ego: np.ndarray, gate: float = 2.0):
    """Hungarian association of predicted with GT tracks by the median distance of the predicted contact point
    to the GT footprint over the common frames (>= 5, GT between 3 and 60 m)."""
    from scipy.optimize import linear_sum_assignment
    ids = list(gt.objs)
    cost = np.full((len(tracks), len(ids)), 1e3)
    for i, tr in enumerate(tracks):
        k_, rel = pred_rel(tr, ego)
        for j, tid in enumerate(ids):
            ob = gt.objs[tid]
            sel = ob["ok"][k_] & (ob["range"][k_] >= 3.0) & (ob["range"][k_] <= 60.0)
            if sel.sum() >= 5:
                cost[i, j] = float(np.median(gt.rect_dist(tid, k_[sel], rel[sel])))
    rr, cc = linear_sum_assignment(cost)
    return {int(i): ids[j] for i, j in zip(rr, cc) if cost[i, j] < gate}, cost


def track_metrics(gt: Gt, tracks: list[dict], ego: np.ndarray) -> dict:
    pairs, _ = match_tracks(gt, tracks, ego)
    bins = {f"{lo}-{hi}": {k: [] for k in SAMPLE_KEYS} for lo, hi in BINS}
    matched: dict[str, set[int]] = {}
    for i, tid in pairs.items():
        tr, ob = tracks[i], gt.objs[tid]
        k_, rel = pred_rel(tr, ego)
        foot = gt.rect_dist(tid, k_, rel)
        good = ob["elig"][k_] & (foot <= 3.0)
        for m in np.flatnonzero(good):
            k = k_[m]
            near, r = ob["near"][k], float(ob["range"][k])
            matched.setdefault(tid, set()).add(int(k))
            b = bins[bin_of(r)]
            b["e_range"].append(float(np.hypot(*rel[m]) - r))
            b["e_cross"].append(float((rel[m] - near) @ (np.array([-near[1], near[0]]) / r)))
            b["rel"].append(abs(float(np.hypot(*rel[m]) - r)) / r)
            b["dv"].append(float(tr["speed"][m] - ob["speed"][k]) * 3.6)
            b["d_foot"].append(float(foot[m]))
            b["d_cen"].append(float(np.hypot(*(rel[m] - ob["c"][k]))))
    elig_tracks = [tid for tid, ob in gt.objs.items() if ob["elig"].sum() >= 22]
    rec = {f"{lo}-{hi}": [0, 0] for lo, hi in BINS}
    for tid in elig_tracks:
        ob = gt.objs[tid]
        for k in np.flatnonzero(ob["elig"]):
            b = rec[bin_of(float(ob["range"][k]))]
            b[1] += 1
            b[0] += int(int(k) in matched.get(tid, ()))
    return {"n_pred": len(tracks), "n_gt_eligible": len(elig_tracks), "n_matched_pred": len(pairs),
            "track_recall": float(len(set(elig_tracks) & set(pairs.values())) / max(len(elig_tracks), 1)),
            "precision_vs_gt_labels": float(len(pairs) / max(len(tracks), 1)), "frame_recall_bins": rec, "bins": bins}


def depth_metrics(gt: Gt, run_dir: Path, stride: int = 3) -> dict:
    """Dense-depth error on (a) the road (flat-road geometry from rig height and attitude) and (b) the near
    footprint corner of eligible GT vehicles, 0.6 m above the road (median of a 7x7 window)."""
    from scipy.spatial.transform import Rotation
    z = np.load(run_dir / "w00" / "dense.npz")
    depth, inst, K = z["depth"].astype(np.float32), z["instance"], z["K"]
    f, cx, cy, phi = gt.meta["f_px"], gt.meta["cx"], gt.meta["cy"], gt.pitch
    # vehicle axes (forward, left, up) -> virtual camera (right, down, forward); the camera looks phi above the horizon
    M = np.array([[0, -1, 0], [math.sin(phi), 0, -math.cos(phi)], [math.cos(phi), 0, math.sin(phi)]])
    H, W = depth.shape[1:]
    v, u = np.mgrid[0:H:6, 0:W:6]
    ray = np.stack([(u + 0.5 - cx) / f, (v + 0.5 - cy) / f, np.ones(u.shape)], -1)
    road = {f"{lo}-{hi}": [] for lo, hi in BINS}
    veh = {f"{lo}-{hi}": [] for lo, hi in BINS}
    for k in range(0, len(depth), stride):
        up_veh = Rotation.from_quat(gt.quat[np.argmin(np.abs(gt.te - gt.tau[k]))]).as_matrix().T @ [0, 0, 1.0]
        den = ray @ -(M @ up_veh)
        zg = gt.cam_h / np.maximum(den, 1e-3)
        zd = depth[k][v, u]
        ok = (den > 0.05) & (zg < 45) & (inst[k][v, u] == 0) & (zd > 0)
        for lo, hi in BINS:
            sel = ok & (zg >= lo) & (zg < hi)
            road[f"{lo}-{hi}"] += (np.abs(zd[sel] - zg[sel]) / zg[sel]).tolist()
        for ob in gt.objs.values():
            if not ob["elig"][k]:
                continue
            n = ob["near"][k]
            p = M @ np.array([n[0], n[1], 0.6 - gt.cam_h])
            if p[2] <= 1:
                continue
            uu, vv = int(f * p[0] / p[2] + cx), int(f * p[1] / p[2] + cy)
            if not (4 <= uu < W - 4 and 4 <= vv < H - 4):
                continue
            win = depth[k][vv - 3:vv + 4, uu - 3:uu + 4]
            win = win[win > 0]
            if len(win) >= 5:
                veh[bin_of(float(np.hypot(*n)))].append(abs(float(np.median(win)) - p[2]) / p[2])

    def stat(a):
        a = np.array(a)
        return {"n": len(a), "absrel_median": float(np.median(a)), "within_10pct": float((a < 0.1).mean())} \
            if len(a) else {"n": 0}
    return {"road": {b: stat(a) for b, a in road.items()}, "vehicle_corner": {b: stat(a) for b, a in veh.items()},
            "focal_est_px": float(K[0]), "focal_true_px": float(f)}


def summarize_bins(bins: dict) -> dict:
    out = {}
    for b, d in bins.items():
        if not d["e_range"]:
            out[b] = {"n": 0}
            continue
        er, ec, rel, dv = (np.array(d[k]) for k in ("e_range", "e_cross", "rel", "dv"))
        foot, cen = np.array(d.get("d_foot", [])), np.array(d.get("d_cen", []))
        out[b] = {"n": len(er), "range_bias_m": float(np.median(er)), "range_mae_m": float(np.abs(er).mean()),
                  "range_p90_m": float(np.percentile(np.abs(er), 90)), "range_rel_median": float(np.median(rel)),
                  "cross_mae_m": float(np.abs(ec).mean()), "speed_mae_kmh": float(np.abs(dv).mean()),
                  "speed_median_abs_kmh": float(np.median(np.abs(dv))),
                  **({"foot_median_m": float(np.median(foot)), "foot_p90_m": float(np.percentile(foot, 90)),
                      "inside_frac": float((foot < 0.05).mean()), "centre_mae_m": float(cen.mean())}
                     if len(foot) else {})}
    return out


def eval_clip(clip_id: str, variant: str = "dense", scale: float = 1.0) -> dict:
    from tsf.ingest import vipe
    from tsf.ingest.npc import npc_tracks
    run = WORK / clip_id / variant
    z = np.load(run / "track.npz")
    raw = vipe.ego_track(z["poses"], z["t"])
    ego = vipe.despike(raw, z["t"])
    gt = Gt(clip_id, len(z["t"]))
    drops: dict = {}
    tracks = npc_tracks(run, ego, z["t"], scale=scale, drops=drops)
    tm = track_metrics(gt, tracks, ego)
    (WORK / clip_id / f"eval_{variant}_samples.json").write_text(json.dumps(tm["bins"]))
    res = {"clip": clip_id, "frames": len(z["t"]), "ego_raw": ego_metrics(gt, raw),
           "ego_despiked": ego_metrics(gt, ego), "n_gt_tracks": len(gt.objs), "drops": drops,
           "tracks": {k: v for k, v in tm.items() if k != "bins"}, "range_bins": summarize_bins(tm["bins"]),
           "depth": depth_metrics(gt, run), "stats": json.loads((run / "w00" / "stats.json").read_text())}
    (WORK / clip_id / f"eval_{variant}.json").write_text(json.dumps(res, indent=1))
    return res


def eval_ego(clip_id: str, variant: str) -> dict:
    """Ego metrics (raw and despiked) of one run directory, with its runtime and memory."""
    from tsf.ingest import vipe
    run = WORK / clip_id / variant
    z = np.load(run / "track.npz")
    raw = vipe.ego_track(z["poses"], z["t"])
    gt = Gt(clip_id, len(z["t"]))
    st = json.loads((run / "w00" / "stats.json").read_text())
    return {"clip": clip_id[:8], "variant": variant, "raw": ego_metrics(gt, raw),
            "despiked": ego_metrics(gt, vipe.despike(raw, z["t"])),
            **{k: st.get(k) for k in ("pipeline_s", "infer_fps", "gpu_peak_reserved_gb", "rss_peak_gb", "height",
                                      "override", "kf_gap_s")}, "focal_px": st["intrinsics"][0],
            "camera_height_m": st["camera_height"].get("median_m")}


def aggregate(clips: list[str], variant: str) -> dict:
    """Pooled numbers over clips: ego metrics per clip and mean, NPC error per range bin from the pooled samples,
    track / frame recall, dense-depth AbsRel pooled by sample count. These are the only figures that leave thor."""
    ev = [json.loads((WORK / c / f"eval_{variant}.json").read_text()) for c in clips]
    pooled = {b: {k: [] for k in SAMPLE_KEYS} for b in (f"{lo}-{hi}" for lo, hi in BINS)}
    for c in clips:
        for b, d in json.loads((WORK / c / f"eval_{variant}_samples.json").read_text()).items():
            for k in d:
                pooled[b][k] += d[k]
    rec = {b: [0, 0] for b in pooled}
    for e in ev:
        for b, (m, n) in e["tracks"]["frame_recall_bins"].items():
            rec[b][0] += m
            rec[b][1] += n
    keys = ("speed_mae_kmh", "heading_err_end_deg", "heading_err_mean_abs_deg", "ate_start_rmse_m", "final_err_m",
            "path_ratio", "ate_sim2_rmse_m", "sim2_scale", "gt_speed_mean_kmh", "gt_heading_change_deg")
    out = {"variant": variant, "clips": len(clips),
           "ego": {k: {"per_clip": [round(e["ego_despiked"][k], 2) for e in ev],
                       "mean": round(float(np.mean([e["ego_despiked"][k] for e in ev])), 2)} for k in keys},
           "ego_raw_speed_mae_kmh": [round(e["ego_raw"]["speed_mae_kmh"], 2) for e in ev],
           "npc_range_bins": summarize_bins(pooled), "frame_recall": {b: [m, n, round(m / max(n, 1), 3)]
                                                                     for b, (m, n) in rec.items()},
           "track_recall": [f"{round(e['tracks']['track_recall'] * e['tracks']['n_gt_eligible'])}/"
                            f"{e['tracks']['n_gt_eligible']}" for e in ev],
           "tracks_pred": [e["tracks"]["n_pred"] for e in ev], "precision_vs_gt_labels": [
               round(e["tracks"]["precision_vs_gt_labels"], 2) for e in ev]}
    dep: dict = {}
    for kind in ("road", "vehicle_corner"):
        for b in pooled:
            n = [e["depth"][kind][b].get("n", 0) for e in ev]
            if sum(n):
                med = [e["depth"][kind][b].get("absrel_median") or 0 for e in ev]
                w10 = [e["depth"][kind][b].get("within_10pct") or 0 for e in ev]
                dep.setdefault(kind, {})[b] = {
                    "n": sum(n), "absrel_median": round(float(np.average(med, weights=n)), 3),
                    "within_10pct": round(float(np.average(w10, weights=n)), 3)}
    out["depth"] = dep
    out["focal_est_px"] = [round(e["depth"]["focal_est_px"], 1) for e in ev]
    out["camera_height_depth_m"] = [round(e["stats"]["camera_height"]["median_m"], 2) for e in ev]
    out["runtime_s"] = [round(e["stats"]["pipeline_s"] + e["stats"].get("dense_s", 0), 0) for e in ev]
    out["gpu_peak_gb"] = [round(e["stats"]["gpu_peak_reserved_gb"], 1) for e in ev]
    out["host_rss_gb"] = [round(e["stats"]["rss_peak_gb"], 1) for e in ev]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "eval", "ego", "report"])
    ap.add_argument("clips", nargs="+")
    ap.add_argument("--variant", default="dense", help="run directory under pai/<clip>/ (dense, dense_f)")
    a = ap.parse_args()
    for c in a.clips:
        if a.cmd == "prep":
            print(prep(c))
        elif a.cmd == "eval":
            print(json.dumps(eval_clip(c, a.variant)))
        elif a.cmd == "ego":
            for v in a.variant.split(","):
                print(json.dumps(eval_ego(c, v)))
    if a.cmd == "report":
        print(json.dumps(aggregate(a.clips, a.variant), indent=1))


if __name__ == "__main__":
    main()
