"""Dashcam video -> metric camera poses (NVIDIA ViPE) -> planar ego track, for UC2 (video -> ego motion).

``run_vipe`` trims/rescales the clip and runs ViPE's ``pose_only`` pipeline (GeoCalib intrinsics, DROID-style
SLAM, MoGe-2 metric keyframe depth) in the ``tsf-vipe`` docker image built by ``deploy/vipe/setup.sh``
(override with ``TSF_VIPE_IMAGE``; weight cache ``TSF_VIPE_CACHE``, default ``~/tsf/dev/vipe/cache``).
Everything stays on the host.

Conventions and assumptions
    * ViPE poses are camera-to-world, OpenCV camera axes (x right, y down, z forward), origin = first frame.
    * Translation is metric: SLAM scale is anchored by MoGe-2's metric keyframe depth, so ``ego_track`` does
      not rescale by default. ``camera_height`` in the run result is the camera height above the road
      re-estimated from ViPE's own point maps; ``metric_scale`` turns it into a correction factor against a
      prior camera height (``CAM_HEIGHT_M`` = 1.4 m, windshield dashcam 1.2-1.6 m) for ``ego_track(scale=)``.
    * Gravity is not observed: it is the mean camera y axis (camera roll about the driving direction ~ 0)
      with the component along the direction of travel removed (the vehicle moves tangent to the ground, so
      camera pitch drops out). The camera optical axis is taken as vehicle heading (no yaw offset).
    * The track is planar: x forward at t0, y left, yaw counter-clockwise (left turn > 0), metres/radians.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

IMAGE = os.environ.get("TSF_VIPE_IMAGE", "tsf-vipe:latest")
CACHE = Path(os.environ.get("TSF_VIPE_CACHE", "~/tsf/dev/vipe/cache")).expanduser()
WORKER = os.environ.get("TSF_VIPE_WORKER")      # dev: bind-mount a local vipe_worker.py over the image's copy
CAM_HEIGHT_M = 1.4


def mem_available_gb() -> float:
    """Host MemAvailable in GB (thor3 shares 122 GB unified memory with vLLM/Cosmos/REFe)."""
    with open("/proc/meminfo") as f:
        return next(int(ln.split()[1]) for ln in f if ln.startswith("MemAvailable:")) / 2**20


def _window(video: Path, out: Path, start_s: float, dur_s: float | None, fps: float | None,
            height: int | None, gpu_mem_gb: float, mem_gb: float, pipeline: str, dynamic_masks: bool,
            intrinsics: str) -> dict:
    """One capped container run on ``video[start_s : start_s + dur_s]``; returns poses, intrinsics, stats."""
    out.mkdir(parents=True, exist_ok=True)
    cmd = ["docker", "run", "--rm", "--runtime", "nvidia", "--gpus", "all", "--ipc", "host",
           "--memory", f"{mem_gb:g}g", "--memory-swap", f"{mem_gb:g}g",
           "--user", f"{os.getuid()}:{os.getgid()}", "-e", "HOME=/tmp",
           "-v", f"{video.parent}:/in:ro", "-v", f"{out}:/work", "-v", f"{CACHE}:/cache",
           *(["-v", f"{Path(WORKER).resolve()}:/opt/tsf/vipe_worker.py:ro"] if WORKER else []), IMAGE,
           "python", "/opt/tsf/vipe_worker.py", f"/in/{video.name}", "--out", "/work", "--pipeline", pipeline,
           "--start-s", str(start_s), "--gpu-mem-gb", str(gpu_mem_gb), "--intrinsics", intrinsics]
    for flag, val in (("--max-seconds", dur_s), ("--fps", fps), ("--height", height)):
        if val:
            cmd += [flag, str(val)]
    if not dynamic_masks:
        cmd.append("--no-instance")
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    runtime = time.perf_counter() - t0
    log = proc.stdout + proc.stderr
    (out / "vipe_run.log").write_text(log)
    if proc.returncode:
        tail = "\n".join(log.splitlines()[-25:])
        raise RuntimeError(f"ViPE failed ({proc.returncode}):\n{tail}")
    stats = json.loads((out / "stats.json").read_text())
    poses = np.load(out / "vipe" / "pose" / "clip.npz")["data"].astype(np.float64)
    fx, fy, cx, cy = np.load(out / "vipe" / "intrinsics" / "clip.npz")["data"][0][:4].astype(float)
    return {"poses": poses, "runtime_s": runtime, "stats": stats, "start_s": start_s,
            "clip": out / "clip.mp4",
            "intrinsics": {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "width": stats["width"],
                           "height": stats["height"], "model": "pinhole"}}


def run_vipe(video: str | Path, out_dir: str | Path, max_seconds: float | None = None,
             fps: float | None = None, *, start_s: float = 0.0, height: int | None = 480,
             gpu_mem_gb: float = 16.0, mem_gb: float = 26.0, window_s: float = 20.0, overlap_s: float = 2.0,
             min_avail_gb: float = 30.0, wait_s: float = 900.0, pipeline: str = "pose_only",
             dynamic_masks: bool = True, intrinsics: str = "moge_fixed") -> dict:
    """Run ViPE on ``video[start_s : start_s + max_seconds]`` (resampled to ``fps``, capped at ``height`` px).

    ViPE holds a whole clip in host RAM (~10 MB/frame at 480p), so the span is processed in windows of
    ``window_s`` s (``overlap_s`` s overlap) in containers capped at ``mem_gb`` host / ``gpu_mem_gb`` GPU
    memory; windows are chained through the overlap (pose of the previous window at the overlap start).
    A launch waits up to ``wait_s`` s for ``min_avail_gb`` free host memory, then raises ``RuntimeError``.
    ``dynamic_masks`` keeps ViPE's GroundingDINO/SAM masking of vehicles, people and sky out of the SLAM.
    ``intrinsics`` picks the focal init: "moge_fixed" (MoGe-2 FOV, median over the window, not optimised in
    the bundle adjustment; default), "moge" (same but optimised) or "geocalib" (ViPE default, which drifted to
    an implausible focal / 10 m camera height on near-static dashcam clips).
    Returns ``poses`` (N,4,4) camera-to-world, ``t`` (N,) video time in s, ``intrinsics`` (first window; fx,
    fy, cx, cy, width, height in clip pixels, pinhole), ``fps``, ``runtime_s`` (wall time incl. trimming,
    model loading, memory waits), ``frames``, ``camera_height`` (m, median over windows), ``windows`` (per
    window clip path, start and worker stats incl. GPU/host peaks). Raises ``RuntimeError`` on failure.
    """
    video, out_dir = Path(video).resolve(), Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    t_start, t_end = time.perf_counter(), start_s + max_seconds if max_seconds else float("inf")
    wins, poses, t, s = [], np.zeros((0, 4, 4)), np.zeros(0), start_s
    while True:
        deadline = time.perf_counter() + wait_s
        while mem_available_gb() < min_avail_gb:
            if time.perf_counter() > deadline:
                raise RuntimeError(f"host MemAvailable {mem_available_gb():.0f} GB < {min_avail_gb:g} GB")
            time.sleep(30)
        dur = min(window_s, t_end - s)
        w = _window(video, out_dir / f"w{len(wins):02d}", s, dur if dur < float("inf") else None, fps, height,
                    gpu_mem_gb, mem_gb, pipeline, dynamic_masks, intrinsics)
        wins.append(w)
        wt = s + np.arange(len(w["poses"])) / w["stats"]["fps"]
        if len(poses):             # chain: previous world pose at the overlap start = this window's frame 0
            j = int(np.argmin(np.abs(t - s)))
            w_poses = poses[j] @ np.linalg.inv(w["poses"][0]) @ w["poses"]
            new = wt > t[-1] + 0.5 / w["stats"]["fps"]
            poses, t = np.concatenate([poses, w_poses[new]]), np.concatenate([t, wt[new]])
        else:
            poses, t = w["poses"], wt
        if len(w["poses"]) / w["stats"]["fps"] < dur - 0.5 or s + dur >= t_end - 1e-6:
            break                  # end of video or requested span covered
        s += window_s - overlap_s
    heights = [x["stats"]["camera_height"]["median_m"] for x in wins if x["stats"]["camera_height"].get("n")]
    res = {"poses": poses, "t": t, "fps": wins[0]["stats"]["fps"], "runtime_s": time.perf_counter() - t_start,
           "frames": len(poses), "intrinsics": wins[0]["intrinsics"],
           "camera_height": float(np.median(heights)) if heights else None,
           "windows": [{"start_s": x["start_s"], "clip": x["clip"], "runtime_s": x["runtime_s"],
                        "stats": x["stats"]} for x in wins]}
    np.savez(out_dir / "track.npz", poses=poses, t=t)
    return res


def metric_scale(height_est: float | None, height: float = CAM_HEIGHT_M, limit: float = 2.0) -> float:
    """Scale taking the estimated camera height to the prior ``height`` (1 if unknown), in 1/limit..limit."""
    if not height_est or height_est <= 0:
        return 1.0
    return float(np.clip(height / height_est, 1 / limit, limit))


def _up_axis(R: np.ndarray, p: np.ndarray, lam: float = 0.02) -> np.ndarray:
    """World-frame unit 'up': mean camera down axis, made orthogonal to the observed travel directions."""
    d0 = R[:, :, 1].mean(0)
    d0 /= np.linalg.norm(d0)
    step = np.diff(p, axis=0)
    n = np.linalg.norm(step, axis=1)
    moving = n > 0.25 * np.percentile(n, 90) if len(n) and n.max() > 0 else np.zeros(len(n), bool)
    if not moving.any():
        return -d0
    u = step[moving] / n[moving, None]
    d = np.linalg.solve(u.T @ u / len(u) + lam * np.eye(3), d0)
    return -d / np.linalg.norm(d)


def ego_track(poses: np.ndarray, t: np.ndarray | None = None, scale: float = 1.0,
              smooth_s: float = 0.5) -> np.ndarray:
    """Planar ego track ``[x, y, yaw, v]`` (N,4) from camera-to-world ``poses`` (N,4,4).

    x forward / y left in metres in the gravity-aligned frame of frame 0, yaw in radians (unwrapped, left
    > 0), v the ground speed in m/s from a centred difference over ``smooth_s`` seconds (m per frame if
    ``t`` is None). ``scale`` multiplies all translations (poses are metric already; see ``metric_scale``).
    """
    poses = np.asarray(poses, float)
    n = len(poses)
    t = np.arange(n, dtype=float) if t is None else np.asarray(t, float)
    R, p = poses[:, :3, :3], poses[:, :3, 3] * scale
    up = _up_axis(R, p)
    f = R[:, :, 2] - (R[:, :, 2] @ up)[:, None] * up
    f /= np.linalg.norm(f, axis=1, keepdims=True)
    left = np.cross(up, f[0])
    rel = p - p[0]
    xy = np.stack([rel @ f[0], rel @ left], axis=1)
    yaw = np.unwrap(np.arctan2(f @ left, f @ f[0]))
    h = max(1, round(smooth_s / np.median(np.diff(t))) // 2) if n > 1 else 1
    i = np.arange(n)
    lo, hi = np.clip(i - h, 0, n - 1), np.clip(i + h, 0, n - 1)
    dt = np.where(hi > lo, t[hi] - t[lo], 1.0)
    v = np.linalg.norm(xy[hi] - xy[lo], axis=1) / dt
    return np.column_stack([xy, yaw, v])


def _slew(v: np.ndarray, dt: np.ndarray, a_max: float) -> np.ndarray:
    out = v.copy()
    for k in range(1, len(v)):
        out[k] = np.clip(v[k], out[k - 1] - a_max * dt[k - 1], out[k - 1] + a_max * dt[k - 1])
    return out


def despike(track: np.ndarray, t: np.ndarray, a_max: float = 6.0, kappa_max: float = 0.2,
            alat_max: float = 8.0) -> np.ndarray:
    """Remove SLAM glitches a car cannot drive from an ``ego_track`` result (N,4) ``[x, y, yaw, v]``.

    Speed is slew-limited to ``a_max`` m/s2 (minimum of a forward and a backward pass, so a spike collapses to
    a small bump); the yaw rate to what that speed allows, |yaw'| <= min(kappa_max v, alat_max / v) (5 m
    turn radius, 8 m/s2 lateral), so a stopped car cannot turn. A step whose raw speed the slew limit rejected
    is a tracking glitch, so its heading change is dropped as well. Steps where any limit acted are re-integrated
    from the gated speed and heading; the others keep ViPE's geometry, rotated by the accumulated heading
    correction (a glitch's spurious rotation does not persist into the rest of the track)."""
    t = np.asarray(t, float)
    x, y, yaw, v = np.asarray(track, float).T
    if len(t) < 3:
        return np.asarray(track, float).copy()
    dt = np.maximum(np.diff(t), 1e-3)
    vf = np.minimum(_slew(v, dt, a_max), _slew(v[::-1], dt[::-1], a_max)[::-1])
    vm = 0.5 * (vf[1:] + vf[:-1])
    step = np.diff(np.column_stack([x, y]), axis=0)
    raw_v = np.linalg.norm(step, axis=1) / dt
    jump = np.abs(raw_v - vm) > 0.5 + 0.2 * vm
    lim = np.minimum(kappa_max * vm, alat_max / np.maximum(vm, 0.1)) * dt + np.radians(0.5) * dt
    dyaw = np.diff(yaw)
    dg = np.where(jump, 0.0, np.clip(dyaw, -lim, lim))
    yg = yaw[0] + np.concatenate([[0.0], np.cumsum(dg)])
    glitch = jump | (np.abs(dyaw - dg) > 1e-9)
    c = yg[1:] - yaw[1:]                                    # accumulated heading correction
    rot = np.stack([step[:, 0] * np.cos(c) - step[:, 1] * np.sin(c),
                    step[:, 0] * np.sin(c) + step[:, 1] * np.cos(c)], 1)
    hm = 0.5 * (yg[1:] + yg[:-1])
    integ = (vm * dt)[:, None] * np.stack([np.cos(hm), np.sin(hm)], 1)
    xy = np.vstack([[x[0], y[0]], [x[0], y[0]] + np.cumsum(np.where(glitch[:, None], integ, rot), axis=0)])
    return np.column_stack([xy, yg, vf])
