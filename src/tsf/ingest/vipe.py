"""Dashcam video -> metric camera poses (NVIDIA ViPE) -> planar ego track, for UC2 (video -> ego motion -> scenario).

``run_vipe`` trims/rescales the clip and runs ViPE's ``pose_only`` pipeline (GeoCalib intrinsics, DROID-style SLAM,
MoGe-2 keyframe depth) in the ``tsf-vipe`` docker image built by ``deploy/vipe/setup.sh`` (override with
``TSF_VIPE_IMAGE``; weight cache ``TSF_VIPE_CACHE``, default ``~/tsf/dev/vipe/cache``). Everything stays on the host.

Conventions and assumptions
    * ViPE poses are camera-to-world, OpenCV camera axes (x right, y down, z forward), world origin = first frame.
    * Translation is metric: the SLAM scale is anchored by MoGe-2's metric keyframe depth, so ``ego_track`` does not
      rescale by default. ``camera_height`` in the run result is the camera height above the road re-estimated from
      ViPE's own point maps; ``metric_scale`` turns it into a correction factor against a prior camera height
      (``CAM_HEIGHT_M`` = 1.4 m, windshield dashcam 1.2-1.6 m) that can be passed to ``ego_track(scale=...)``.
    * Gravity is not observed: it is the mean camera y axis (camera roll about the driving direction ~ 0), with the
      component along the direction of travel removed (the vehicle moves tangent to the ground, so any camera
      pitch drops out). The camera optical axis is taken as vehicle heading (no yaw offset).
    * The track is planar: x forward at t0, y left, yaw counter-clockwise (left turn > 0), all in metres/radians.
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
CAM_HEIGHT_M = 1.4


def run_vipe(video: str | Path, out_dir: str | Path, max_seconds: float | None = None, fps: float | None = None,
             *, start_s: float = 0.0, height: int | None = 720, gpu_mem_gb: float = 24.0,
             pipeline: str = "pose_only") -> dict:
    """Run ViPE on ``video[start_s : start_s + max_seconds]`` (resampled to ``fps`` and capped at ``height`` px).

    Returns ``poses`` (N,4,4) camera-to-world, ``t`` (N,) video time in s, ``intrinsics`` (fx, fy, cx, cy, width,
    height in clip pixels, pinhole), ``fps``, ``runtime_s`` (wall time incl. trimming and model loading),
    ``frames``, plus ``camera_height`` (m, from ViPE's point maps) and ``stats`` (worker timings, GPU peak).
    Raises ``RuntimeError`` with the container log tail on failure.
    """
    video, out_dir = Path(video).resolve(), Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    cmd = ["docker", "run", "--rm", "--runtime", "nvidia", "--gpus", "all", "--ipc", "host",
           "--user", f"{os.getuid()}:{os.getgid()}", "-e", "HOME=/tmp",
           "-v", f"{video.parent}:/in:ro", "-v", f"{out_dir}:/work", "-v", f"{CACHE}:/cache", IMAGE,
           "python", "/opt/tsf/vipe_worker.py", f"/in/{video.name}", "--out", "/work", "--pipeline", pipeline,
           "--start-s", str(start_s), "--gpu-mem-gb", str(gpu_mem_gb)]
    if max_seconds:
        cmd += ["--max-seconds", str(max_seconds)]
    if fps:
        cmd += ["--fps", str(fps)]
    if height:
        cmd += ["--height", str(height)]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    runtime = time.perf_counter() - t0
    (out_dir / "vipe_run.log").write_text(proc.stdout + proc.stderr)
    if proc.returncode:
        raise RuntimeError(f"ViPE failed ({proc.returncode}):\n" + "\n".join((proc.stdout + proc.stderr).splitlines()[-25:]))
    stats = json.loads((out_dir / "stats.json").read_text())
    poses = np.load(out_dir / "vipe" / "pose" / "clip.npz")["data"].astype(np.float64)
    fx, fy, cx, cy = np.load(out_dir / "vipe" / "intrinsics" / "clip.npz")["data"][0][:4].astype(float)
    t = start_s + np.arange(len(poses)) / stats["fps"]
    return {"poses": poses, "t": t, "fps": stats["fps"], "runtime_s": runtime, "frames": len(poses),
            "intrinsics": {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "width": stats["width"], "height": stats["height"],
                           "model": "pinhole"},
            "camera_height": stats["camera_height"].get("median_m"), "stats": stats}


def metric_scale(height_est: float | None, height: float = CAM_HEIGHT_M, limit: float = 2.0) -> float:
    """Scale factor taking an estimated camera height to the prior ``height`` (1.0 if unknown), clipped to 1/limit..limit."""
    if not height_est or height_est <= 0:
        return 1.0
    return float(np.clip(height / height_est, 1 / limit, limit))


def _up_axis(R: np.ndarray, p: np.ndarray, lam: float = 0.02) -> np.ndarray:
    """World-frame unit 'up': mean camera down axis, made orthogonal to the observed directions of travel."""
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


def ego_track(poses: np.ndarray, t: np.ndarray | None = None, scale: float = 1.0, smooth_s: float = 0.5) -> np.ndarray:
    """Planar ego track ``[x, y, yaw, v]`` (N,4) from camera-to-world ``poses`` (N,4,4).

    x forward / y left in metres in the gravity-aligned frame of frame 0, yaw in radians (unwrapped, left > 0), v the
    ground speed in m/s from a centred difference over ``smooth_s`` seconds (m per frame if ``t`` is None).
    ``scale`` multiplies all translations (poses are metric already; see ``metric_scale``).
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
    h = max(1, int(round(smooth_s / np.median(np.diff(t)))) // 2) if n > 1 else 1
    i = np.arange(n)
    lo, hi = np.clip(i - h, 0, n - 1), np.clip(i + h, 0, n - 1)
    dt = np.where(hi > lo, t[hi] - t[lo], 1.0)
    v = np.linalg.norm(xy[hi] - xy[lo], axis=1) / dt
    return np.column_stack([xy, yaw, v])
