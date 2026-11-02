"""UC2 ego motion, speed from Cosmos3-Edge inverse dynamics and heading from ViPE.

ViPE's metric translation collapses on highway-speed footage (PhysicalAI-AV windows at 55-86 km/h: speed MAE
21-47 km/h; the YouTube dashcam clips: 0.8 m in 20 s on a 100 km/h road), while its rotation stays good (yaw-rate MAE
0.63 deg/s). The inverse-dynamics model keeps the speed (median MAE 4.1 km/h, p90 8.4) but turns worse (1.4 deg/s).
``fuse_ego`` keeps the ViPE yaw and puts the Cosmos speed along it: on 12 ground-truth windows speed MAE median 3.9 /
p90 8.5 km/h, 1 s RPE 1.4 m (ViPE alone 5.7 / 40.8 km/h, 1.8 m).

``idm_speed`` cuts 6 s windows (61 frames, 10 fps) from the video and asks a running ``tsf-cosmos-idm`` server; windows
already computed (this run's ``idm/`` folder, ``TSF_IDM_CACHE``) are reused. Without cache or server it raises, and the
caller falls back to ViPE alone. ``TSF_IDM_AUTOSTART=1`` lets it start (and stop) the container itself when
``tsf.cosmos.idm.free_to_start`` allows; it is a GPU-heavy job, schedule it like one.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

WINDOW_S = 6.0
CACHE = Path(os.environ.get("TSF_IDM_CACHE", "~/tsf/dev/idm/cosmos/edge")).expanduser()


def window_starts(duration_s: float, start_s: float = 0.0, total_s: float | None = None,
                  window_s: float = WINDOW_S) -> list[float]:
    """Start times of the windows covering ``[start_s, start_s + duration_s]``; those that would run past the end of the
    video (``total_s``) are pulled back so that they end with it."""
    out: list[float] = []
    for k in range(max(1, int(np.ceil(duration_s / window_s - 1e-6)))):
        s = start_s + k * window_s
        if total_s is not None:
            s = min(s, max(0.0, total_s - window_s - 0.1))
        if not out or abs(s - out[-1]) > 1e-6:
            out.append(s)
    return out


def _cached(stem: str, s: float, dirs: list[Path]) -> Path | None:
    names = [f"{stem}@{s:g}.json", f"{stem}_{int(s):02d}.s0.json", f"{stem.rsplit('_', 1)[0]}_{int(s):02d}.s0.json"]
    return next((d / n for d in dirs for n in names if (d / n).is_file()), None)


def _reachable(base_url: str) -> bool:
    import httpx
    try:
        return httpx.get(f"{base_url}/v1/models", timeout=3).status_code == 200
    except httpx.HTTPError:
        return False


def idm_speed(video, start_s: float, duration_s: float, workdir, *, base_url: str | None = None,
              cache_dirs: tuple = (), conv: dict | None = None) -> dict:
    """Ego speed profile of ``[start_s, start_s + duration_s]`` from inverse-dynamics windows.

    Returns ``t`` (video time, s, 0.1 s steps), ``v`` (m/s, despiked, calibrated ``tsf.cosmos.idm.CONVENTION``),
    ``windows``, ``cached`` (windows reused), ``latency_s`` (server time per computed window). Raises when a window is
    neither cached nor computable (no server)."""
    import cv2

    from tsf.cosmos import idm
    video, wd = Path(video), Path(workdir)
    wd.mkdir(parents=True, exist_ok=True)
    base_url = base_url or idm.DEFAULT_URL
    dirs = [wd, *map(Path, cache_dirs), CACHE]
    cap = cv2.VideoCapture(str(video)) if video.is_file() else None
    fps = (cap.get(cv2.CAP_PROP_FPS) or 30.0) if cap else 30.0
    total = (cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps) if cap and cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else None
    if cap:
        cap.release()
    samples: dict[int, float] = {}
    n_cached, lat, started = 0, [], False
    try:
        for s in window_starts(duration_s, start_s, total):
            hit = _cached(video.stem, s, dirs)
            if hit is not None:
                action, n_cached = np.asarray(json.loads(hit.read_text())["action"], float), n_cached + 1
            else:
                if not _reachable(base_url):
                    if os.environ.get("TSF_IDM_AUTOSTART") != "1" or idm.free_to_start():
                        raise RuntimeError(f"no cached inverse-dynamics window at {s:g} s and no server at {base_url}")
                    idm.serve()
                    started = True
                    idm.wait_ready(base_url)
                clip = wd / f"{video.stem}@{s:g}.mp4"
                idm.window_video(video, clip, round(s * fps), src_fps=fps)
                t0 = time.perf_counter()
                res = idm.inverse_dynamics(clip, base_url=base_url)
                lat.append(time.perf_counter() - t0)
                action = res["action"]
                (wd / f"{video.stem}@{s:g}.json").write_text(
                    json.dumps({"action": action.tolist(), "latency_s": lat[-1]}))
                clip.unlink(missing_ok=True)
            v = idm.ego_track_from_actions(action, conv, despike=True)[:, 3]
            for k, x in enumerate(v):
                samples[round((s + k / idm.FPS) * 10)] = float(x)
    finally:
        if started:
            idm.stop()
    keys = sorted(samples)
    return {"t": np.array(keys) / 10.0, "v": np.array([samples[k] for k in keys]), "windows": len(window_starts(
        duration_s, start_s, total)), "cached": n_cached, "latency_s": lat}


def fuse_ego(track: np.ndarray, t: np.ndarray, t_idm: np.ndarray, v_idm: np.ndarray) -> np.ndarray:
    """Planar ego track ``[x, y, yaw, v]`` with the yaw of ``track`` (ViPE) and the speed ``v_idm`` (m/s at ``t_idm``,
    held beyond its ends) integrated along that heading from the first position of ``track``."""
    track, t = np.asarray(track, float), np.asarray(t, float)
    v = np.interp(t, t_idm, v_idm)
    yaw = track[:, 2]
    vm, ym, dt = 0.5 * (v[1:] + v[:-1]), 0.5 * (yaw[1:] + yaw[:-1]), np.diff(t)
    step = np.column_stack([vm * np.cos(ym) * dt, vm * np.sin(ym) * dt])
    xy = track[0, :2] + np.vstack([np.zeros(2), np.cumsum(step, axis=0)])
    return np.column_stack([xy, yaw, v])
