"""UC2 plausibility run without ground truth: ViPE on dashcam clips -> ego-track sanity numbers.

    PYTHONPATH=~/tsf/dev/vipe/pkg python check_clips.py ~/data/youtube/*.mp4 [--max-seconds 20] [--fps 15]

Writes <out>/<stem><tag>/report.json (aggregates only) next to ViPE's raw outputs; prints one JSON line per
clip. Windowed, memory-capped runs: see ``tsf.ingest.vipe.run_vipe``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tsf.ingest.vipe import ego_track, metric_scale, run_vipe

SPEED_BAND = (3.0, 25.0)       # m/s mean-speed band for urban/suburban dashcam footage (~11-90 km/h)
HEIGHT_BAND = (1.0, 1.9)       # plausible windshield dashcam height above the road, m


def road_flow(clip: Path, start_s: float) -> dict[int, float]:
    """Median Farneback flow (px/frame at 640 px width) in the road region, per video second."""
    import cv2

    cap, prev, per, i = cv2.VideoCapture(str(clip)), None, {}, 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    while (ok := cap.read())[0]:
        g = cv2.cvtColor(cv2.resize(ok[1], (640, 360)), cv2.COLOR_BGR2GRAY)
        if prev is not None and i % 2 == 0:
            fl = cv2.calcOpticalFlowFarneback(prev, g, None, 0.5, 3, 15, 3, 5, 1.2, 0)[210:330, 100:540]
            mag = float(np.median(np.hypot(fl[..., 0], fl[..., 1])))
            per.setdefault(int(start_s + i / fps), []).append(mag)
        prev, i = g, i + 1
    return {k: float(np.median(v)) for k, v in per.items()}


def rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def report(res: dict) -> dict:
    tr = ego_track(res["poses"], res["t"])
    t, v, yaw = res["t"], tr[:, 3], tr[:, 2]
    dur = float(t[-1] - t[0])
    path = float(np.linalg.norm(np.diff(tr[:, :2], axis=0), axis=1).sum())
    yaw_rate = np.gradient(yaw, t)
    h, intr, wins = res["camera_height"], res["intrinsics"], res["windows"]
    flow: dict[int, float] = {}
    for w in wins:
        flow.update(road_flow(w["clip"], w["start_s"]))
    secs = sorted(k for k in flow if t[0] <= k < t[-1])
    v_sec = np.array([v[(t >= k) & (t < k + 1)].mean() for k in secs])
    f_sec = np.array([flow[k] for k in secs])
    stats = [w["stats"] for w in wins]
    return {
        "frames": res["frames"], "windows": len(wins), "duration_s": dur, "fps": res["fps"],
        "runtime_s": res["runtime_s"], "infer_fps": float(np.mean([s["infer_fps"] for s in stats])),
        "model_load_s": float(np.mean([s["model_load_s"] for s in stats])),
        "gpu_peak_reserved_gb": max(s["gpu_peak_reserved_gb"] for s in stats),
        "rss_peak_gb": max(s["rss_peak_gb"] for s in stats),
        "path_m": path, "net_m": float(np.hypot(*tr[-1, :2])), "mean_speed_ms": path / dur,
        "speed_kmh_p5_p50_p95_max": [float(x * 3.6) for x in (*np.percentile(v, [5, 50, 95]), v.max())],
        "mean_speed_in_band": bool(SPEED_BAND[0] <= path / dur <= SPEED_BAND[1]),
        "path_band_m": [SPEED_BAND[0] * dur, SPEED_BAND[1] * dur],
        "yaw_total_deg": float(np.degrees(yaw[-1])),
        "yaw_range_deg": [float(np.degrees(yaw.min())), float(np.degrees(yaw.max()))],
        "max_yaw_rate_dps": float(np.degrees(np.abs(yaw_rate).max())),
        "max_lat_acc_ms2": float(np.abs(v * yaw_rate).max()),
        "forward_fraction": float((np.diff(tr[:, 0]) > 0).mean()),
        "speed_by_second_kmh": [round(float(x) * 3.6, 1) for x in v_sec],
        "road_flow_by_second_px": [round(float(x), 2) for x in f_sec],
        "rank_corr_speed_vs_flow": rank_corr(v_sec, f_sec) if len(secs) > 3 else None,
        "intrinsics": intr, "hfov_deg": float(np.degrees(2 * np.arctan(intr["width"] / 2 / intr["fx"]))),
        "camera_height_m": h,
        "camera_height_by_window_m": [s["camera_height"].get("median_m") for s in stats],
        "camera_height_in_band": None if h is None else bool(HEIGHT_BAND[0] <= h <= HEIGHT_BAND[1]),
        "scale_to_1p4m": metric_scale(h),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=Path("~/tsf/dev/vipe/out").expanduser())
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--fps", type=float, default=None)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--no-masks", action="store_true")
    ap.add_argument("--intrinsics", default="moge_fixed", choices=["geocalib", "moge", "moge_fixed"])
    ap.add_argument("--tag", default="", help="suffix of the per-clip output dir")
    a = ap.parse_args()
    for video in a.videos:
        out = a.out / (video.stem + a.tag)
        try:
            res = run_vipe(video, out, max_seconds=a.max_seconds, fps=a.fps, height=a.height,
                           dynamic_masks=not a.no_masks, intrinsics=a.intrinsics)
        except RuntimeError as e:
            print(json.dumps({"clip": video.stem, "error": str(e)[:600]}), flush=True)
            continue
        rep = {"clip": video.stem, "intrinsics_init": a.intrinsics, "height_px": a.height, **report(res)}
        (out / "report.json").write_text(json.dumps(rep, indent=1))
        print(json.dumps(rep), flush=True)


if __name__ == "__main__":
    main()
