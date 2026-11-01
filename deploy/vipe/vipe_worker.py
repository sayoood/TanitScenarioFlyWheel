"""Container-side ViPE runner: trim/rescale a clip, run the pose-only pipeline, write poses and diagnostics.

Copied to /opt/tsf/vipe_worker.py by the Dockerfile; driven by tsf.ingest.vipe.run_vipe.
Outputs (under --out): clip.mp4, vipe/pose/clip.npz, vipe/intrinsics/clip.npz, stats.json.
"""
from __future__ import annotations

import argparse
import json
import resource
import subprocess
import time
from pathlib import Path

import numpy as np
import torch


def prep_clip(src: Path, dst: Path, start_s: float, max_seconds: float | None, fps: float | None,
              height: int | None) -> None:
    """Trim, resample and downscale ``src`` into an H.264 mp4 (ViPE reads full-resolution frames to the GPU)."""
    import imageio_ffmpeg

    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-ss", f"{start_s:.3f}", "-i", str(src)]
    if max_seconds:
        cmd += ["-t", f"{max_seconds:.3f}"]
    vf = []
    if fps:
        vf.append(f"fps={fps}")
    if height:
        vf.append(f"scale=-2:'min({height},ih)'")
    if vf:
        cmd += ["-vf", ",".join(vf)]
    cmd += ["-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "14", "-pix_fmt", "yuv420p", str(dst)]
    subprocess.run(cmd, check=True)


def camera_height(slam_map_path: Path, poses: np.ndarray, n_max: int = 16, seed: int = 0) -> dict:
    """Camera height above the road from ViPE's own keyframe point maps (metric via the MoGe-2 depth prior).

    Per keyframe: RANSAC plane over points ahead of and below the camera whose normal is near the camera y axis;
    height = plane distance from the camera centre. Dynamic objects are already masked out of the map.
    """
    m = torch.load(slam_map_path, weights_only=False, map_location="cpu")
    xyz = m["dense_disp_xyz"].numpy().astype(np.float64)
    pack = m["dense_disp_packinfo"][:, 0].numpy()
    inds = list(m["dense_disp_frame_inds"])
    rng = np.random.default_rng(seed)
    heights, ratios = [], []
    for k in np.unique(np.linspace(0, len(inds) - 1, min(n_max, len(inds))).astype(int)):
        s, c = int(pack[k][0]), int(pack[k][1])
        T = poses[int(inds[k])]
        pc = (xyz[s:s + c] - T[:3, 3]) @ T[:3, :3]
        pc = pc[(pc[:, 2] > 3) & (pc[:, 2] < 30) & (pc[:, 1] > 0.3) & (np.abs(pc[:, 0]) < 0.5 * pc[:, 2] + 2)]
        if len(pc) < 300:
            continue
        best, best_n = None, 0
        for _ in range(300):
            p = pc[rng.choice(len(pc), 3, replace=False)]
            n = np.cross(p[1] - p[0], p[2] - p[0])
            if (nn := np.linalg.norm(n)) < 1e-9:
                continue
            n = n / nn
            if n[1] < 0:
                n = -n
            if n[1] < 0.85:
                continue
            inl = np.abs(pc @ n - p[0] @ n) < 0.1
            if inl.sum() > best_n:
                best, best_n = inl, int(inl.sum())
        if best is None or best_n < 200:
            continue
        q = pc[best]
        c0 = q.mean(0)
        n = np.linalg.svd(q - c0, full_matrices=False)[2][-1]
        if n[1] < 0:
            n = -n
        heights.append(float(c0 @ n))
        ratios.append(best_n / len(pc))
    if not heights:
        return {"n": 0}
    h = np.array(heights)
    return {"n": len(h), "median_m": float(np.median(h)), "p25_m": float(np.percentile(h, 25)),
            "p75_m": float(np.percentile(h, 75)), "inlier_ratio": float(np.mean(ratios))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("--out", type=Path, default=Path("/work"))
    ap.add_argument("--pipeline", default="pose_only")
    ap.add_argument("--start-s", type=float, default=0.0)
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--fps", type=float, default=None)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--gpu-mem-gb", type=float, default=24.0)
    a = ap.parse_args()

    a.out.mkdir(parents=True, exist_ok=True)
    clip = a.out / "clip.mp4"
    t0 = time.perf_counter()
    prep_clip(a.video, clip, a.start_s, a.max_seconds, a.fps, a.height)
    prep_s = time.perf_counter() - t0

    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, a.gpu_mem_gb * 2**30 / total))

    from vipe import make_pipeline
    from vipe.config import parse_typed_config
    from vipe.streams.base import ProcessedVideoStream
    from vipe.streams.raw_mp4_stream import RawMp4Stream

    vipe_out = a.out / "vipe"
    args = parse_typed_config("default", hydra_args=[
        f"pipeline={a.pipeline}", f"pipeline.output.path={vipe_out}", "pipeline.output.save_artifacts=true",
        "pipeline.output.save_slam_map=true", "pipeline.output.save_viz=false"])
    t0 = time.perf_counter()
    pipe = make_pipeline(args.pipeline)
    load_s = [0.0]
    get = pipe.model_cache.get

    def timed_get(key, builder):
        t = time.perf_counter()
        try:
            return get(key, builder)
        finally:
            load_s[0] += time.perf_counter() - t

    pipe.model_cache.get = timed_get
    torch.cuda.reset_peak_memory_stats()
    stream = ProcessedVideoStream(RawMp4Stream(clip), []).cache(desc="Reading video stream")
    n_frames, fps_out = len(stream), float(stream.fps())
    t1 = time.perf_counter()
    pipe.run(stream)
    torch.cuda.synchronize()
    t2 = time.perf_counter()

    pose = np.load(vipe_out / "pose" / "clip.npz")["data"]
    intr = np.load(vipe_out / "intrinsics" / "clip.npz")["data"]
    h, w = stream.frame_size()
    stats = {
        "pipeline": a.pipeline, "frames": int(n_frames), "fps": fps_out, "width": int(w), "height": int(h),
        "prep_s": prep_s, "read_s": t1 - t0, "pipeline_s": t2 - t1, "model_load_s": load_s[0],
        "infer_fps": n_frames / max(t2 - t1 - load_s[0], 1e-6),
        "gpu_peak_allocated_gb": torch.cuda.max_memory_allocated() / 2**30,
        "gpu_peak_reserved_gb": torch.cuda.max_memory_reserved() / 2**30,
        "gpu_cap_gb": a.gpu_mem_gb,
        "rss_peak_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
        "intrinsics": [float(x) for x in intr[0][:4]], "poses": list(pose.shape),
        "camera_height": camera_height(vipe_out / "vipe" / "clip_slam_map.pt", pose),
    }
    (a.out / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
