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


def use_moge_fov(n: int = 7) -> None:
    """Initialise the pinhole focal length from MoGe-2's own FOV estimate (median over ``n`` frames spread
    over the clip) instead of GeoCalib's (first 2 s only). MoGe-2's metric depth is conditioned on the FOV,
    so a FOV consistent with its depth head keeps the metric scale coherent."""
    import vipe.pipeline.default as default
    from vipe.pipeline.processors import CachedVideoStream, IntrinsicEstimationProcessor
    from vipe.priors.depth import make_depth_model
    from vipe.utils.cameras import CameraType

    class MogeFovProcessor(IntrinsicEstimationProcessor):
        def __init__(self, video_stream, camera_type=CameraType.PINHOLE, model_cache=None):
            super().__init__(video_stream)
            moge = model_cache.get("depth/moge2-l", lambda: make_depth_model("moge2-l"))
            frames = CachedVideoStream(video_stream)
            fov = []
            for i in np.linspace(0, len(video_stream) - 1, n).astype(int):
                rgb = frames[int(i)].rgb.moveaxis(-1, 0)[None]
                with torch.no_grad():
                    out = moge.model.infer(rgb, fov_x=None, use_fp16=True, apply_mask=False,
                                           force_projection=False)
                fov.append(2 * np.arctan(0.5 / float(out["intrinsics"][0, 1, 1])))
            self.fov_y = float(np.median(fov))
            self.camera_type = camera_type

    default.GeoCalibIntrinsicsProcessor = MogeFovProcessor


def prep_clip(src: Path, dst: Path, start_s: float, max_seconds: float | None, fps: float | None,
              height: int | None) -> None:
    """Trim, resample and downscale ``src`` to an H.264 mp4 (ViPE moves full-resolution frames to the GPU)."""
    import imageio_ffmpeg

    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-ss", f"{start_s:.3f}",
           "-i", str(src)]
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


def ground_height(pc: np.ndarray, rng: np.random.Generator, min_inliers: int = 200
                  ) -> tuple[float, float] | None:
    """RANSAC road plane in camera-frame points (N,3; x right, y down, z forward) -> (height m, inlier ratio).

    Candidates are points 3-40 m ahead and below the camera; the plane normal must be near the camera y axis.
    """
    pc = pc[(pc[:, 2] > 3) & (pc[:, 2] < 40) & (pc[:, 1] > 0.3)]
    if len(pc) < 300:
        return None
    best, best_n = None, 0
    for _ in range(300):
        p = pc[rng.choice(len(pc), 3, replace=False)]
        n = np.cross(p[1] - p[0], p[2] - p[0])
        if (nn := np.linalg.norm(n)) < 1e-9:
            continue
        n = n / nn if n[1] > 0 else -n / nn
        if n[1] < 0.85:
            continue
        inl = np.abs(pc @ n - p[0] @ n) < 0.05 + 0.005 * p[:, 2].mean()
        if inl.sum() > best_n:
            best, best_n = inl, int(inl.sum())
    if best is None or best_n < min_inliers:
        return None
    q = pc[best]
    c0 = q.mean(0)
    n = np.linalg.svd(q - c0, full_matrices=False)[2][-1]
    return float(abs(c0 @ n)), best_n / len(pc)


def summarize(heights: list[float], ratios: list[float]) -> dict:
    if not heights:
        return {"n": 0}
    h = np.array(heights)
    return {"n": len(h), "median_m": float(np.median(h)), "p25_m": float(np.percentile(h, 25)),
            "p75_m": float(np.percentile(h, 75)), "inlier_ratio": float(np.mean(ratios))}


def camera_height_map(slam_map_path: Path, poses: np.ndarray, n_max: int = 16, seed: int = 0) -> dict:
    """Camera height above the road from ViPE's SLAM keyframe point map (multi-view-consistent points)."""
    m = torch.load(slam_map_path, weights_only=False, map_location="cpu")
    xyz = m["dense_disp_xyz"].numpy().astype(np.float64)
    pack = m["dense_disp_packinfo"][:, 0].numpy()
    inds = list(m["dense_disp_frame_inds"])
    rng = np.random.default_rng(seed)
    res = []
    for k in np.unique(np.linspace(0, len(inds) - 1, min(n_max, len(inds))).astype(int)):
        s, c = int(pack[k][0]), int(pack[k][1])
        T = poses[int(inds[k])]
        if (r := ground_height((xyz[s:s + c] - T[:3, 3]) @ T[:3, :3], rng)) is not None:
            res.append(r)
    return summarize([r[0] for r in res], [r[1] for r in res])


def camera_height_depth(pipe, clip: Path, intrinsics: np.ndarray, n: int = 8, seed: int = 0) -> dict:
    """Camera height above the road from MoGe-2 metric depth on ``n`` frames, unprojected with ViPE's K.

    Same prior that anchors the SLAM scale, conditioned on ViPE's focal length: a wrong focal or scale shows
    up as an implausible height (depth, hence height, scales with the focal length).
    """
    import cv2
    from vipe.priors.depth.base import DepthEstimationInput

    model = pipe.model_cache.get("depth/" + pipe.slam_cfg.keyframe_depth, lambda: None)
    fx, fy, cx, cy = (float(x) for x in intrinsics[:4])
    cap = cv2.VideoCapture(str(clip))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    rng = np.random.default_rng(seed)
    res = []
    for i in np.unique(np.linspace(0, total - 1, n).astype(int)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, bgr = cap.read()
        if not ok:
            continue
        rgb = torch.as_tensor(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).float().div(255).cuda()
        out = model.estimate(DepthEstimationInput(rgb=rgb, intrinsics=torch.tensor([fx, fy, cx, cy]).cuda()))
        z = out.metric_depth.float().cpu().numpy()
        v, u = np.mgrid[:z.shape[0], :z.shape[1]]
        ok = (z > 0)[::4, ::4] & (v[::4, ::4] > cy)
        z, u, v = z[::4, ::4][ok], u[::4, ::4][ok], v[::4, ::4][ok]
        if (r := ground_height(np.stack([(u - cx) / fx * z, (v - cy) / fy * z, z], 1), rng)) is not None:
            res.append(r)
    cap.release()
    return summarize([r[0] for r in res], [r[1] for r in res])


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
    ap.add_argument("--no-instance", action="store_true", help="skip object and sky masks")
    ap.add_argument("--intrinsics", choices=["geocalib", "moge", "moge_fixed"], default="geocalib",
                    help="focal init: GeoCalib (ViPE default) or MoGe-2; moge_fixed also freezes it in BA")
    a = ap.parse_args()

    a.out.mkdir(parents=True, exist_ok=True)
    clip = a.out / "clip.mp4"
    t0 = time.perf_counter()
    prep_clip(a.video, clip, a.start_s, a.max_seconds, a.fps, a.height)
    prep_s = time.perf_counter() - t0

    # Thor: cuDNN (CUDA 13.4 forward-compat) raises on batch-norm / transposed convs and silently returns
    # wrong convolution results (MoGe-2 depth came out near-constant); native kernels match the CPU reference.
    torch.backends.cudnn.enabled = False
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, a.gpu_mem_gb * 2**30 / total))

    from vipe import make_pipeline
    from vipe.config import parse_typed_config
    from vipe.streams.base import ProcessedVideoStream
    from vipe.streams.raw_mp4_stream import RawMp4Stream

    vipe_out = a.out / "vipe"
    args = parse_typed_config("default", hydra_args=[
        f"pipeline={a.pipeline}", f"pipeline.output.path={vipe_out}", "pipeline.output.save_artifacts=true",
        "pipeline.output.save_slam_map=true", "pipeline.output.save_viz=false",
        *(["pipeline.init.instance=null"] if a.no_instance else []),
        *(["pipeline.slam.optimize_intrinsics=false"] if a.intrinsics == "moge_fixed" else [])])
    if a.intrinsics != "geocalib":
        use_moge_fov()
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
        "pipeline": a.pipeline, "instance_masks": not a.no_instance,
        "intrinsics_init": a.intrinsics, "frames": int(n_frames), "fps": fps_out,
        "width": int(w), "height": int(h),
        "prep_s": prep_s, "read_s": t1 - t0, "pipeline_s": t2 - t1, "model_load_s": load_s[0],
        "infer_fps": n_frames / max(t2 - t1 - load_s[0], 1e-6),
        "gpu_peak_allocated_gb": torch.cuda.max_memory_allocated() / 2**30,
        "gpu_peak_reserved_gb": torch.cuda.max_memory_reserved() / 2**30,
        "gpu_cap_gb": a.gpu_mem_gb,
        "rss_peak_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
        "intrinsics": [float(x) for x in intr[0][:4]], "poses": list(pose.shape),
        "camera_height": camera_height_depth(pipe, clip, intr[0]),
        "camera_height_map": camera_height_map(vipe_out / "vipe" / "clip_slam_map.pt", pose),
    }
    (a.out / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
