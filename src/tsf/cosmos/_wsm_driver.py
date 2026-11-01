"""Runs *inside the WSM venv* (needs torch / moderngl / hydra / cosmos_auto); never imports ``tsf``.

Renders one RDS-HQ clip through the framework's WSM v4 renderer (``cosmos_auto.wm.cli``), one camera at a
time so every camera gets its own wall time. Equivalent to

    cosmos-auto-wm input_root=... output_root=... version=v4 s3_spec=no_upload \
        camera_spec.camera_list='[<cam>]' input_fps_spec.input_pose_fps=<fps> resolution_spec=<res> \
        output_frame_spec.target_render_fps=<fps> output_frame_spec.target_chunk_frame=<N> \
        output_frame_spec.overlap_frame=0 output_frame_spec.max_chunk=1 clip_id_json=<clip> skip_resume=true

but calls the renderer implementation in-process (no Ray cluster): ``ray.is_initialized()`` is false there, so
the renderer opens EGL device 0 itself. Prints one JSON line (timings + projection check) on stdout.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

PROBES = {"ahead_20m": (20.0, 0.0, 0.8), "left_10m": (0.0, 10.0, 0.8), "right_10m": (0.0, -10.0, 0.8),
          "behind_15m": (-15.0, 0.0, 0.8), "ahead_60m": (60.0, 0.0, 0.8)}


def check_projection(input_root: str, clip_id: str, cameras: list[str]) -> dict:
    """Load the clip with the renderer's own reader + F-theta model and project probe points that are known in
    the vehicle frame (vehicle_to_world[0] is the identity, so frame-0 world == vehicle)."""
    import numpy as np
    from cosmos_auto.wm.camera.ftheta import FThetaCamera
    from cosmos_auto.wm.wds_utils import get_sample

    pose = get_sample(Path(input_root) / "pose" / f"{clip_id}.tar")
    intr = get_sample(Path(input_root) / "ftheta_intrinsic" / f"{clip_id}.tar")
    pts = np.array(list(PROBES.values()), float)
    out = {}
    for cam in cameras:
        P0 = pose[f"000000.pose.{cam}.npy"]
        cm = FThetaCamera.from_numpy(intr[f"ftheta_intrinsic.{cam}.npy"], device="cpu")
        xyd = np.asarray(cm.get_xy_and_depth(pts, P0))
        ok = (xyd[:, 2] > 0) & (xyd[:, 0] >= 0) & (xyd[:, 0] < cm.width) & (xyd[:, 1] >= 0) & (xyd[:, 1] < cm.height)
        out[cam] = {"size": [int(cm.width), int(cm.height)],
                    "probes_in_view": [k for k, v in zip(PROBES, ok) if v],
                    "uv_depth": {k: [round(float(a), 1) for a in xyd[i]] for i, k in enumerate(PROBES)}}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", required=True)
    ap.add_argument("--clip-id", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cameras", required=True, help="comma separated RDS-HQ camera names")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--chunk", type=int, default=297)
    ap.add_argument("--resolution", default="resize_1280x720")
    ap.add_argument("--timings", default=None, help="write the JSON result here as well")
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--extra", nargs="*", default=[], help="additional hydra overrides")
    a = ap.parse_args()
    cams = [c for c in a.cameras.split(",") if c]
    res: dict = {"clip_id": a.clip_id, "cameras": {}, "fps_out": a.fps, "chunk_frames": a.chunk}
    t_imp = time.perf_counter()
    import numpy as np  # noqa: F401
    res["projection_check"] = check_projection(a.input_root, a.clip_id, cams)
    if not a.check_only:
        from hydra import compose, initialize_config_dir

        import cosmos_auto.wm.cli as cli
        res["import_s"] = round(time.perf_counter() - t_imp, 2)
        conf = Path(cli.__file__).parent / "conf"
        for cam in cams:
            ov = [f"input_root={a.input_root}", f"output_root={a.out}", "version=v4", "s3_spec=no_upload",
                  f"camera_spec.camera_list=[{cam}]", f"input_fps_spec.input_pose_fps={a.fps}",
                  f"resolution_spec={a.resolution}", f"output_frame_spec.target_render_fps={a.fps}",
                  f"output_frame_spec.target_chunk_frame={a.chunk}", "output_frame_spec.overlap_frame=0",
                  "output_frame_spec.max_chunk=1", f"clip_id_json={a.clip_id}", "skip_resume=true",
                  "allow_short_final_chunk=true", *a.extra]
            with initialize_config_dir(config_dir=str(conf), version_base=None):
                cfg = compose(config_name="render", overrides=ov)
            t = time.perf_counter()
            try:
                cli._render_sample_hdmap_v4_impl(a.clip_id, cfg)
                err = None
            except Exception as e:  # keep going with the other cameras
                err = f"{type(e).__name__}: {e}"
                traceback.print_exc()
            dt = time.perf_counter() - t
            mp4 = Path(a.out) / "world_scenario" / f"ftheta_{cam}" / f"{a.clip_id}_0.mp4"
            res["cameras"][cam] = {"render_s": round(dt, 2), "fps_render": round(a.chunk / dt, 2) if dt > 0 else None,
                                   "mp4": str(mp4) if mp4.exists() else None, "error": err}
    js = json.dumps(res)
    if a.timings:
        Path(a.timings).write_text(js)
    print("WSM_RESULT " + js, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
