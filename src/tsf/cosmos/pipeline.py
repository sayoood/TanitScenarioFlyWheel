"""End-to-end: TSIR example -> TanitSim -> RDS-HQ -> WSM v4 control videos -> mosaic (+ BEV, PNG checks).

    python -m tsf.cosmos.pipeline examples/uc1_left_turn_intersection.json runs/wsm/uc1 [--t0 12.0] [--lidar]

Everything lands under ``out_dir``: ``rds_hq/`` (clip tars + ``tsf_export`` report), ``wsm/`` (per-camera MP4s,
``wsm_timings.json``, logs), ``bev.mp4`` (TanitSim bird's-eye view of the same window), ``mosaic.mp4``,
``frames/`` (PNG spot checks), ``rds_bev_*.png`` (map + objects + cameras drawn from the exported tars) and
``summary.json`` with all timings.
"""
from __future__ import annotations

import argparse
import io
import json
import tarfile
import time
from pathlib import Path

import numpy as np


def _read_tar(path: Path) -> dict:
    out = {}
    with tarfile.open(path) as tf:
        for m in tf:
            raw = tf.extractfile(m).read()
            key = m.name.split(".", 1)[1]
            out[key] = np.load(io.BytesIO(raw)) if key.endswith(".npy") else json.loads(raw)
    return out


def rds_bev_png(clip_root, clip_id: str, frame: int, png, half: float = 70.0, ppm: float = 9.0) -> Path:
    """Top-down drawing of one frame straight from the RDS-HQ tars (world frame, ego-centred): lane lines,
    boundaries, crosswalks, stop lines, signal heads (lit by state), objects, camera frusta. Independent of WSM,
    so it separates exporter errors from renderer errors."""
    import cv2
    root = Path(clip_root)
    rd = lambda comp: _read_tar(root / comp / f"{clip_id}.tar")   # noqa: E731
    pose = rd("pose")
    cams = sorted({k.split(".")[2] for k in pose})
    v2w = np.load(root / "tsf_export" / f"{clip_id}.vehicle_to_world.npy")[frame]   # rear-axle pose in the world frame
    cx, cy = v2w[0, 3], v2w[1, 3]
    S = int(2 * half * ppm)
    img = np.full((S, S, 3), (28, 28, 32), np.uint8)
    px = lambda p: np.round(((np.asarray(p)[..., :2] - [cx, cy]) * [1, -1] * ppm + S / 2) * 16).astype(np.int32)  # noqa: E731

    def poly(pts, col, closed=False, thick=1, fill=False):
        p = px(np.asarray(pts))
        if fill:
            cv2.fillPoly(img, [p], col, cv2.LINE_AA, shift=4)
        else:
            cv2.polylines(img, [p], closed, col, thick, cv2.LINE_AA, shift=4)

    for lab in rd("3d_crosswalks")["crosswalks.json"]["labels"]:
        poly(lab["labelData"]["shape3d"]["surface"]["vertices"], (139, 93, 255)[::-1], fill=True)
    for lab in rd("3d_road_boundaries")["road_boundaries.json"]["labels"]:
        poly(lab["labelData"]["shape3d"]["polyline3d"]["vertices"], (191, 255, 207)[::-1], thick=2)
    for lab in rd("3d_lanelines")["lanelines.json"]["labels"]:
        a = {x["name"]: x["enumsList"]["enumsList"][0] for x in lab["labelData"]["shape3d"]["attributes"]}
        col = (0, 255, 255) if a["colors"] == "YELLOW" else (255, 255, 255)
        v = np.asarray(lab["labelData"]["shape3d"]["polyline3d"]["vertices"])
        if "DASHED" in a["styles"]:
            for i in range(0, len(v) - 6, 18):          # coarse dashes
                poly(v[i:i + 6], col)
        else:
            poly(v, col, thick=1)
    for lab in rd("3d_wait_lines")["wait_lines.json"]["labels"]:
        poly(lab["labelData"]["shape3d"]["polyline3d"]["vertices"], (59, 179, 108), thick=3)
    for lab in rd("3d_road_markings")["road_markings.json"]["labels"]:
        v = np.asarray(lab["labelData"]["shape3d"]["surface"]["vertices"])
        poly(v, (244, 244, 244), closed=True)
        poly(np.array([v[2:4].mean(0), v[0:2].mean(0)]), (244, 244, 244), thick=2)     # rear -> front
    for lab in rd("3d_traffic_signs")["traffic_signs.json"]["labels"]:
        v = np.asarray(lab["labelData"]["shape3d"]["cuboid3d"]["vertices"])
        poly(v[[2, 3, 7, 6]][:, :2], (255, 2, 8), closed=True, thick=2)
    tl = rd("3d_traffic_lights")["traffic_lights.json"]["labels"]
    st = rd("3d_traffic_lights_lightnet_status")["aggregated_states.json"]["traffic_light_states"]
    for i, lab in enumerate(tl):
        v = np.asarray(lab["labelData"]["shape3d"]["cuboid3d"]["vertices"])
        s = st[str(i)]["state"][frame]
        col = {"red": (50, 50, 240), "yellow": (0, 200, 255), "green": (132, 240, 130)}.get(s.split("_")[0], (150, 150, 150))
        poly(v[[2, 3, 7, 6]][:, :2], col, closed=True, thick=4)
        c = px(v[:, :2].mean(0))
        cv2.putText(img, "L" if "left" in s else "", tuple(c // 16 + [6, 4]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)
    objs = rd("all_object_info")[f"{frame:06d}.all_object_info.json"]
    palette = {"Car": (255, 206, 126), "Truck": (64, 192, 255), "Pedestrian": (171, 124, 255), "Cyclist": (198, 208, 102),
               "Others": (125, 136, 166)}
    for tid, o in objs.items():
        T = np.asarray(o["object_to_world"])
        l, w, _ = o["object_lwh"]
        corners = np.array([[l / 2, w / 2], [l / 2, -w / 2], [-l / 2, -w / 2], [-l / 2, w / 2]])
        p = corners @ T[:2, :2].T + T[:2, 3]
        poly(p, palette.get(o["object_type"], (200, 200, 200)), closed=True, thick=2)
        poly(np.array([p[0], p[1]]), (255, 255, 255), thick=3)                              # front edge
        c = px(T[:2, 3]) // 16
        cv2.putText(img, tid, (int(c[0]) + 5, int(c[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)
    # ego box (not part of all_object_info): rear axle at (cx, cy)
    rep = json.loads((root / "tsf_export" / f"{clip_id}.json").read_text())
    Lego = rep["ego"]["length"]
    Rv = v2w[:2, :2]
    ego_box = np.array([[-0.3 * Lego, 0.9], [0.7 * Lego, 0.9], [0.7 * Lego, -0.9], [-0.3 * Lego, -0.9]]) @ Rv.T + [cx, cy]
    poly(ego_box, (214, 230, 46), closed=True, thick=2)
    # camera frusta (horizontal FOV from the name)
    for cam in cams:
        P = pose[f"{frame:06d}.pose.{cam}.npy"]
        fov = float(cam.rsplit("_", 1)[-1].replace("fov", ""))
        fwd = P[:2, 2] / max(np.linalg.norm(P[:2, 2]), 1e-9)
        a0 = np.arctan2(fwd[1], fwd[0])
        o = P[:2, 3]
        for da in (-fov / 2, fov / 2):
            q = o + 12 * np.array([np.cos(a0 + np.radians(da)), np.sin(a0 + np.radians(da))])
            poly(np.array([o, q]), (0, 180, 0), thick=1)
        cv2.circle(img, tuple((px(o) // 16).tolist()), 3, (0, 255, 0), -1)
    cv2.putText(img, f"{clip_id}  frame {frame}  world frame, ego-centred, x fwd(up in clip start)", (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)
    png = Path(png)
    png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(png), img)
    return png


def bev_video(sim, log, t0: float, n: int, fps: int, out, size=(960, 540), hold: int = 3) -> Path:
    """TanitSim BEV (render2d) of the clip window, one rendered frame per ``hold`` output frames."""
    import imageio.v2 as imageio

    from tsf.sim.render2d import Renderer
    r = Renderer(sim, log, size)
    dt = float(log.t[1] - log.t[0])
    wr = imageio.get_writer(str(out), fps=fps, codec="libx264", macro_block_size=1,
                            output_params=["-crf", "20", "-preset", "fast", "-pix_fmt", "yuv420p"])
    last = None
    for k in range(n):
        if k % hold == 0 or last is None:
            last = r.frame(int(min(len(log.t) - 1, round((t0 + k / fps) / dt))))
        wr.append_data(last)
    wr.close()
    return Path(out)


def run_example(example, out_dir, *, clip_id: str | None = None, t0: float | None = None, duration: float = 9.9,
                cameras=None, resolution: str = "resize_1280x720", workers: int = 1, bev: bool = True,
                lidar: bool = False, rig: str = "hyperion") -> dict:
    """Run one TSIR example end to end and return (and save) the summary with timings."""
    from tsf.cosmos import rds_hq, wsm
    from tsf.verify import ladder

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tm: dict = {}
    t = time.perf_counter()
    rep = ladder.run(Path(example).read_text(), out / "sim", qc=False)
    if not hasattr(rep, "_log"):
        raise RuntimeError(f"scenario failed before simulation: {rep.diagnostics()}")
    sim, log = rep._sim, rep._log
    tm["sim_s"] = round(time.perf_counter() - t, 2)
    cid = clip_id or sim.sc.meta.id
    root = out / "rds_hq"
    # one extra pose frame: the renderer still emits 297 frames, the LiDAR renderer gets the end pose of sweep 99
    rds_hq.export_clip(sim, log, root, cid, t0=t0, duration=duration, rig=rig, cameras=cameras, overwrite=True,
                       pad_frames=1)
    erep = rds_hq.export_report(root, cid)
    tm["export_s"] = erep["timings"]["export_s"]
    n, fps, t0r = erep["render_frames"], erep["fps"], erep["t0_s"]
    # exporter-side bird's-eye views from the tars (first / middle / last frame)
    bev_pngs = [str(rds_bev_png(root, cid, f, out / f"rds_bev_f{f:03d}.png")) for f in (0, n // 2, n - 1)]
    t = time.perf_counter()
    videos = wsm.render_wsm(root, cid, out / "wsm", cameras, fps=fps, resolution=resolution, workers=workers)
    tm["render_wall_s"] = round(time.perf_counter() - t, 2)
    wt = wsm.render_wsm.last_timings
    tm["render_s_per_camera"], tm["render_fps_per_camera"] = wt["render_s"], wt["render_fps"]
    extra = {}
    if bev:
        t = time.perf_counter()
        extra["TANITSIM BEV"] = bev_video(sim, log, t0r, n, fps, out / "bev.mp4")
        tm["bev_s"] = round(time.perf_counter() - t, 2)
    t = time.perf_counter()
    mos = wsm.mosaic(videos, out / "mosaic.mp4", extra=extra, title=cid, t0=t0r, fps=fps)
    tm["mosaic_s"] = round(time.perf_counter() - t, 2)
    stub = wsm.write_request_stub(videos, out / "transfer_auto", num_frames=n, fps=fps, name=cid)
    pngs = []
    for idx in (0, n // 3, 2 * n // 3, n - 1):
        pngs += wsm.extract_frames(mos, out / "frames", [idx], "mosaic_")
    pngs += wsm.extract_frames(videos[next(iter(videos))], out / "frames", [0, n // 2, n - 1])
    summary = {"example": str(example), "clip_id": cid, "t0_s": t0r, "frames": n, "fps": fps,
               "cameras": list(videos), "videos": {k: str(v) for k, v in videos.items()}, "mosaic": str(mos),
               "bev_video": str(extra.get("TANITSIM BEV", "")), "request_stub": str(stub), "rds_bev_png": bev_pngs, "frames_png": [str(p) for p in pngs],
               "timings": tm, "export": {k: erep[k] for k in ("window_choice", "ego", "objects", "map_labels",
                                                              "traffic_lights", "signal_state_observations", "validator")},
               "projection_check": wt.get("projection_check")}
    if lidar:
        from tsf.cosmos import lidar as _lidar
        summary["lidar"] = _lidar.render_lidar(root, cid, out / "lidar")
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("example")
    ap.add_argument("out_dir")
    ap.add_argument("--clip-id")
    ap.add_argument("--t0", type=float)
    ap.add_argument("--duration", type=float, default=9.9)
    ap.add_argument("--cameras", help="comma separated; default the 7 standard Hyperion cameras")
    ap.add_argument("--resolution", default="resize_1280x720")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--no-bev", action="store_true")
    ap.add_argument("--lidar", action="store_true")
    a = ap.parse_args()
    s = run_example(a.example, a.out_dir, clip_id=a.clip_id, t0=a.t0, duration=a.duration,
                    cameras=a.cameras.split(",") if a.cameras else None, resolution=a.resolution, workers=a.workers,
                    bev=not a.no_bev, lidar=a.lidar)
    print(json.dumps(s["timings"], indent=1))
    print("mosaic:", s["mosaic"])


if __name__ == "__main__":
    main()
