"""Render an RDS-HQ clip into WSM v4 control videos and tile them into a labelled multi-camera mosaic.

``render_wsm`` shells out to ``_wsm_driver.py`` inside the WSM venv (``TSF_WSM_PYTHON``, default
``~/tsf/venvs/wsm/bin/python``), which calls NVIDIA's ``cosmos_auto.wm`` v4 renderer (headless EGL, one camera
per call so each camera is timed). Settings follow the Auto recipe in the framework README: version=v4,
30 fps, 297-frame chunk, ``resize_1280x720``.

Per-run timings are written to ``<out_dir>/wsm_timings.json`` and stay in ``render_wsm.last_timings``.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import numpy as np

DEFAULT_PY = "~/tsf/venvs/wsm/bin/python"
CAMERAS_7 = (
    "camera_front_wide_120fov", "camera_front_tele_30fov", "camera_cross_left_120fov", "camera_cross_right_120fov",
    "camera_rear_left_70fov", "camera_rear_right_70fov", "camera_rear_tele_30fov",
)
# mosaic cell of each camera on a 3x3 grid (row, col); (2,0) and (2,2) hold the BEV / info panels
LAYOUT = {
    "camera_cross_left_120fov": (0, 0), "camera_front_wide_120fov": (0, 1), "camera_cross_right_120fov": (0, 2),
    "camera_rear_left_70fov": (1, 0), "camera_front_tele_30fov": (1, 1), "camera_rear_right_70fov": (1, 2),
    "camera_rear_tele_30fov": (2, 1),
}
FREE_CELLS = [(2, 0), (2, 2)]


def wsm_python() -> str:
    return str(Path(os.environ.get("TSF_WSM_PYTHON", DEFAULT_PY)).expanduser())


def clip_cameras(clip_root, clip_id: str) -> tuple[list[str], int]:
    """(camera names, number of pose frames) read from the clip's pose tar (member names only)."""
    cams: dict[str, int] = {}
    with tarfile.open(Path(clip_root) / "pose" / f"{clip_id}.tar") as tf:
        for name in tf.getnames():
            parts = name.split(".")             # <clip>.<frame>.pose.<camera>.npy
            if len(parts) >= 5 and parts[2] == "pose":
                cams[parts[3]] = cams.get(parts[3], 0) + 1
    return list(cams), (max(cams.values()) if cams else 0)


def render_wsm(clip_root, clip_id: str, out_dir, cameras=None, *, fps: int = 30, resolution: str = "resize_1280x720",
               workers: int = 1, python: str | None = None, check: bool = True, extra_overrides=(),
               timeout: float | None = None) -> dict[str, Path]:
    """Render WSM v4 control MP4s for ``clip_id`` found under the RDS-HQ root ``clip_root``.

    cameras: RDS-HQ camera names (default: the clip's 7 standard Hyperion cameras). ``workers`` > 1 renders
    disjoint camera subsets in parallel processes. Returns ``{camera: mp4_path}``; videos are at
    ``<out_dir>/world_scenario/ftheta_<camera>/<clip_id>_0.mp4`` (1280x720 / ``fps`` / <= 297 frames).
    Raises RuntimeError if a camera failed or is missing. ``render_wsm.last_timings`` holds the timing dict.
    """
    clip_root, out_dir = Path(clip_root).resolve(), Path(out_dir).resolve()
    avail, T = clip_cameras(clip_root, clip_id)
    cams = list(cameras) if cameras else ([c for c in CAMERAS_7 if c in avail] or avail)
    miss = [c for c in cams if c not in avail]
    if miss:
        raise KeyError(f"cameras {miss} have no pose in {clip_root}/pose/{clip_id}.tar (have {avail})")
    chunk = min(297, T)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(exist_ok=True)
    py = python or wsm_python()
    driver = Path(__file__).with_name("_wsm_driver.py")
    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    groups = [cams[i::max(1, workers)] for i in range(max(1, workers))]
    groups = [g for g in groups if g]
    t_wall = time.perf_counter()
    procs = []
    for gi, g in enumerate(groups):
        tj = out_dir / "logs" / f"timings_{gi}.json"
        cmd = [py, str(driver), "--input-root", str(clip_root), "--clip-id", clip_id, "--out", str(out_dir),
               "--cameras", ",".join(g), "--fps", str(fps), "--chunk", str(chunk), "--resolution", resolution,
               "--timings", str(tj)]
        if extra_overrides:
            cmd += ["--extra", *extra_overrides]
        log = open(out_dir / "logs" / f"render_{gi}.log", "w")
        procs.append((subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env), tj, log, g))
    results = {"cameras": {}, "projection_check": {}}
    for p, tj, log, g in procs:
        rc = p.wait(timeout=timeout)
        log.close()
        if rc != 0 or not tj.exists():
            tail = Path(log.name).read_text()[-2500:]
            raise RuntimeError(f"WSM renderer failed for {g} (rc={rc}); log {log.name}:\n{tail}")
        part = json.loads(tj.read_text())
        results["cameras"].update(part["cameras"])
        results["projection_check"].update(part.get("projection_check", {}))
        results.setdefault("import_s", part.get("import_s"))
    wall = time.perf_counter() - t_wall
    videos, errs = {}, []
    for cam in cams:
        r = results["cameras"].get(cam, {})
        if r.get("mp4") and not r.get("error"):
            videos[cam] = Path(r["mp4"])
        else:
            errs.append(f"{cam}: {r.get('error') or 'no output'}")
    timings = {"clip_id": clip_id, "frames_per_video": chunk, "fps_out": fps, "resolution": resolution,
               "workers": len(groups), "wall_s": round(wall, 2), "python": py,
               "render_s": {c: results["cameras"][c]["render_s"] for c in results["cameras"]},
               "render_fps": {c: results["cameras"][c]["fps_render"] for c in results["cameras"]},
               "sum_render_s": round(sum(v["render_s"] for v in results["cameras"].values()), 2),
               "import_s": results.get("import_s"), "projection_check": results["projection_check"],
               "errors": errs}
    (out_dir / "wsm_timings.json").write_text(json.dumps(timings, indent=1))
    render_wsm.last_timings = timings
    if errs:
        raise RuntimeError("WSM render errors: " + "; ".join(errs))
    return videos


render_wsm.last_timings = {}


# ============================================================================== mosaic
def _label(cam: str) -> str:
    n = cam.replace("camera_", "")
    fov = n.rsplit("_", 1)[-1].replace("fov", "")
    return f"{n.rsplit('_', 1)[0].replace('_', ' ').upper()}  {fov} deg"


def _caption(img: np.ndarray, text: str, cv2) -> None:
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(img, (0, 0), (w + 14, h + 12), (0, 0, 0), -1)
    cv2.putText(img, text, (7, h + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)


def mosaic(videos: dict, out_mp4, *, extra: dict | None = None, title: str = "", t0: float = 0.0,
           fps: int | None = None, cell=(640, 360), max_frames: int | None = None) -> Path:
    """Tile per-camera control videos into one labelled grid MP4 (3x3 cells of ``cell`` px = 1920x1080 by default).

    videos: ``{camera_name: mp4}``; the seven Hyperion cameras go to fixed cells (front wide top-centre,
    cross cameras top-left/right, rear cameras left/right of the front tele, rear tele bottom-centre).
    extra: ``{label: mp4}`` of additional panels (e.g. the TanitSim BEV) placed into the free bottom corners;
    the last free cell shows clip title, simulated time and frame counter.
    """
    import cv2
    import imageio.v2 as imageio

    out_mp4 = Path(out_mp4)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    cw, ch = cell
    panels: list[tuple[tuple[int, int], str, Path]] = []
    info_cell = FREE_CELLS[-1]                   # bottom-right: title / time / frame counter
    free = list(FREE_CELLS[:-1])                 # bottom-left: BEV or other extra panel
    for cam, p in videos.items():
        pos = LAYOUT.get(cam) or (free.pop(0) if free else None)
        if pos is not None:
            panels.append((pos, _label(cam), Path(p)))
    for lab, p in (extra or {}).items():
        if free:
            panels.append((free.pop(0), lab, Path(p)))
    caps = [cv2.VideoCapture(str(p)) for _, _, p in panels]
    src_fps = caps[0].get(cv2.CAP_PROP_FPS) if caps else 30.0
    fps = fps or int(round(src_fps or 30))
    n = min(int(c.get(cv2.CAP_PROP_FRAME_COUNT)) for c in caps)
    if max_frames:
        n = min(n, max_frames)
    wr = imageio.get_writer(str(out_mp4), fps=fps, codec="libx264", macro_block_size=1,
                            output_params=["-crf", "20", "-preset", "fast", "-pix_fmt", "yuv420p"])
    try:
        for k in range(n):
            canvas = np.zeros((3 * ch, 3 * cw, 3), np.uint8)
            for ((r, c), lab, _), cap in zip(panels, caps):
                ok, fr = cap.read()
                if not ok:
                    fr = np.zeros((ch, cw, 3), np.uint8)
                else:
                    fr = cv2.cvtColor(cv2.resize(fr, (cw, ch), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
                _caption(fr, lab, cv2)
                cv2.rectangle(fr, (0, 0), (cw - 1, ch - 1), (60, 60, 60), 1)
                canvas[r * ch:(r + 1) * ch, c * cw:(c + 1) * cw] = fr
            r, c = info_cell
            box = canvas[r * ch:(r + 1) * ch, c * cw:(c + 1) * cw]
            box[:] = (18, 18, 22)
            for i, line in enumerate([title or "WSM v4 control mosaic", f"t = {t0 + k / fps:6.2f} s (sim)",
                                      f"frame {k + 1}/{n}  @ {fps} fps"]):
                cv2.putText(box, line, (16, 50 + 44 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8 if i else 0.7,
                            (235, 235, 235), 1, cv2.LINE_AA)
            wr.append_data(canvas)
    finally:
        wr.close()
        for c in caps:
            c.release()
    return out_mp4


def write_request_stub(videos: dict, out_dir, *, num_frames: int = 297, fps: int = 30, resolution: str = "720",
                       name: str = "tsf_scene") -> Path:
    """Skeleton Cosmos3-Nano-Transfer-Auto request (WSM text-to-video) for rendered controls.

    Copies each control MP4 to ``<out_dir>/controls/<camera_key>.mp4`` and writes ``<out_dir>/request.stub.json`` in
    the schema of the framework's ``assets/daytime_scene/transfer_auto_wsm.json``. **Captions are not created**:
    ``caption_path`` points at ``captions/<camera_key>.json`` sidecars that still have to be authored
    (``{"caption_structured": {"chunk_0_297": {"start_frame": 0, "end_frame": 297, "caption": "..."}}}``).
    """
    import shutil
    out_dir = Path(out_dir)
    (out_dir / "controls").mkdir(parents=True, exist_ok=True)
    views = []
    for cam, p in videos.items():
        shutil.copyfile(p, out_dir / "controls" / f"{cam}.mp4")
        views.append({"camera_key": cam, "control_path": f"controls/{cam}.mp4", "caption_path": f"captions/{cam}.json"})
    req = {"name": name, "prompt_from_camera": "camera_front_wide_120fov", "model_mode": "text2video",
           "resolution": resolution, "aspect_ratio": "16,9", "num_frames": num_frames, "fps": fps,
           "num_inference_steps": 35, "guidance_scale": 3, "flow_shift": 10, "seed": 0, "wsm": {},
           "multiview": {"views": views}}
    path = out_dir / "request.stub.json"
    path.write_text(json.dumps(req, indent=2))
    return path


def extract_frames(video, out_dir, indices, prefix: str = "") -> list[Path]:
    """Save selected frames of an MP4 as PNG (for visual checks)."""
    import cv2
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    paths = []
    for i in indices:
        i = n - 1 if i < 0 else min(i, n - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, fr = cap.read()
        if ok:
            p = out_dir / f"{prefix}{Path(video).stem}_f{i:03d}.png"
            cv2.imwrite(str(p), fr)
            paths.append(p)
    cap.release()
    return paths


if __name__ == "__main__":      # python -m tsf.cosmos.wsm <rds_root> <clip_id> <out_dir> [cam,cam,...]
    root, cid, out = sys.argv[1:4]
    vids = render_wsm(root, cid, out, sys.argv[4].split(",") if len(sys.argv) > 4 else None)
    print(json.dumps({k: str(v) for k, v in vids.items()}, indent=1))
    print(json.dumps(render_wsm.last_timings, indent=1))
