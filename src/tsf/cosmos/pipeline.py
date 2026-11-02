"""End-to-end: TSIR example -> TanitSim -> RDS-HQ -> WSM v4 control videos -> mosaic (+ BEV, PNG checks).

    python -m tsf.cosmos.pipeline examples/uc1_left_turn_intersection.json runs/wsm/uc1 [--t0 12.0] [--lidar]

Everything lands under ``out_dir``: ``rds_hq/`` (clip tars + ``tsf_export`` report), ``wsm/`` (per-camera MP4s,
``wsm_timings.json``, logs), ``bev.mp4`` (TanitSim bird's-eye view of the same window), ``mosaic.mp4``,
``frames/`` (PNG spot checks), ``rds_bev_*.png`` (map + objects + cameras drawn from the exported tars) and
``summary.json`` with all timings.

Sensor rendering (camera video + LiDAR from Cosmos3-Nano-Sim-Auto's joint mode) for other code to call:

    from tsf.cosmos import pipeline
    pipeline.ensure_server(views=3, lidar=True)                       # once; refuses politely if another server runs
    r = pipeline.render_sensors("examples/uc1_left_turn_intersection.json", "runs/sensors/uc1",
                                appearance={"weather": "heavy_rain", "time_of_day": "21:30"})
    r["videos"], r["lidar"]["pointclouds_dir"], r["lidar"]["checks"], r["mosaic"]   # see ``render_sensors``

CLI: ``python -m tsf.cosmos.pipeline sensors <tsir.json> <out_dir> [--frames 97] [--appearance JSON] [--no-lidar] [--ensure-server]``.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import sys
import tarfile
import threading
import time
from pathlib import Path

import numpy as np

DEFAULT_URL = os.environ.get("COSMOS_URL", "http://127.0.0.1:8100")   # = generate.DEFAULT_URL, duplicated: light import
SENSOR_CAMERAS = ("camera_front_wide_120fov", "camera_cross_left_120fov", "camera_cross_right_120fov")


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
                            output_params=["-crf", "20", "-preset", "fast"])
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


# ============================================================================== sensors: camera + LiDAR from Sim-Auto
class CosmosBusyError(RuntimeError):
    """Another Cosmos server (or too little free memory) blocks starting the Sim-Auto server; nothing was touched."""


def _mem_available_gb() -> float:
    m = {k: int(v.split()[0]) for k, v in (ln.split(":", 1) for ln in Path("/proc/meminfo").read_text().splitlines())}
    return m["MemAvailable"] / 2**20


def wait_for_memory(min_gb: float = 35.0, timeout: float = 0.0, interval: float = 20.0, what: str = "this step") -> float:
    """Block until ``MemAvailable`` >= ``min_gb`` (Thor's rule for any process above ~4 GB: the 122 GB are shared with the
    LLM, REFe and the Cosmos servers, and a kernel OOM kill hits the resident services). ``timeout`` 0 = check once.
    Raises :class:`CosmosBusyError` when still short after ``timeout`` s; returns the available GB."""
    end = time.monotonic() + timeout
    while True:
        avail = _mem_available_gb()
        if avail >= min_gb:
            return avail
        if time.monotonic() >= end:
            raise CosmosBusyError(f"only {avail:.0f} GB memory available (< {min_gb:.0f} GB) before {what}: wait and retry")
        time.sleep(interval)


def ensure_server(views: int = 3, lidar: bool = True, *, base_url: str = DEFAULT_URL, timeout: float = 1800.0,
                  min_available_gb: float = 62.0, evict_dirs=("~/.cache/huggingface", "~/tsf", "~/data"),
                  owner: str | None = None) -> dict:
    """Make sure a Sim-Auto server that can render ``views`` cameras (+ LiDAR) is up on this machine (Thor).

    * a running ``tsf-cosmos-sim`` sized for at least ``views`` views (and joint LiDAR if asked) is used as it is;
    * any other running ``tsf-cosmos-*`` container (another model, or a Sim server sized too small) is NOT touched: raises
      :class:`CosmosBusyError` so the caller can wait and retry (one Cosmos server fits next to ``tsf-llm``);
    * otherwise: evicts the page cache of our own files (unified memory: CUDA's free memory excludes cache), refuses with
      :class:`CosmosBusyError` if ``MemAvailable`` is below ``min_available_gb`` (62 GB = the server's ~43-46 GB, a
      joint window's working set and a margin; never less than the 35 GB Thor rule), starts
      ``generate.serve("sim", kv_views=views, lidar=lidar)``, keeps evicting while the weights load and waits until ready.
    Returns ``{started, server, kv_views, kv_lidar, ready_s, ...}``. Stop a server you started with ``generate.stop("sim")``.
    """
    from tsf.cosmos import generate as g

    running = g.running_servers()
    if running:
        if running == ["tsf-cosmos-sim"]:
            env = g.server_env("tsf-cosmos-sim")
            kv, kl = int(env.get("TSF_KV_VIEWS", "0") or 0), env.get("TSF_KV_LIDAR", "0") not in ("", "0")
            chunk_ok = int(env.get("TSF_LIDAR_CHUNK", "0") or 0) >= 1
            if kv >= views and (kl or not lidar) and (chunk_ok or not lidar):
                t = g.wait_ready("sim", base_url, timeout)
                return {"started": False, "server": "tsf-cosmos-sim", "kv_views": kv, "kv_lidar": kl, "ready_s": round(t, 1)}
            raise CosmosBusyError(f"tsf-cosmos-sim is running (owner {g.server_owner()!r}) but sized for {kv} views, lidar={kl}, "
                                  f"lidar chunk={env.get('TSF_LIDAR_CHUNK')}; need >= {views} views, lidar={lidar} with "
                                  "TSF_LIDAR_CHUNK set. It is not mine to stop: wait for its owner, then retry.")
        raise CosmosBusyError(f"another Cosmos server is running: {running}. One server fits next to tsf-llm; wait and retry "
                              "(do not stop it unless you started it).")
    g.evict_page_cache(evict_dirs)
    avail = wait_for_memory(min_available_gb, what="starting the Sim-Auto server")
    stop = threading.Event()

    def _evict():
        while not stop.wait(3.0):
            g.evict_page_cache(evict_dirs)

    th = threading.Thread(target=_evict, daemon=True)
    th.start()
    try:
        g.serve("sim", kv_views=views, lidar=lidar, owner=owner or os.environ.get("TSF_OWNER") or "pipeline.ensure_server")
        try:
            ready = g.wait_ready("sim", base_url, timeout)
        except BaseException:
            g.stop("sim")
            raise
    finally:
        stop.set()
        th.join()
    return {"started": True, "server": "tsf-cosmos-sim", "kv_views": views, "kv_lidar": lidar, "ready_s": round(ready, 1),
            "mem_available_gb_before": round(avail, 1), "mem_available_gb_after": round(_mem_available_gb(), 1)}


# -- captions with the environment (template fallback) ------------------------------------------------------------------
_CLEAR_WORDS = re.compile(r"\b(sunlit|sunlight|sunny|shadows?|blue sky|bright|crisp|clear|dry|glossy|high clouds|sharp)\b", re.I)


def describe_environment(env: dict) -> tuple[str, str]:
    """``(setting, remind)`` sentences for time of day, sky / weather and road surface of a TSIR ``environment``."""
    e = {"time_of_day": "12:00", "weather": "clear", "road_surface": "dry", "fog_visibility_m": 100000.0, **(env or {})}
    hh, mm = (int(x) for x in str(e["time_of_day"]).split(":"))
    h, w, surf = hh + mm / 60.0, str(e["weather"]), str(e["road_surface"])
    if h < 5 or h >= 21.0:
        tod = (f"It is night, {e['time_of_day']}, and it is dark: the only light comes from headlights, street lamps, "
               "illuminated windows and signal heads, and the sky is black")
    elif h < 7:
        tod = "It is early morning with a pale gray-blue dawn light and long soft shadows"
    elif h < 10:
        tod = "It is morning with low warm sunlight"
    elif h < 14:
        tod = "It is around midday with high, even daylight"
    elif h < 17.5:
        tod = "It is afternoon with warm daylight"
    elif h < 19.5:
        tod = "It is early evening with low golden light and long shadows"
    else:
        tod = "It is dusk with a dim purple-orange sky and the street lamps and headlights switched on"
    sky = {"clear": "The sky is clear.", "cloudy": "The sky is covered by gray clouds and the light is soft and diffuse.",
           "rain": "Steady rain falls from a gray overcast sky and fine rain streaks catch the light of the headlights.",
           "heavy_rain": "Heavy rain pours from a dark overcast sky with dense rain streaks and spray rising from the tires.",
           "snow": "Snow falls steadily from a low gray sky and flakes drift through the headlight beams.",
           "fog": f"Thick fog hangs over the road with visibility of only about {float(e['fog_visibility_m']):.0f} meters; "
                  "distant objects fade into pale gray haze."}.get(w, "")
    if surf == "dry" and w in ("rain", "heavy_rain"):
        surf = "wet"
    road = {"dry": "The pavement is dry asphalt with sharp paint.",
            "wet": "The pavement is dark, wet and reflective, with lane paint and lights mirrored in puddles.",
            "snow": "The pavement is covered by a thin layer of snow with dark tire tracks.",
            "ice": "The pavement is covered by a glassy sheet of ice."}.get(surf, "")
    setting = f"{tod}. {sky} {road}".replace("..", ".")
    remind = f"{sky} {road} The lighting stays the same from the first frame to the last."
    return setting, remind


def environment_captions(scenario_id: str, cams, frames: int, env: dict, fps: float = 30.0) -> dict[str, str]:
    """Deterministic template captions (``tsf.cosmos.captions``) re-written for ``env``: the setting / reminder sentences are
    replaced by :func:`describe_environment` and, when the environment is not clear day, sentences that contradict it
    (sunlit, shadows, blue sky, dry, glossy...) are dropped. Same camera descriptions and scenario content otherwise."""
    from tsf.cosmos import captions as C

    t = C.TEMPLATES[C.scenario_kind(scenario_id)]
    h = int(str(env.get("time_of_day", "12:00"))[:2])
    adverse = env.get("weather", "clear") not in ("clear",) or not 6 <= h < 19 or env.get("road_surface", "dry") != "dry"
    split = lambda x: re.split(r"(?<=[.!?])\s+", x.strip())            # noqa: E731
    filt = (lambda x: " ".join(q for q in split(x) if not _CLEAR_WORDS.search(q))) if adverse else (lambda x: x)
    setting, remind = describe_environment(env)
    geo = "The scene is" + t["setting"].split("The scene is", 1)[1]
    out = {}
    for cam in cams:
        grp = "front" if "front" in cam else "side" if "cross" in cam else "rear"
        blocks = []
        for i, ((a, b), text) in enumerate(zip(C._bounds(frames, fps), t["view"][cam])):
            parts = [f"{setting} {geo}" if i == 0 else remind, filt(text), filt(t["detail"][grp][i])]
            if i == 2:
                parts += [filt(t["closing"][grp]), C._STEADY if not adverse else "Colors stay natural and consistent."]
            blocks.append(f"[{a:g}s-{b:g}s]: " + " ".join(q for q in parts if q))
        out[cam] = "\n\n".join(blocks)
    return out


def make_captions(tsir: dict, cams, frames: int, appearance: dict | None, captions=None, *, fps: float = 30.0) -> tuple[object, str]:
    """``(captions, source)``: ``captions`` as given (``source='given'``), else the LLM captioner with ``appearance`` as
    environment overrides (``'llm'``), else the deterministic templates adapted to the environment (``'template'``)."""
    if captions is not None:
        return captions, "given"
    env = {**tsir.get("environment", {}), **(appearance or {})}
    try:
        from tsf.cosmos.llm_captions import llm_captions
        return llm_captions(tsir, tuple(cams), frames=frames, fps=fps, appearance=appearance), "llm"
    except Exception as exc:                     # LLM down / schema failure: fall back, but say so
        print(f"[render_sensors] LLM captions failed ({type(exc).__name__}: {str(exc)[:120]}); using templates", file=sys.stderr)
        return environment_captions(tsir["meta"]["id"], cams, frames, env, fps), "template"


# -- mosaic ---------------------------------------------------------------------------------------------------------------
def _read_frames(path, cell, hold: int = 1, n: int | None = None) -> list[np.ndarray]:
    """All frames of a video, letterboxed into ``cell`` (RGB), each repeated ``hold`` times (10 Hz LiDAR -> 30 fps)."""
    import cv2
    cw, ch = cell
    cap = cv2.VideoCapture(str(path))
    out = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        sc = min(cw / fr.shape[1], ch / fr.shape[0])
        im = cv2.resize(fr, (max(1, int(fr.shape[1] * sc)), max(1, int(fr.shape[0] * sc))), interpolation=cv2.INTER_AREA)
        box = np.zeros((ch, cw, 3), np.uint8)
        y, x = (ch - im.shape[0]) // 2, (cw - im.shape[1]) // 2
        box[y:y + im.shape[0], x:x + im.shape[1]] = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        out += [box] * hold
        if n and len(out) >= n:
            break
    cap.release()
    return out[:n] if n else out


def sensors_mosaic(cams: dict, extras: list[tuple[str, object, int]], out_mp4, *, title: str = "", t0: float = 0.0, fps: int = 30,
                   frames: int | None = None, cell=(640, 360)) -> Path:
    """Grid MP4, 3 columns of ``cell`` px: generated camera videos first, then ``extras`` = ``[(label, mp4, hold)]`` (TanitSim
    BEV, LiDAR BEV, LiDAR range images; ``hold`` repeats each frame, 3 for the 10 Hz LiDAR videos); title strip on top."""
    import cv2
    import imageio.v2 as imageio
    cw, ch = cell
    pref = ["camera_cross_left_120fov", "camera_front_wide_120fov", "camera_cross_right_120fov"]      # left | front | right
    order = [c for c in pref if c in cams] + [c for c in cams if c not in pref]
    panels = [(c.replace("camera_", "").replace("_", " "), _read_frames(cams[c], cell, 1, frames)) for c in order]
    panels += [(lab, _read_frames(p, cell, hold, frames)) for lab, p, hold in extras if p and Path(p).exists()]
    n = min(len(f) for _, f in panels)
    cols = 3
    rows = -(-len(panels) // cols)
    out_mp4 = Path(out_mp4)
    wr = imageio.get_writer(str(out_mp4), fps=fps, codec="libx264", macro_block_size=1,
                            output_params=["-crf", "20", "-preset", "fast"])
    try:
        for k in range(n):
            canvas = np.zeros((36 + rows * ch, cols * cw, 3), np.uint8)
            cv2.putText(canvas, f"{title}   t = {t0 + k / fps:6.2f} s   frame {k + 1}/{n}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (235, 235, 235), 1, cv2.LINE_AA)
            for i, (lab, fr) in enumerate(panels):
                r, c = divmod(i, cols)
                cell_img = fr[k].copy()
                (tw, th), _ = cv2.getTextSize(lab, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                cv2.rectangle(cell_img, (0, 0), (tw + 12, th + 12), (0, 0, 0), -1)
                cv2.putText(cell_img, lab, (6, th + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
                canvas[36 + r * ch:36 + (r + 1) * ch, c * cw:(c + 1) * cw] = cell_img
            wr.append_data(canvas)
    finally:
        wr.close()
    return out_mp4


# -- render_sensors -------------------------------------------------------------------------------------------------------
WEATHER_PHYSICS_AUTO = True        # "auto" = apply the LiDAR weather degradation for adverse environments (see lidar_weather)


def _as_tsir(scenario) -> tuple[dict, str]:
    """``(tsir dict, raw json text)`` of a TSIR dict, a path to one, or a JSON string."""
    if isinstance(scenario, dict):
        return scenario, json.dumps(scenario)
    if isinstance(scenario, (str, Path)) and not str(scenario).lstrip().startswith("{"):
        txt = Path(scenario).read_text()
    else:
        txt = str(scenario)
    return json.loads(txt), txt


def _geometry_key(txt: str, t0, frames: int, cams, rig: str, resolution: str, lidar: bool) -> str:
    return hashlib.sha1(json.dumps([txt, t0, frames, list(cams), rig, resolution, lidar]).encode()).hexdigest()[:16]


def geometry_bev(txt: str, gdir, frames: int, *, sim=None, log=None, min_available_gb: float = 35.0, wait_s: float = 900.0) -> Path:
    """``<gdir>/bev.mp4``: TanitSim bird's-eye view of the clip window (30 fps, ``frames`` frames). The map raster of
    ``tsf.sim.render2d`` peaks at ~4.6 GB RSS, so this is a scenario-only step that runs next to the other renderers (guarded
    by ``wait_for_memory``) and never while a Cosmos server holds the memory."""
    from tsf.cosmos import rds_hq
    from tsf.verify import ladder

    gdir = Path(gdir)
    out = gdir / "bev.mp4"
    if out.exists():
        return out
    wait_for_memory(min_available_gb, wait_s, what="the TanitSim BEV render (4.6 GB)")
    if sim is None:
        rep = ladder.run(txt, gdir / "sim", qc=False)
        sim, log = rep._sim, rep._log
    cid = json.loads((gdir / "geometry.json").read_text())["clip_id"]
    return bev_video(sim, log, rds_hq.export_report(gdir / "rds_hq", cid)["t0_s"], frames, 30, out)


def prepare_geometry(txt: str, gdir, cams, frames: int, *, t0: float | None = None, lidar: bool = True, rig: str = "hyperion",
                     resolution: str = "resize_1280x720", clip_id: str | None = None, min_available_gb: float = 35.0,
                     wait_s: float = 900.0) -> tuple[dict, object, object]:
    """Scenario-only stages of :func:`render_sensors`: TanitSim -> RDS-HQ export (``pad_frames=1``) -> WSM v4 controls ->
    HD-map LiDAR control, all under ``gdir`` (``rds_hq/``, ``wsm/``, ``lidar/``, ``sim/``). Skipped when ``gdir/geometry.json``
    holds the same key (scenario text, window, cameras, rig, resolution, lidar). Returns ``(info, sim, log)`` with
    ``info = {key, timings, clip_id, frames, reused}``; ``sim`` / ``log`` are None when reused. Needs no Cosmos server.
    Waits up to ``wait_s`` for ``min_available_gb`` free memory before the renderers (WSM, LiDAR range map) start."""
    from tsf.cosmos import lidar as L
    from tsf.cosmos import rds_hq, wsm
    from tsf.verify import ladder

    gdir = Path(gdir)
    gdir.mkdir(parents=True, exist_ok=True)
    tsir = json.loads(txt)
    cid = clip_id or tsir["meta"]["id"]
    key = _geometry_key(txt, t0, frames, cams, rig, resolution, lidar)
    gfile = gdir / "geometry.json"
    geo = json.loads(gfile.read_text()) if gfile.exists() else {}
    if geo.get("key") == key and (gdir / "wsm").exists():
        return {**geo, "reused": True}, None, None
    wait_for_memory(min_available_gb, wait_s, what="the WSM / LiDAR renderers")
    gt: dict = {}
    t = time.perf_counter()
    rep = ladder.run(txt, gdir / "sim", qc=False)
    if not hasattr(rep, "_log"):
        raise RuntimeError(f"scenario failed before simulation: {rep.diagnostics()}")
    sim, log = rep._sim, rep._log
    gt["sim_s"] = round(time.perf_counter() - t, 2)
    rds_hq.export_clip(sim, log, gdir / "rds_hq", cid, t0=t0, duration=frames / 30.0, rig=rig, cameras=list(cams), overwrite=True,
                       pad_frames=1)
    gt["export_s"] = rds_hq.export_report(gdir / "rds_hq", cid)["timings"]["export_s"]
    t = time.perf_counter()
    wsm.render_wsm(gdir / "rds_hq", cid, gdir / "wsm", list(cams), fps=30, resolution=resolution)
    gt["wsm_s"] = round(time.perf_counter() - t, 2)
    if lidar:
        gt["lidar_control_s"] = L.render_lidar(gdir / "rds_hq", cid, gdir / "lidar", max_sweeps=L.required_sweeps(frames))["wall_s"]
    info = {"key": key, "timings": gt, "clip_id": cid, "frames": frames, "reused": False}
    gfile.write_text(json.dumps(info, indent=1))
    t = time.perf_counter()
    (gdir / "bev.mp4").unlink(missing_ok=True)
    geometry_bev(txt, gdir, frames, sim=sim, log=log)
    gt["bev_s"] = round(time.perf_counter() - t, 2)
    gfile.write_text(json.dumps(info, indent=1))
    return info, sim, log


def _coherent_appearance(appearance: dict | None) -> dict | None:
    """An appearance override that names rain / snow without a road surface gets the matching surface (the TSIR's own
    ``road_surface`` is usually ``dry``, which would make the captions describe a dry road in the rain)."""
    if not appearance:
        return appearance
    a = dict(appearance)
    if "road_surface" not in a:
        if a.get("weather") in ("rain", "heavy_rain", "drizzle"):
            a["road_surface"] = "wet"
        elif a.get("weather") == "snow":
            a["road_surface"] = "snow"
    return a


def render_sensors(scenario, out_dir, *, cameras=SENSOR_CAMERAS, frames: int = 97, t0: float | None = None, captions=None,
                   appearance: dict | None = None, lidar: bool = True, seed: int = 0, base_url: str = DEFAULT_URL,
                   res: str = "480", weather_physics="auto", geometry_dir=None, rig: str = "hyperion",
                   resolution: str = "resize_1280x720", clip_id: str | None = None, timeout: float = 7200.0,
                   window_frames: int = 49, window_min_available_gb: float = 15.0) -> dict:
    """Render one scenario's sensor data: ``cameras`` video + LiDAR from Cosmos3-Nano-Sim-Auto (joint mode).

    ``scenario``: TSIR dict or path. Stages (timed in ``timings``): TanitSim -> RDS-HQ export (window ``t0`` s, ``frames``
    camera frames at 30 fps, auto window if None) -> WSM v4 controls -> HD-map LiDAR control (``lidar=True``) -> captions
    (``captions`` dict/dir as ``generate.resolve_captions``; None = ``tsf.cosmos.llm_captions`` with ``appearance`` as
    environment override, templates adapted to the environment if the LLM is down) -> Sim-Auto joint request (needs
    ``ensure_server(views=len(cameras), lidar=lidar)``; ``frames`` must be 4k+1) -> point clouds, videos, checks ->
    ``mosaic.mp4`` (cameras + LiDAR BEV + range images + TanitSim BEV). ``appearance`` ({"weather", "time_of_day", ...},
    TSIR environment keys) only changes captions and the weather post-process, never the simulated geometry.
    ``window_frames``: a clip longer than this is generated in windows of at most ``window_frames`` camera frames that overlap
    by one frame (``generate.sim_windows``: each window conditioned on the previous one's last frame, LiDAR sweeps joined;
    finished windows are reused when a crashed run is repeated). 97 frames x 3 views in ONE request overran Thor's unified
    memory (MemAvailable 0, tsf-llm OOM-killed); the working set beyond the idle server is ~13-18 GB for 25 as well as 49
    frames, so 49 (fewest seams) is the default and the cap. Before every window ``MemAvailable``
    must be at least ``window_min_available_gb`` (waits up to 10 min, then raises :class:`CosmosBusyError`).
    ``weather_physics``: ``"auto"`` (apply :func:`tsf.cosmos.lidar_weather.degrade` when the environment is fog / rain /
    snow, see ``WEATHER_PHYSICS_AUTO``), True, False. ``geometry_dir``: reuse the scenario-only stages (sim, export, WSM,
    LiDAR control) between calls that only differ in appearance / seed (keyed by scenario, window, cameras).

    Returns (and writes ``<out_dir>/sensors.json``): ``videos`` {camera: mp4}, ``mosaic``, ``bev_video`` (TanitSim),
    ``lidar`` {``generated`` (raw model output safetensors), ``control``, ``pointclouds_dir`` (+ ``pointclouds_raw_dir`` if a
    physics degradation was applied), ``bev_video``, ``range_compare``, ``checks`` (``lidar_checks.json``),
    ``checks_summary``, ``weather``}, ``captions``, ``timings``, ``memory``, ``cosmos`` (request result). Point clouds are
    ``sweep_NNN.npz``: ``xyz`` [N, 3] in the LiDAR frame at sweep start (x fwd, y left, z up, motion compensated),
    ``intensity``, ``range``, ``row``, ``col``, ``t_rel`` and the sweep's lidar_to_world poses.
    """
    from tsf.cosmos import generate as g
    from tsf.cosmos import lidar as L
    from tsf.cosmos import lidar_weather as W
    from tsf.cosmos import rds_hq

    cams = [c for c in g.CAMERA_ORDER if c in cameras]
    if not cams or len(cams) != len(set(cameras)):
        raise ValueError(f"cameras must be a subset of {g.CAMERA_ORDER}, got {list(cameras)}")
    if (frames - 1) % 4:
        raise ValueError("frames must be 4k+1 (e.g. 49, 97, 201, 297)")
    fps = 30
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tsir, txt = _as_tsir(scenario)
    sid = tsir["meta"]["id"]
    cid = clip_id or sid
    appearance = _coherent_appearance(appearance)
    env = W.effective_environment({**tsir.get("environment", {}), **(appearance or {})})
    tm: dict = {}
    err: dict = {}

    # ---- geometry (scenario only): TanitSim -> RDS-HQ -> WSM -> LiDAR control, reusable between appearance variants
    gdir = Path(geometry_dir) if geometry_dir else out
    geo, sim, log = prepare_geometry(txt, gdir, cams, frames, t0=t0, lidar=lidar, rig=rig, resolution=resolution, clip_id=cid)
    tm.update(geo["timings"], geometry_reused=geo["reused"])
    root = gdir / "rds_hq"
    erep = rds_hq.export_report(root, cid)
    if erep["render_frames"] < frames:
        raise RuntimeError(f"the simulation only provides {erep['render_frames']} frames from t0={erep['t0_s']} s, asked for {frames}")
    control = gdir / "lidar" / "controls" / cid / "lidar_control.safetensors"
    sweeps = L.load_sweep_poses(gdir / "lidar" / "lidar_rangemap_metadata" / f"{cid}.npz") if lidar else None

    # ---- captions
    t = time.perf_counter()
    caps, cap_src = make_captions(tsir, cams, frames, appearance, captions, fps=fps)
    if isinstance(caps, dict) and all(isinstance(v, str) and not v.endswith(".json") for v in caps.values()):
        from tsf.cosmos.captions import write_sidecars
        write_sidecars(caps, out / "captions", frames=frames)
    tm["captions_s"] = round(time.perf_counter() - t, 2)

    # ---- Sim-Auto (joint camera + LiDAR when lidar=True)
    cosmos = g.sim_windows(gdir / "wsm", out / "cosmos", cams, frames, res, caps, window_frames=window_frames, seed=seed,
                           scenario=sid, lidar_control=control if lidar else None, base_url=base_url, timeout=timeout,
                           before_window=lambda k: wait_for_memory(window_min_available_gb, timeout=600.0,
                                                                  what=f"Sim-Auto window {k} ({window_frames} frames)"))
    tm["cosmos_s"] = cosmos["request_s"]

    # ---- LiDAR post-processing (a failure here keeps the camera result and is reported)
    lid: dict = {"enabled": bool(lidar)}
    if lidar:
        try:
            t = time.perf_counter()
            lid.update(_lidar_products(out, cosmos, sweeps, root, cid, env, weather_physics, seed, ego_length=erep["ego"]["length"]))
            tm["lidar_post_s"] = round(time.perf_counter() - t, 2)
        except Exception:
            import traceback
            err["lidar_post"] = traceback.format_exc()
            lid["error"] = err["lidar_post"].strip().splitlines()[-1]

    # ---- TanitSim BEV (scenario-only, made with the geometry) + mosaic
    t = time.perf_counter()
    bev = gdir / "bev.mp4"
    if not bev.exists():
        try:
            bev = geometry_bev(txt, gdir, frames, sim=sim, log=log, wait_s=0.0)      # old geometry dir: needs 35 GB free
        except Exception as exc:                                                    # e.g. a server holds the memory
            err["bev"] = f"{type(exc).__name__}: {exc}"
            bev = None
    extras = [("TanitSim BEV", bev, 1)]
    if lid.get("bev_video"):
        extras += [("LiDAR BEV (generated)", lid["bev_video"], 3),
                   ("LiDAR range: control / generated / intensity", lid.get("range_compare"), 3)]
    mos = sensors_mosaic({c: cosmos["videos"][c] for c in cams}, extras, out / "mosaic.mp4",
                         title=f"{cid}  {env['weather']} {env['time_of_day']}", t0=erep["t0_s"], fps=fps, frames=frames)
    tm["mosaic_s"] = round(time.perf_counter() - t, 2)

    res_d = {"scenario": sid, "clip_id": cid, "out_dir": str(out), "frames": frames, "fps": fps, "cameras": cams, "seed": seed,
             "t0_s": erep["t0_s"], "appearance": appearance or {}, "environment": env,
             "captions": {"source": cap_src, "dir": str(out / "captions")},
             "videos": cosmos["videos"], "mosaic": str(mos), "bev_video": str(bev) if bev else None, "lidar": lid,
             "timings": tm, "memory": {k: cosmos[k] for k in ("mem_used_before_gb", "mem_used_peak_gb", "mem_used_peak_delta_gb")},
             "cosmos": {k: v for k, v in cosmos.items() if k not in ("videos", "controls", "last_frames")},
             "windows": cosmos.get("windows"),
             "geometry_dir": str(gdir), "errors": err}
    (out / "sensors.json").write_text(json.dumps(res_d, indent=1, default=str))
    return res_d


def _lidar_products(out: Path, cosmos: dict, sweeps: dict, rds_root: Path, cid: str, env: dict, weather_physics, seed: int, *,
                    ego_length: float) -> dict:
    """Point clouds / videos / checks of the generated LiDAR, plus the optional weather degradation (raw kept)."""
    from tsf.cosmos import lidar as L
    from tsf.cosmos import lidar_weather as W

    gen = L.read_safetensors(cosmos["lidar"]["generated"])[0]["frames"]
    ctl = L.read_safetensors(cosmos["lidar"]["control"])[0]["frames"]       # the trimmed control that was sent
    n = min(gen.shape[1], ctl.shape[1], len(sweeps["frame_ids"]))
    fids = sweeps["frame_ids"][:n]
    objs = L.read_objects(rds_root, cid, fids + [f + 3 for f in fids])
    sw = {"frame_ids": fids, "poses": sweeps["poses"][:n]}
    ld = out / "lidar"
    apply = (WEATHER_PHYSICS_AUTO if weather_physics == "auto" else bool(weather_physics)) and W.is_adverse(env)
    res: dict = {"generated": cosmos["lidar"]["generated"], "control": cosmos["lidar"]["control"], "sweeps": n,
                 "descriptor": cosmos["lidar"]["descriptor"], "weather": {"physics_applied": False}}
    if apply:
        raw = L.process_sweeps(gen, ctl, ld, sweeps=sw, objects=objs, name="lidar_raw", points_dir="pointclouds_raw",
                               ego_length=ego_length, videos=False, title="raw generated")
        deg, rep = W.degrade(gen[:, :n], env, seed)
        L.write_safetensors(ld / "lidar_degraded.safetensors", {"frames": deg}, {"tsf_weather": json.dumps(rep["environment"])})
        (ld / "weather_degradation.json").write_text(json.dumps({k: v for k, v in rep.items() if k != "sweeps"}, indent=1))
        chk = L.process_sweeps(deg, ctl, ld, sweeps=sw, objects=objs, name="lidar", ego_length=ego_length,
                               title=f"{env['weather']} (physics degraded)")
        res["weather"] = {"physics_applied": True, "report": str(ld / "weather_degradation.json"), "totals": rep["totals"],
                          "alpha_per_m": rep["alpha_per_m"], "raw_checks": str(ld / "lidar_raw_checks.json"),
                          "raw_summary": raw["summary"], "degraded": str(ld / "lidar_degraded.safetensors")}
        res["pointclouds_raw_dir"] = str(ld / "pointclouds_raw")
    else:
        chk = L.process_sweeps(gen, ctl, ld, sweeps=sw, objects=objs, name="lidar", ego_length=ego_length, title="generated")
    res.update(pointclouds_dir=str(ld / "pointclouds"), bev_video=chk["paths"]["bev"], range_compare=chk["paths"]["range_compare"],
               checks=str(ld / "lidar_checks.json"),
               checks_summary={"summary": chk["summary"], "objects": {k: v for k, v in chk["objects"].items() if k != "per_object"},
                               "warnings": chk["warnings"]})
    return res


def sensors_main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(prog="python -m tsf.cosmos.pipeline sensors", description=render_sensors.__doc__.splitlines()[0])
    ap.add_argument("scenario")
    ap.add_argument("out_dir")
    ap.add_argument("--frames", type=int, default=97)
    ap.add_argument("--t0", type=float)
    ap.add_argument("--cameras", help="comma separated camera keys")
    ap.add_argument("--appearance", help='JSON, e.g. {"weather":"heavy_rain","time_of_day":"21:30"}')
    ap.add_argument("--no-lidar", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--geometry-dir")
    ap.add_argument("--window-frames", type=int, default=49, help="longest single Sim-Auto request (clips are chained in windows)")
    ap.add_argument("--ensure-server", action="store_true", help="start Sim-Auto if no Cosmos server runs (refuses if one does)")
    a = ap.parse_args(argv)
    cams = tuple(a.cameras.split(",")) if a.cameras else SENSOR_CAMERAS
    if a.ensure_server:
        print(json.dumps(ensure_server(views=len(cams), lidar=not a.no_lidar)))
    r = render_sensors(a.scenario, a.out_dir, cameras=cams, frames=a.frames, t0=a.t0, appearance=json.loads(a.appearance or "null"),
                       lidar=not a.no_lidar, seed=a.seed, geometry_dir=a.geometry_dir, window_frames=a.window_frames)
    print(json.dumps({k: r[k] for k in ("timings", "memory", "mosaic")}, indent=1))
    print(json.dumps(r["lidar"].get("checks_summary", {}), indent=1))


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "sensors":
        return sensors_main(sys.argv[2:])
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
