"""Render WSM control videos into camera video with NVIDIA Cosmos3-Nano Auto on Thor (vLLM-Omni Videos API).

    wsm_dir (``runs/wsm/<scenario>``)  --  per-camera WSM v4 MP4s from ``tsf.cosmos.wsm.render_wsm``
      transfer()    Cosmos3-Nano-Transfer-Auto, offline, WSM -> text2video, 35 steps / CFG 3 / shift 10
      sim_chunk()   Cosmos3-Nano-Sim-Auto, causal 4-step student; WSM-only or first-frame (I2V) conditioned,
                    ``sim_rollout`` chains chunks through the last generated frame
      compare_mosaic()  control | generated grid MP4

One Cosmos server at a time (``serve``); it runs next to ``tsf-llm`` in the 122 GB unified memory, so the Sim/Transfer
containers use their own name and port (``tsf-cosmos-<kind>``, 8100). Run on thor with the repo venv (needs ``httpx``,
``numpy``, ``cv2``, ffmpeg/ffprobe on PATH; ``serve`` needs docker). Commands: ``deploy/cosmos/README.md``.

    from tsf.cosmos import generate as g
    g.serve("sim"); g.wait_ready("sim")
    r = g.sim_chunk(wsm_dir, out_dir, ["camera_front_wide_120fov"], frames=49, res="480", captions=None, scenario="uc1_left_turn_intersection")
"""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path

import httpx

from tsf.cosmos.captions import CAMERAS_7, caption_window, template_captions

CAMERA_ORDER = CAMERAS_7                         # canonical Auto order = trained camera ids 0..6
MODELS = {                                       # kind -> (HF repo, vLLM-Omni pipeline class)
    "transfer": ("nvidia/Cosmos3-Nano-Transfer-Auto", "Cosmos3MultiviewPipeline"),
    "sim": ("nvidia/Cosmos3-Nano-Sim-Auto", "Cosmos3NanoSimAutoPipeline"),
}
SIZES = {"480": (832, 480), "720": (1280, 720)}  # per-camera output, training-matched 16:9 presets
DEFAULT_URL = os.environ.get("COSMOS_URL", "http://127.0.0.1:8100")
IMAGE = "vllm/vllm-omni:cosmos3-aarch64"
NEGATIVE = ("The driving-camera video has unstable shading, with faint, dirty-looking blotches drifting across the sky and other "
            "smooth surfaces. Surface textures crawl, fine details shimmer, and object edges wobble. Motion is excessively smeared, "
            "leaving long trails and ghosted outlines. The image remains unnaturally soft, vehicles deform, road geometry drifts, "
            "and exposure changes abruptly. The scene has a computer-generated, game-rendered appearance, with plastic-looking "
            "materials and artificially exaggerated contrast. Objects morph, merge, split, or abruptly appear or disappear.")


# ============================================================================== inputs
def find_controls(wsm_dir, cameras=None) -> dict[str, Path]:
    """``{camera_key: control mp4}`` under a ``runs/wsm/<scenario>`` dir (``wsm/world_scenario`` or ``transfer_auto/controls``)."""
    root = Path(wsm_dir)
    cams = list(cameras) if cameras else list(CAMERA_ORDER)
    out = {}
    for cam in cams:
        cands = [*root.glob(f"wsm/world_scenario/ftheta_{cam}/*_0.mp4"), *root.glob(f"world_scenario/ftheta_{cam}/*_0.mp4"),
                 root / "transfer_auto" / "controls" / f"{cam}.mp4", root / "controls" / f"{cam}.mp4"]
        hit = next((p for p in cands if p.is_file()), None)
        if hit is None:
            raise FileNotFoundError(f"no WSM control for {cam} under {root}")
        out[cam] = hit
    return out


_FFMPEG: list[str] = []


def ffmpeg_exe() -> str:
    """ffmpeg with libx264: the system one if it has the encoder (Thor's apt build only has NVENC), else imageio-ffmpeg's."""
    if not _FFMPEG:
        exe = shutil.which("ffmpeg")
        if exe is None or b"libx264" not in subprocess.run([exe, "-hide_banner", "-encoders"], capture_output=True).stdout:
            import imageio_ffmpeg
            exe = imageio_ffmpeg.get_ffmpeg_exe()
        _FFMPEG.append(exe)
    return _FFMPEG[0]


def _ff(*args: str) -> None:
    r = subprocess.run([ffmpeg_exe(), "-v", "error", "-y", *args], capture_output=True, text=True)
    if r.returncode or r.stderr.strip():
        raise RuntimeError(f"ffmpeg failed: {r.stderr[-800:]}")


def probe(path) -> dict:
    """``{frames, fps, width, height}`` of a video (counted decode through OpenCV)."""
    import cv2
    cap = cv2.VideoCapture(str(path))
    info = {"fps": cap.get(cv2.CAP_PROP_FPS), "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}
    n = 0
    while cap.grab():
        n += 1
    cap.release()
    return {"frames": n, **info}


def trim_control(src, dst, *, start: int, frames: int, fps: float = 30.0) -> Path:
    """Copy frames ``[start, start+frames)`` of ``src`` to ``dst`` (H.264, yuv420p, no resize)."""
    _ff("-i", str(src), "-vf", f"trim=start_frame={start}:end_frame={start + frames},setpts=N/({fps}*TB)", "-r", str(fps),
        "-fps_mode", "cfr", "-an", "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", "-frames:v", str(frames), str(dst))
    return Path(dst)


def extract_frame(video, png, index: int = -1) -> Path:
    """Save frame ``index`` (negative = from the end) of ``video`` as PNG; used as the next chunk's RGB condition."""
    import cv2
    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, index if index >= 0 else n + index)
    ok, fr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"cannot read frame {index} of {video}")
    cv2.imwrite(str(png), fr)
    return Path(png)


def resolve_captions(captions, cameras, frames: int, scenario: str | None, fps: float = 30.0) -> dict[str, str]:
    """Per-camera caption text. ``captions``: dict camera -> text | sidecar path, or a directory of ``<camera>.json``, or
    None (deterministic template for ``scenario``)."""
    if captions is None:
        if scenario is None:
            raise ValueError("pass captions=... or scenario=... (template captions)")
        return template_captions(scenario, cameras, frames=frames, fps=fps)
    if isinstance(captions, (str, Path)):
        captions = {c: Path(captions) / f"{c}.json" for c in cameras}
    out = {}
    for cam in cameras:
        v = captions[cam]
        if isinstance(v, Path) or (isinstance(v, str) and v.endswith(".json") and Path(v).is_file()):
            out[cam] = caption_window(json.loads(Path(v).read_text()), frames, fps)
        else:
            out[cam] = str(v)
    return out


# ============================================================================== memory sampling
class MemSampler:
    """Background sampler of system memory (unified memory => CPU + GPU) from ``/proc/meminfo`` and container RSS-free totals."""

    def __init__(self, interval: float = 0.5):
        self.interval, self.samples, self._stop = interval, [], threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    @staticmethod
    def used_gb() -> float:
        m = {k: int(v.split()[0]) for k, v in (ln.split(":", 1) for ln in Path("/proc/meminfo").read_text().splitlines())}
        return (m["MemTotal"] - m["MemAvailable"]) / 2**20

    def _run(self) -> None:
        while not self._stop.is_set():
            self.samples.append(self.used_gb())
            self._stop.wait(self.interval)

    def __enter__(self):
        self.base = self.used_gb()
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()
        self.peak = max(self.samples + [self.base])

    def summary(self) -> dict:
        return {"mem_used_before_gb": round(self.base, 1), "mem_used_peak_gb": round(self.peak, 1),
                "mem_used_peak_delta_gb": round(self.peak - self.base, 1)}


# ============================================================================== server (docker)
def serve(kind: str, *, port: int = 8100, kv_views: int = 1, work_dir=None, name: str | None = None,
          extra_args: tuple[str, ...] = (), dry_run: bool = False) -> list[str]:
    """Start ``tsf-cosmos-<kind>`` (detached) and return the docker argv. Refuses while another ``tsf-cosmos-*`` runs.

    ``work_dir`` (default ``~/tsf/dev/cosmos_gen``, mounted at /work) holds ``deploy/sim_auto_thor.yaml`` (small KV pool
    fraction), ``patches/pipeline_cosmos3_nano_sim_auto.py`` and the JIT caches (``cache/``; the first Sim-Auto request spends
    ~200 s compiling FlashAttention-4 kernels for sm_110 unless the cache is warm). Weights come from the host HF cache, offline.
    ``kv_views`` (Sim only): the stock pipeline sizes its paged KV ring for 7 views + LiDAR (27 GB); the patch sizes it for
    ``kv_views`` RGB-only views (1: 4 GB, 3: 12 GB, 7: 27 GB) so Sim-Auto fits next to ``tsf-llm``. Requests may use up to
    ``kv_views`` cameras.
    """
    repo, cls = MODELS[kind]
    name = name or f"tsf-cosmos-{kind}"
    work = Path(work_dir or "~/tsf/dev/cosmos_gen").expanduser()
    running = subprocess.run(["docker", "ps", "--format", "{{.Names}}"], capture_output=True, text=True).stdout.split()
    others = [n for n in running if n.startswith("tsf-cosmos-") and n != name]
    if others and not dry_run:
        raise RuntimeError(f"another Cosmos server is running: {others}; stop it first (one model fits next to tsf-llm)")
    cmd = ["docker", "run", "-d", "--name", name, "--runtime", "nvidia", "--gpus", "all", "--ipc=host",
           "-e", "HF_HUB_OFFLINE=1", "-e", "VLLM_OMNI_VIDEO_SYNC_TIMEOUT=3600", "-e", "VLLM_OMNI_DLO_DP_WAVE_TIMEOUT=3600",
           "-e", "FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1", "-e", "FLASH_ATTENTION_CUTE_DSL_CACHE_DIR=/work/cache/fa_cute",
           "-v", f"{Path.home()}/.cache/huggingface:/root/.cache/huggingface", "-v", f"{work}:/work",
           "-v", f"{work}/cache/triton_{kind}:/root/.triton", "-p", f"127.0.0.1:{port}:8000", "-w", "/app/vllm-omni"]
    if kind == "sim":
        mod = "/usr/local/lib/python3.12/dist-packages/vllm_omni/diffusion/models/cosmos3_nano_sim_auto/pipeline_cosmos3_nano_sim_auto.py"
        cmd += ["-e", f"TSF_KV_VIEWS={kv_views}", "-v", f"{work}/patches/pipeline_cosmos3_nano_sim_auto.py:{mod}:ro"]
    cmd += [IMAGE, "vllm", "serve", repo, "--revision", "main", "--omni", "--model-class-name", cls, "--no-guardrails",
            "--host", "0.0.0.0", "--port", "8000", "--init-timeout", "1800"]
    if kind == "sim":
        cmd += ["--deploy-config", "/work/deploy/sim_auto_thor.yaml", "--tensor-parallel-size", "1"]
    else:
        cmd += ["--diffusion-compile-granularity", "regional", "--no-diffusion-compile-dynamic"]
    cmd += list(extra_args)
    if not dry_run:
        (work / "cache" / f"triton_{kind}").mkdir(parents=True, exist_ok=True)
        (work / "cache" / "fa_cute").mkdir(parents=True, exist_ok=True)
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)
        subprocess.run(cmd, check=True)
    return cmd


def stop(kind: str, name: str | None = None) -> None:
    """Stop and remove this module's server container (never touches ``tsf-llm``)."""
    name = name or f"tsf-cosmos-{kind}"
    if not name.startswith("tsf-cosmos-"):
        raise ValueError("refusing to stop a container that is not a tsf-cosmos-* server")
    subprocess.run(["docker", "rm", "-f", name], capture_output=True)


def wait_ready(kind: str, base_url: str = DEFAULT_URL, timeout: float = 1800, interval: float = 5.0) -> float:
    """Poll ``/v1/models`` until the server lists the model; returns seconds waited."""
    repo = MODELS[kind][0]
    t0, deadline = time.monotonic(), time.monotonic() + timeout
    while True:
        try:
            r = httpx.get(f"{base_url}/v1/models", timeout=10)
            if r.status_code == 200 and repo in [m["id"] for m in r.json().get("data", [])]:
                return time.monotonic() - t0
        except httpx.HTTPError:
            pass
        if time.monotonic() > deadline:
            raise TimeoutError(f"{repo} not ready at {base_url} after {timeout:.0f}s")
        time.sleep(interval)


# ============================================================================== requests
def _build_request(kind: str, controls: dict[str, Path], captions: dict[str, str], *, frames: int, fps: float, res: str,
                   seed: int, steps: int, guidance: float, shift: float,
                   first_frames: dict[str, Path] | None) -> tuple[dict, list[Path]]:
    if res not in SIZES:
        raise ValueError(f"res must be one of {sorted(SIZES)}")
    if (frames - 1) % 4:
        raise ValueError("frames must be 4k+1 (e.g. 49, 61, 97, 201, 297)")
    cams = [c for c in CAMERA_ORDER if c in controls]
    if len(cams) != len(controls):
        raise ValueError(f"unknown cameras: {sorted(set(controls) - set(CAMERA_ORDER))}")
    uploads, views = [], []
    for cam in cams:
        v = {"camera_key": cam, "prompt": captions[cam], "control_reference_index": len(uploads)}
        uploads.append(controls[cam])
        if first_frames:
            v["vision_reference_index"] = len(uploads)
            uploads.append(first_frames[cam])
        views.append(v)
    w, h = SIZES[res]
    extra: dict = {"wsm": {}, "multiview": {"views": views}, "resolution": res, "aspect_ratio": "16,9"}
    if first_frames:
        extra["multiview"]["condition_video_as_image"] = True
    if kind == "sim":
        extra["multiview"]["num_frames"] = frames
        extra.update(session_id=f"tsf-{uuid.uuid4().hex[:12]}", reset=True, close_session=True)
        form = {"num_inference_steps": "4", "guidance_scale": "1.0", "enable_frame_interpolation": "false"}
    else:
        extra["per_view_negative_prompt"] = NEGATIVE
        form = {"num_inference_steps": str(steps), "guidance_scale": str(guidance), "flow_shift": str(shift)}
    form.update(prompt=captions[cams[0]], model=MODELS[kind][0], size=f"{w}x{h}", fps=str(fps), num_frames=str(frames),
                seed=str(seed), extra_params=json.dumps(extra))
    return form, uploads


def _post_sync(form: dict, uploads: list[Path], out_mp4: Path, base_url: str, timeout: float) -> float:
    mime = {".mp4": "video/mp4", ".png": "image/png", ".jpg": "image/jpeg"}
    files, fhs = [], []
    try:
        for p in uploads:
            fh = open(p, "rb")
            fhs.append(fh)
            files.append(("input_references", (p.name, fh, mime.get(p.suffix, "application/octet-stream"))))
        t0 = time.perf_counter()
        r = httpx.post(f"{base_url}/v1/videos/sync", data=form, files=files, headers={"Accept": "video/mp4"},
                       timeout=httpx.Timeout(timeout, connect=30))
        dt = time.perf_counter() - t0
    finally:
        for fh in fhs:
            fh.close()
    if r.is_error or not r.headers.get("content-type", "").startswith("video/mp4"):
        out_mp4.with_suffix(".error.txt").write_text(r.text[:20000])
        raise RuntimeError(f"HTTP {r.status_code} from /v1/videos/sync: {r.text[:1500]}")
    out_mp4.write_bytes(r.content)
    return dt


def _split(camera_major: Path, out_dir: Path, cams: list[str], frames: int, fps: float, w: int, h: int) -> dict[str, Path]:
    info = probe(camera_major)
    if (info["frames"], info["width"], info["height"]) != (len(cams) * frames, w, h):
        raise RuntimeError(f"unexpected camera-major response {info}; expected {len(cams) * frames} frames at {w}x{h}")
    outs = {}
    for i, cam in enumerate(cams):
        p = out_dir / f"vision_view{i:02d}_{cam}.mp4"
        _ff("-i", str(camera_major), "-vf", f"trim=start_frame={i * frames}:end_frame={(i + 1) * frames},setpts=N/({fps}*TB)",
            "-r", str(fps), "-fps_mode", "cfr", "-an", "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", str(p))
        outs[cam] = p
    return outs


def _run(kind: str, wsm_dir, out_dir, cameras, frames, res, captions, *, start, seed, steps, guidance, shift, fps, scenario,
         first_frames, base_url, timeout) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cams = [c for c in CAMERA_ORDER if c in (cameras or CAMERA_ORDER)]
    if not cams:
        raise ValueError("no cameras")
    scenario = scenario or Path(wsm_dir).name
    src = find_controls(wsm_dir, cams)
    total = probe(src[cams[0]])["frames"]
    if start + frames > total:
        raise ValueError(f"controls have {total} frames; need {start + frames}")
    (out_dir / "controls").mkdir(exist_ok=True)
    ctrl = {c: trim_control(src[c], out_dir / "controls" / f"{c}.mp4", start=start, frames=frames, fps=fps) for c in cams}
    caps = resolve_captions(captions, cams, frames, scenario, fps)
    form, uploads = _build_request(kind, ctrl, caps, frames=frames, fps=fps, res=res, seed=seed, steps=steps,
                                   guidance=guidance, shift=shift, first_frames=first_frames)
    (out_dir / "request.json").write_text(json.dumps({**form, "extra_params": json.loads(form["extra_params"]),
                                                      "uploads": [str(u) for u in uploads]}, indent=1))
    cm = out_dir / "multiview_camera_major.mp4"
    with MemSampler() as ms:
        dt = _post_sync(form, uploads, cm, base_url, timeout)
    w, h = SIZES[res]
    videos = _split(cm, out_dir, cams, frames, fps, w, h)
    n_view_frames = frames * len(cams)
    res_d = {"kind": kind, "scenario": scenario, "cameras": cams, "frames": frames, "res": res, "start_frame": start,
             "seed": seed, "request_s": round(dt, 2), "s_per_frame_view": round(dt / n_view_frames, 3),
             "s_per_frame": round(dt / frames, 3), "videos": {c: str(p) for c, p in videos.items()},
             "controls": {c: str(p) for c, p in ctrl.items()}, "camera_major": str(cm), **ms.summary()}
    (out_dir / "result.json").write_text(json.dumps(res_d, indent=1))
    return res_d


def transfer(wsm_dir, out_dir, cameras=None, frames: int = 297, res: str = "480", captions=None, *, start: int = 0,
             seed: int = 0, steps: int = 35, guidance: float = 3.0, shift: float = 10.0, fps: float = 30.0,
             scenario: str | None = None, base_url: str = DEFAULT_URL, timeout: float = 7200) -> dict:
    """Offline Transfer-Auto: WSM controls (+ captions) -> camera videos (server: ``serve("transfer")``).

    ``frames`` is 4k+1 starting at control frame ``start``; ``res`` '480' (832x480) or '720' (1280x720); ``captions`` as in
    ``resolve_captions`` (None = template for ``scenario``, default the wsm_dir name). Writes ``vision_view<NN>_<camera>.mp4``,
    ``controls/``, ``request.json``, ``result.json`` (timings, ``s_per_frame_view``, peak unified memory) into ``out_dir``.
    """
    return _run("transfer", wsm_dir, out_dir, cameras, frames, res, captions, start=start, seed=seed, steps=steps,
                guidance=guidance, shift=shift, fps=fps, scenario=scenario, first_frames=None, base_url=base_url, timeout=timeout)


def sim_chunk(wsm_dir, out_dir, cameras=None, frames: int = 49, res: str = "480", captions=None, *, start: int = 0,
              first_frames: dict[str, Path] | None = None, seed: int = 0, fps: float = 30.0, scenario: str | None = None,
              base_url: str = DEFAULT_URL, timeout: float = 7200) -> dict:
    """Sim-Auto rollout of controls ``[start, start+frames)`` (server: ``serve("sim")``; 4 steps, guidance 1, fixed).

    ``first_frames``: ``{camera: png/mp4}`` RGB condition for the window's first frame (I2V); None = WSM-only text2video. The
    result carries ``last_frames`` (PNG per camera) to condition the next chunk (see ``sim_rollout``).
    """
    r = _run("sim", wsm_dir, out_dir, cameras, frames, res, captions, start=start, seed=seed, steps=4, guidance=1.0, shift=0.0,
             fps=fps, scenario=scenario, first_frames=first_frames, base_url=base_url, timeout=timeout)
    r["last_frames"] = {c: str(extract_frame(p, Path(out_dir) / f"last_{c}.png", -1)) for c, p in r["videos"].items()}
    return r


def sim_rollout(wsm_dir, out_dir, cameras=None, chunks: int = 3, chunk_frames: int = 49, res: str = "480", captions=None, *,
                first_frames: dict[str, Path] | None = None, seed: int = 0, scenario: str | None = None, **kw) -> dict:
    """Closed-loop style rollout: ``chunks`` windows of ``chunk_frames`` where window k+1 starts from the last generated frame
    of window k (windows overlap by one control frame). ``first_frames`` seeds window 0 (None = WSM-only)."""
    out_dir, results, ff = Path(out_dir), [], first_frames
    for k in range(chunks):
        r = sim_chunk(wsm_dir, out_dir / f"chunk{k:02d}", cameras, chunk_frames, res, captions, start=k * (chunk_frames - 1),
                      first_frames=ff, seed=seed + k, scenario=scenario, **kw)
        results.append(r)
        ff = {c: Path(p) for c, p in r["last_frames"].items()}
    tot = sum(r["request_s"] for r in results)
    return {"chunks": results, "total_s": round(tot, 2), "frames": chunks * (chunk_frames - 1) + 1}


# ============================================================================== mosaic
def compare_mosaic(controls: dict[str, Path], generated: dict[str, Path], out_mp4, *, title: str = "", max_width: int = 1920,
                   fps: float = 30.0) -> Path:
    """Grid MP4: per camera the WSM control (top) over the generated video (bottom); up to 4 camera columns per row."""
    import cv2
    import imageio.v2 as imageio
    import numpy as np

    cams = [c for c in CAMERA_ORDER if c in generated]
    cols = min(len(cams), 4)
    rows = math.ceil(len(cams) / cols)
    cw = max_width // cols
    ch = int(round(cw * 9 / 16))
    caps = {c: (cv2.VideoCapture(str(controls[c])), cv2.VideoCapture(str(generated[c]))) for c in cams}
    n = min(int(min(a.get(cv2.CAP_PROP_FRAME_COUNT), b.get(cv2.CAP_PROP_FRAME_COUNT))) for a, b in caps.values())
    hh = rows * 2 * ch + 36
    wr = imageio.get_writer(str(out_mp4), fps=fps, codec="libx264", macro_block_size=1,
                            output_params=["-crf", "20", "-preset", "fast", "-pix_fmt", "yuv420p"])
    try:
        for k in range(n):
            canvas = np.zeros((hh, cols * cw, 3), np.uint8)
            cv2.putText(canvas, f"{title}   frame {k + 1}/{n}   top: WSM control   bottom: Cosmos3-Nano generated", (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (235, 235, 235), 1, cv2.LINE_AA)
            for i, c in enumerate(cams):
                r, col = divmod(i, cols)
                for j, cap in enumerate(caps[c]):
                    ok, fr = cap.read()
                    if not ok:
                        continue
                    fr = cv2.cvtColor(cv2.resize(fr, (cw, ch), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
                    if j == 0:
                        cv2.putText(fr, c.replace("camera_", ""), (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                                    cv2.LINE_AA)
                    y = 36 + (r * 2 + j) * ch
                    canvas[y:y + ch, col * cw:(col + 1) * cw] = fr
            wr.append_data(canvas)
    finally:
        wr.close()
        for a, b in caps.values():
            a.release()
            b.release()
    return Path(out_mp4)
