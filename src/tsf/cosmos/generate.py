"""Render WSM control videos into camera video with NVIDIA Cosmos3-Nano Auto on Thor (vLLM-Omni Videos API).

    wsm_dir (``runs/wsm/<scenario>``)  --  per-camera WSM v4 MP4s from ``tsf.cosmos.wsm.render_wsm``
      transfer()    Cosmos3-Nano-Transfer-Auto, offline, WSM -> text2video, 35 steps / CFG 3 / shift 10
      sim_chunk()   Cosmos3-Nano-Sim-Auto, causal 4-step student; WSM-only or first-frame (I2V) conditioned,
                    ``sim_rollout`` chains chunks through the last generated frame. With ``lidar_control=<safetensors>``
                    the request is JOINT camera + LiDAR (async API, see below)
      compare_mosaic()  control | generated grid MP4

One Cosmos server at a time (``serve``); it runs next to ``tsf-llm`` in the 122 GB unified memory, so the Sim/Transfer
containers use their own name and port (``tsf-cosmos-<kind>``, 8100). Run on thor with the repo venv (needs ``httpx``,
``numpy``, ``cv2``, ffmpeg/ffprobe on PATH; ``serve`` needs docker). Commands: ``deploy/cosmos/README.md``.

    from tsf.cosmos import generate as g
    g.serve("sim"); g.wait_ready("sim")
    r = g.sim_chunk(wsm_dir, out_dir, ["camera_front_wide_120fov"], frames=49, res="480", captions=None, scenario="uc1_left_turn_intersection")

Joint camera + LiDAR (``serve("sim", kv_views=3, lidar=True)`` sizes the KV ring for it)
    The checkpoint is joint (``lidar_vae/``): one request generates the camera views and the LiDAR sweeps together,
    the LiDAR guided by an HD-map range-map control (``tsf.cosmos.lidar.render_lidar``) and by the same captions.
    Wire format: ``POST /v1/videos`` (ASYNC; ``/v1/videos/sync`` rejects LiDAR output) multipart with ``input_references``
    = [per camera: WSM control mp4 (+ first-frame png)] + [LiDAR control ``.safetensors``, one tensor ``frames`` float32
    ``[3, T, 128, 1800]`` = range m, occupied, occupied; ``T = round(frames * 10 / 30)`` sweeps, the first ``T`` of the
    control are used] and ``extra_params = {..., "lidar": {"control_reference_index": <i>, "return_output": true}}``
    (optionally ``"condition_reference_index"``: one measured sweep). Camera fps must be 30. Poll ``GET /v1/videos/{id}``
    until ``completed``; the job JSON carries ``lidar`` = {url, shape, fps, num_frames, channels (range, intensity,
    validity), units, apply_validity_mask, validity_threshold, range_projection}; ``GET /v1/videos/{id}/content`` is the
    camera-major mp4, ``GET .../lidar`` the generated safetensors (``frames`` ``[3, T, 128, 1800]`` float32).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
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


def extract_frames(video, picks: dict[int, Path]) -> dict[int, Path]:
    """Save the frames ``{index: png}`` of ``video``, decoding front to back (exact indices, no seeking)."""
    import cv2
    cap, i, left = cv2.VideoCapture(str(video)), 0, set(picks)
    while left:
        ok, fr = cap.read()
        if not ok:
            break
        if i in left:
            cv2.imwrite(str(picks[i]), fr)
            left.discard(i)
        i += 1
    cap.release()
    if left:
        raise RuntimeError(f"{video} has {i} frames, wanted {sorted(left)}")
    return {k: Path(p) for k, p in picks.items()}


def resolve_captions(captions, cameras, frames: int, scenario: str | None, fps: float = 30.0, tsir=None) -> dict[str, str]:
    """Per-camera caption text. ``captions``: dict camera -> text | sidecar path, or a directory of ``<camera>.json``, or
    None (deterministic template for ``scenario``; ``tsir`` adds its static objects / setting sentences)."""
    if captions is None:
        if scenario is None:
            raise ValueError("pass captions=... or scenario=... (template captions)")
        return template_captions(scenario, cameras, frames=frames, fps=fps, tsir=tsir)
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
def serve(kind: str, *, port: int = 8100, kv_views: int = 1, lidar: bool = False, lidar_chunk: int = 4, work_dir=None, name: str | None = None,
          extra_args: tuple[str, ...] = (), dry_run: bool = False, owner: str | None = None,
          force: bool = False, kv_res: str | None = None) -> list[str]:
    """Start ``tsf-cosmos-<kind>`` (detached) and return the docker argv. Refuses while another ``tsf-cosmos-*`` runs.

    ``work_dir`` (default ``~/tsf/dev/cosmos_gen``, mounted at /work) holds ``deploy/sim_auto_thor.yaml`` (small KV pool
    fraction), ``patches/pipeline_cosmos3_nano_sim_auto.py`` and the JIT caches (``cache/``; the first Sim-Auto request spends
    ~200 s compiling FlashAttention-4 kernels for sm_110 unless the cache is warm). Weights come from the host HF cache, offline.
    ``kv_views`` (Sim only): the stock pipeline sizes its paged KV ring for 7 views + LiDAR (27 GB); the patch sizes it for
    ``kv_views`` RGB-only views (1: 4 GB, 3: 12 GB, 7: 27 GB) so Sim-Auto fits next to ``tsf-llm``. Requests may use up to
    ``kv_views`` cameras. ``lidar=True`` (``TSF_KV_LIDAR=1``) sizes it for joint camera + LiDAR requests instead (the
    LiDAR tokens of every chunk are added to the frame unit); a server started without it cannot serve joint requests.
    ``lidar_chunk`` (``TSF_LIDAR_CHUNK``, with ``lidar=True``): sweeps per streaming chunk of the FP32 LiDAR VAE (checkpoint: 20);
    20 sweeps of 128 x 1808 at once exhaust the unified memory next to the resident LLM and get the worker OOM-killed, 4 does not.
    ``kv_res="720"`` (``TSF_KV_RES``, Sim only): size the ring for 720p requests, the resolution the model was trained at (the
    checkpoint default is 480p and a 720p request overruns a 480p ring), with ``deploy/sim_auto_thor_720.yaml`` (larger pool).

    Refuses (RuntimeError) while ANY ``tsf-cosmos-*`` container runs, including one with the same name: a caller must never
    replace another caller's live server (``force=True`` removes a running container of the SAME name only; use it for your
    own restarts). The container is labelled ``tsf.owner=<owner>`` (default ``$TSF_OWNER`` or ``<script>:<pid>``);
    ``server_owner()`` reads it back.
    """
    repo, cls = MODELS[kind]
    name = name or f"tsf-cosmos-{kind}"
    work = Path(work_dir or "~/tsf/dev/cosmos_gen").expanduser()
    running = running_servers()
    others = [n for n in running if n != name]
    if others and not dry_run:
        raise RuntimeError(f"another Cosmos server is running: {others} (owner {[server_owner(n) for n in others]}); "
                           "stop it first (one model fits next to tsf-llm)")
    if name in running and not force and not dry_run:
        raise RuntimeError(f"{name} is already running (owner {server_owner(name)!r}); refusing to replace it. Reuse it, wait for "
                           "its owner, or pass force=True if it is yours")
    who = owner or os.environ.get("TSF_OWNER") or f"{Path(sys.argv[0] or 'python').name}:{os.getpid()}"
    cmd = ["docker", "run", "-d", "--name", name, "--label", f"tsf.owner={who}", "--runtime", "nvidia", "--gpus", "all", "--ipc=host",
           "-e", "HF_HUB_OFFLINE=1", "-e", "VLLM_OMNI_VIDEO_SYNC_TIMEOUT=3600", "-e", "VLLM_OMNI_DLO_DP_WAVE_TIMEOUT=3600",
           "-e", "FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1", "-e", "FLASH_ATTENTION_CUTE_DSL_CACHE_DIR=/work/cache/fa_cute",
           "-v", f"{Path.home()}/.cache/huggingface:/root/.cache/huggingface", "-v", f"{work}:/work",
           "-v", f"{work}/cache/triton_{kind}:/root/.triton", "-p", f"127.0.0.1:{port}:8000", "-w", "/app/vllm-omni"]
    if kind == "sim":
        mod = "/usr/local/lib/python3.12/dist-packages/vllm_omni/diffusion/models/cosmos3_nano_sim_auto/pipeline_cosmos3_nano_sim_auto.py"
        cmd += ["-e", f"TSF_KV_VIEWS={kv_views}", "-e", f"TSF_KV_LIDAR={int(bool(lidar))}",
                "-e", f"TSF_LIDAR_CHUNK={int(lidar_chunk) if lidar else 0}", "-e", f"TSF_KV_RES={kv_res or ''}",
                "-v", f"{work}/patches/pipeline_cosmos3_nano_sim_auto.py:{mod}:ro"]
    cmd += [IMAGE, "vllm", "serve", repo, "--revision", "main", "--omni", "--model-class-name", cls, "--no-guardrails",
            "--host", "0.0.0.0", "--port", "8000", "--init-timeout", "1800"]
    if kind == "sim":
        deploy = "sim_auto_thor_720.yaml" if kv_res == "720" else "sim_auto_thor.yaml"
        cmd += ["--deploy-config", f"/work/deploy/{deploy}", "--tensor-parallel-size", "1"]
    else:
        cmd += ["--diffusion-compile-granularity", "regional", "--no-diffusion-compile-dynamic"]
    cmd += list(extra_args)
    if not dry_run:
        (work / "cache" / f"triton_{kind}").mkdir(parents=True, exist_ok=True)
        (work / "cache" / "fa_cute").mkdir(parents=True, exist_ok=True)
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)
        subprocess.run(cmd, check=True)
    return cmd


def evict_page_cache(dirs=("~/.cache/huggingface", "~/tsf", "~/data"), *, seconds: float = 0.0, interval: float = 3.0) -> int:
    """Drop the page cache of our own files below ``dirs`` (``posix_fadvise(DONTNEED)``, unprivileged): on Thor's unified
    memory the CUDA free-memory query does not count reclaimable cache, and Sim-Auto sizes its KV pool from it. One pass
    (``seconds=0``) or repeated every ``interval`` s for ``seconds`` (blocking; run it in a thread while a server loads).
    Returns the number of files advised."""
    n, end = 0, time.time() + seconds
    while True:
        for d in dirs:
            for root, _, files in os.walk(os.path.expanduser(d)):
                for f in files:
                    try:
                        fd = os.open(os.path.join(root, f), os.O_RDONLY)
                        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                        os.close(fd)
                        n += 1
                    except OSError:
                        pass
        if time.time() >= end:
            return n
        time.sleep(interval)


def running_servers() -> list[str]:
    """Names of the running ``tsf-cosmos-*`` containers."""
    r = subprocess.run(["docker", "ps", "--format", "{{.Names}}"], capture_output=True, text=True)
    return [n for n in r.stdout.split() if n.startswith("tsf-cosmos-")]


def server_env(name: str = "tsf-cosmos-sim") -> dict[str, str]:
    """Environment of a running server container (``TSF_KV_VIEWS`` / ``TSF_KV_LIDAR`` tell what it was sized for)."""
    r = subprocess.run(["docker", "inspect", name, "--format", "{{json .Config.Env}}"], capture_output=True, text=True)
    return dict(e.split("=", 1) for e in json.loads(r.stdout or "[]") if "=" in e)


def server_owner(name: str = "tsf-cosmos-sim") -> str | None:
    """``tsf.owner`` label of a (running) container, None if unlabelled / absent."""
    r = subprocess.run(["docker", "inspect", name, "--format", "{{index .Config.Labels \"tsf.owner\"}}"], capture_output=True, text=True)
    return (r.stdout.strip() or None) if r.returncode == 0 else None


def check_joint_server(base_url: str = DEFAULT_URL, name: str = "tsf-cosmos-sim") -> dict | None:
    """Refuse a joint camera+LiDAR request when the local server was not started for it: the container needs
    ``TSF_KV_LIDAR=1`` (KV ring sized for LiDAR) and ``TSF_LIDAR_CHUNK`` >= 1 (small LiDAR-VAE chunks; the stock 20-sweep
    chunks exhaust the unified memory and get the worker OOM-killed). Returns the relevant env, or None when it cannot be
    checked (remote ``base_url`` or no such local container)."""
    from urllib.parse import urlparse
    if urlparse(base_url).hostname not in ("127.0.0.1", "localhost", "::1"):
        return None
    try:
        env = server_env(name)
    except (OSError, ValueError):
        return None
    if not env:
        return None
    got = {k: env.get(k) for k in ("TSF_KV_VIEWS", "TSF_KV_LIDAR", "TSF_LIDAR_CHUNK")}
    if got["TSF_KV_LIDAR"] in (None, "", "0") or int(got["TSF_LIDAR_CHUNK"] or 0) < 1:
        raise RuntimeError(f"{name} was not started for joint camera+LiDAR requests ({got}); a joint request on it can exhaust "
                           f"the unified memory. Restart it with serve('sim', kv_views=N, lidar=True) / pipeline.ensure_server("
                           f"lidar=True) when its owner is done (owner: {server_owner(name)!r}).")
    return got


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
                   first_frames: dict[str, Path] | None, lidar_control: Path | None = None,
                   lidar_condition: Path | None = None) -> tuple[dict, list[Path]]:
    if res not in SIZES:
        raise ValueError(f"res must be one of {sorted(SIZES)}")
    if (frames - 1) % 4:
        raise ValueError("frames must be 4k+1 (e.g. 49, 61, 97, 201, 297)")
    if lidar_control is not None and kind != "sim":
        raise ValueError("joint camera+LiDAR is a Sim-Auto feature")
    if lidar_control is not None and abs(fps - 30.0) > 1e-6:
        raise ValueError("joint requests need camera fps 30 (the LiDAR source clock)")
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
    if lidar_control is not None:                    # numeric uploads, referenced by index into input_references
        lid = {"control_reference_index": len(uploads), "return_output": True}
        uploads.append(lidar_control)
        if lidar_condition is not None:
            lid["condition_reference_index"] = len(uploads)
            uploads.append(lidar_condition)
        extra["lidar"] = lid
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
        for p in map(Path, uploads):                 # chained first frames arrive as str paths
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


def _post_async(form: dict, uploads: list[Path], out_mp4: Path, out_lidar: Path | None, base_url: str, timeout: float,
                poll: float = 2.0) -> tuple[float, dict]:
    """Joint request through the async API: ``POST /v1/videos``, poll ``GET /v1/videos/{id}`` until ``completed`` (or raise
    with the server's error), download the camera-major mp4 and (when the job has one) the LiDAR safetensors, then
    ``DELETE`` the job (frees the server's copies). Returns ``(wall seconds from POST to files on disk, job json)``;
    the job json has ``inference_time_s``, ``stage_durations``, ``peak_memory_mb`` and the ``lidar`` descriptor."""
    mime = {".mp4": "video/mp4", ".png": "image/png", ".jpg": "image/jpeg", ".safetensors": "application/octet-stream"}
    files, fhs = [], []
    t0 = time.perf_counter()
    try:
        for p in map(Path, uploads):
            fh = open(p, "rb")
            fhs.append(fh)
            files.append(("input_references", (p.name, fh, mime.get(p.suffix, "application/octet-stream"))))
        r = httpx.post(f"{base_url}/v1/videos", data=form, files=files, timeout=httpx.Timeout(600, connect=30))
    finally:
        for fh in fhs:
            fh.close()
    if r.is_error:
        out_mp4.with_suffix(".error.txt").write_text(r.text[:20000])
        raise RuntimeError(f"HTTP {r.status_code} from POST /v1/videos: {r.text[:1500]}")
    vid = r.json()["id"]
    job: dict = r.json()
    try:
        while job.get("status") not in ("completed", "failed"):
            if time.perf_counter() - t0 > timeout:
                raise TimeoutError(f"video job {vid} not finished after {timeout:.0f}s (status {job.get('status')})")
            time.sleep(poll)
            g = httpx.get(f"{base_url}/v1/videos/{vid}", timeout=60)
            job = g.json()
        if job["status"] == "failed":
            out_mp4.with_suffix(".error.txt").write_text(json.dumps(job, indent=1)[:20000])
            raise RuntimeError(f"video job {vid} failed: {job.get('error')}")
        c = httpx.get(f"{base_url}/v1/videos/{vid}/content", timeout=httpx.Timeout(600, connect=30))
        if c.is_error:
            raise RuntimeError(f"HTTP {c.status_code} from GET /v1/videos/{vid}/content: {c.text[:500]}")
        out_mp4.write_bytes(c.content)
        if job.get("lidar"):
            if out_lidar is None:
                raise RuntimeError("the job returned LiDAR but no destination was given")
            lr = httpx.get(f"{base_url}{job['lidar']['url']}", timeout=httpx.Timeout(600, connect=30))
            if lr.is_error:
                raise RuntimeError(f"HTTP {lr.status_code} from GET {job['lidar']['url']}: {lr.text[:500]}")
            out_lidar.write_bytes(lr.content)
        dt = time.perf_counter() - t0
    finally:
        try:
            httpx.delete(f"{base_url}/v1/videos/{vid}", timeout=60)
        except httpx.HTTPError:
            pass
    return dt, job


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
         first_frames, base_url, timeout, lidar_control=None, lidar_condition=None, tsir=None) -> dict:
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
    lid_ctrl = lid_cond = None
    n_sweeps = 0
    if lidar_control is not None:                    # sweeps matching the camera window (10 Hz vs 30 fps)
        from tsf.cosmos import lidar as _lidar
        check_joint_server(base_url)
        if start % 3:
            raise ValueError("joint requests need a window start that is a multiple of 3 camera frames (one sweep)")
        n_sweeps = _lidar.required_sweeps(frames, fps)
        lid_ctrl = _lidar.trim_control(lidar_control, out_dir / "controls" / "lidar_control.safetensors", n_sweeps,
                                       start=start // 3)
        lid_cond = Path(lidar_condition) if lidar_condition is not None else None
    caps = resolve_captions(captions, cams, frames, scenario, fps, tsir)
    form, uploads = _build_request(kind, ctrl, caps, frames=frames, fps=fps, res=res, seed=seed, steps=steps,
                                   guidance=guidance, shift=shift, first_frames=first_frames, lidar_control=lid_ctrl,
                                   lidar_condition=lid_cond)
    (out_dir / "request.json").write_text(json.dumps({**form, "extra_params": json.loads(form["extra_params"]),
                                                      "uploads": [str(u) for u in uploads]}, indent=1))
    cm = out_dir / "multiview_camera_major.mp4"
    lid_out = out_dir / "lidar_generated.safetensors" if lid_ctrl is not None else None
    job: dict = {}
    with MemSampler() as ms:
        if lid_ctrl is not None:
            dt, job = _post_async(form, uploads, cm, lid_out, base_url, timeout)
        else:
            dt = _post_sync(form, uploads, cm, base_url, timeout)
    w, h = SIZES[res]
    videos = _split(cm, out_dir, cams, frames, fps, w, h)
    n_view_frames = frames * len(cams)
    res_d = {"kind": kind, "scenario": scenario, "cameras": cams, "frames": frames, "res": res, "start_frame": start,
             "seed": seed, "request_s": round(dt, 2), "s_per_frame_view": round(dt / n_view_frames, 3),
             "s_per_frame": round(dt / frames, 3), "videos": {c: str(p) for c, p in videos.items()},
             "controls": {c: str(p) for c, p in ctrl.items()}, "camera_major": str(cm), **ms.summary()}
    if lid_ctrl is not None:
        desc = job.get("lidar")
        if not desc:
            raise RuntimeError(f"joint request completed without LiDAR output: {json.dumps(job)[:800]}")
        shape = tuple(desc["shape"])
        if shape[:1] != (3,) or shape[2:] != (128, 1800) or shape[1] < n_sweeps:
            raise RuntimeError(f"unexpected LiDAR output {desc}; expected [3, >={n_sweeps}, 128, 1800]")
        (out_dir / "lidar_generated.json").write_text(json.dumps({"lidar": desc, "job": {k: v for k, v in job.items()
                                                                                          if k != "lidar"}}, indent=1))
        res_d["lidar"] = {"generated": str(lid_out), "control": str(lid_ctrl), "sweeps": n_sweeps, "shape": list(shape),
                          "descriptor": desc, "server": {"inference_time_s": job.get("inference_time_s"),
                                                         "stage_durations": job.get("stage_durations"),
                                                         "peak_memory_mb": job.get("peak_memory_mb")}}
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
              lidar_control=None, lidar_condition=None, base_url: str = DEFAULT_URL, timeout: float = 7200, tsir=None) -> dict:
    """Sim-Auto rollout of controls ``[start, start+frames)`` (server: ``serve("sim")``; 4 steps, guidance 1, fixed).

    ``first_frames``: ``{camera: png/mp4}`` RGB condition for the window's first frame (I2V); None = WSM-only text2video. The
    result carries ``last_frames`` (PNG per camera) to condition the next chunk (see ``sim_rollout``).

    ``lidar_control``: path of an HD-map LiDAR control (``tsf.cosmos.lidar.render_lidar``, ``[3, T, 128, 1800]`` safetensors,
    any ``T >= (start + frames) / 3``) makes the request JOINT: the control is trimmed to the ``round(frames * 10 / 30)``
    sweeps of the window (``start`` must be a multiple of 3, ``fps`` 30), sent through the async API, and the result gets
    ``lidar`` = {generated (``lidar_generated.safetensors``, ``[3, T, 128, 1800]``: range m, intensity, validity), control
    (the trimmed one), sweeps, shape, descriptor (server metadata), server (inference_time_s, stage_durations,
    peak_memory_mb)}; ``lidar_generated.json`` holds the job json. The server must run with ``serve(..., lidar=True)``.
    ``lidar_condition``: optional one-sweep measured safetensors that conditions the first sweep.
    """
    r = _run("sim", wsm_dir, out_dir, cameras, frames, res, captions, start=start, seed=seed, steps=4, guidance=1.0, shift=0.0,
             fps=fps, scenario=scenario, first_frames=first_frames, base_url=base_url, timeout=timeout,
             lidar_control=lidar_control, lidar_condition=lidar_condition, tsir=tsir)
    # the next chunk's conditioning frames come from the server's camera-major video: the per-camera split is a second
    # lossy encode, and chaining on it cost 13 % sharpness at every hand-off (Laplacian variance 93 -> 81)
    n = r["frames"]
    picks = {i * n + n - 1: Path(out_dir) / f"last_{c}.png" for i, c in enumerate(r["cameras"])}
    extract_frames(r["camera_major"], picks)
    r["last_frames"] = {c: str(picks[i * n + n - 1]) for i, c in enumerate(r["cameras"])}
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


# ============================================================================== windows (long clips in several requests)
WINDOW_FRAMES = 49          # longest 3-view joint request on Thor next to tsf-llm (coordinator cap). Measured: the working set beyond the idle
                            # server is ~13-18 GB for 25 AND 49 frames (it does not shrink with shorter windows), 97 frames overran 24 GB
                            # (MemAvailable 0, tsf-llm OOM-killed), so use the longest allowed window: fewer seams, same peak
_CAP_BLOCK = re.compile(r"(?m)^\[([0-9.]+)s-([0-9.]+)s\]:[ \t]*")


def window_caption(text: str, t_a: float, t_b: float, min_len: float = 0.15) -> str:
    """Blocks ``[a s-b s]: text`` of a caption restricted to ``[t_a, t_b)`` seconds and shifted to start at 0 (a window of a
    longer clip). Blocks cut shorter than ``min_len`` s are absorbed by their neighbour; the last block ends at the window end.
    A caption without time blocks is returned unchanged."""
    parts = _CAP_BLOCK.split(text.strip())
    if len(parts) < 4:
        return text
    blocks = [(float(parts[i]), float(parts[i + 1]), parts[i + 2].strip()) for i in range(1, len(parts) - 2, 3)]
    sel = [[max(a, t_a) - t_a, min(b, t_b) - t_a, body] for a, b, body in blocks if min(b, t_b) - max(a, t_a) > 1e-6]
    if not sel:
        return text
    sel[-1][1] = max(sel[-1][1], t_b - t_a)
    for blk in [b for b in sel if b[1] - b[0] < min_len and len(sel) > 1]:
        i = sel.index(blk)
        j = i + 1 if i + 1 < len(sel) else i - 1
        if j > i:
            sel[j][0] = blk[0]
        else:
            sel[j][1] = blk[1]
        sel.remove(blk)
    return "\n\n".join(f"[{a:.2f}s-{b:.2f}s]: {body}" for a, b, body in sel)


def concat_windows(parts: list, dst, *, fps: float = 30.0) -> Path:
    """Join consecutive window videos into one H.264 clip; every window after the first repeats the previous window's last
    frame (it was its condition), so its first frame is dropped."""
    filt = ["[0:v]setpts=PTS-STARTPTS[v0]"] + [f"[{i}:v]trim=start_frame=1,setpts=PTS-STARTPTS[v{i}]" for i in range(1, len(parts))]
    filt.append("".join(f"[v{i}]" for i in range(len(parts))) + f"concat=n={len(parts)}:v=1:a=0[v]")
    args = [a for p in parts for a in ("-i", str(p))]
    _ff(*args, "-filter_complex", ";".join(filt), "-map", "[v]", "-r", str(fps), "-fps_mode", "cfr", "-an", "-c:v", "libx264",
        "-crf", "12", "-pix_fmt", "yuv420p", str(dst))
    return Path(dst)


def _load_window(wdir: Path, key: str, cams: list[str]) -> dict | None:
    """Result of a window finished by an earlier call with the same request key (None if absent / different / incomplete)."""
    try:
        if json.loads((wdir / "window.json").read_text())["key"] != key:
            return None
        r = json.loads((wdir / "result.json").read_text())
        last = {c: str(wdir / f"last_{c}.png") for c in cams}
        if not all(Path(v).is_file() for v in [*r["videos"].values(), *last.values()]):
            return None
        if "lidar" in r and not Path(r["lidar"]["generated"]).is_file():
            return None
        return {**r, "last_frames": last, "resumed": True}
    except (OSError, ValueError, KeyError):
        return None


def sim_windows(wsm_dir, out_dir, cameras=None, frames: int = 97, res: str = "480", captions=None, *, window_frames: int = WINDOW_FRAMES,
                start: int = 0, first_frames: dict[str, Path] | None = None, seed: int = 0, fps: float = 30.0,
                scenario: str | None = None, lidar_control=None, base_url: str = DEFAULT_URL, timeout: float = 7200,
                before_window=None, resume: bool = True, tsir=None, drift_guard: bool = True) -> dict:
    """:func:`sim_chunk` for a clip longer than one request can hold: ``frames`` (4k+1) camera frames from control frame
    ``start`` in windows of at most ``window_frames`` that overlap by one frame; window k+1 is conditioned (I2V) on the last
    generated frame of window k, its captions are the clip's captions cut to its time span, its seed is ``seed + k``. The
    LiDAR sweeps of consecutive windows are generated from the HD-map control alone (a sweep cannot be conditioned on the
    previous window's last sweep, which lies half a window earlier than the next window's first one) and concatenated.
    ``before_window(k)`` runs before each request (memory guard). ``resume``: a window that an earlier call finished with the
    same request (captions, start, frames, seed, resolution) is reused, so a crashed run continues where it stopped. One
    window = plain :func:`sim_chunk`. ``drift_guard``: the hand-off frames are colour- and sharpness-corrected toward the
    first window's (:class:`tsf.cosmos.drift.DriftGuard`, never re-anchored: a delivered clip must not jump). Returns the sim_chunk
    dict for the whole clip (``videos`` 4k+1 frames, ``lidar`` generated / control with all sweeps) plus ``windows``
    (per-window start, frames, request_s, memory delta, server peak). Why short windows: on Thor's unified memory a 3-view
    joint request needs ~13.7 GB beyond the resident server for 49 frames and more than 24 GB for 97 (the OOM killer then
    takes tsf-llm), see ``WINDOW_FRAMES``."""
    out_dir = Path(out_dir)
    if frames <= window_frames:
        return sim_chunk(wsm_dir, out_dir, cameras, frames, res, captions, start=start, first_frames=first_frames, seed=seed,
                         fps=fps, scenario=scenario, lidar_control=lidar_control, base_url=base_url, timeout=timeout, tsir=tsir)
    stride = window_frames - 1
    if stride % (12 if lidar_control is not None else 4) or (frames - 1) % 4:
        raise ValueError("windows need window_frames-1 a multiple of 12 (joint) / 4 and frames = 4k+1")
    out_dir.mkdir(parents=True, exist_ok=True)
    cams = [c for c in CAMERA_ORDER if c in (cameras or CAMERA_ORDER)]
    scenario = scenario or Path(wsm_dir).name
    caps = resolve_captions(captions, cams, frames, scenario, fps, tsir)
    runs, ff, s = [], first_frames, start
    from tsf.cosmos.drift import DriftGuard
    guard = DriftGuard(reanchor_below=0.0, veil_below=0.0) if drift_guard else None
    while True:
        wf = min(window_frames, start + frames - s)
        k = len(runs)
        caps_k = {c: window_caption(caps[c], (s - start) / fps, (s - start + wf) / fps) for c in cams}
        wdir = out_dir / f"window{k:02d}"
        key = hashlib.sha1(json.dumps([caps_k, s, wf, seed + k, res, lidar_control is not None, ff is not None], sort_keys=True)
                           .encode()).hexdigest()[:16]
        r = _load_window(wdir, key, cams) if resume else None
        if r is None:
            if before_window is not None:
                before_window(k)
            r = sim_chunk(wsm_dir, wdir, cams, wf, res, caps_k, start=s, first_frames=ff, seed=seed + k, fps=fps,
                          scenario=scenario, lidar_control=lidar_control, base_url=base_url, timeout=timeout)
            (wdir / "window.json").write_text(json.dumps({"key": key, "start": s, "frames": wf, "seed": seed + k}))
        runs.append(r)
        ff = {c: Path(p) for c, p in r["last_frames"].items()}
        if guard is not None:
            if k == 0:
                guard.reference(ff)
            else:
                ff, r["drift"] = guard(ff, wdir / "cond")
        s += wf - 1
        if s >= start + frames - 1:
            break
    videos = {c: str(concat_windows([r["videos"][c] for r in runs], out_dir / f"{c}.mp4", fps=fps)) for c in cams}
    for c in cams:
        n = probe(videos[c])["frames"]
        if n != frames:
            raise RuntimeError(f"joined {c} has {n} frames, expected {frames}")
    src = find_controls(wsm_dir, cams)
    (out_dir / "controls").mkdir(exist_ok=True)
    ctrl = {c: str(trim_control(src[c], out_dir / "controls" / f"{c}.mp4", start=start, frames=frames, fps=fps)) for c in cams}
    dt = sum(r["request_s"] for r in runs)
    res_d = {"kind": "sim", "scenario": scenario, "cameras": cams, "frames": frames, "res": res, "start_frame": start, "seed": seed,
             "window_frames": window_frames, "request_s": round(dt, 2), "s_per_frame_view": round(dt / (frames * len(cams)), 3),
             "s_per_frame": round(dt / frames, 3), "videos": videos, "controls": ctrl, "last_frames": runs[-1]["last_frames"],
             "mem_used_before_gb": runs[0]["mem_used_before_gb"], "mem_used_peak_gb": max(r["mem_used_peak_gb"] for r in runs),
             "mem_used_peak_delta_gb": max(r["mem_used_peak_delta_gb"] for r in runs),
             "windows": [{"dir": str(out_dir / f"window{i:02d}"), "start_frame": r["start_frame"], "frames": r["frames"],
                          "request_s": r["request_s"], "mem_used_before_gb": r["mem_used_before_gb"],
                          "mem_used_peak_gb": r["mem_used_peak_gb"], "mem_used_peak_delta_gb": r["mem_used_peak_delta_gb"],
                          "resumed": bool(r.get("resumed")), "server": r.get("lidar", {}).get("server")} for i, r in enumerate(runs)]}
    if lidar_control is not None:
        import numpy as np

        from tsf.cosmos import lidar as _lidar
        n_sweeps = _lidar.required_sweeps(frames, fps)
        gen = np.concatenate([_lidar.read_safetensors(r["lidar"]["generated"])[0]["frames"][:, :r["lidar"]["sweeps"]] for r in runs], axis=1)
        if gen.shape[1] != n_sweeps:
            raise RuntimeError(f"joined LiDAR has {gen.shape[1]} sweeps, expected {n_sweeps}")
        gpath = out_dir / "lidar_generated.safetensors"
        _lidar.write_safetensors(gpath, {"frames": gen}, {"tsf_windows": json.dumps([r["frames"] for r in runs])})
        cpath = _lidar.trim_control(lidar_control, out_dir / "controls" / "lidar_control.safetensors", n_sweeps, start=start // 3)
        desc = dict(runs[-1]["lidar"]["descriptor"], shape=list(gen.shape), num_frames=int(gen.shape[1]))
        res_d["lidar"] = {"generated": str(gpath), "control": str(cpath), "sweeps": n_sweeps, "shape": list(gen.shape), "descriptor": desc,
                          "server": {"inference_time_s": round(sum(r["lidar"]["server"]["inference_time_s"] or 0 for r in runs), 2),
                                     "peak_memory_mb": max(r["lidar"]["server"]["peak_memory_mb"] or 0 for r in runs),
                                     "windows": [r["lidar"]["server"] for r in runs]}}
    (out_dir / "result.json").write_text(json.dumps(res_d, indent=1))
    return res_d



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
