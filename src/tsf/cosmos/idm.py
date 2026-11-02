"""Ego motion from video with NVIDIA Cosmos3-Edge inverse dynamics (vLLM-Omni on Thor): video -> ego actions -> track.

    window_video()        61 frames @ 10 fps, 832x480 H.264 window cut from any dashcam video
    serve() / stop()      ``tsf-cosmos-idm`` container (one heavy model at a time; ``free_to_start`` checks the rules)
    inverse_dynamics()    POST /v1/videos (action_mode=inverse_dynamics, domain av) + poll -> (60, 9) raw action
    actions_to_poses()    raw action -> camera-to-world poses (frozen convention below) -> ``vipe.ego_track`` planar track
    pai_gt()              ground-truth camera poses of a PhysicalAI-AV window (virtual pinhole camera of deploy/vipe)
    metrics()             speed / yaw-rate MAE, ATE (rigid SE(2) / SE(3), no scale), 1 s RPE of a track vs ground truth

Action convention (frozen, see ``CONVENTION``; calibrated on PhysicalAI-AV, evidence in ``scripts/idm_compare.py``)
    The model returns one 9-vector per frame step: ``[t(3), r1(3), r2(3)]``. ``t`` is the translation of camera ``i+1``
    expressed in the axes of camera ``i`` (OpenCV: x right, y down, z forward), ``r1``/``r2`` are the first two COLUMNS
    of the relative rotation (6D encoding; third column = r1 x r2 after Gram-Schmidt). Camera-to-world poses chain as
    ``P[i+1] = P[i] @ [R_i | t_i]``. The 60 actions of a 61-frame, 10 fps window therefore are the 60 inter-frame steps.
Everything is data-local: videos and tracks stay on the host that runs the container (thor3).
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import httpx
import numpy as np

REPO = "nvidia/Cosmos3-Edge"
IMAGE = "vllm/vllm-omni:cosmos3-aarch64"
NAME = "tsf-cosmos-idm"
DEFAULT_URL = os.environ.get("COSMOS_IDM_URL", "http://127.0.0.1:8100")
FRAMES, FPS, SIZE = 61, 10.0, (832, 480)         # input window: 6 s at 10 fps, 480p (card: AV 61 frames @ 10 fps)
ACTION_DIM, ACTION_STEPS = 9, 60

# Frozen by ``scripts/idm_compare.py calibrate`` on 5 PhysicalAI-AV windows (calib_edge.json on thor3). ``scale`` / ``rot_scale``
# are logged/Cosmos ratios of translation (m) and rotation angle for the 90 deg virtual pinhole camera of deploy/vipe (the model
# over-estimates both by the same ~1.4x: it assumes a wider field of view than the 88.6 deg crop it is shown).
CONVENTION = {"rot6d": "cols", "chain": "right", "scale": 0.733, "rot_scale": 0.699, "lag": 0}   # chain: right P[i+1]=P[i]@T, left P[i+1]=T@P[i]


# ============================================================================== serving
def config() -> dict:
    """``deploy/cosmos/idm_edge_thor.yaml`` (serving + request defaults); {} when the file is not next to this checkout."""
    import yaml
    for root in (Path(__file__).resolve().parents[3], Path.home() / "tsf" / "TanitScenarioFlyWheel"):
        p = root / "deploy" / "cosmos" / "idm_edge_thor.yaml"
        if p.is_file():
            return yaml.safe_load(p.read_text())
    return {}


def mem_available_gb() -> float:
    with open("/proc/meminfo") as f:
        return next(int(ln.split()[1]) for ln in f if ln.startswith("MemAvailable:")) / 2**20


def free_to_start(min_avail_gb: float = 40.0, t1_log: str = "~/tsf/logs/t1.log") -> list[str]:
    """Reasons why a GPU-heavy container must not start now (empty list = go): another ``tsf-vipe`` or ``tsf-cosmos-*``
    container is running, the Cosmos T1 closed loop has not finished, or less than ``min_avail_gb`` is available."""
    why = []
    ps = subprocess.run(["docker", "ps", "--format", "{{.Names}} {{.Image}}"], capture_output=True, text=True).stdout
    for ln in ps.splitlines():
        name, _, image = ln.partition(" ")
        if "tsf-vipe" in image or "tsf-vipe" in name or (name.startswith("tsf-cosmos-") and name != NAME):
            why.append(f"container running: {ln}")
    p = Path(t1_log).expanduser()
    if p.exists():
        txt = p.read_text(errors="replace")
        if "T1 exit" not in txt and "cosmos failed" not in txt:
            why.append("T1 closed loop not finished")
    elif Path("~/tsf/dev/t1_after_api.sh").expanduser().exists() and subprocess.run(
            ["pgrep", "-f", "^/bin/bash .*t1_after_api[.]sh"], capture_output=True).returncode == 0:
        why.append("t1_after_api.sh pending")
    if mem_available_gb() < min_avail_gb:
        why.append(f"MemAvailable {mem_available_gb():.0f} GB < {min_avail_gb:g} GB")
    return why


def serve(*, port: int = 8100, work_dir=None, name: str = NAME, extra_args: tuple[str, ...] = (),
          deploy_config: str | None = None, dry_run: bool = False) -> list[str]:
    """Start ``tsf-cosmos-idm`` (detached; ``vllm serve nvidia/Cosmos3-Edge --omni``) and return the docker argv.

    Weights come from the host HF cache (offline), JIT caches from ``work_dir`` (default ``~/tsf/dev/cosmos_gen``, same
    as ``tsf.cosmos.generate``). ``deploy_config`` is a path inside the container (``/work/deploy/idm_edge_thor.yaml``).
    """
    work = Path(work_dir or "~/tsf/dev/cosmos_gen").expanduser()
    cfg = config()
    env = cfg.get("env") or {"HF_HUB_OFFLINE": "1", "VLLM_OMNI_VIDEO_SYNC_TIMEOUT": "3600", "FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED": "1",
                            "FLASH_ATTENTION_CUTE_DSL_CACHE_DIR": "/work/cache/fa_cute"}
    cmd = ["docker", "run", "-d", "--name", name, "--runtime", "nvidia", "--gpus", "all", "--ipc=host",
           *[x for k, v in env.items() for x in ("-e", f"{k}={v}")],
           "-v", f"{Path.home()}/.cache/huggingface:/root/.cache/huggingface", "-v", f"{work}:/work",
           "-v", f"{work}/cache/triton_idm:/root/.triton", "-p", f"127.0.0.1:{port}:8000", "-w", "/app/vllm-omni",
           cfg.get("image", IMAGE), "vllm", "serve", REPO,
           *map(str, cfg.get("server_args") or ["--omni", "--no-guardrails", "--host", "0.0.0.0", "--port", "8000", "--init-timeout", "1800"])]
    if deploy_config:
        cmd += ["--deploy-config", deploy_config]
    cmd += list(extra_args)
    if not dry_run:
        (work / "cache" / "triton_idm").mkdir(parents=True, exist_ok=True)
        (work / "cache" / "fa_cute").mkdir(parents=True, exist_ok=True)
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)
        subprocess.run(cmd, check=True)
    return cmd


def stop(name: str = NAME) -> None:
    if not name.startswith("tsf-cosmos-"):
        raise ValueError("refusing to stop a container that is not a tsf-cosmos-* server")
    subprocess.run(["docker", "rm", "-f", name], capture_output=True)


def wait_ready(base_url: str = DEFAULT_URL, timeout: float = 1800, interval: float = 5.0, name: str = NAME) -> float:
    """Poll ``/v1/models`` until the server lists the model; fails fast when the container died."""
    t0 = time.monotonic()
    while True:
        try:
            r = httpx.get(f"{base_url}/v1/models", timeout=10)
            if r.status_code == 200 and REPO in [m["id"] for m in r.json().get("data", [])]:
                return time.monotonic() - t0
        except httpx.HTTPError:
            pass
        alive = subprocess.run(["docker", "ps", "-q", "--filter", f"name=^{name}$"], capture_output=True, text=True).stdout.strip()
        if not alive:
            raise RuntimeError(f"{name} exited while loading; docker logs {name}")
        if time.monotonic() - t0 > timeout:
            raise TimeoutError(f"{REPO} not ready at {base_url} after {timeout:.0f}s")
        time.sleep(interval)


def container_mem_gb(name: str = NAME) -> float | None:
    """cgroup memory (GB) of the container (CPU + pinned/unified allocations charged to it), None if unreadable."""
    cid = subprocess.run(["docker", "inspect", "-f", "{{.Id}}", name], capture_output=True, text=True).stdout.strip()
    for p in (f"/sys/fs/cgroup/system.slice/docker-{cid}.scope/memory.current", f"/sys/fs/cgroup/memory/docker/{cid}/memory.usage_in_bytes"):
        try:
            return int(Path(p).read_text()) / 2**30
        except (OSError, ValueError):
            pass
    return None


# ============================================================================== inputs
def window_video(src, dst, start_frame: int, *, src_fps: float | None = None, frames: int = FRAMES, fps: float = FPS,
                 size: tuple[int, int] = SIZE, ffmpeg: str | None = None) -> Path:
    """61-frame, 10 fps H.264 window of ``src`` starting at ``start_frame`` (every ``round(src_fps/fps)``-th frame).

    Frames are scaled to the target height and centre-cropped (or edge-padded) to the target width: no anisotropic squeeze.
    """
    import cv2
    cap = cv2.VideoCapture(str(src))
    sfps = src_fps or cap.get(cv2.CAP_PROP_FPS)
    step = int(round(sfps / fps))
    w, h = size
    if ffmpeg is None:
        from tsf.cosmos.generate import ffmpeg_exe
        ffmpeg = ffmpeg_exe()
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ff = subprocess.Popen([ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
                           "-r", f"{fps:g}", "-i", "-", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "12",
                           "-pix_fmt", "yuv420p", str(dst)], stdin=subprocess.PIPE)
    n = 0
    for k in range(frames):
        for _ in range(step - 1 if k else 0):
            if not cap.grab():
                break
        ok, bgr = cap.read()
        if not ok:
            break
        sh, sw = bgr.shape[:2]
        if sh != h:
            bgr = cv2.resize(bgr, (round(sw * h / sh), h), interpolation=cv2.INTER_AREA)
        sw = bgr.shape[1]
        if sw >= w:
            x0 = (sw - w) // 2
            bgr = bgr[:, x0:x0 + w]
        else:
            bgr = cv2.copyMakeBorder(bgr, 0, 0, (w - sw) // 2, w - sw - (w - sw) // 2, cv2.BORDER_REPLICATE)
        ff.stdin.write(np.ascontiguousarray(bgr).tobytes())
        n += 1
    cap.release()
    ff.stdin.close()
    ff.wait()
    if n < frames:
        raise ValueError(f"{src}: only {n}/{frames} frames from frame {start_frame}")
    return Path(dst)


# ============================================================================== request
def inverse_dynamics(video, *, base_url: str = DEFAULT_URL, seed: int = 0, steps: int = 30, guidance: float = 1.0,
                     flow_shift: float = 10.0, timeout: float = 1800, poll: float = 0.5, view_point: str = "ego_view",
                     domain: str = "av") -> dict:
    """One inverse-dynamics request on a 61-frame, 10 fps window. Returns ``action`` (60, 9) float32 (raw model output),
    ``latency_s`` (submit -> completed, wall clock) and the server's final job record minus the action payload."""
    extra = {"action_mode": "inverse_dynamics", "domain_name": domain, "action_chunk_size": ACTION_STEPS, "image_size": 480,
             "view_point": view_point, "raw_action_dim": ACTION_DIM, "guardrails": False}
    data = {"prompt": "You are an autonomous vehicle planning system.", "num_frames": str(FRAMES), "fps": str(int(FPS)),
            "num_inference_steps": str(steps), "guidance_scale": str(guidance), "flow_shift": str(flow_shift),
            "seed": str(seed), "extra_params": json.dumps(extra)}
    path = Path(video)
    t0 = time.perf_counter()
    with path.open("rb") as fh:
        r = httpx.post(f"{base_url}/v1/videos", data=data, files={"input_reference": (path.name, fh, "video/mp4")},
                       timeout=httpx.Timeout(300, connect=30))
    if r.is_error:
        raise RuntimeError(f"HTTP {r.status_code} from /v1/videos: {r.text[:1500]}")
    job = r.json()
    while True:
        g = httpx.get(f"{base_url}/v1/videos/{job['id']}", timeout=30)
        g.raise_for_status()
        final = g.json()
        if final.get("status") == "completed":
            break
        if final.get("status") in {"failed", "cancelled"}:
            raise RuntimeError(json.dumps(final)[:2000])
        if time.perf_counter() - t0 > timeout:
            raise TimeoutError(f"job {job['id']} still {final.get('status')} after {timeout:.0f}s")
        time.sleep(poll)
    dt = time.perf_counter() - t0
    act = final.get("action")
    if not act or "data" not in act:
        raise RuntimeError(f"no action in response: {json.dumps(final)[:1500]}")
    a = np.asarray(act["data"], np.float32)
    if a.shape != (ACTION_STEPS, ACTION_DIM):
        raise RuntimeError(f"unexpected action shape {a.shape}")
    return {"action": a, "latency_s": dt, "job": {k: v for k, v in final.items() if k != "action"}}


# ============================================================================== action -> poses -> track
def rot6d(r1: np.ndarray, r2: np.ndarray, layout: str = "cols") -> np.ndarray:
    """(N,3),(N,3) -> (N,3,3) rotations by Gram-Schmidt; ``layout`` 'cols': r1, r2 are the first two columns, 'rows': rows."""
    x = r1 / np.linalg.norm(r1, axis=-1, keepdims=True)
    y = r2 - (x * r2).sum(-1, keepdims=True) * x
    y /= np.linalg.norm(y, axis=-1, keepdims=True)
    z = np.cross(x, y)
    return np.stack([x, y, z], axis=-1 if layout == "cols" else -2)


def action_steps(action: np.ndarray, conv: dict | None = None) -> np.ndarray:
    """(60, 9) raw action -> (60, 4, 4) inter-frame transforms ``T_i`` (camera i+1 in camera i axes, metres)."""
    c = {**CONVENTION, **(conv or {})}
    a = np.asarray(action, np.float64)
    T = np.tile(np.eye(4), (len(a), 1, 1))
    R = rot6d(a[:, 3:6], a[:, 6:9], c["rot6d"])
    if c["rot_scale"] != 1.0:                     # angle scaling about the same axis (unit calibration, see module docstring)
        from scipy.spatial.transform import Rotation
        R = Rotation.from_rotvec(Rotation.from_matrix(R).as_rotvec() * c["rot_scale"]).as_matrix()
    T[:, :3, :3] = R
    T[:, :3, 3] = a[:, :3] * c["scale"]
    if c["lag"]:                                  # action i belongs to step i+lag (edge steps repeat)
        idx = np.clip(np.arange(len(a)) - c["lag"], 0, len(a) - 1)
        T = T[idx]
    return T


def actions_to_poses(action: np.ndarray, conv: dict | None = None) -> np.ndarray:
    """(60, 9) raw action -> (61, 4, 4) camera-to-world poses, frame 0 = identity (OpenCV camera axes)."""
    T = action_steps(action, conv)
    c = {**CONVENTION, **(conv or {})}
    P = np.tile(np.eye(4), (len(T) + 1, 1, 1))
    for i, t in enumerate(T):
        P[i + 1] = P[i] @ t if c["chain"] == "right" else t @ P[i]
    return P


def ego_track_from_actions(action: np.ndarray, conv: dict | None = None, *, despike: bool = False, fps: float = FPS) -> np.ndarray:
    """(60, 9) raw action -> planar ``[x, y, yaw, v]`` (61, 4) via ``tsf.ingest.vipe.ego_track`` (same code path as ViPE)."""
    from tsf.ingest import vipe
    P = actions_to_poses(action, conv)
    t = np.arange(len(P)) / fps
    tr = vipe.ego_track(P, t)
    return vipe.despike(tr, t) if despike else tr


# ============================================================================== ground truth (PhysicalAI-AV windows)
def _rx(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def pai_gt(clip_id: str, start_frame: int, *, pai_root=None, work=None, frames: int = FRAMES, step: int = 3,
           pitch_up_deg: float | None = None) -> dict:
    """Ground truth of the window whose frames are ``start_frame + step*i`` of ``<work>/<clip>/pinhole.mp4``.

    Returns ``poses`` (61,4,4) virtual-camera-to-anchor-world (OpenCV axes, from the logged ego motion at the video frame
    timestamps and the calibrated camera extrinsics; the virtual camera is pitched up ``pitch_up_deg`` from the real axis, as
    in ``deploy/vipe/pai_validate.py``), ``t`` (61,) seconds from the first frame, ``rig_speed`` (61,) m/s logged speed
    (|v| of the logged velocity) and ``frame_times`` (61,) video timestamps in s.
    """
    import pandas as pd
    from scipy.spatial.transform import Rotation, Slerp
    pai_root = Path(pai_root or os.environ.get("TSF_PAI_AV", Path.home() / "data" / "pai_av"))
    work = Path(work or os.environ.get("TSF_IDM_WORK", Path.home() / "tsf" / "dev" / "idm" / "pai"))
    ego_p = pai_root / "x" / "ego" / f"{clip_id}.egomotion.parquet"
    e = pd.read_parquet(ego_p).sort_values("timestamp")
    ts = pd.read_parquet(work / clip_id / "timestamps.parquet")["timestamp"].to_numpy() / 1e6
    idx = start_frame + step * np.arange(frames)
    ft = ts[idx]
    te = e["timestamp"].to_numpy() / 1e6
    keep = (te >= ft[0] - 0.5) & (te <= ft[-1] + 0.5)
    te, e = te[keep], e[keep]
    rot = Slerp(te, Rotation.from_quat(e[["qx", "qy", "qz", "qw"]].to_numpy()))(ft)
    pos = np.stack([np.interp(ft, te, e[c].to_numpy()) for c in "xyz"], 1)
    spd = np.interp(ft, te, np.linalg.norm(e[["vx", "vy", "vz"]].to_numpy(), axis=1))
    mod = _pai_validate()
    cam, R_vc, t_vc = mod.rig_camera(clip_id)
    pitch = np.radians(mod.PITCH_UP if pitch_up_deg is None else pitch_up_deg)
    R_cam = R_vc @ _rx(pitch)
    P = np.tile(np.eye(4), (frames, 1, 1))
    Rw = rot.as_matrix()
    P[:, :3, :3] = Rw @ R_cam
    P[:, :3, 3] = pos + Rw @ t_vc
    return {"poses": P, "t": ft - ft[0], "rig_speed": spd, "frame_times": ft, "rig_yaw": np.unwrap(rot.as_euler("ZYX")[:, 0])}


def _pai_validate():
    """``deploy/vipe/pai_validate.py`` (virtual pinhole camera + rig extrinsics), imported by path; not edited here."""
    import importlib.util
    here = Path(__file__).resolve()
    for root in (here.parents[3], Path.home() / "tsf" / "TanitScenarioFlyWheel", Path.home() / "tsf" / "dev" / "vipe" / "repo"):
        p = root / "deploy" / "vipe" / "pai_validate.py"
        if p.is_file():
            spec = importlib.util.spec_from_file_location("tsf_pai_validate", p)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    raise FileNotFoundError("deploy/vipe/pai_validate.py not found")


def relative_steps(P: np.ndarray) -> np.ndarray:
    """(N,4,4) camera-to-world -> (N-1,4,4) ``P[i]^-1 P[i+1]`` (the action convention)."""
    return np.linalg.inv(P[:-1]) @ P[1:]


# ============================================================================== metrics
def resample_poses(P: np.ndarray, t: np.ndarray, t_new: np.ndarray) -> np.ndarray:
    """Interpolate camera-to-world poses (slerp + linear) onto ``t_new`` (clamped to the span of ``t``)."""
    from scipy.spatial.transform import Rotation, Slerp
    tn = np.clip(t_new, t[0], t[-1])
    R = Slerp(t, Rotation.from_matrix(P[:, :3, :3]))(tn).as_matrix()
    out = np.tile(np.eye(4), (len(tn), 1, 1))
    out[:, :3, :3] = R
    out[:, :3, 3] = np.stack([np.interp(tn, t, P[:, :3, 3][:, k]) for k in range(3)], 1)
    return out


def rates(track: np.ndarray, t: np.ndarray, smooth_s: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """(speed m/s, yaw rate rad/s) from a planar track ``[x, y, yaw, v]`` (centred difference of the yaw over ``smooth_s``)."""
    n = len(t)
    h = max(1, round(smooth_s / np.median(np.diff(t))) // 2)
    i = np.arange(n)
    lo, hi = np.clip(i - h, 0, n - 1), np.clip(i + h, 0, n - 1)
    dt = np.where(hi > lo, t[hi] - t[lo], 1.0)
    return track[:, 3], (track[hi, 2] - track[lo, 2]) / dt


def umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rigid alignment (rotation R, translation t; scale fixed at 1) minimising |R src + t - dst|; works for 2D and 3D."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    U, _, Vt = np.linalg.svd((dst - mu_d).T @ (src - mu_s))
    S = np.eye(len(mu_s))
    S[-1, -1] = np.sign(np.linalg.det(U @ Vt))
    R = U @ S @ Vt
    return R, mu_d - R @ mu_s


def ate(est: np.ndarray, gt: np.ndarray) -> float:
    R, t = umeyama(est, gt)
    return float(np.sqrt(((est @ R.T + t - gt) ** 2).sum(1).mean()))


def rpe(P_est: np.ndarray, P_gt: np.ndarray, delta: int) -> tuple[float, float]:
    """Mean translation (m) and rotation (deg) error of the relative pose over ``delta`` frames (3D camera poses)."""
    err_t, err_r = [], []
    for i in range(len(P_est) - delta):
        Q = np.linalg.inv(relative_steps(P_gt[[i, i + delta]])[0]) @ relative_steps(P_est[[i, i + delta]])[0]
        err_t.append(np.linalg.norm(Q[:3, 3]))
        err_r.append(np.degrees(np.arccos(np.clip((np.trace(Q[:3, :3]) - 1) / 2, -1, 1))))
    return float(np.mean(err_t)), float(np.mean(err_r))


def metrics(P_est: np.ndarray, P_gt: np.ndarray, t: np.ndarray, *, despike: bool = False, rpe_s: float = 1.0) -> dict:
    """Window metrics of estimated vs ground-truth camera-to-world poses on the common grid ``t`` (both (N,4,4)).

    Both tracks are built by ``vipe.ego_track`` (the UC2 code path, optionally ``vipe.despike``-d). Returns speed MAE
    (km/h), yaw-rate MAE (deg/s), ``ate_se2`` (m, planar tracks), ``ate_se3`` (m, 3D camera centres), 1 s RPE translation
    (m) and rotation (deg), mean speeds and the speed bias (est/gt mean)."""
    from tsf.ingest import vipe
    tr_e, tr_g = (vipe.ego_track(P, t) for P in (P_est, P_gt))
    if despike:
        tr_e, tr_g = vipe.despike(tr_e, t), vipe.despike(tr_g, t)
    ve, we = rates(tr_e, t)
    vg, wg = rates(tr_g, t)
    d = max(1, round(rpe_s / np.median(np.diff(t))))
    rt, rr = rpe(P_est, P_gt, d)
    return {"speed_mae_kmh": float(np.mean(np.abs(ve - vg)) * 3.6), "yaw_rate_mae_dps": float(np.degrees(np.mean(np.abs(we - wg)))),
            "ate_se2_m": ate(tr_e[:, :2], tr_g[:, :2]), "ate_se3_m": ate(P_est[:, :3, 3], P_gt[:, :3, 3]),
            "rpe_1s_trans_m": rt, "rpe_1s_rot_deg": rr, "speed_est_kmh": float(ve.mean() * 3.6),
            "speed_gt_kmh": float(vg.mean() * 3.6), "path_est_m": float(np.sum(np.linalg.norm(np.diff(tr_e[:, :2], axis=0), axis=1))),
            "path_gt_m": float(np.sum(np.linalg.norm(np.diff(tr_g[:, :2], axis=0), axis=1)))}
