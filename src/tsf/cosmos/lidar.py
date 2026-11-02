"""LiDAR for the Sim-Auto joint camera+LiDAR mode: HD-map range-map control, generated range maps -> point clouds,
quality checks and videos.

Control (geometry only; framework ``cosmos-auto-lidar-rangemap``)
  ``render_lidar`` ray-casts map layers + object cuboids onto the Pandar128-style 128 x 1800 lattice with sweep poses
  from the exported ``vehicle_pose`` tar and the nominal top-LiDAR mount (``--allow-default-extrinsic``; the Hyperion
  asset has no LiDAR entry) and writes ``lidar_control.safetensors``: one tensor ``frames`` float32 ``[3, T, 128, 1800]``,
  channels (nearest range in m, occupied, occupied), the exact contract of the vLLM-Omni Sim-Auto ``lidar`` request.
  297 camera frames at 30 fps = 99 sweeps at 10 Hz and need 298 pose frames (RDS-HQ export with ``pad_frames=1``);
  for a shorter camera window ``trim_control`` cuts the first ``required_sweeps(frames)`` sweeps.

Generated LiDAR (Sim-Auto response, ``lidar_generated.safetensors``)
  ``frames`` float32 ``[3, T, 128, 1800]``, channels (range m, intensity 0..1, validity 0/1). The lattice is the one
  of the control: ring ``r`` has the elevation of Hesai Pandar128 beam ``r`` (ring 0 = top, +14.4 deg, ring 127 =
  -25.0 deg), column ``k`` looks at azimuth ``pi - (k + 0.5) * 2 pi / 1800`` (column 0 = rear, centre column = forward,
  clockwise spin), x forward, y left, z up. Cell ``(r, k)`` holds the nearest return along that ray, so
  ``xyz = range * ray[r, k]``. A column is fired at relative sweep time ``k / 1799``; the map is "as captured" in the
  moving sensor frame, so points are motion compensated to the sweep-start frame with the sweep's start/end poses.

``process_sweeps`` writes per-sweep point clouds (``.npz``), the BEV video, the control-vs-generated range-image
video and ``lidar_checks.json``. Needs numpy + scipy + cv2 + imageio (the repo venv); safetensors is read/written
with a tiny built-in reader (no ``safetensors`` / torch dependency).
"""
from __future__ import annotations

import json
import math
import os
import struct
import subprocess
import tarfile
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

DEFAULT_PY = "~/tsf/venvs/wsm/bin/python"
ROWS, COLS = 128, 1800
CAMERA_FPS, LIDAR_FPS = 30.0, 10.0
MIN_RANGE_M, MAX_RANGE_M = 5.0, 100.0
VALID_THRESHOLD = 0.5
# nominal top-LiDAR mount of cosmos_auto.lidar.pose_source.DEFAULT_LIDAR_TO_VEHICLE (rear-axle ground projection, FLU)
LIDAR_X_M, LIDAR_Z_M = 1.14, 1.94
PALETTE = {"Car": (126, 206, 255), "Truck": (255, 192, 64), "Pedestrian": (255, 124, 171), "Cyclist": (102, 208, 198),
           "Others": (166, 136, 125)}                                         # RGB, same as the exporter BEV
_ST_DT = {"F32": "<f4", "F16": "<f2", "F64": "<f8", "I64": "<i8", "I32": "<i4", "I16": "<i2", "I8": "i1", "U8": "u1",
          "U16": "<u2", "BOOL": "?"}


# ============================================================================== safetensors (numpy only)
def read_safetensors(path, *, mmap: bool = True) -> tuple[dict[str, np.ndarray], dict]:
    """``({name: array}, metadata)``; arrays are read-only memory maps when ``mmap`` (no copy of a 270 MB control)."""
    path = Path(path)
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    meta = header.pop("__metadata__", None) or {}
    base, out = 8 + n, {}
    for name, h in header.items():
        a, b = h["data_offsets"]
        dt, shape = np.dtype(_ST_DT[h["dtype"]]), tuple(h["shape"])
        if mmap and b > a:
            out[name] = np.memmap(path, dtype=dt, mode="r", offset=base + a, shape=shape)
        else:
            with open(path, "rb") as f:
                f.seek(base + a)
                out[name] = np.frombuffer(f.read(b - a), dtype=dt).reshape(shape)
    return out, meta


def write_safetensors(path, tensors: dict[str, np.ndarray], metadata: dict | None = None) -> Path:
    """Write little-endian safetensors (atomic ``.part`` rename)."""
    inv = {np.dtype(v): k for k, v in _ST_DT.items()}
    header, off, arrs = {}, 0, []
    for name, a in tensors.items():
        a = np.ascontiguousarray(a)
        a = a.astype(a.dtype.newbyteorder("<"), copy=False)
        header[name] = {"dtype": inv[np.dtype(a.dtype)], "shape": list(a.shape), "data_offsets": [off, off + a.nbytes]}
        off += a.nbytes
        arrs.append(a)
    if metadata:
        header["__metadata__"] = {str(k): str(v) for k, v in metadata.items()}
    hb = json.dumps(header, separators=(",", ":")).encode()
    hb += b" " * (-len(hb) % 8)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        for a in arrs:
            f.write(a.tobytes())
    tmp.replace(path)
    return path


# ============================================================================== control
def required_sweeps(frames: int, camera_fps: float = CAMERA_FPS, lidar_fps: float = LIDAR_FPS) -> int:
    """Sweeps a ``frames``-frame camera window needs (the server's rule: ``round(frames * lidar_fps / camera_fps)``)."""
    n = round(frames * lidar_fps / camera_fps)
    if n < 1:
        raise ValueError("the camera window must cover at least one LiDAR sweep")
    return n


def render_lidar(clip_root, clip_id: str, out_dir, *, max_sweeps: int | None = None, prefilter: bool = False,
                 device: str = "cpu", python: str | None = None, camera_name: str = "camera_front_wide_120fov",
                 timeout: float | None = None) -> dict:
    """Run the range-map renderer; returns ``{outputs, sweeps, control, metadata, wall_s, ...}`` and writes
    ``lidar_timings.json``. ``control`` is the validated ``lidar_control.safetensors`` (``[3, T, 128, 1800]`` float32),
    ``metadata`` the per-sweep pose npz (``pose_list`` ``[T, 2, 4, 4]`` = lidar_to_world at sweep start / end)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    py = python or str(Path(os.environ.get("TSF_WSM_PYTHON", DEFAULT_PY)).expanduser())
    cmd = [py, "-m", "cosmos_auto.lidar.cli", "--input-root", str(clip_root), "--output-root", str(out_dir),
           "--clip-id", clip_id, "--camera-name", camera_name, "--allow-default-extrinsic", "--export-control",
           "--no-vis", "--device", device]
    if max_sweeps:
        cmd += ["--max-frames", str(max_sweeps)]
    if prefilter:
        cmd += ["--prefilter"]
    t = time.perf_counter()
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    wall = time.perf_counter() - t
    (out_dir / "lidar.log").write_text(r.stdout + "\n--- stderr ---\n" + r.stderr)
    if r.returncode != 0:
        raise RuntimeError(f"cosmos-auto-lidar-rangemap failed (rc={r.returncode}):\n{r.stderr[-2000:]}")
    res = {}
    for line in r.stdout.splitlines():
        if line.startswith("{") and '"clip_id"' in line:
            res = json.loads(line)
    outs = res.get("outputs", {})
    ctl = Path(outs.get("lidar_control", out_dir / "controls" / clip_id / "lidar_control.safetensors"))
    meta = Path(outs.get("metadata", out_dir / "lidar_rangemap_metadata" / f"{clip_id}.npz"))
    res.update(wall_s=round(wall, 2), device=device, prefilter=prefilter, cmd=" ".join(cmd), control=str(ctl),
               metadata=str(meta), sweeps=validate_control(ctl)[1])
    (out_dir / "lidar_timings.json").write_text(json.dumps(res, indent=1))
    return res


def validate_control(path) -> tuple[tuple[int, ...], int]:
    """Check the Sim-Auto control contract (single tensor ``frames`` float32 ``[3, T, 128, 1800]``, finite, range >= 0,
    channels 1-2 in [0, 1]; the last two are sampled on the first/last sweep). Returns ``(shape, T)``."""
    arrs, _ = read_safetensors(path)
    if list(arrs) != ["frames"]:
        raise ValueError(f"{path}: a LiDAR control holds exactly one tensor 'frames', got {list(arrs)}")
    a = arrs["frames"]
    if a.dtype != np.float32 or a.ndim != 4 or a.shape[0] != 3 or a.shape[2:] != (ROWS, COLS) or a.shape[1] < 1:
        raise ValueError(f"{path}: need float32 [3, T, {ROWS}, {COLS}], got {a.dtype} {a.shape}")
    for t in {0, a.shape[1] - 1}:
        x = np.asarray(a[:, t])
        if not np.isfinite(x).all() or (x[0] < 0).any() or (x[1:] < 0).any() or (x[1:] > 1).any():
            raise ValueError(f"{path}: sweep {t} has non-finite values, negative range or channels outside [0, 1]")
    return tuple(a.shape), int(a.shape[1])


def trim_control(src, dst, sweeps: int, *, start: int = 0) -> Path:
    """Copy sweeps ``[start, start + sweeps)`` of a control to ``dst`` (what a camera window of
    ``required_sweeps(frames)`` sweeps needs; the server rejects a control shorter than that)."""
    arrs, meta = read_safetensors(src)
    a = arrs["frames"]
    if start < 0 or start + sweeps > a.shape[1]:
        raise ValueError(f"control has {a.shape[1]} sweeps, asked for [{start}, {start + sweeps})")
    out = write_safetensors(dst, {"frames": np.ascontiguousarray(a[:, start:start + sweeps], dtype=np.float32)},
                            {**meta, "tsf_trim": json.dumps({"source": str(src), "start": start, "sweeps": sweeps})})
    validate_control(out)
    return out


# ============================================================================== lattice and point clouds
@lru_cache(maxsize=1)
def beam_elevations() -> np.ndarray:
    """``[128]`` ring elevations in rad, ring 0 = top (Hesai Pandar128 default calibration of the control renderer)."""
    d = json.loads((Path(__file__).parent / "data" / "pandar128_rows.json").read_text())
    return np.asarray(d["row_elevations_rad"], np.float64)


@lru_cache(maxsize=2)
def ray_directions(n_cols: int = COLS) -> np.ndarray:
    """``[128, n_cols, 3]`` unit rays through the cell centres, identical to
    ``cosmos_auto.lidar.sensor_model.SpinningLidarModel(n_columns=n_cols).ray_directions()`` (x fwd, y left, z up)."""
    az = np.pi - (np.arange(n_cols) + 0.5) * (2 * np.pi / n_cols)
    el = beam_elevations()[:, None]
    return np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el) + 0 * az], axis=-1).astype(np.float32)


def valid_mask(frames, i: int, thr: float = VALID_THRESHOLD) -> np.ndarray:
    """``[128, 1800]`` bool: sweep ``i`` cells with a return (validity >= ``thr`` and range > 0)."""
    return (np.asarray(frames[2, i]) >= thr) & (np.asarray(frames[0, i]) > 0)


def _pose_interp(T0, T1, t):
    """Rotation matrices ``[N, 3, 3]`` and translations ``[N, 3]`` of the sensor at relative sweep times ``t`` (slerp / lerp
    between ``T0`` and ``T1``), evaluated once per distinct time (a sweep has only 1800 firing times)."""
    from scipy.spatial.transform import Rotation, Slerp
    t = np.clip(np.asarray(t, np.float64), 0.0, 1.0)
    u, inv = np.unique(t, return_inverse=True)
    Ru = Slerp([0.0, 1.0], Rotation.from_matrix(np.stack([T0[:3, :3], T1[:3, :3]])))(u).as_matrix()
    tu = T0[:3, 3][None] + (T1[:3, 3] - T0[:3, 3])[None] * u[:, None]
    return Ru[inv], tu[inv]


def sweep_to_world(xyz, t_rel, T0, T1) -> np.ndarray:
    """Points of the moving sensor frame (fired at relative sweep time ``t_rel``) -> world, with the sweep's
    start/end ``lidar_to_world`` poses (slerp / lerp, the renderer's rolling-shutter model)."""
    R, tr = _pose_interp(T0, T1, t_rel)
    return np.einsum("nij,nj->ni", R, np.asarray(xyz, np.float64)) + tr


def start_to_world(xyz, T0) -> np.ndarray:
    """Points of the sweep-start sensor frame -> world."""
    return np.asarray(xyz, np.float64) @ T0[:3, :3].T + T0[:3, 3]


def world_to_sensor(pw, T) -> np.ndarray:
    """World points -> the sensor frame of pose ``T`` (``lidar_to_world``)."""
    return (np.asarray(pw, np.float64) - T[:3, 3]) @ T[:3, :3]


def sweep_points(frames, i: int, *, poses=None, thr: float = VALID_THRESHOLD) -> dict:
    """Point cloud of sweep ``i``. ``poses`` = ``(T_start, T_end)`` ``lidar_to_world`` makes ``xyz`` motion compensated
    to the sweep-start sensor frame; without it ``xyz`` is the as-captured cloud. Keys: ``xyz`` ``[N, 3]`` m,
    ``intensity``, ``range``, ``row``, ``col``, ``t_rel`` (firing time 0..1), ``compensated`` (bool)."""
    m = valid_mask(frames, i, thr)
    row, col = np.nonzero(m)
    rng = np.asarray(frames[0, i])[row, col].astype(np.float32)
    xyz = rng[:, None] * ray_directions()[row, col]
    t_rel = (col / (COLS - 1)).astype(np.float32)
    comp = poses is not None
    if comp:
        xyz = world_to_sensor(sweep_to_world(xyz, t_rel, *poses), poses[0]).astype(np.float32)
    return {"xyz": xyz.astype(np.float32), "intensity": np.asarray(frames[1, i])[row, col].astype(np.float32),
            "range": rng, "row": row.astype(np.uint8), "col": col.astype(np.uint16), "t_rel": t_rel,
            "compensated": np.bool_(comp)}


def save_pointcloud(path, cloud: dict, poses=None, **extra) -> Path:
    """``.npz`` of one sweep: ``xyz`` (sweep-start LiDAR frame, x fwd / y left / z up), ``intensity`` plus the
    lattice indices, and ``lidar_to_world_start/end`` when the poses are known."""
    d = dict(cloud, **extra)
    if poses is not None:
        d["lidar_to_world_start"], d["lidar_to_world_end"] = (np.asarray(p, np.float64) for p in poses)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **d)
    return path


# ============================================================================== poses and objects
def load_sweep_poses(meta_npz) -> dict:
    """Sweep poses written by the control renderer (``lidar_rangemap_metadata/<clip>.npz``): ``frame_ids`` (camera frame
    at which each sweep starts), ``poses`` ``[T, 2, 4, 4]`` lidar_to_world (start, end), ``pose_source``."""
    z = np.load(meta_npz)
    return {"frame_ids": [int(s) for s in z["frame_indices"]], "poses": np.asarray(z["pose_list"], np.float64),
            "timestamps_us": np.asarray(z["timestamps_list"]), "pose_source": str(z["pose_source"])}


def read_objects(rds_root, clip_id: str, frames) -> dict[int, dict]:
    """``{frame: {track_id: {type, lwh, T}}}`` from the RDS-HQ ``all_object_info`` tar (``T`` = object_to_world, the
    geometric box centre, local x along the length)."""
    want, out, suf = set(int(f) for f in frames), {}, ".all_object_info.json"
    with tarfile.open(Path(rds_root) / "all_object_info" / f"{clip_id}.tar") as tf:
        for m in tf:
            if not m.name.endswith(suf):
                continue
            f = int(m.name[: -len(suf)].rsplit(".", 1)[-1])
            if f in want:
                out[f] = {tid: {"type": o["object_type"], "lwh": np.asarray(o["object_lwh"], float),
                                "T": np.asarray(o["object_to_world"], float)}
                          for tid, o in json.loads(tf.extractfile(m).read()).items()}
    return out


def _obj_pose(o0: dict, o1: dict | None, t):
    """Centre ``[N, 3]`` and yaw ``[N]`` of an object at fractions ``t`` between its pose in two frames (lerp)."""
    t = np.asarray(t, np.float64)
    c0, y0 = o0["T"][:3, 3], math.atan2(o0["T"][1, 0], o0["T"][0, 0])
    if o1 is None:
        return np.tile(c0, (len(t), 1)), np.full(len(t), y0)
    c1, y1 = o1["T"][:3, 3], math.atan2(o1["T"][1, 0], o1["T"][0, 0])
    dy = (y1 - y0 + math.pi) % (2 * math.pi) - math.pi
    return c0[None] + (c1 - c0)[None] * t[:, None], y0 + dy * t


def points_in_box(pw, t_rel, o0: dict, o1: dict | None, margin: float = 0.3) -> np.ndarray:
    """Bool mask of world points inside the (moving) cuboid of an object, evaluated at each point's firing time."""
    c, yaw = _obj_pose(o0, o1, t_rel)
    d = np.asarray(pw, np.float64) - c
    cs, sn = np.cos(yaw), np.sin(yaw)
    x, y = cs * d[:, 0] + sn * d[:, 1], -sn * d[:, 0] + cs * d[:, 1]
    l, w, h = o0["lwh"]
    return (np.abs(x) <= l / 2 + margin) & (np.abs(y) <= w / 2 + margin) & (np.abs(d[:, 2]) <= h / 2 + margin)


# ============================================================================== checks
def fit_ground(xyz, *, seed: int = 0, iters: int = 200, max_r: float = 35.0, tol: float = 0.12) -> dict:
    """RANSAC plane through near-ground points (z within 0.8..3.0 m below the sensor, r < ``max_r``) of one sweep in the
    sensor frame. ``height_m`` = distance sensor -> plane (nominal 1.94), ``tilt_deg`` = plane normal vs. z."""
    xyz = np.asarray(xyz, np.float64)
    r = np.hypot(xyz[:, 0], xyz[:, 1])
    cand = xyz[(r < max_r) & (xyz[:, 2] < -LIDAR_Z_M + 1.1) & (xyz[:, 2] > -LIDAR_Z_M - 1.1)]
    out = {"candidates": int(len(cand)), "inliers": 0, "height_m": None, "tilt_deg": None, "sectors_covered": 0}
    if len(cand) < 50:
        return out
    rng = np.random.default_rng(seed)
    if len(cand) > 20000:                                   # RANSAC on a subsample, inliers counted on all
        cand_all, cand = cand, cand[rng.choice(len(cand), 20000, replace=False)]
    else:
        cand_all = cand
    best, best_n = None, 0
    for _ in range(iters):
        p = cand[rng.choice(len(cand), 3, replace=False)]
        n = np.cross(p[1] - p[0], p[2] - p[0])
        nn = np.linalg.norm(n)
        if nn < 1e-6:
            continue
        n = n / nn
        if abs(n[2]) < 0.9:
            continue
        cnt = int((np.abs((cand - p[0]) @ n) < tol).sum())
        if cnt > best_n:
            best, best_n = (n, p[0]), cnt
    if best is None:
        return out
    n, p0 = best
    cand = cand_all
    inl = np.abs((cand - p0) @ n) < tol
    A = np.c_[cand[inl][:, :2], np.ones(int(inl.sum()))]            # refine: z = a x + b y + c
    (a, b, c), *_ = np.linalg.lstsq(A, cand[inl][:, 2], rcond=None)
    ang = np.degrees(np.arctan2(cand[inl][:, 1], cand[inl][:, 0]))
    sec = np.histogram(ang, bins=np.arange(-180, 181, 10))[0]
    out.update(inliers=int(inl.sum()), height_m=round(float(-c / np.sqrt(1 + a * a + b * b)), 3),
               tilt_deg=round(float(np.degrees(np.arctan(np.hypot(a, b)))), 2), sectors_covered=int((sec >= 10).sum()))
    return out


def ground_present(g: dict, min_inliers: int = 300) -> bool:
    return (g["inliers"] >= min_inliers and g["height_m"] is not None and abs(g["height_m"] - LIDAR_Z_M) < 0.35
            and g["tilt_deg"] < 6.0)


def _dilate_wrap(mask: np.ndarray, rows: int = 2, cols: int = 4) -> np.ndarray:
    from scipy.ndimage import maximum_filter
    return maximum_filter(mask.astype(np.uint8), size=(2 * rows + 1, 2 * cols + 1), mode=("nearest", "wrap")).astype(bool)


def cloud_metrics(frames, control=None, *, thr: float = VALID_THRESHOLD) -> dict:
    """Per-sweep statistics from range maps alone (no poses): ``n_points``, ``valid_fraction``, ``max_range_m``,
    ``p50/p99_range_m``, ``mean_intensity``, ``frac_beyond_50m`` and ``near_clutter`` = returns closer than 15 m,
    0.4..4 m above the ground and with no control return within +-2 rows / +-4 columns (airborne returns the HD-map
    geometry does not explain: spray, droplets, backscatter). Used by the weather comparison."""
    rays_z = ray_directions()[..., 2]
    out = []
    for i in range(frames.shape[1]):
        m = valid_mask(frames, i, thr)
        r = np.asarray(frames[0, i])[m]
        row = {"n_points": int(m.sum()), "valid_fraction": round(float(m.mean()), 5)}
        if r.size:
            row.update(max_range_m=float(r.max()), p50_range_m=float(np.percentile(r, 50)),
                       p99_range_m=float(np.percentile(r, 99)), mean_intensity=float(np.asarray(frames[1, i])[m].mean()),
                       frac_beyond_50m=float((r > 50).mean()))
            height = np.asarray(frames[0, i]) * rays_z + LIDAR_Z_M
            air = m & (np.asarray(frames[0, i]) < 15) & (height > 0.4) & (height < 4.0)
            if control is not None:
                air &= ~_dilate_wrap(valid_mask(control, min(i, control.shape[1] - 1), thr))
            row["near_clutter"] = int(air.sum())
        else:
            row.update(max_range_m=0.0, p50_range_m=0.0, p99_range_m=0.0, mean_intensity=0.0, frac_beyond_50m=0.0,
                       near_clutter=0)
        out.append(row)
    return {"per_sweep": out, "mean": {k: round(float(np.mean([s[k] for s in out])), 4) for k in out[0]}}


def lidar_checks(frames, control, sweeps: dict, objects: dict[int, dict] | None, *, thr: float = VALID_THRESHOLD,
                 obj_range: float = 40.0, min_obj_points: int = 10, clouds: list[dict] | None = None) -> dict:
    """Quality checks of generated ``frames`` against the HD-map ``control`` (same sweeps, same lattice).

    Per sweep: points, valid fraction, median / p90 |generated - control| range on pixels valid in both, share of
    control returns that the generation also has within 1 m ("agree"), ground plane fit. Objects: for every exported
    cuboid whose centre is within ``obj_range`` m of the sensor at sweep start, the points inside it (cuboid grown by
    0.3 m, evaluated at each point's firing time) for generated and control; ``recall_ge10`` = share of those
    (object, sweep) pairs with >= ``min_obj_points`` generated points, ``recall_ge10_given_control`` the same among
    pairs where the control itself has >= ``min_obj_points`` (visible, unoccluded objects).
    """
    n = frames.shape[1]
    poses, fids = sweeps["poses"], sweeps["frame_ids"]
    per, diffs_all, pairs = [], [], []
    for i in range(n):
        T0, T1 = poses[i]
        g_m = valid_mask(frames, i, thr)
        c_m = valid_mask(control, i, thr)
        both = g_m & c_m
        gr, cr = np.asarray(frames[0, i]), np.asarray(control[0, i])
        d = np.abs(gr[both] - cr[both])
        diffs_all.append(d)
        cl = clouds[i] if clouds is not None else sweep_points(frames, i, poses=(T0, T1), thr=thr)
        cc = sweep_points(control, i, poses=(T0, T1), thr=thr)
        gnd = fit_ground(cl["xyz"], seed=i)
        gnd_c = fit_ground(cc["xyz"], seed=i)
        row = {"sweep": i, "frame": fids[i], "points": int(g_m.sum()), "valid_fraction": round(float(g_m.mean()), 5),
               "control_points": int(c_m.sum()), "both_valid": int(both.sum()),
               "median_abs_range_err_m": round(float(np.median(d)), 3) if d.size else None,
               "p90_abs_range_err_m": round(float(np.percentile(d, 90)), 3) if d.size else None,
               "control_agree_within_1m": round(float((d < 1.0).sum() / max(1, c_m.sum())), 4),
               "ground": {**gnd, "present": ground_present(gnd)}, "ground_control": {**gnd_c, "present": ground_present(gnd_c)}}
        if objects is not None and fids[i] in objects:
            o0s, o1s = objects[fids[i]], objects.get(fids[i] + 3, {})
            pg = start_to_world(cl["xyz"], T0)                      # compensated cloud (start frame) -> world
            tg = cl["t_rel"]
            pc = start_to_world(cc["xyz"], T0)
            tc = cc["t_rel"]
            for tid, o0 in o0s.items():
                if np.hypot(*(o0["T"][:2, 3] - T0[:2, 3])) > obj_range:
                    continue
                o1 = o1s.get(tid)
                ng = int(points_in_box(pg, tg, o0, o1).sum())
                nc = int(points_in_box(pc, tc, o0, o1).sum())
                pairs.append({"sweep": i, "track": tid, "type": o0["type"], "n_generated": ng, "n_control": nc,
                              "dist_m": round(float(np.hypot(*(o0["T"][:2, 3] - T0[:2, 3]))), 1)})
            row["objects_within_range"] = sum(1 for p in pairs if p["sweep"] == i)
        per.append(row)
    d_all = np.concatenate(diffs_all) if diffs_all else np.zeros(0)
    summ = {"sweeps": n, "mean_points_per_sweep": round(float(np.mean([r["points"] for r in per])), 1),
            "min_points_per_sweep": int(min(r["points"] for r in per)),
            "mean_valid_fraction": round(float(np.mean([r["valid_fraction"] for r in per])), 5),
            "mean_control_points_per_sweep": round(float(np.mean([r["control_points"] for r in per])), 1),
            "median_abs_range_err_m": round(float(np.median(d_all)), 3) if d_all.size else None,
            "p90_abs_range_err_m": round(float(np.percentile(d_all, 90)), 3) if d_all.size else None,
            "control_agree_within_1m": round(float(np.mean([r["control_agree_within_1m"] for r in per])), 4),
            "ground_present_sweeps": int(sum(r["ground"]["present"] for r in per)),
            "ground_control_present_sweeps": int(sum(r["ground_control"]["present"] for r in per)),
            "ground_height_m_median": _med([r["ground"]["height_m"] for r in per]),
            "ground_tilt_deg_median": _med([r["ground"]["tilt_deg"] for r in per])}
    obj = {"range_m": obj_range, "min_points": min_obj_points, "pairs": len(pairs)}
    if pairs:
        ge = np.array([p["n_generated"] >= min_obj_points for p in pairs])
        ce = np.array([p["n_control"] >= min_obj_points for p in pairs])
        obj.update(recall_ge10=round(float(ge.mean()), 4), control_ge10=round(float(ce.mean()), 4),
                   recall_ge10_given_control=round(float(ge[ce].mean()), 4) if ce.any() else None,
                   median_points_generated=float(np.median([p["n_generated"] for p in pairs])),
                   median_points_control=float(np.median([p["n_control"] for p in pairs])))
        per_obj = {}
        for p in pairs:
            per_obj.setdefault(p["track"], []).append(p)
        obj["per_object"] = {t: {"type": ps[0]["type"], "sweeps_in_range": len(ps),
                                 "frac_ge10_generated": round(float(np.mean([q["n_generated"] >= min_obj_points for q in ps])), 3),
                                 "frac_ge10_control": round(float(np.mean([q["n_control"] >= min_obj_points for q in ps])), 3),
                                 "median_points_generated": float(np.median([q["n_generated"] for q in ps])),
                                 "median_points_control": float(np.median([q["n_control"] for q in ps])),
                                 "median_dist_m": float(np.median([q["dist_m"] for q in ps]))} for t, ps in per_obj.items()}
    warn = []
    if summ["ground_present_sweeps"] < n:
        warn.append(f"no ground plane in {n - summ['ground_present_sweeps']}/{n} sweeps")
    if summ["median_abs_range_err_m"] is not None and summ["median_abs_range_err_m"] > 2.0:
        warn.append(f"median |generated - control| range {summ['median_abs_range_err_m']} m > 2 m")
    if obj.get("recall_ge10_given_control") is not None and obj["recall_ge10_given_control"] < 0.8:
        warn.append(f"object recall {obj['recall_ge10_given_control']} < 0.8 for objects the control sees")
    return {"summary": summ, "objects": obj, "per_sweep": per, "object_pairs": pairs, "warnings": warn,
            "definitions": {"valid": f"validity >= {thr} and range > 0", "range_err": "pixels valid in both generated and control",
                            "object": f"cuboid +0.3 m margin, centre within {obj_range} m of the sensor at sweep start, "
                                      "point positions in the world frame at their firing time",
                            "ground": "RANSAC plane over points 0.8..3.0 m below the sensor within 35 m; present = >= 300 inliers, "
                                      "height 1.94 +- 0.35 m, tilt < 6 deg"}}


def _med(xs):
    xs = [x for x in xs if x is not None]
    return round(float(np.median(xs)), 3) if xs else None


# ============================================================================== videos
def _writer(path, fps: float):
    import imageio.v2 as imageio
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return imageio.get_writer(str(path), fps=fps, codec="libx264", macro_block_size=1,
                              output_params=["-crf", "18", "-preset", "fast"])


def _turbo(x: np.ndarray) -> np.ndarray:
    """``x`` in [0, 1] -> RGB uint8 (cv2 TURBO)."""
    import cv2
    v = (np.clip(x, 0, 1) * 255).astype(np.uint8).reshape(-1, 1)
    return cv2.applyColorMap(v, cv2.COLORMAP_TURBO).reshape(*x.shape, 3)[..., ::-1]


def render_bev(xyz, intensity, boxes: list | None = None, *, half: float = 50.0, size: int = 800, color: str = "height",
               title: str = "", ego_length: float = 4.8, ego_width: float = 1.9) -> np.ndarray:
    """RGB top-down view in the sensor frame: ego at the centre, +x up, +y left; points coloured by height above the
    ground (``color='height'``, turbo) or intensity; ``boxes`` = ``[(corners [4, 2] sensor frame, type)]``."""
    import cv2
    ppm = size / (2 * half)
    img = np.full((size, size, 3), 14, np.uint8)
    c = size // 2
    for r in range(10, int(half) + 1, 10):
        cv2.circle(img, (c, c), int(r * ppm), (46, 46, 52), 1, cv2.LINE_AA)
        cv2.putText(img, f"{r} m", (c + 4, c - int(r * ppm) + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (90, 90, 100), 1, cv2.LINE_AA)
    cv2.line(img, (c, 0), (c, size), (40, 40, 46), 1)
    cv2.line(img, (0, c), (size, c), (40, 40, 46), 1)
    xyz = np.asarray(xyz)
    if len(xyz):
        u = (c - xyz[:, 1] * ppm).astype(int)
        v = (c - xyz[:, 0] * ppm).astype(int)
        ok = (u >= 1) & (u < size - 1) & (v >= 1) & (v < size - 1)
        val = (xyz[:, 2] + LIDAR_Z_M + 0.5) / 4.5 if color == "height" else np.clip(np.asarray(intensity) * 2.0, 0, 1)
        col = _turbo(val[ok])
        order = np.argsort(xyz[ok][:, 2])                                  # higher points on top
        u, v, col = u[ok][order], v[ok][order], col[order]
        for du in (0, 1):
            for dv in (0, 1):
                img[v + dv, u + du] = col
    for corners, typ in boxes or []:
        pts = np.stack([c - corners[:, 1] * ppm, c - corners[:, 0] * ppm], 1)
        cv2.polylines(img, [np.round(pts * 16).astype(np.int32)], True, PALETTE.get(typ, (200, 200, 200)), 1, cv2.LINE_AA, 4)
        cv2.line(img, tuple(np.round(pts[0]).astype(int).tolist()), tuple(np.round(pts[1]).astype(int).tolist()),
                 (255, 255, 255), 2, cv2.LINE_AA)
    x0, x1 = -LIDAR_X_M - 0.3 * ego_length, -LIDAR_X_M + 0.7 * ego_length
    e = np.array([[x1, ego_width / 2], [x1, -ego_width / 2], [x0, -ego_width / 2], [x0, ego_width / 2]])
    cv2.polylines(img, [np.round(np.stack([c - e[:, 1] * ppm, c - e[:, 0] * ppm], 1) * 16).astype(np.int32)], True,
                  (46, 230, 214), 2, cv2.LINE_AA, 4)
    cv2.putText(img, title, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.putText(img, "height above ground -0.5 .. 4 m" if color == "height" else "intensity", (8, size - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 160), 1, cv2.LINE_AA)
    return img


def box_corners(o0: dict, o1: dict | None, T0, t: float = 0.5) -> np.ndarray:
    """Footprint of an object at sweep fraction ``t`` in the sweep-start sensor frame, ``[4, 2]`` (front-left first)."""
    c, yaw = _obj_pose(o0, o1, [t])
    l, w, _ = o0["lwh"]
    loc = np.array([[l / 2, w / 2], [l / 2, -w / 2], [-l / 2, -w / 2], [-l / 2, w / 2]])
    R = np.array([[np.cos(yaw[0]), -np.sin(yaw[0])], [np.sin(yaw[0]), np.cos(yaw[0])]])
    pw = np.c_[loc @ R.T + c[0, :2], np.full(4, c[0, 2])]
    return world_to_sensor(pw, T0)[:, :2]


def range_compare_frame(frames, control, i: int, title: str = "", vscale: int = 2) -> np.ndarray:
    """RGB image stacking control range, generated range and generated intensity of sweep ``i`` (1800 px wide,
    each strip ``128 * vscale`` px high, invalid cells black)."""
    import cv2
    strips = []
    for lab, src, chan, kind in (("HD-map control range, 0..100 m", control, 0, "range"),
                                 ("generated range, 0..100 m", frames, 0, "range"),
                                 ("generated intensity", frames, 1, "gray")):
        x = np.asarray(src[chan, i])
        a = _turbo(x / MAX_RANGE_M) if kind == "range" else np.repeat((np.clip(x, 0, 1) * 255).astype(np.uint8)[..., None], 3, -1)
        a[~valid_mask(src, i)] = 0
        a = np.repeat(a, vscale, axis=0)
        cv2.putText(a, lab, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        strips.append(a)
    top = np.zeros((26, COLS, 3), np.uint8)
    cv2.putText(top, title + "   columns: rear (left edge) > left > forward (centre) > right > rear", (6, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
    return np.concatenate([top, *strips], 0)


# ============================================================================== orchestration
def process_sweeps(frames, control, out_dir, *, sweeps: dict, objects: dict[int, dict] | None = None, name: str = "lidar",
                   ego_length: float = 4.8, thr: float = VALID_THRESHOLD, videos: bool = True, save_points: bool = True,
                   points_dir: str = "pointclouds", video_fps: float = LIDAR_FPS, title: str = "generated") -> dict:
    """Point clouds, BEV video, range-image comparison video and checks for generated ``frames`` ``[3, T, 128, 1800]``.

    Writes under ``out_dir``: ``<points_dir>/sweep_<i>.npz``, ``<name>_bev.mp4``, ``<name>_range_compare.mp4`` and
    ``<name>_checks.json``. ``sweeps`` = :func:`load_sweep_poses`, ``objects`` = :func:`read_objects` (frames at the sweep
    starts and +3). Returns the checks dict plus ``paths``.
    """
    out_dir = Path(out_dir)
    n = min(frames.shape[1], control.shape[1], len(sweeps["frame_ids"]))
    frames_, control_ = frames[:, :n], control[:, :n]
    poses, fids = sweeps["poses"], sweeps["frame_ids"]
    clouds = [sweep_points(frames_, i, poses=tuple(poses[i]), thr=thr) for i in range(n)]
    paths: dict = {"pointclouds": []}
    if save_points:
        for i, cl in enumerate(clouds):
            paths["pointclouds"].append(str(save_pointcloud(out_dir / points_dir / f"sweep_{i:03d}.npz", cl, tuple(poses[i]),
                                                            frame=fids[i], sweep=i)))
    checks = lidar_checks(frames_, control_, {"poses": poses[:n], "frame_ids": fids[:n]}, objects, thr=thr, clouds=clouds)
    checks["weather_stats"] = cloud_metrics(frames_, control_, thr=thr)["mean"]
    if videos:
        bev = _writer(out_dir / f"{name}_bev.mp4", video_fps)
        cmp_ = _writer(out_dir / f"{name}_range_compare.mp4", video_fps)
        try:
            for i in range(n):
                boxes = []
                if objects is not None and fids[i] in objects:
                    o1s = objects.get(fids[i] + 3, {})
                    boxes = [(box_corners(o, o1s.get(tid), poses[i][0]), o["type"]) for tid, o in objects[fids[i]].items()
                             if np.hypot(*(o["T"][:2, 3] - poses[i][0][:2, 3])) < 75]
                img = render_bev(clouds[i]["xyz"], clouds[i]["intensity"], boxes, ego_length=ego_length,
                                 title=f"LiDAR BEV ({title})  sweep {i + 1}/{n}  t={fids[i] / CAMERA_FPS:5.2f}s  "
                                       f"{len(clouds[i]['xyz'])} pts")
                bev.append_data(img)
                cmp_.append_data(range_compare_frame(frames_, control_, i, f"{title}  sweep {i + 1}/{n}"))
        finally:
            bev.close()
            cmp_.close()
        paths.update(bev=str(out_dir / f"{name}_bev.mp4"), range_compare=str(out_dir / f"{name}_range_compare.mp4"))
    checks["paths"] = paths
    (out_dir / f"{name}_checks.json").write_text(json.dumps(checks, indent=1, default=float))
    return checks
