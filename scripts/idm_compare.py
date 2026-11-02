"""Cosmos3-Edge inverse dynamics vs ViPE vs logged ego motion on PhysicalAI-AV + the three YouTube clips (thor3 only).

PhysicalAI-AV (internal-validation licence) and the YouTube clips never leave thor3, nor do frames, videos or tracks
derived from them; only metrics (numbers) are reported. Work dir ``~/tsf/dev/idm`` (override ``TSF_IDM_DIR``).

    plan         pick 5 calibration + 30 evaluation PAI windows (6 s, front camera) -> plan.json
    prep         render each window as a virtual pinhole clip (30 fps, 854x480; same optics as deploy/vipe/pai_validate)
    cosmos       serve Cosmos3-Edge when thor3 is free, run all windows (calib, eval, youtube), stop the server
    calibrate    fit / verify the action convention against the logged ego motion (calibration windows only)
    vipe         run ViPE on the same evaluation windows when thor3 is free (wall-clock budget ``--budget-min``)
    report       metrics Cosmos vs ViPE vs logged ego (eval) + YouTube agreement -> results.json

Run with the repo venv on thor:  python scripts/idm_compare.py <command>
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tsf.cosmos import idm                                          # noqa: E402
from tsf.cosmos.generate import MemSampler                           # noqa: E402

DIR = Path(os.environ.get("TSF_IDM_DIR", "~/tsf/dev/idm")).expanduser()
PAI = Path(os.environ.get("TSF_PAI_AV", Path.home() / "data" / "pai_av"))
os.environ.setdefault("TSF_IDM_WORK", str(DIR / "pai"))
WORK = Path(os.environ["TSF_IDM_WORK"])
YT = Path("~/data/youtube").expanduser()
YT_IDS = ["dToIIqqvSt0", "qdAI_quNcqw", "amuldGGgNc4"]
CAM, CHUNK = "camera_front_wide_120fov", "chunk_2800"
WIN_FRAMES, PINHOLE_FRAMES = 6 * 30, 6 * 30 + 1                       # 30 fps source window (6 s) and frames rendered
GRID = np.arange(idm.FRAMES) / idm.FPS                                # 0..6 s, 0.1 s


def jload(p):
    return json.loads(Path(p).read_text())


def jsave(p, obj):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(obj, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o)))


def name_of(w: dict) -> str:
    return f"{w['clip'][:8]}_{w['f0']:03d}"


# ============================================================================== plan / prep
def _ego(clip):
    import pandas as pd
    return pd.read_parquet(PAI / "x" / "ego" / f"{clip}.egomotion.parquet").sort_values("timestamp")


def cmd_plan(a):
    import pandas as pd
    from scipy.spatial.transform import Rotation
    with zipfile.ZipFile(PAI / "camera" / CAM / f"{CAM}.{CHUNK}.zip") as z:
        clips = sorted({n.split(".")[0] for n in z.namelist() if n.endswith(".mp4")})
    clips = [c for c in clips if (PAI / "x" / "ego" / f"{c}.egomotion.parquet").exists()]
    rng = np.random.default_rng(a.seed)
    cand = []
    for c in clips:
        e = _ego(c)
        te = e.timestamp.to_numpy() / 1e6
        yaw = np.unwrap(Rotation.from_quat(e[["qx", "qy", "qz", "qw"]].to_numpy()).as_euler("ZYX")[:, 0])
        spd = np.linalg.norm(e[["vx", "vy", "vz"]].to_numpy(), axis=1)
        for f0 in range(0, 425, 30):
            t0 = f0 / 30 - 0.078
            if te.min() > t0 - 1 or te.max() < t0 + 7:
                continue
            m = (te >= t0) & (te <= t0 + 6)
            if m.sum() < 400:
                continue
            cand.append({"clip": c, "f0": f0, "speed_kmh": float(spd[m].mean() * 3.6), "speed_start_kmh": float(spd[m][:20].mean() * 3.6),
                         "speed_end_kmh": float(spd[m][-20:].mean() * 3.6), "yaw_deg": float(np.degrees(yaw[m][-1] - yaw[m][0]))})
    df = pd.DataFrame(cand)
    mov = df[df.speed_kmh > 10]
    calib = []

    def take(row):
        calib.append(row.to_dict())
        return df[df["clip"] != row["clip"]]
    pool = mov
    for key in ("turn", "fast", "accel", "median_curve", "random"):
        if key == "turn":
            r = pool.loc[pool.yaw_deg.abs().idxmax()]
        elif key == "fast":
            r = pool.loc[pool.speed_kmh.idxmax()]
        elif key == "accel":
            r = pool.loc[(pool.speed_end_kmh - pool.speed_start_kmh).abs().idxmax()]
        elif key == "median_curve":
            r = pool.loc[(pool.yaw_deg.abs() - pool.yaw_deg.abs().median()).abs().idxmin()]
        else:
            r = pool.iloc[int(rng.integers(len(pool)))]
        df, pool = take(r), pool[pool["clip"] != r["clip"]]
    rest = sorted(set(df["clip"]))
    pick = rng.choice(rest, size=min(a.n_eval, len(rest)), replace=False)
    evals = [df[df["clip"] == c].iloc[int(rng.integers((df["clip"] == c).sum()))].to_dict() for c in pick]
    for w in calib + evals:
        w["name"] = name_of(w)
    plan = {"seed": a.seed, "calib": calib, "eval": evals, "n_candidates": len(cand), "n_clips": len(clips)}
    jsave(DIR / "plan.json", plan)
    for k in ("calib", "eval"):
        d = pd.DataFrame(plan[k])
        print(k, len(d), "mean speed km/h", d.speed_kmh.describe()[["min", "50%", "max"]].round(1).tolist(),
              "|yaw| deg median/max", d.yaw_deg.abs().median().round(1), d.yaw_deg.abs().max().round(1),
              "standstill(<5km/h)", int((d.speed_kmh < 5).sum()))


def parse_variant(v: str | None) -> tuple[float | None, float | None, str]:
    """'h110p12' -> (hfov 110 deg, pitch-up 12 deg, '@h110p12'); None -> the deploy/vipe default camera (90 deg, 12 deg)."""
    if not v:
        return None, None, ""
    import re
    m = re.fullmatch(r"h(\d+)p(-?\d+)", v)
    if not m:
        raise ValueError("variant must look like h110p12")
    return float(m[1]), float(m[2]), "@" + v


def prep_window(clip: str, f0: int, variant: str | None = None) -> Path:
    """Front-wide f-theta mp4 frames [f0, f0+181) -> virtual pinhole clip (854x480, 90 deg, pitched up), 30 fps."""
    import cv2
    import imageio_ffmpeg
    mod = idm._pai_validate()
    hfov, pitch, suf = parse_variant(variant)
    if hfov is not None:
        mod.HFOV, mod.PITCH_UP = hfov, pitch                              # our in-process copy of the module only
    out = WORK / clip
    out.mkdir(parents=True, exist_ok=True)
    dst = out / f"pinhole{suf}_{f0:03d}.mp4"
    if dst.exists():
        return dst
    ts = out / "timestamps.parquet"
    src = out / "front_wide.mp4"
    with zipfile.ZipFile(PAI / "camera" / CAM / f"{CAM}.{CHUNK}.zip") as z:
        if not ts.exists():
            ts.write_bytes(z.read(f"{clip}.{CAM}.timestamps.parquet"))
        if not src.exists():
            src.write_bytes(z.read(f"{clip}.{CAM}.mp4"))
    cam, _, _ = mod.rig_camera(clip)
    rays, _ = mod.virtual_rays()
    px, ok = cam.ray2pixel(rays)
    ss = 2
    mapx = px[:, 0].reshape(mod.OUT_H * ss, mod.OUT_W * ss).astype(np.float32)
    mapy = px[:, 1].reshape(mod.OUT_H * ss, mod.OUT_W * ss).astype(np.float32)
    cap = cv2.VideoCapture(str(src))
    for _ in range(f0):
        cap.grab()
    ff = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                           "-s", f"{mod.OUT_W}x{mod.OUT_H}", "-r", "30", "-i", "-", "-an", "-c:v", "libx264", "-preset", "veryfast",
                           "-crf", "12", "-pix_fmt", "yuv420p", str(dst)], stdin=subprocess.PIPE)
    for _ in range(PINHOLE_FRAMES):
        okf, bgr = cap.read()
        if not okf:
            break
        big = cv2.remap(bgr, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        ff.stdin.write(cv2.resize(big, (mod.OUT_W, mod.OUT_H), interpolation=cv2.INTER_AREA).tobytes())
    cap.release()
    ff.stdin.close()
    ff.wait()
    src.unlink(missing_ok=True)
    return dst


def cmd_prep(a):
    plan = jload(DIR / "plan.json")
    suf = parse_variant(a.variant)[2]
    for w in sum((plan[k] for k in a.sets.split(",")), []):
        t0 = time.time()
        p = prep_window(w["clip"], w["f0"], a.variant)
        win = DIR / "win" / f"{w['name']}{suf}.mp4"
        win.parent.mkdir(parents=True, exist_ok=True)
        if not win.exists():
            idm.window_video(p, win, 0, src_fps=30.0)
        print(w["name"], f"{time.time() - t0:.1f}s", flush=True)
    for yid in YT_IDS if "youtube" in a.sets else []:
        for s in (0, 6, 12, 18):
            win = DIR / "win" / f"yt_{yid}_{s:02d}.mp4"
            if not win.exists():
                idm.window_video(YT / f"yt_{yid}_0.mp4", win, s * 30, src_fps=30.0)
    print("prep done")


# ============================================================================== cosmos session
class CgroupPeak:
    """Peak of the container's cgroup memory while a request runs."""

    def __enter__(self):
        self.peak, self._stop = 0.0, threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def _run(self):
        while not self._stop.is_set():
            m = idm.container_mem_gb()
            self.peak = max(self.peak, m or 0.0)
            self._stop.wait(0.5)

    def __exit__(self, *e):
        self._stop.set()
        self._t.join()


def run_window(video: Path, out_json: Path, *, seed: int, base_url: str, steps: int) -> dict:
    with MemSampler(0.25) as ms, CgroupPeak() as cg:
        r = idm.inverse_dynamics(video, base_url=base_url, seed=seed, steps=steps)
    rec = {"video": video.name, "seed": seed, "steps": steps, "latency_s": r["latency_s"], "action": r["action"],
           "host_used_before_gb": ms.base, "host_used_peak_gb": ms.peak, "host_used_peak_delta_gb": ms.peak - ms.base,
           "min_mem_available_gb": float(idm.mem_available_gb()), "container_mem_peak_gb": cg.peak, "ts": time.strftime("%F %T")}
    jsave(out_json, rec)
    return rec


def wait_free(min_gb: float, poll: float = 20.0, max_wait_h: float = 12.0):
    t0 = time.time()
    while True:
        why = idm.free_to_start(min_gb)
        if not why:
            return
        print(time.strftime("%T"), "waiting:", "; ".join(why), flush=True)
        if time.time() - t0 > max_wait_h * 3600:
            raise TimeoutError("thor3 not free: " + "; ".join(why))
        time.sleep(poll)


def cmd_cosmos(a):
    plan = jload(DIR / "plan.json")
    names = {"calib": [w["name"] for w in plan["calib"]], "eval": [w["name"] for w in plan["eval"]],
             "youtube": [f"yt_{y}_{s:02d}" for y in YT_IDS for s in (0, 6, 12, 18)]}
    suf = parse_variant(a.variant)[2]
    todo = [(k, n + (suf if k != "youtube" else "")) for k in a.sets.split(",") for n in names[k]]
    out = DIR / "cosmos" / a.tag
    if not a.no_serve:
        heavy_ok_wait()
        wait_free(a.min_avail_gb)
        cmd = idm.serve(deploy_config=a.deploy_config, extra_args=tuple(a.extra.split()) if a.extra else ())
        print("serve:", " ".join(cmd[:12]), "...", flush=True)
        print(f"ready after {idm.wait_ready():.0f} s", flush=True)
    try:
        if not (out / "_warmup.json").exists() or a.rewarm:
            r = run_window(DIR / "win" / f"{todo[0][1]}.mp4", out / "_warmup.json", seed=99, base_url=a.url, steps=a.steps)
            print(f"warm-up request {r['latency_s']:.1f} s (first request: compile / cache warm)", flush=True)
        for k, n in todo:
            for seed in a.seeds:
                p = out / f"{n}.s{seed}.json"
                if p.exists():
                    continue
                r = run_window(DIR / "win" / f"{n}.mp4", p, seed=seed, base_url=a.url, steps=a.steps)
                print(k, n, f"seed {seed}: {r['latency_s']:.1f} s, host +{r['host_used_peak_delta_gb']:.1f} GB, "
                      f"container peak {r['container_mem_peak_gb']:.1f} GB", flush=True)
    finally:
        if not a.no_serve and not a.keep:
            idm.stop()
            print("server stopped", flush=True)


# ============================================================================== calibrate
def _gt(w, variant: str | None = None):
    return idm.pai_gt(w["clip"], w["f0"], pitch_up_deg=parse_variant(variant)[1])


def _signed_perms():
    import itertools
    out = []
    for perm in itertools.permutations(range(3)):
        for sg in itertools.product((1, -1), repeat=3):
            Q = np.zeros((3, 3))
            for i, (p, s) in enumerate(zip(perm, sg)):
                Q[i, p] = s
            out.append(Q)
    return out


def cmd_calibrate(a):
    plan = jload(DIR / "plan.json")
    rep, rows = {}, []
    acts, gts = [], []
    for w in plan["calib"]:
        p = DIR / "cosmos" / a.tag / f"{w['name']}{parse_variant(a.variant)[2]}.s0.json"
        if not p.exists():
            continue
        acts.append(np.asarray(jload(p)["action"], np.float64))
        gts.append(_gt(w, a.variant))
        rows.append(w)
    if not acts:
        raise SystemExit("no calibration results yet (run `cosmos --sets calib`)")
    # --- 1. frame axes + unit: translation of camera i+1 in camera i axes vs logged
    tg = np.concatenate([idm.relative_steps(g["poses"])[:, :3, 3] for g in gts])
    te = np.concatenate([x[:, :3] for x in acts])
    res = []
    for Q in _signed_perms():
        m = te @ Q.T
        s = float((m * tg).sum() / (m * m).sum())
        res.append((float(np.linalg.norm(tg - s * m) / np.linalg.norm(tg)), s, Q))
    res.sort(key=lambda r: r[0])
    rep["axes_top3"] = [{"rel_err": r[0], "scale": r[1], "Q": r[2].astype(int).tolist()} for r in res[:3]]
    rep["identity_axes"] = next({"rel_err": r[0], "scale": r[1]} for r in res if np.allclose(r[2], np.eye(3)))
    M = np.linalg.lstsq(te, tg, rcond=None)[0].T
    rep["lstsq_matrix_gt_from_cosmos"] = M
    print("axes (best signed permutation first):", json.dumps(rep["axes_top3"][0]), "identity:", rep["identity_axes"])
    # --- 2. per-step distance: unit, lag
    d_g = [np.linalg.norm(idm.relative_steps(g["poses"])[:, :3, 3], axis=1) for g in gts]
    d_e = [np.linalg.norm(x[:, :3], axis=1) for x in acts]
    lag_res = {}
    for lag in (-2, -1, 0, 1, 2):
        num = den = err = tot = 0.0
        for dg, de in zip(d_g, d_e):
            x, y = (de[:len(de) - lag], dg[lag:]) if lag >= 0 else (de[-lag:], dg[:len(dg) + lag])
            num += (x * y).sum(); den += (x * x).sum(); err += ((x - y) ** 2).sum(); tot += (y ** 2).sum()
        s = num / den
        lag_res[lag] = {"scale": float(s), "rmse_m": float(np.sqrt(err / sum(len(d) - abs(lag) for d in d_e))),
                        "rel_err_unscaled": float(np.sqrt(err / tot))}
    rep["distance_by_lag"] = lag_res
    print("per-step distance vs logged, lag (action i vs step i+lag):", json.dumps(lag_res))
    scale = lag_res[0]["scale"]
    rep["distance_scale_ratio_per_window"] = [float(np.sum(de * dg) / np.sum(de * de)) for dg, de in zip(d_g, d_e)]
    # --- 3. rotation encoding x chaining: compare reconstructed tracks with the logged poses
    table = {}
    for layout in ("cols", "rows"):
        for chain in ("right", "left"):
            ms = [idm.metrics(idm.actions_to_poses(x, {"rot6d": layout, "chain": chain, "scale": 1.0}), g["poses"], g["t"])
                  for x, g in zip(acts, gts)]
            table[f"{layout}/{chain}"] = {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}
    rep["rotation_chain_grid"] = table
    for k, v in table.items():
        print(f"  {k:12s} speed MAE {v['speed_mae_kmh']:.2f} km/h  yaw MAE {v['yaw_rate_mae_dps']:.2f} deg/s  ATE se2 {v['ate_se2_m']:.2f} m  "
              f"RPE {v['rpe_1s_trans_m']:.2f} m / {v['rpe_1s_rot_deg']:.2f} deg")
    # yaw-rate slope for the winning combination (sign test): decoded rotation about the vertical vs logged rig yaw
    slopes = {}
    for layout in ("cols", "rows"):
        yaw_e = []
        for x, g in zip(acts, gts):
            T = idm.action_steps(x, {"rot6d": layout})
            rv = np.degrees(np.arctan2(T[:, 0, 2] - T[:, 2, 0], T[:, 0, 0] + T[:, 2, 2]))     # rotation about camera y (down): + = right
            yaw_e.append(rv)
        yaw_g = [-np.degrees(np.diff(g["rig_yaw"])) for g in gts]                             # rig yaw ccw+ -> right turn = -
        e, gg = np.concatenate(yaw_e), np.concatenate(yaw_g)
        slopes[layout] = {"slope": float((e * gg).sum() / (e * e).sum()), "corr": float(np.corrcoef(e, gg)[0, 1])}
    rep["yaw_rate_slope_cosmos_to_logged"] = slopes
    print("yaw-rate regression (logged = slope * cosmos, per step):", json.dumps(slopes))
    best = min(table, key=lambda k: table[k]["ate_se2_m"] + table[k]["yaw_rate_mae_dps"])
    # rotation unit: angle about the vertical-ish camera y axis, summed over 1 s so per-step noise does not attenuate the fit
    from scipy.spatial.transform import Rotation
    ye, yg = [], []
    for x, g in zip(acts, gts):
        re_ = Rotation.from_matrix(idm.action_steps(x, {"rot6d": best.split("/")[0]})[:, :3, :3]).as_rotvec()[:, 1]
        rg = Rotation.from_matrix(idm.relative_steps(g["poses"])[:, :3, :3]).as_rotvec()[:, 1]
        ye.append(re_.reshape(-1, 10).sum(1)); yg.append(rg.reshape(-1, 10).sum(1))
    ye, yg = np.concatenate(ye), np.concatenate(yg)
    rot_scale = float((ye * yg).sum() / (ye * ye).sum())
    rep["rot_scale_fit_1s_sums"] = {"rot_scale": rot_scale, "corr": float(np.corrcoef(ye, yg)[0, 1])}
    print("rotation unit (logged = s * cosmos, 1 s sums):", rep["rot_scale_fit_1s_sums"])
    rep["chosen"] = {"rot6d": best.split("/")[0], "chain": best.split("/")[1], "scale": round(scale, 3), "rot_scale": round(rot_scale, 3), "lag": 0}
    print("chosen:", rep["chosen"])
    # evidence on the unscaled / scaled convention
    ms = [idm.metrics(idm.actions_to_poses(x, {**rep["chosen"], "scale": 1.0, "rot_scale": 1.0}), g["poses"], g["t"]) for x, g in zip(acts, gts)]
    ms_s = [idm.metrics(idm.actions_to_poses(x, rep["chosen"]), g["poses"], g["t"]) for x, g in zip(acts, gts)]
    rep["chosen_metrics_scale1"] = ms
    rep["chosen_metrics_scale_fit"] = ms_s
    jsave(DIR / f"calib_{a.tag}{parse_variant(a.variant)[2]}.json", rep)
    print("speed MAE km/h per calib window (scale 1):", [round(m["speed_mae_kmh"], 2) for m in ms],
          "| scale fit:", [round(m["speed_mae_kmh"], 2) for m in ms_s])


# ============================================================================== ViPE
HOLD_FLAG = Path("~/tsf/dev/flags/uc2_reruns.done").expanduser()
HEAVY_OK = Path("~/tsf/dev/heavy_ok.sh").expanduser()


def heavy_ok_wait(owner: str = "idm-agent") -> None:
    """thor3 scheduling gate: block until no other owner holds an exclusive window (``~/tsf/dev/heavy_ok.sh``)."""
    if HEAVY_OK.exists():
        while subprocess.run([str(HEAVY_OK), owner]).returncode:
            print(time.strftime("%T"), "heavy_ok.sh: another owner holds an exclusive window", flush=True)
            subprocess.run([str(HEAVY_OK), owner, "--wait"])


def hold_for_uc2_reruns(state: dict, after_t1_s: float = 3600.0, poll: float = 20.0) -> None:
    """Coordinator rule: the ViPE owner's UC2 reruns go first. Block until ``~/tsf/dev/flags/uc2_reruns.done`` exists or the T1
    container ``tsf-cosmos-sim`` has been gone for ``after_t1_s`` seconds, whichever comes first (only once per run)."""
    if state.get("held"):
        return
    while True:
        names = subprocess.run(["docker", "ps", "--format", "{{.Names}}"], capture_output=True, text=True).stdout.split()
        if "tsf-cosmos-sim" in names:
            state["t1_gone"] = None
        elif state.get("t1_gone") is None:
            state["t1_gone"] = time.time()
        if HOLD_FLAG.exists() or (state.get("t1_gone") and time.time() - state["t1_gone"] >= after_t1_s):
            state["held"] = True
            print(time.strftime("%T"), "UC2 hold released:", "flag present" if HOLD_FLAG.exists() else "60 min after T1", flush=True)
            return
        print(time.strftime("%T"), "holding for the UC2 reruns (T1 running)" if state.get("t1_gone") is None else
              f"holding for the UC2 reruns ({(time.time() - state['t1_gone']) / 60:.0f} min after T1)", flush=True)
        time.sleep(poll * 6)


def _vipe_module():
    """The current ``tsf.ingest.vipe`` (run_vipe with ``gate``/retries) and its dev worker, as the ViPE owner runs them from
    ``~/tsf/dev/vipe/repo`` (thor3's main checkout lags behind); imported by path, nothing is edited."""
    import importlib.util
    for root in (Path.home() / "tsf" / "dev" / "vipe" / "repo", ROOT):
        p = root / "src" / "tsf" / "ingest" / "vipe.py"
        if p.is_file() and "gate" in p.read_text():
            w = root / "deploy" / "vipe" / "vipe_worker.py"
            if w.is_file():
                os.environ.setdefault("TSF_VIPE_WORKER", str(w))
            spec = importlib.util.spec_from_file_location("tsf_vipe_current", p)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    raise FileNotFoundError("no tsf/ingest/vipe.py with run_vipe(gate=...) found")


def heavy_claim(minutes: float, owner: str = "idm-agent") -> None:
    """Wait for thor3's scheduling gate, then claim an exclusive window of ``minutes`` (``heavy_ok.sh --wait`` then ``--claim``)."""
    if not HEAVY_OK.exists():
        return
    while True:
        subprocess.run([str(HEAVY_OK), owner, "--wait"])
        if subprocess.run([str(HEAVY_OK), owner, "--claim", f"{minutes:g}"]).returncode == 0:
            return


def heavy_release(owner: str = "idm-agent") -> None:
    if HEAVY_OK.exists():
        subprocess.run([str(HEAVY_OK), owner, "--release"])


def cmd_vipe(a):
    vipe = _vipe_module()
    plan = jload(DIR / "plan.json")
    state = {"t0": None, "t1_gone": a.t1_gone_epoch}                          # budget counts from the first window launch
    out = DIR / "vipe"
    if a.grace_s:
        time.sleep(a.grace_s)
    batch_end, done = 0.0, 0
    try:
        for w in plan["eval"]:
            res = out / w["name"] / "result.json"
            if res.exists():
                done += 1
                continue
            if state["t0"] and time.time() > state["t0"] + a.budget_min * 60:
                print("budget exhausted", flush=True)
                break
            if time.time() > batch_end - 5 * 60:                              # claim thor3 in batches <= 40 min, yield in between
                if batch_end:
                    heavy_release()
                    time.sleep(a.gap_s)
                hold_for_uc2_reruns(state)
                heavy_claim(a.batch_min)
                batch_end = time.time() + a.batch_min * 60
                print(time.strftime("%T"), f"claimed thor3 for {a.batch_min:g} min", flush=True)

            def gate():
                heavy_ok_wait()
                wait_free(a.min_avail_gb)
                state["t0"] = state["t0"] or time.time()
            try:
                vr = vipe.run_vipe(prep_window(w["clip"], w["f0"]), out / w["name"], max_seconds=6.0, fps=a.fps, height=480,
                                   min_avail_gb=20.0, wait_s=1800, gate=gate)
            except Exception as e:                                           # a killed container (mem guard) is a failed window
                print(w["name"], "FAILED:", str(e)[-300:], flush=True)
                jsave(out / w["name"] / "failed.json", {"error": str(e)[-2000:], "ts": time.strftime("%F %T")})
                continue
            st = vr["windows"][0]["stats"]
            rec = {"name": w["name"], "container_wall_s": vr["windows"][0]["runtime_s"], "frames": vr["frames"], "fps": vr["fps"],
                   "worker_s": st["prep_s"] + st["read_s"] + st["pipeline_s"], "model_load_s": st.get("model_load_s"),
                   "infer_fps": st.get("infer_fps"), "gpu_peak_reserved_gb": st.get("gpu_peak_reserved_gb"),
                   "rss_peak_gb": st.get("rss_peak_gb"), "camera_height": vr["camera_height"]}
            np.savez(out / w["name"] / "poses.npz", poses=vr["poses"], t=vr["t"])
            jsave(res, rec)
            done += 1
            print(w["name"], f"worker {rec['worker_s']:.0f} s (container {rec['container_wall_s']:.0f} s)  gpu {rec['gpu_peak_reserved_gb']:.1f} GB  "
                  f"rss {rec['rss_peak_gb']:.1f} GB  ({done}/{len(plan['eval'])})", flush=True)
    finally:
        heavy_release()


# ============================================================================== report
def _stat(v):
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    return {"n": int(len(v)), "median": float(np.median(v)), "p90": float(np.percentile(v, 90)), "mean": float(v.mean())} if len(v) else {"n": 0}


def cmd_report(a):
    from tsf.ingest import vipe
    plan = jload(DIR / "plan.json")
    conv = jload(DIR / f"calib_{a.tag}.json")["chosen"] if a.conv == "calib" else idm.CONVENTION
    conv = {**idm.CONVENTION, **{k: v for k, v in conv.items() if k in idm.CONVENTION}, **({"scale": a.scale, "rot_scale": a.scale} if a.scale is not None else {})}
    rows = []
    for w in plan["eval"]:
        pc = DIR / "cosmos" / a.tag / f"{w['name']}.s0.json"
        if not pc.exists():
            continue
        cj = jload(pc)
        gt = _gt(w)
        row = {"name": w["name"], "gt_speed_kmh": w["speed_kmh"], "gt_yaw_deg": w["yaw_deg"], "cosmos_latency_s": cj["latency_s"],
               "cosmos_host_peak_delta_gb": cj["host_used_peak_delta_gb"], "cosmos_container_peak_gb": cj["container_mem_peak_gb"]}
        P = idm.actions_to_poses(np.asarray(cj["action"]), conv)
        for despike in (False, True):
            m = idm.metrics(P, gt["poses"], gt["t"], despike=despike)
            row["cosmos" + ("_ds" if despike else "")] = m
        pv = DIR / "vipe" / w["name"] / "result.json"
        if pv.exists():
            vj = jload(pv)
            vs = jload(DIR / "vipe" / w["name"] / "w00" / "stats.json")
            vj["runtime_s"] = vs["prep_s"] + vs["read_s"] + vs["pipeline_s"]       # worker wall: trim + read + model load + SLAM
            vj["model_load_s"], vj["gpu_peak_reserved_gb"], vj["rss_peak_gb"] = vs["model_load_s"], vs["gpu_peak_reserved_gb"], vs["rss_peak_gb"]
            z = np.load(DIR / "vipe" / w["name"] / "poses.npz")
            Pv = idm.resample_poses(z["poses"], z["t"], gt["t"])
            for despike in (False, True):
                row["vipe" + ("_ds" if despike else "")] = idm.metrics(Pv, gt["poses"], gt["t"], despike=despike)
            row.update({"vipe_runtime_s": vj["runtime_s"], "vipe_infer_s": vj["runtime_s"] - (vj.get("model_load_s") or 0),
                        "vipe_gpu_gb": vj["gpu_peak_reserved_gb"], "vipe_rss_gb": vj["rss_peak_gb"], "vipe_container_wall_s": vj.get("container_wall_s")})
        rows.append(row)
    both = [r for r in rows if "vipe" in r]
    agg = {"convention": conv, "n_cosmos": len(rows), "n_both": len(both)}
    keys = ("speed_mae_kmh", "yaw_rate_mae_dps", "ate_se2_m", "ate_se3_m", "rpe_1s_trans_m", "rpe_1s_rot_deg")
    for sel, rr in (("all_cosmos_windows", rows), ("windows_with_vipe", both)):
        for meth in ("cosmos", "cosmos_ds", "vipe", "vipe_ds"):
            if meth.startswith("vipe") and sel == "all_cosmos_windows":
                continue
            agg[f"{sel}/{meth}"] = {k: _stat([r[meth][k] for r in rr if meth in r]) for k in keys}
    agg["runtime_s"] = {"cosmos_latency": _stat([r["cosmos_latency_s"] for r in rows]),
                        "vipe_worker_total": _stat([r.get("vipe_runtime_s") for r in rows]), "vipe_infer_only": _stat([r.get("vipe_infer_s") for r in rows]),
                        "vipe_container_wall": _stat([r.get("vipe_container_wall_s") for r in rows])}
    agg["memory_gb"] = {"cosmos_host_peak_delta": _stat([r["cosmos_host_peak_delta_gb"] for r in rows]),
                        "cosmos_container_peak": _stat([r["cosmos_container_peak_gb"] for r in rows]),
                        "vipe_gpu_reserved": _stat([r.get("vipe_gpu_gb") for r in rows]), "vipe_rss": _stat([r.get("vipe_rss_gb") for r in rows])}
    # ---- YouTube agreement (no ground truth)
    yt = []
    for y in YT_IDS:
        trk = np.load(DIR.parent / "vipe" / "out" / f"yt_{y}_0" / "track.npz")
        tr_v = vipe.despike(vipe.ego_track(trk["poses"], trk["t"]), trk["t"])
        for s in (0, 6, 12, 18):
            pc = DIR / "cosmos" / a.tag / f"yt_{y}_{s:02d}.s0.json"
            if not pc.exists():
                continue
            act = np.asarray(jload(pc)["action"])
            tc = s + GRID
            rec = {"clip": y, "start_s": s}
            for tag, ds in (("raw", False), ("despike", True)):
                tr_c = idm.ego_track_from_actions(act, conv, despike=ds)
                vv = np.interp(tc, trk["t"], tr_v[:, 3])
                vc = tr_c[:, 3]
                wv = idm.rates(np.column_stack([np.zeros(len(tc)), np.zeros(len(tc)), np.interp(tc, trk["t"], tr_v[:, 2]), vv]), tc)[1]
                wc = idm.rates(tr_c, tc)[1]
                rec[tag] = {"speed_mae_kmh": float(np.mean(np.abs(vc - vv)) * 3.6), "speed_cosmos_kmh": float(vc.mean() * 3.6),
                            "speed_vipe_kmh": float(vv.mean() * 3.6), "yaw_rate_mae_dps": float(np.degrees(np.mean(np.abs(wc - wv)))),
                            "speed_corr": float(np.corrcoef(vc, vv)[0, 1]) if vv.std() > 1e-3 and vc.std() > 1e-3 else None}
            yt.append(rec)
    agg["youtube"] = {"windows": yt}
    for tag in ("raw", "despike"):
        sel = [r for r in yt if r["start_s"] < 20]
        agg["youtube"][tag] = {"speed_mae_kmh": _stat([r[tag]["speed_mae_kmh"] for r in sel]),
                               "yaw_rate_mae_dps": _stat([r[tag]["yaw_rate_mae_dps"] for r in sel]),
                               "speed_corr": _stat([r[tag]["speed_corr"] for r in sel]),
                               "mean_speed_ratio_cosmos_over_vipe": float(np.sum([r[tag]["speed_cosmos_kmh"] for r in sel]) / max(np.sum([r[tag]["speed_vipe_kmh"] for r in sel]), 1e-9))}
    jsave(DIR / f"results_{a.tag}.json", {"aggregate": agg, "windows": rows})
    print(json.dumps({k: v for k, v in agg.items() if k != "youtube"}, indent=1)[:6000])
    print("youtube:", json.dumps({k: agg["youtube"][k] for k in ("raw", "despike")}, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("plan"); p.add_argument("--seed", type=int, default=7); p.add_argument("--n-eval", type=int, default=30); p.set_defaults(f=cmd_plan)
    p = sp.add_parser("prep"); p.add_argument("--variant"); p.add_argument("--sets", default="calib,eval,youtube"); p.set_defaults(f=cmd_prep)
    p = sp.add_parser("cosmos")
    p.add_argument("--sets", default="calib,eval,youtube"); p.add_argument("--tag", default="edge"); p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--steps", type=int, default=30); p.add_argument("--url", default=idm.DEFAULT_URL); p.add_argument("--min-avail-gb", type=float, default=40.0)
    p.add_argument("--no-serve", action="store_true"); p.add_argument("--keep", action="store_true"); p.add_argument("--rewarm", action="store_true")
    p.add_argument("--deploy-config"); p.add_argument("--extra", default=""); p.add_argument("--variant"); p.set_defaults(f=cmd_cosmos)
    p = sp.add_parser("calibrate"); p.add_argument("--tag", default="edge"); p.add_argument("--variant"); p.set_defaults(f=cmd_calibrate)
    p = sp.add_parser("vipe"); p.add_argument("--budget-min", type=float, default=120.0); p.add_argument("--fps", type=float, default=15.0)
    p.add_argument("--min-avail-gb", type=float, default=40.0); p.add_argument("--grace-s", type=float, default=0.0)
    p.add_argument("--gap-s", type=float, default=90.0); p.add_argument("--batch-min", type=float, default=40.0)
    p.add_argument("--t1-gone-epoch", type=float, help="when the T1 container was last seen gone (resume of a restarted loop)"); p.set_defaults(f=cmd_vipe)
    p = sp.add_parser("report"); p.add_argument("--tag", default="edge"); p.add_argument("--conv", choices=["calib", "frozen"], default="frozen")
    p.add_argument("--scale", type=float); p.set_defaults(f=cmd_report)
    a = ap.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
