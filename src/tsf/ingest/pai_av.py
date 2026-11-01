"""UC4/UC5: PhysicalAI-AV snippet (GT egomotion + auto-labelled obstacle tracks) -> TSIR.

load      : egomotion (anchor frame, ~100 Hz) and obstacle.offline (rig frame, ~10 Hz) of one clip
frenet    : the ego path, low-passed over ~25 m so lane changes stay lateral offsets, is the road reference;
            every track is mapped to (s, d) along it
road      : piecewise-constant-curvature segments of the reference; lane counts and the ego lane from the
            lateral distribution of same-direction and oncoming traffic
behaviour : speed profiles -> speed_change actions (Douglas-Peucker breakpoints); lane index changes ->
            lane_change actions; static vehicles keep their lateral offset (partially blocking a lane)
fidelity  : the lifted TSIR is simulated in TanitSim and compared with the log in road coordinates
Dataset licence: internal validation only, so inputs and lifted scenarios stay on thor3.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

ROOT = Path(os.environ.get("TSF_PAI_AV", Path.home() / "data" / "pai_av" / "x"))
CLASS = {"automobile": "car", "car": "car", "truck": "truck", "heavy_truck": "truck", "bus": "bus", "van": "van",
         "trailer": "truck", "motorcycle": "motorcycle", "bicycle": "bicycle", "cyclist": "bicycle",
         "person": "pedestrian", "pedestrian": "pedestrian"}
W = 3.5  # nominal lane width


@dataclass
class Track:
    id: str
    cls: str
    t: np.ndarray
    xy: np.ndarray            # (N, 2) anchor frame
    yaw: np.ndarray
    lwh: tuple[float, float, float]
    sd: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))   # Frenet (s, d) along the reference


@dataclass
class Clip:
    id: str
    t: np.ndarray             # ego, uniform 10 Hz from 0
    xy: np.ndarray
    yaw: np.ndarray
    v: np.ndarray
    tracks: list[Track]


def _yaw(qx, qy, qz, qw):
    return np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy ** 2 + qz ** 2))


def load_clip(clip_id: str, root: Path = ROOT, hz: float = 10.0) -> Clip:
    e = pd.read_parquet(root / "ego" / f"{clip_id}.egomotion.parquet").sort_values("timestamp")
    te = e.timestamp.to_numpy() / 1e6
    yaw_e = np.unwrap(_yaw(e.qx, e.qy, e.qz, e.qw).to_numpy())
    t = np.arange(0.0, min(te.max(), 20.0), 1.0 / hz)
    ex, ey, eyaw = (np.interp(t, te, a) for a in (e.x.to_numpy(), e.y.to_numpy(), yaw_e))
    v = np.interp(t, te, np.hypot(e.vx, e.vy).to_numpy())
    tracks = []
    op = root / "obs" / f"{clip_id}.obstacle.offline.parquet"
    if op.exists():
        o = pd.read_parquet(op)
        tr = o.reference_frame_timestamp_us.to_numpy() / 1e6
        rx, ry, ryaw = (np.interp(tr, te, a) for a in (e.x.to_numpy(), e.y.to_numpy(), yaw_e))
        c, s = np.cos(ryaw), np.sin(ryaw)
        o = o.assign(X=rx + c * o.center_x - s * o.center_y, Y=ry + s * o.center_x + c * o.center_y,
                     YAW=ryaw + _yaw(o.orientation_x, o.orientation_y, o.orientation_z, o.orientation_w),
                     T=o.timestamp_us.to_numpy() / 1e6)
        for tid, g in o.sort_values("T").groupby("track_id"):
            if len(g) < 5:
                continue
            cls = CLASS.get(str(g.label_class.mode().iloc[0]).lower(), "car")
            tracks.append(Track(str(tid), cls, g["T"].to_numpy(), g[["X", "Y"]].to_numpy(), np.unwrap(g.YAW.to_numpy()),
                                (float(g.size_x.median()), float(g.size_y.median()), float(g.size_z.median()))))
    return Clip(clip_id, t, np.stack([ex, ey], 1), eyaw, v, tracks)


# ----------------------------------------------------------------------------- reference + Frenet
@dataclass
class Reference:
    s: np.ndarray       # arc length, 0 at the ego start, extended back/forward
    xy: np.ndarray
    hdg: np.ndarray
    kappa: np.ndarray
    back: float

    def frenet(self, pts: np.ndarray) -> np.ndarray:
        d2 = ((pts[:, None, :] - self.xy[None, :, :]) ** 2).sum(-1)
        i = d2.argmin(1)
        dv = pts - self.xy[i]
        h = self.hdg[i]
        along = dv[:, 0] * np.cos(h) + dv[:, 1] * np.sin(h)
        lat = -dv[:, 0] * np.sin(h) + dv[:, 1] * np.cos(h)
        return np.stack([self.s[i] + along, lat], 1)


def reference(clip: Clip, sigma_m: float = 25.0, back: float = 120.0, fwd: float = 200.0, ds: float = 1.0) -> Reference:
    seg = np.hypot(*np.diff(clip.xy, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    keep = np.concatenate([[True], seg > 1e-3])
    s, xy = s[keep], clip.xy[keep]
    su = np.arange(0.0, s[-1] + 1e-6, ds)
    x, y = np.interp(su, s, xy[:, 0]), np.interp(su, s, xy[:, 1])
    k = max(1.0, sigma_m / ds)
    x, y = gaussian_filter1d(x, k, mode="nearest"), gaussian_filter1d(y, k, mode="nearest")
    h = np.unwrap(np.arctan2(np.gradient(y), np.gradient(x)))
    h0, h1 = h[:5].mean(), h[-5:].mean()
    nb, nf = int(back / ds), int(fwd / ds)
    sb = -np.arange(nb, 0, -1) * ds
    xb, yb = x[0] + sb * math.cos(h0), y[0] + sb * math.sin(h0)
    sf = np.arange(1, nf + 1) * ds
    xf, yf = x[-1] + sf * math.cos(h1), y[-1] + sf * math.sin(h1)
    S = np.concatenate([sb, su, su[-1] + sf])
    X, Y = np.concatenate([xb, x, xf]), np.concatenate([yb, y, yf])
    H = np.concatenate([np.full(nb, h0), h, np.full(nf, h1)])
    K = gaussian_filter1d(np.gradient(H, S), 3.0)
    return Reference(S, np.stack([X, Y], 1), H, K, back)


def road_segments(ref: Reference, piece: float = 20.0, kmin: float = 2.5e-4) -> list[dict]:
    out, s0 = [], ref.s[0]
    while s0 < ref.s[-1] - 1.0:
        s1 = min(s0 + piece, ref.s[-1])
        k = float(ref.kappa[(ref.s >= s0) & (ref.s < s1)].mean())
        if abs(k) < kmin:
            if out and out[-1]["kind"] == "line":
                out[-1]["length"] = round(out[-1]["length"] + s1 - s0, 2)
            else:
                out.append({"kind": "line", "length": round(s1 - s0, 2)})
        else:
            out.append({"kind": "arc", "length": round(s1 - s0, 2), "curvature_start": round(k, 6),
                        "curvature_end": round(k, 6)})
        s0 = s1
    return out


# ----------------------------------------------------------------------------- behaviour
def _rdp(t: np.ndarray, v: np.ndarray, tol: float) -> list[int]:
    def rec(i, j):
        if j <= i + 1:
            return []
        line = v[i] + (v[j] - v[i]) * (t[i + 1:j] - t[i]) / max(t[j] - t[i], 1e-9)
        err = np.abs(v[i + 1:j] - line)
        k = int(err.argmax())
        if err[k] < tol:
            return []
        m = i + 1 + k
        return rec(i, m) + [m] + rec(m, j)
    return [0] + rec(0, len(t) - 1) + [len(t) - 1]


def _lane_speed(ref: Reference, sd: np.ndarray, ds_dt: np.ndarray) -> np.ndarray:
    """Reference-line s-rate -> speed along a lane at lateral offset d (a lane left of a left curve is shorter),
    so the simulated actor, which drives its lane path, keeps the logged road-coordinate progress."""
    return ds_dt * np.clip(1 - np.interp(sd[:, 0], ref.s, ref.kappa) * sd[:, 1], 0.5, 1.5)


def speed_actions(aid: str, t: np.ndarray, v: np.ndarray, t0: float, tol: float = 1.0) -> list[dict]:
    idx = _rdp(t, v, tol)
    acts = []
    for a, b in zip(idx[:-1], idx[1:]):
        dv = v[b] - v[a]
        if abs(dv) < tol or t[a] - t0 < 0.05 and abs(dv) < 1.5 * tol:
            continue
        acc = float(np.clip(abs(dv) / max(t[b] - t[a], 0.1), 0.3, 10.0))
        acts.append({"id": f"{aid}_v{len(acts)}", "type": "speed_change", "actor": aid,
                     "start": {"kind": "sim_time", "t": round(max(0.0, t[a] - t0), 2)},
                     "target_speed": round(max(0.0, float(v[b])), 2), "accel": round(acc, 2)})
    return acts


def lane_actions(aid: str, t: np.ndarray, d: np.ndarray, t0: float) -> list[dict]:
    k = np.round(d / W).astype(int)
    acts, cur = [], k[0]
    for i in range(1, len(k)):
        if k[i] != cur and abs(d[i] - k[i] * W) < 1.0:          # settled in the new lane (hysteresis)
            j = i
            while j > 0 and abs(d[j] - cur * W) > 0.4:          # back to where the manoeuvre began
                j -= 1
            dur = float(np.clip(t[i] - t[j] + 1.0, 2.0, 8.0))
            acts.append({"id": f"{aid}_lc{len(acts)}", "type": "lane_change", "actor": aid,
                         "start": {"kind": "sim_time", "t": round(max(0.0, t[j] - t0), 2)},
                         "direction": "left" if k[i] > cur else "right", "lanes": int(abs(k[i] - cur)),
                         "duration": round(dur, 1)})
            cur = k[i]
    return acts


def offset_actions(aid: str, t: np.ndarray, d: np.ndarray, t0: float, thr: float = 0.45, min_s: float = 1.5) -> list[dict]:
    """Sustained in-lane lateral excursions (nudges) -> lane_offset actions out and back."""
    r = gaussian_filter1d(d - np.round(d / W) * W, 3)
    on = np.abs(r) > thr
    acts, i = [], 0
    while i < len(on):
        if not on[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(on) and on[j + 1]:
            j += 1
        if t[j] - t[i] >= min_s and abs(d[i] - d[j]) < W / 2:   # not part of a lane change
            for when, target in ((max(t[i] - 1.0, t[0]), float(np.median(r[i:j + 1]))), (t[j], 0.0)):
                if when - t0 < 0.05 and target == 0.0 and not acts:
                    continue
                acts.append({"id": f"{aid}_off{len(acts)}", "type": "lane_offset", "actor": aid,
                             "start": {"kind": "sim_time", "t": round(max(0.0, when - t0), 2)},
                             "offset": round(float(np.clip(target, -2.5, 2.5)), 2), "duration": 2.0})
        i = j + 1
    return acts


def _sample(tr: Track, tq: np.ndarray) -> np.ndarray:
    return np.stack([np.interp(tq, tr.t, tr.sd[:, 0]), np.interp(tq, tr.t, tr.sd[:, 1])], 1)


# ----------------------------------------------------------------------------- lift
def lift(clip: Clip, t0: float = 0.0, duration: float | None = None, max_actors: int = 12, title: str = "",
         description: str = "", polarity: str = "positive") -> dict:
    """Returns a TSIR dict (+ `_log` with the Frenet ground truth used for the fidelity check)."""
    T = clip.t[-1]
    duration = float(min(duration or T - t0, T - t0))
    ref = reference(clip, back=200.0, fwd=450.0)   # room for late (back-extrapolated) appearances
    ego_sd = ref.frenet(clip.xy)
    for tr in clip.tracks:
        tr.sd = ref.frenet(tr.xy)
    i0 = int(round(t0 / (clip.t[1] - clip.t[0])))
    te = clip.t[i0:]
    ego_d0 = ego_sd[i0, 1]
    # candidate actors: present near t0 or later interacting with the ego, inside the road corridor
    cands = []
    for tr in clip.tracks:
        if tr.t[-1] < t0 + 0.5 or tr.t[0] > t0 + duration - 1.0:
            continue
        tq = te[(te >= tr.t[0]) & (te <= tr.t[-1])]
        sd = _sample(tr, tq)
        ego_q = np.stack([np.interp(tq, clip.t, ego_sd[:, 0]), np.interp(tq, clip.t, ego_sd[:, 1])], 1)
        dist = np.hypot(*(sd - ego_q).T).min()
        dh = abs(math.remainder(float(np.median(tr.yaw)) - float(np.interp(np.median(tr.t), clip.t, clip.yaw)), 2 * math.pi))
        if np.abs(sd[:, 1]).min() > 4.5 * W or dist > 90:
            continue
        cands.append((dist, tr, dh))
    cands.sort(key=lambda c: c[0])
    cands = cands[:max_actors]
    # lane topology from the lateral distribution (lane index relative to the ego lane, + = left)
    same = [np.median(np.round((_sample(tr, tr.t)[:, 1] - ego_d0) / W)) for _, tr, dh in cands if dh < math.pi / 4]
    # oncoming lanes are anchored at the nearest oncoming vehicle (a median, if any, is compressed away)
    opp_d = [float(np.median(_sample(tr, tr.t)[:, 1] - ego_d0)) for _, tr, dh in cands if dh > 3 * math.pi / 4]
    opp_d0 = min(opp_d) if opp_d else 0.0
    opp = [int(np.round((x - opp_d0) / W)) + 1 for x in opp_d]
    ego_k = np.round((ego_sd[i0:, 1] - ego_d0) / W)
    n_left = int(max([0] + [k for k in same if k > 0] + list(ego_k[ego_k > 0])))
    n_right = int(max([0] + [-k for k in same if k < 0] + list(-ego_k[ego_k < 0])))
    n_fwd = min(6, n_left + n_right + 1)
    n_bwd = int(min(6, max([0] + opp)))
    ego_lane = n_left + 1
    s_off = ref.back                                  # road s = reference s + back
    actors, actions = [], []
    ego_s0 = float(ego_sd[i0, 0]) + s_off
    ds_ego = _lane_speed(ref, ego_sd, gaussian_filter1d(np.gradient(ego_sd[:, 0], clip.t), 5))
    actors.append({"id": "ego", "role": "ego", "category": "car", "initial": {"road": 0, "direction": "forward",
                   "lane": ego_lane, "s": round(ego_s0, 2)}, "initial_speed": round(float(ds_ego[i0]), 2)})
    actions += speed_actions("ego", te, ds_ego[i0:], t0)
    actions += lane_actions("ego", te, ego_sd[i0:, 1] - ego_d0, t0)
    actions += offset_actions("ego", te, ego_sd[i0:, 1] - ego_d0, t0)
    gt = {"ego": {"t": (te - t0).tolist(), "s": (ego_sd[i0:, 0] + s_off).tolist(), "d": (ego_sd[i0:, 1] - ego_d0).tolist()}}
    for n, (dist, tr, dh) in enumerate(cands):
        aid = f"{tr.cls}_{n}"
        ts = np.arange(max(tr.t[0], t0), min(tr.t[-1], t0 + duration), 0.1)
        if len(ts) < 3:
            continue
        sd = _sample(tr, ts)
        v = _lane_speed(ref, sd, gaussian_filter1d(np.gradient(sd[:, 0], ts), 4))
        sd[:, 1] -= ego_d0
        L = float(ref.s[-1] - ref.s[0])
        s_fwd = sd[:, 0] + s_off                              # road s in the forward direction
        if dh > 3 * math.pi / 4:                              # oncoming: backward lanes, s runs from the road end
            if n_bwd == 0:
                continue
            k = int(np.clip(np.round((np.median(sd[:, 1]) - opp_d0) / W) + 1, 1, n_bwd))
            s_gt = L - s_fwd
            s0 = s_gt[0] - abs(v[0]) * (ts[0] - t0)          # back-extrapolate late appearances to t0
            init = {"road": 0, "direction": "backward", "lane": k, "s": round(float(np.clip(s0, 1, L - 1)), 2)}
            # lane centre in the ego frame; a median (opp_d0 beyond the first lane) stays compressed, a narrow
            # road does not, so queued/parked oncoming cars keep the real gap the ego squeezes through
            centre = (n_left + k) * W + max(0.0, opp_d0 - (n_left + 1) * W)
            off = -(sd[:, 1] - centre)                            # + = left of the actor's own heading
            if np.abs(v).max() < 0.8:
                init["t_offset"] = round(float(np.clip(off.mean(), -2.5, 2.5)), 2)
            elif abs(np.median(off[:10])) > 0.4:
                init["t_offset"] = round(float(np.clip(np.median(off[:10]), -2.5, 2.5)), 2)
            speed = abs(float(np.median(v[:5])))
            acts = speed_actions(aid, ts, np.abs(v), t0)
        else:
            lane = int(np.clip(ego_lane - int(np.round(sd[0, 1] / W)), 1, n_fwd))
            s_gt = s_fwd
            s0 = s_gt[0] - v[0] * (ts[0] - t0)
            init = {"road": 0, "direction": "forward", "lane": lane, "s": round(float(s0), 2)}
            off = sd[:, 1] - (ego_lane - lane) * W                # offset from its lane centre
            if np.abs(v).max() < 0.8:                         # parked: keep its offset from the lane centre
                init["t_offset"] = round(float(np.clip(off.mean(), -2.5, 2.5)), 2)
            elif abs(np.median(off[:10])) > 0.4:              # moving but riding off-centre (e.g. a wide truck)
                init["t_offset"] = round(float(np.clip(np.median(off[:10]), -2.5, 2.5)), 2)
            speed = max(0.0, float(np.median(v[:5])))
            acts = (speed_actions(aid, ts, np.maximum(v, 0), t0) + lane_actions(aid, ts, sd[:, 1], t0)
                    + (offset_actions(aid, ts, sd[:, 1], t0) if np.abs(v).max() >= 0.8 else []))
        cat = tr.cls if tr.cls in ("car", "truck", "bus", "van", "motorcycle", "bicycle", "pedestrian") else "car"
        if cat == "pedestrian":
            continue                                             # crossing pedestrians: TODO cross actions
        actors.append({"id": aid, "role": "npc", "category": cat, "initial": init, "initial_speed": round(speed, 2),
                       "dimensions": [round(x, 2) for x in tr.lwh]})
        actions += acts
        gt[aid] = {"t": (ts - t0).tolist(), "s": s_gt.tolist(), "d": sd[:, 1].tolist(), "direction": init["direction"]}
    seg_total = ref.s[-1] - ref.s[0]
    tsir = {
        "meta": {"id": f"pai_{clip.id[:8]}", "title": title or f"PhysicalAI-AV snippet {clip.id[:8]}",
                 "description": description, "source": "negative_snippet" if polarity == "negative" else "sensor_snippet",
                 "polarity": polarity, "tags": ["PhysicalAI-AV", "log_lifted"]},
        "road": {"type": "linear", "segments": road_segments(ref), "speed_limit_kph": 100,
                 "forward": {"driving_lanes": n_fwd, "lane_width": W},
                 "backward": {"driving_lanes": n_bwd, "lane_width": W} if n_bwd else None},
        "actors": actors, "actions": actions,
        "assertions": [{"kind": "no_collision"}],
        "parameters": [], "duration": round(duration, 1), "dt": 0.05,
    }
    tsir["_log"] = {"gt": gt, "road_length": round(seg_total, 1), "lanes": [n_fwd, n_bwd], "ego_lane": ego_lane}
    return tsir


# ----------------------------------------------------------------------------- fidelity
def fidelity(sim, log, gt: dict) -> dict:
    """Simulated vs logged motion in road coordinates: ADE/FDE of s, lane agreement, per actor."""
    r = sim.m.roads["0"]
    ss = np.arange(0.0, r.length, 0.5)
    x, y, h = r.ref(ss)
    out = {}
    for aid, g in gt.items():
        if aid not in log.ids:
            continue
        j = log.ids.index(aid)
        keep = np.asarray(g["t"]) <= log.t[-1] + 1e-6          # compare only within the simulated horizon
        if keep.sum() < 2:
            continue
        tq = np.asarray(g["t"])[keep]
        sx = np.interp(tq, log.t, log.states[:, j, 0])
        sy = np.interp(tq, log.t, log.states[:, j, 1])
        d2 = (sx[:, None] - x[None]) ** 2 + (sy[:, None] - y[None]) ** 2
        i = d2.argmin(1)
        along = (sx - x[i]) * np.cos(h[i]) + (sy - y[i]) * np.sin(h[i])
        lat = -(sx - x[i]) * np.sin(h[i]) + (sy - y[i]) * np.cos(h[i])
        s_sim = ss[i] + along
        s_gt = np.asarray(g["s"])[keep]
        if g.get("direction") == "backward":
            s_sim = r.length - s_sim
        err = np.abs(s_sim - s_gt)
        out[aid] = {"ade_s": round(float(err.mean()), 2), "fde_s": round(float(err[-1]), 2), "n": int(len(tq))}
    npc = [v["ade_s"] for k, v in out.items() if k != "ego"]
    return {"ego": out.get("ego"), "npc_mean_ade_s": round(float(np.mean(npc)), 2) if npc else None, "per_actor": out}


# ----------------------------------------------------------------------------- consistency + parameters
def deconflict(tsir: dict, workdir, max_iter: int = 6, protect: tuple[str, ...] = ()):
    """Lifting errors show up as collisions in re-simulation: pull the offending actor's lateral offset back
    toward its lane centre, else drop the actor (recorded in meta). Returns (tsir, report, dropped)."""
    from tsf.verify import ladder
    dropped = []
    for _ in range(max_iter):
        rep = ladder.run(tsir, workdir, qc=False)
        cols = rep.metrics.get("collisions") if rep.metrics else None
        if rep.passed or not cols:
            break
        other = cols[0]["with"]
        a = next((x for x in tsir["actors"] if x["id"] == other), None)
        if a is None or a["role"] == "ego" or other in protect:
            break
        if abs(a["initial"].get("t_offset", 0.0)) > 0.3:
            a["initial"]["t_offset"] = round(a["initial"]["t_offset"] * 0.4, 2)
        else:
            tsir["actors"] = [x for x in tsir["actors"] if x["id"] != other]
            tsir["actions"] = [x for x in tsir["actions"] if x["actor"] != other]
            dropped.append(other)
    if dropped:
        tsir["meta"]["description"] += f" Dropped by consistency check: {', '.join(dropped)}."
    return tsir, rep, dropped


def add_parameters(tsir: dict, n_actors: int = 2) -> dict:
    """Logical parameters for augmentation: initial gap and first action of the closest NPCs."""
    ego = next(a for a in tsir["actors"] if a["role"] == "ego")
    npcs = sorted((a for a in tsir["actors"] if a["role"] != "ego" and a["initial"]["direction"] == "forward"),
                  key=lambda a: abs(a["initial"]["s"] - ego["initial"]["s"]))[:n_actors]
    params = []
    for a in npcs:
        j = tsir["actors"].index(a)
        params.append({"name": f"{a['id']}_s0", "path": f"/actors/{j}/initial/s", "distribution": "uniform",
                       "low": round(a["initial"]["s"] - 8, 1), "high": round(a["initial"]["s"] + 8, 1)})
        i = next((k for k, x in enumerate(tsir["actions"]) if x["actor"] == a["id"]), None)
        if i is not None:
            st = tsir["actions"][i]["start"]["t"]
            params.append({"name": f"{a['id']}_t_{tsir['actions'][i]['type']}", "path": f"/actions/{i}/start/t",
                           "distribution": "uniform", "low": round(max(0.0, st - 1.5), 2), "high": round(st + 1.5, 2)})
    tsir["parameters"] = params[:3]
    return tsir
