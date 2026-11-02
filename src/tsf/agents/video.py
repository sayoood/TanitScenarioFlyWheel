"""UC2: single dash-cam video (no ego data) -> ego motion + measured road users + VLM scene -> verified TSIR.

ego    : ViPE metric camera poses -> planar ego track (heading, turns from the integrated yaw, lane changes);
         the speed comes from Cosmos3-Edge inverse dynamics when available (``tsf.ingest.ego_speed``: ViPE's
         translation collapses at speed), ViPE alone otherwise
npc    : dense depth + instance tracks -> other road users with category, gap, lane, speed, manoeuvres
         (``tsf.ingest.npc``)
scene  : Qwen3.8 (vision) reads keyframes and writes a structured description (road, lanes, signals, road
         users relative to the ego, timed events, appearance)
lift   : all become one request for the UC1 Lifter; the ego profile is a hard constraint (pinned on every proposal,
         assertions included), checked afterwards by comparing the simulated ego speed with the measured one (MAE)
         and the measured road users with the simulated ones
Videos and everything derived from them stay on thor3.
"""
from __future__ import annotations

import base64
import json
import math
import os
import time
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import linear_sum_assignment

from tsf.agents.lifter import Lifter, LiftResult, _slug
from tsf.agents.llm import LLM
from tsf.ir.scenario import DEFAULT_DIMS

YOUTUBE = Path(os.environ.get("TSF_YOUTUBE", Path.home() / "data" / "youtube"))


def resolve(video: str) -> Path:
    if video.startswith("youtube:"):
        vid = video.split(":", 1)[1]
        hits = sorted(YOUTUBE.glob(f"yt_{vid}_*.mp4"))
        if not hits:
            raise FileNotFoundError(f"{video}: not downloaded to {YOUTUBE}")
        return hits[0]
    return Path(video).expanduser()


def keyframes(video: Path, n: int = 6, max_seconds: float | None = None, width: int = 896) -> list[tuple[float, str]]:
    import cv2
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    last = min(total - 1, int((max_seconds or 1e9) * fps))
    out = []
    for i in np.linspace(0, last, n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, img = cap.read()
        if not ok:
            continue
        h, w = img.shape[:2]
        img = cv2.resize(img, (width, int(h * width / w)))
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        out.append((round(i / fps, 2), base64.b64encode(buf.tobytes()).decode()))
    cap.release()
    return out


def ego_summary(track: np.ndarray, t: np.ndarray) -> dict:
    """track (N, 4) [x, y, yaw, v] from ViPE -> speed profile, turns and lane changes."""
    v = gaussian_filter1d(track[:, 3], 3)
    secs = np.arange(0, t[-1] + 1e-6, 1.0)
    yaw = np.unwrap(track[:, 2])
    turns, i0 = [], 0
    rate = np.gradient(gaussian_filter1d(yaw, 5), t)
    on = np.abs(rate) > math.radians(6)                          # sustained yaw rate > 6 deg/s
    while i0 < len(on):
        if on[i0]:
            j = i0
            while j + 1 < len(on) and on[j + 1]:
                j += 1
            dpsi = math.degrees(yaw[j] - yaw[i0])
            if abs(dpsi) > 35:
                turns.append({"t_start": round(float(t[i0]), 1), "t_end": round(float(t[j]), 1),
                              "direction": "left" if dpsi > 0 else "right", "deg": round(dpsi)})
            i0 = j + 1
        else:
            i0 += 1
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(track[:, :2], axis=0).T))])
    ref = np.stack([gaussian_filter1d(track[:, 0], 25), gaussian_filter1d(track[:, 1], 25)], 1)
    h = np.arctan2(np.gradient(ref[:, 1]), np.gradient(ref[:, 0]))
    d = -(track[:, 0] - ref[:, 0]) * np.sin(h) + (track[:, 1] - ref[:, 1]) * np.cos(h)
    lcs = []
    for k in range(1, len(t)):
        j = np.searchsorted(t, t[k] - 5.0)
        if abs(d[k] - d[j]) > 2.6 and (not lcs or t[k] - lcs[-1]["t"] > 6) and not turns_near(turns, t[k]):
            lcs.append({"t": round(float(t[j]), 1), "direction": "left" if d[k] > d[j] else "right"})
    return {"duration_s": round(float(t[-1]), 1), "distance_m": round(float(s[-1]), 1),
            "speed_kmh_per_s": [round(float(np.interp(x, t, v)) * 3.6, 1) for x in secs],
            "turns": turns, "lane_changes": lcs, "total_heading_change_deg": round(math.degrees(yaw[-1] - yaw[0]))}


def turns_near(turns: list[dict], tt: float) -> bool:
    return any(x["t_start"] - 2 <= tt <= x["t_end"] + 2 for x in turns)


class SceneActor(BaseModel):
    category: Literal["car", "suv", "van", "truck", "bus", "motorcycle", "bicycle", "pedestrian"]
    relation: Literal["ahead_same_lane", "ahead_left_lane", "ahead_right_lane", "oncoming", "crossing_from_left",
                      "crossing_from_right", "behind", "parked", "adjacent_left", "adjacent_right"]
    approx_distance_m: float = Field(..., ge=0, le=200)
    behavior: str = Field(description="what it does over the clip, with times if visible")


class VideoScene(BaseModel):
    summary: str = Field(description="2-3 sentences: what happens in the clip")
    road_type: Literal["signalized_intersection", "unsignalized_intersection", "urban_street", "highway",
                       "rural_road", "roundabout", "parking_lot"]
    lanes_same_direction: int = Field(..., ge=1, le=6)
    lanes_oncoming: int = Field(..., ge=0, le=6)
    ego_lane_from_left: int = Field(..., ge=1, le=6)
    traffic_lights: bool
    speed_limit_kph_guess: float = Field(..., ge=10, le=140)
    time_of_day: Literal["day", "dawn_dusk", "night"]
    weather: Literal["clear", "cloudy", "rain", "fog", "snow"]
    road_surface: Literal["dry", "wet", "snow", "ice"]
    actors: list[SceneActor] = Field(..., max_length=8)
    events: list[str] = Field(description="timed events, e.g. 't=6 s: white SUV cuts in from the right'")


def describe_scene(frames: list[tuple[float, str]], ego: dict, llm: LLM) -> dict:
    content = [{"type": "text", "text": "Dash-cam keyframes (timestamps in s) of one clip, plus the ego motion "
                "estimated by visual SLAM (metric). Describe the traffic scene for re-creating it in a simulator. "
                "Ego motion: " + json.dumps(ego, separators=(",", ":"))}]
    for ts, b64 in frames:
        content += [{"type": "text", "text": f"t = {ts} s"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]
    msgs = [{"role": "system", "content": "You are a traffic-scene analyst for automated-driving test design. "
             "Be factual; report only what the frames and the ego motion support; estimate distances."},
            {"role": "user", "content": content}]
    return llm.chat(msgs, schema=VideoScene.model_json_schema(), role="scene_vlm", max_tokens=2500)


def request_text(scene: dict, ego: dict, npcs: list[dict] | None = None) -> str:
    """Request for the Lifter. ``npcs`` (``npc_summary`` of the measured tracks) replaces the model's guesses about
    the other road users with measured categories, gaps, lanes, speeds and manoeuvres."""
    actors = scene_actor_text(scene, npcs)
    man = [f"turns {x['direction']} between t={x['t_start']} s and t={x['t_end']} s" for x in ego["turns"]]
    man += [f"changes lane to the {x['direction']} at t={x['t']} s" for x in ego["lane_changes"]]
    return (f"Re-create this real dash-cam scenario. {scene['summary']} Road: {scene['road_type'].replace('_', ' ')}, "
            f"{scene['lanes_same_direction']} lane(s) in the ego direction, {scene['lanes_oncoming']} oncoming, ego in "
            f"lane {scene['ego_lane_from_left']} from the left, "
            f"{'with' if scene['traffic_lights'] else 'without'} traffic "
            f"lights, speed limit about {scene['speed_limit_kph_guess']:.0f} km/h; {scene['time_of_day']}, "
            f"{scene['weather']}, {scene['road_surface']} road. Ego (measured, must be reproduced): speed profile in "
            f"km/h at 1 s steps {ego['speed_kmh_per_s']}; the ego {', '.join(man) or 'keeps its lane'}. Other road "
            f"users: {actors}. The camera looks forward: no vehicle starts beside or behind the ego, and the ego "
            f"starts at s = 30 m or more so that there is road behind it. Events: "
            f"{'; '.join(scene['events']) or 'none'}. Duration {ego['duration_s']:.0f} s.")


ACTION_BY_FIELD = (("target_speed", "speed_change"), ("direction", "lane_change"), ("time_gap", "follow"),
                   ("offset", "lane_offset"), ("side", "cross"), ("target", "cut_in"))


def infer_action_types(tsir: dict) -> dict:
    """The schema leaves ``type`` optional (it has a default), so the model sometimes copies an action without it
    and the discriminated union rejects the TSIR in every repair round (a cut-in did, dToIIqqvSt0/amuld reruns):
    restore it from the action's distinguishing field."""
    for a in tsir.get("actions", []):
        if isinstance(a, dict) and "type" not in a:
            a["type"] = next((t for f, t in ACTION_BY_FIELD if f in a), "speed_change")
    return tsir


def move_signs_off_the_road(tsir: dict, verge_m: float = 2.0) -> dict:
    """The model puts roadside panels (an overhead gantry on a highway) at ``t_offset`` +10 m of the outermost lane
    - which lands inside the carriageway, where every driver of the simulation stops for it (amuld replay: the
    traffic ahead braked to a standstill and the pinned ego with it). Panels inside a travelling lane move to
    ``verge_m`` beyond the outer edge of their carriageway (linear roads only)."""
    from tsf.ir.scenario import SIGN_FACING_KINDS
    road = tsir.get("road", {})
    if road.get("type") != "linear":
        return tsir
    side = {"forward": road.get("forward") or {}, "backward": road.get("backward") or {}}
    for o in tsir.get("static_objects", []):
        pos = o.get("position", {})
        d = pos.get("direction", "forward")
        if o.get("kind") not in SIGN_FACING_KINDS or d not in side or not side[d]:
            continue
        n, w = side[d].get("driving_lanes", 2), side[d].get("lane_width", 3.5)
        sgn = -1.0 if d == "forward" else 1.0                      # forward lanes lie right of the reference line
        c = sgn * (pos.get("lane", 1) - 0.5) * w                   # lateral position of the lane centre (left > 0)
        off = float(pos.get("t_offset", 0.0))
        if 0.0 <= sgn * (c + off) <= n * w:
            pos["t_offset"] = round(sgn * (n * w + verge_m) - c, 2)
    return tsir


def pin_ego_speed(tsir: dict, t: np.ndarray, v: np.ndarray) -> dict:
    """Replace the ego's speed actions with RDP segments of the measured ViPE profile. The model transcribes a
    stop-and-go profile poorly (dToIIqqvSt0: one 'accelerate to 50 km/h' for a 16 km/h peak, 13 km/h MAE)."""
    from tsf.ingest.pai_av import speed_actions
    ego = next((a for a in tsir.get("actors", []) if a.get("role") == "ego"), None)
    if ego is None:
        return tsir
    tsir["actions"] = [a for a in tsir.get("actions", [])
                       if not (a.get("actor") == ego["id"] and a.get("type") == "speed_change")]
    ego["initial_speed"] = round(float(v[0]), 2)
    acts = speed_actions(ego["id"], t, v, 0.0, tol=0.5)
    # a constant profile (parked camera) still needs one ego maneuver for the OpenSCENARIO act
    tsir["actions"] += acts or [{"id": f"{ego['id']}_v0", "type": "speed_change", "actor": ego["id"],
                                 "start": {"kind": "sim_time", "t": 0.0}, "target_speed": round(float(v[0]), 2),
                                 "accel": 1.0}]
    return tsir


def pin_ego_assertions(tsir: dict, t: np.ndarray, v: np.ndarray, margin: float = 1.0) -> dict:
    """Make the ego's speed oracle agree with the measured profile (the model asserted 27-28.5 m/s for a parked
    camera): ``ego_speed_between`` becomes [min - margin, max + margin] m/s; ``ego_reaches_s`` and
    ``max_decel_below`` are only relaxed, never tightened; ``ego_stops_before_line`` is dropped when the measured
    ego never stopped."""
    ego = next((a for a in tsir.get("actors", []) if a.get("role") == "ego"), None)
    if ego is None or len(t) < 2:
        return tsir
    keep = np.asarray(t) <= float(tsir.get("duration") or t[-1]) + 1e-6
    tt, vv = (np.asarray(t)[keep], np.asarray(v)[keep]) if keep.sum() >= 2 else (np.asarray(t), np.asarray(v))
    dist = float(np.sum(0.5 * (vv[1:] + vv[:-1]) * np.diff(tt)))
    decel = float(max(0.0, -np.gradient(vv, tt).min()))
    out = []
    for a in tsir.get("assertions", []):
        if a.get("actor", "ego") in ("ego", ego["id"]):
            k = a.get("kind")
            if k == "ego_speed_between":
                a["value"] = round(max(0.0, float(vv.min()) - margin), 2)
                a["value2"] = round(float(vv.max()) + margin, 2)
            elif k == "ego_reaches_s" and a.get("value") is not None:
                a["value"] = round(min(float(a["value"]), 0.9 * dist), 1)
            elif k == "max_decel_below" and a.get("value") is not None:
                a["value"] = round(max(float(a["value"]), 1.25 * decel + 0.5), 2)
            elif k == "ego_stops_before_line" and vv.min() >= 0.3:
                continue
        out.append(a)
    tsir["assertions"] = out
    return tsir


def speed_fidelity(rep, ego: dict) -> dict:
    """Simulated vs ViPE ego speed (MAE, km/h) over the common horizon."""
    if not hasattr(rep, "_log"):
        return {}
    log = rep._log
    e = log.roles.index("ego")
    secs = np.arange(len(ego["speed_kmh_per_s"]), dtype=float)
    secs = secs[secs <= log.t[-1]]
    sim = np.interp(secs, log.t, log.states[:, e, 3]) * 3.6
    ref = np.asarray(ego["speed_kmh_per_s"])[: len(secs)]
    return {"speed_mae_kmh": round(float(np.abs(sim - ref).mean()), 2), "n": len(secs)}


# ----------------------------------------------------------------------------------------------- measured NPCs
CAM_TO_BUMPER = 2.0         # m, windshield camera behind the ego front bumper (measured gaps are camera -> rear face)
FAMILY = {"car": "car", "suv": "car", "van": "car", "truck": "heavy", "bus": "heavy", "motorcycle": "two",
          "bicycle": "two", "pedestrian": "ped", "person": "ped"}
LANE_EVENTS = ("cut_in", "leaves_ego_lane", "lane_change")
MEASURED_TO_TSIR = {"car": "car", "truck": "truck", "bus": "bus", "motorcycle": "motorcycle", "bicycle": "bicycle",
                    "person": "pedestrian"}


def lane_phrase(k: int) -> str:
    if k == 0:
        return "in the ego lane"
    n = abs(k)
    nth = "" if n == 1 else {2: "second ", 3: "third "}.get(n, f"{n}th ")
    return f"in the {nth}lane to the {'left' if k > 0 else 'right'}"


def select_npcs(npcs: list[dict], max_n: int = 8, min_conf: float = 0.3, scene: dict | None = None) -> list[dict]:
    """Measured road users worth putting in the request: confident, at or ahead of the ego, nearest first. With a
    ``scene`` the road decides what can exist: nothing oncoming on a road without oncoming lanes (the opposite
    carriageway is not simulated), nothing crossing away from a junction or signal, nothing beyond the road edge."""
    keep = [n for n in npcs if n["confidence"] >= min_conf and n["gap0_m"] > -4 and n["relation"] != "behind"]
    if scene is not None:
        junction = (any(w in scene.get("road_type", "") for w in ("junction", "intersection", "roundabout"))
                    or bool(scene.get("traffic_lights")) or "intersection" in scene.get("summary", "").lower())
        k, same, opp = (int(scene.get(f, d)) for f, d in (("ego_lane_from_left", 1), ("lanes_same_direction", 1),
                                                          ("lanes_oncoming", 0)))
        left, right = 3.5 * (max(k, 1) - 1 + opp) + 5.25, 3.5 * max(same - k, 0) + 5.25      # road edge + verge
        keep = [n for n in keep if not (n["relation"] == "oncoming" and not opp)
                and not (n["relation"].startswith("crossing") and not junction) and -right <= n["lat0_m"] <= left]
    return sorted(keep, key=lambda n: abs(n["gap0_m"]) / max(n["confidence"], 0.1))[:max_n]


def gap_start(n: dict) -> float:
    """Camera-to-rear-face gap at t = 0 of a measured user: its track extrapolated back at the first-seen speed with
    the ego's own travel accounted for (``gap_start_m``); one first seen late that would have started beside or
    behind the ego keeps its first-seen gap, at most 8 m."""
    g = float(n.get("gap_start_m", n["gap0_m"]))
    return g if g > 4.0 or n["t0"] <= 0.5 else min(float(n["gap0_m"]), 8.0)


def npc_line(n: dict) -> str:
    """One measured road user as a sentence for the request."""
    gap = max(0.5, gap_start(n) - CAM_TO_BUMPER)
    side = "left" if n["lat0_m"] > 0 else "right"
    where = {"oncoming": f"oncoming, {abs(n['lat0_m']):.1f} m to the {side} of the ego",
             "crossing_from_left": "crossing from the left", "crossing_from_right": "crossing from the right",
             "parked": f"stationary, {abs(n['lat0_m']):.1f} m to the {side} of the ego lane centre",
             }.get(n["relation"], lane_phrase(n["lane0"]))
    kmh = n["speed0_kmh"]
    ev, far = [], n["gap0_m"] > 50.0                      # lateral position is too noisy far away for lane events
    for e in n["events"]:
        if n["relation"] == "parked" or (far and e["type"] in LANE_EVENTS) or e.get("gap_m", 0.0) > 45.0:
            continue
        if e["type"] == "cut_in":
            ev.append(f"cuts into the ego lane at t={e['t']} s (gap {max(0.5, e['gap_m'] - CAM_TO_BUMPER):.0f} m)")
        elif e["type"] == "leaves_ego_lane":
            ev.append(f"leaves the ego lane at t={e['t']} s")
        elif e["type"] == "lane_change":
            ev.append(f"changes lane at t={e['t']} s")
        else:
            ev.append(f"{'stops' if e['type'] == 'stops' else 'brakes'} from {e['from_kmh']} to {e['to_kmh']} km/h "
                      f"at t={e['t']} s")
    first = "" if n["t0"] < 0.5 else (f" (from t=0, extrapolated; first visible at t={n['t0']} s, "
                                      f"{max(0.5, n['gap0_m'] - CAM_TO_BUMPER):.0f} m ahead then)")
    speed = "" if n["relation"] == "parked" else f", {'stationary' if kmh < 3 else f'{kmh:.0f} km/h'}"
    return (f"{n['category']} {gap:.0f} m {'ahead' if gap_start(n) >= 0 else 'alongside'} {where}{speed}{first}"
            + (f", {', '.join(ev)}" if ev else "") + f", visible until t={n['t1']} s")


def scene_actor_text(scene: dict, npcs: list[dict] | None) -> str:
    """Road users for the request. A forward camera cannot see behind the ego, so VLM actors 'behind' are dropped;
    with measured tracks the VLM only adds actors the measurement does not cover."""
    vlm = [a for a in scene["actors"] if a["relation"] != "behind"]
    if not npcs:
        return "; ".join(f"a {a['category']} {a['relation'].replace('_', ' ')} at about {a['approx_distance_m']:.0f} m "
                         f"({a['behavior']})" for a in vlm) or "no other relevant road users"
    sel = select_npcs(npcs, scene=scene)
    extra = [a for a in vlm if not any(FAMILY.get(a["category"]) == FAMILY.get(MEASURED_TO_TSIR.get(n["category"], ""))
                                       and abs(a["approx_distance_m"] - n["gap0_m"]) < 12 for n in sel)]
    parts = [f"{i + 1}) {npc_line(n)}" for i, n in enumerate(sel)] or ["none measured"]
    txt = ("measured from the video by depth and tracking (gaps at t = 0 are bumper-to-bumper in metres, speeds are "
           "ground speeds; reproduce them): " + "; ".join(parts))
    if extra:
        txt += ("; additionally seen by the scene analysis (approximate): " +
                "; ".join(f"a {a['category']} {a['relation'].replace('_', ' ')} at about {a['approx_distance_m']:.0f} m"
                          for a in extra[:3]))
    return txt


def drop_actor(tsir: dict, aid: str) -> dict:
    """Remove an actor with its actions, assertions and parameter ranges (JSON pointers are re-indexed)."""
    ia = next(i for i, a in enumerate(tsir["actors"]) if a["id"] == aid)

    def refs(x) -> bool:
        if isinstance(x, dict):
            return any(v == aid or refs(v) for v in x.values())
        return isinstance(x, list) and any(refs(v) for v in x)

    acts = tsir.get("actions", [])
    drop_act = {i for i, a in enumerate(acts) if refs(a)}
    while True:                                           # actions that wait for a dropped action go too
        gone = {acts[i]["id"] for i in drop_act}
        more = {i for i, a in enumerate(acts) if i not in drop_act and a.get("start", {}).get("action_id") in gone}
        if not more:
            break
        drop_act |= more
    tsir["actors"].pop(ia)
    tsir["actions"] = [a for i, a in enumerate(tsir.get("actions", [])) if i not in drop_act]
    tsir["assertions"] = [a for a in tsir.get("assertions", []) if a.get("actor") != aid]

    def shift(path: str):
        parts = path.split("/")
        for key, gone in (("actors", [ia]), ("actions", sorted(drop_act))):
            if len(parts) > 2 and parts[1] == key and parts[2].isdigit():
                j = int(parts[2])
                if j in gone:
                    return None
                parts[2] = str(j - sum(g < j for g in gone))
        return "/".join(parts)

    params = []
    for p in tsir.get("parameters", []):
        q = shift(p["path"])
        if q is not None:
            params.append({**p, "path": q})
    tsir["parameters"] = params
    return tsir


def _len(actor: dict) -> float:
    return float((actor.get("dimensions") or [DEFAULT_DIMS.get(actor.get("category", "car"), (4.6,))[0]])[0])


def ego_lane_of(tsir: dict) -> tuple[dict, dict] | None:
    ego = next((a for a in tsir.get("actors", []) if a.get("role") == "ego"), None)
    if ego is None or ego.get("initial", {}).get("direction", "forward") not in ("forward", "inbound"):
        return None
    return ego, ego["initial"]


def pin_npcs(tsir: dict, npcs: list[dict], ego_s_min: float = 30.0, max_cost: float = 3.0) -> dict:
    """Impose measured road users on a proposal, only where an actor matches a measured track robustly.

    * No NPC may start at, beside-overlapping or behind the ego in its lane (a forward camera cannot see behind;
      the model placed four followers at s = 0 next to the ego): such actors are dropped with their actions.
    * Linear roads: the ego starts at s >= ``ego_s_min`` so that there is road behind it (every actor of the road
      moves along with it, gaps unchanged). On a junction approach (direction ``inbound``, s counts down to the
      junction) only the first rule applies.
    * A forward-direction NPC whose category family, lane and longitudinal position agree with a measured track
      (Hungarian assignment, cost < ``max_cost``) takes the measured initial position, lane, speed and speed
      profile; a measured cut-in is added when the actor has no lane action. Unmatched actors stay as proposed.
    """
    from scipy.optimize import linear_sum_assignment

    from tsf.ingest.pai_av import speed_actions
    for a in tsir.get("actors", []):                    # the model leaves defaulted fields out
        a.setdefault("initial", {}).update({k: a["initial"].get(k, d) for k, d in
                                            (("road", 0), ("direction", "forward"), ("lane", 1), ("s", 0.0))})
    got = ego_lane_of(tsir)
    if got is None:
        return tsir
    ego, ei = got
    road = tsir.get("road", {})
    if road.get("type") == "linear" and ei["s"] < ego_s_min:
        delta, length = ego_s_min - ei["s"], sum(x["length"] for x in road.get("segments", [{"length": 1000.0}]))
        for a in tsir["actors"]:                          # move the whole scene along the road, gaps unchanged
            ini = a["initial"]
            if ini.get("road", 0) == ei.get("road", 0):
                ini["s"] = round(min(ini["s"] + delta, length - 1.0) if ini.get("direction", "forward") == "forward"
                                 else max(ini["s"] - delta, 1.0), 2)
    l_ego, sgn = _len(ego), -1.0 if ei["direction"] == "inbound" else 1.0
    fwd = [a for a in tsir["actors"] if a.get("role", "npc") == "npc" and a["initial"]["direction"] == ei["direction"]
           and a["initial"].get("road", 0) == ei.get("road", 0)]
    for a in list(fwd):
        l_a = _len(a)
        if a["initial"]["lane"] == ei["lane"] and sgn * (a["initial"]["s"] - ei["s"]) < (l_ego + l_a) / 2 + 1.0:
            drop_actor(tsir, a["id"])
            fwd.remove(a)
    if road.get("type") != "linear":
        return tsir
    cand = select_npcs(npcs)
    cand = [n for n in cand if n["relation"] not in ("oncoming", "crossing_from_left", "crossing_from_right")]
    cost = np.full((len(fwd), len(cand)), 1e3)
    for i, a in enumerate(fwd):
        for j, n in enumerate(cand):
            if FAMILY.get(a["category"]) != FAMILY.get(MEASURED_TO_TSIR.get(n["category"], "")):
                continue
            s_meas = ei["s"] + (gap_start(n) - CAM_TO_BUMPER) + (l_ego + _len(a)) / 2
            d_lane = abs(a["initial"]["lane"] - (ei["lane"] - n["lane0"]))
            cost[i, j] = abs(a["initial"]["s"] - s_meas) / 10.0 + 1.5 * d_lane
    if len(fwd) and len(cand):
        for i, j in zip(*linear_sum_assignment(cost)):
            if cost[i, j] >= max_cost:
                continue
            a, n = fwd[i], cand[j]
            lane = ei["lane"] - n["lane0"]
            if not 1 <= lane <= (road.get("forward") or {}).get("driving_lanes", 6):
                continue
            l_a = _len(a)
            v0 = max(0.0, n["speed0_kmh"] / 3.6)
            s0 = ei["s"] + (gap_start(n) - CAM_TO_BUMPER) + (l_ego + l_a) / 2
            a["initial"].update(lane=lane, s=round(float(s0), 2))
            off = n["lat0_m"] - n["lane0"] * 3.5
            if abs(off) > 0.5:
                a["initial"]["t_offset"] = round(float(np.clip(off, -2.0, 2.0)), 2)
            a["initial_speed"] = round(v0, 2)
            prof = n.get("profile")
            if prof and len(prof["t"]) > 2:
                tsir["actions"] = [x for x in tsir.get("actions", [])
                                   if not (x.get("actor") == a["id"] and x.get("type") == "speed_change")]
                tsir["actions"] += speed_actions(a["id"], np.asarray(prof["t"]), np.asarray(prof["v_kmh"]) / 3.6, 0.0,
                                                 tol=0.8)
            cut = next((e for e in n["events"] if e["type"] == "cut_in"), None)
            if cut and not any(x.get("actor") == a["id"] and x.get("type") in ("cut_in", "lane_change")
                               for x in tsir.get("actions", [])):
                tsir["actions"].append({"id": f"{a['id']}_cutin", "type": "cut_in", "actor": a["id"],
                                        "target": ego["id"],
                                        "start": {"kind": "sim_time", "t": round(max(0.0, cut["t"] - 1.5), 2)},
                                        "gap": round(float(np.clip(cut["gap_m"] - CAM_TO_BUMPER, 4, 40)), 1),
                                        "duration": 3.0})
    return tsir


def npc_fidelity(rep, npcs: list[dict], ego_lane_w: float = 3.5) -> dict:
    """Simulated vs measured road users at the time the measurement first sees them: initial bumper gap (m) and
    speed (km/h) of the matched pairs, lane agreement, how many measured users have a simulated counterpart, and
    how many simulated NPCs start at or behind the ego (they cannot be observed)."""
    if not hasattr(rep, "_log"):
        return {}
    log = rep._log
    e = log.roles.index("ego")
    cand = [n for n in select_npcs(npcs) if n["relation"] not in ("oncoming", "crossing_from_left",
                                                                   "crossing_from_right") and n["t0"] < log.t[-1]]

    def state(j: int, t: float):
        x, y, yaw, v = (float(np.interp(t, log.t, log.states[:, j, c])) for c in range(4))
        return x, y, yaw, v

    def rel(j: int, t: float):
        xe, ye, ye_yaw, _ = state(e, t)
        x, y, _, v = state(j, t)
        c, s = math.cos(ye_yaw), math.sin(ye_yaw)
        d = (x - xe, y - ye)
        return d[0] * c + d[1] * s, -d[0] * s + d[1] * c, v

    npc_idx = [j for j, r in enumerate(log.roles) if r != "ego"]
    behind = sum(1 for j in npc_idx if abs(rel(j, 0.0)[1]) < ego_lane_w / 2 and
                 rel(j, 0.0)[0] < (log.lwh[e, 0] + log.lwh[j, 0]) / 2 + 1.0)
    cost = np.full((len(cand), len(npc_idx)), 1e3)
    meas = []
    for i, n in enumerate(cand):
        t = max(n["t0"], 0.0)
        gap_m = n["gap0_m"] - CAM_TO_BUMPER
        meas.append((t, gap_m))
        for k, j in enumerate(npc_idx):
            if FAMILY.get(log.categories[j]) != FAMILY.get(MEASURED_TO_TSIR.get(n["category"], "")):
                continue
            lon, lat, _ = rel(j, t)
            gap = lon - (log.lwh[e, 0] + log.lwh[j, 0]) / 2
            cost[i, k] = abs(gap - gap_m) / 10.0 + 1.5 * abs(round(lat / ego_lane_w) - n["lane0"])
    out = {"npc_measured": len(cand), "npc_sim": len(npc_idx), "npc_start_behind_ego": behind}
    pairs = []
    if len(cand) and len(npc_idx):
        pairs = [(i, k) for i, k in zip(*linear_sum_assignment(cost)) if cost[i, k] < 3.0]
    out["npc_matched"] = len(pairs)
    if pairs:
        gaps, spds, lanes = [], [], []
        for i, k in pairs:
            t, gap_m = meas[i]
            lon, lat, v = rel(npc_idx[k], t)
            gaps.append(abs(lon - (log.lwh[e, 0] + log.lwh[npc_idx[k], 0]) / 2 - gap_m))
            spds.append(abs(v * 3.6 - cand[i]["speed0_kmh"]))
            lanes.append(round(lat / ego_lane_w) == cand[i]["lane0"])
        out |= {"npc_gap_mae_m": round(float(np.mean(gaps)), 2), "npc_speed_mae_kmh": round(float(np.mean(spds)), 2),
                "npc_lane_agree": round(float(np.mean(lanes)), 2)}
    return out


def lift_video(video: str, workdir: Path, llm: LLM | None = None, max_seconds: float = 20.0, on_stage=None,
               npc: bool = True, ego_speed: str = "auto", vipe_dir: str | Path | None = None) -> dict:
    """``npc``: also measure the other road users (dense depth + instance tracks, ~25 % more ViPE time); a failure
    there is reported in ``timing['npc_error']`` and the request falls back to the scene analysis alone.
    ``ego_speed``: ``'auto'`` takes the ego speed from Cosmos3-Edge inverse dynamics and the heading from ViPE
    (``tsf.ingest.ego_speed``; ViPE's translation collapses at speed) and falls back to ViPE alone with the reason in
    ``timing['ego_error']``; ``'vipe'`` never asks Cosmos; ``'idm'`` fails instead of falling back.
    ``vipe_dir``: reuse the ViPE run stored there (same video and span) instead of running ViPE."""
    from tsf.ingest import ego_speed as ego_mod
    from tsf.ingest import npc as npc_mod
    from tsf.ingest import vipe
    stage = on_stage or (lambda *a: None)
    llm = llm or LLM()
    wd = Path(workdir)
    wd.mkdir(parents=True, exist_ok=True)
    path = resolve(video)
    stage("vipe", "running", path.name)
    t0 = time.perf_counter()
    vr = (vipe.load_run(Path(vipe_dir).expanduser()) if vipe_dir
          else vipe.run_vipe(path, wd / "vipe", max_seconds=max_seconds, fps=15, dense=npc))
    track = vipe.despike(vipe.ego_track(vr["poses"], vr["t"]), vr["t"])
    ego_vipe = ego_summary(track, np.asarray(vr["t"]))
    t_vipe = time.perf_counter() - t0
    ego_src, ego_err, t_idm = "vipe", None, 0.0
    if ego_speed != "vipe":
        t3 = time.perf_counter()
        try:
            sp = ego_mod.idm_speed(path, float(vr["t"][0]), float(vr["t"][-1] - vr["t"][0]), wd / "idm")
            track = ego_mod.fuse_ego(track, np.asarray(vr["t"]), sp["t"], sp["v"])
            ego_src = "cosmos_speed+vipe_yaw"
        except Exception as e:  # no server / no cached window: the ViPE track still lifts a scenario
            if ego_speed == "idm":
                raise
            ego_err = f"{type(e).__name__}: {str(e)[:200]}"
        t_idm = time.perf_counter() - t3
    ego = ego_summary(track, np.asarray(vr["t"]))
    npcs, t_npc, npc_err, drops = [], 0.0, None, {}
    if npc:
        t2 = time.perf_counter()
        try:
            tracks = npc_mod.npc_tracks(vr, track, vr["t"], drops=drops)
            npcs = npc_mod.npc_summary(tracks)
            npc_mod.save_tracks(tracks, wd / "npc_tracks.json")
        except Exception as e:  # noqa: BLE001 - the ego motion alone still lifts a scenario
            npc_err = f"{type(e).__name__}: {e}"
        t_npc = time.perf_counter() - t2
    stage("vipe", "ok", f"{vr['frames']} frames in {vr['runtime_s']:.0f} s; {ego['distance_m']} m, "
                        f"{len(ego['turns'])} turn(s), {len(ego['lane_changes'])} lane change(s); "
                        + (f"{len(npcs)} road user(s) tracked" if npc_err is None
                           else f"no NPC tracks ({npc_err[:80]})"))
    stage("scene", "running", "")
    t1 = time.perf_counter()
    scene = describe_scene(keyframes(path, 6, max_seconds), ego, llm)
    stage("scene", "ok", scene["summary"][:200])
    t_scene = time.perf_counter() - t1
    query = request_text(scene, ego, npcs)
    (wd / "uc2_request.json").write_text(
        json.dumps({"ego": ego, "ego_source": ego_src, "ego_vipe": ego_vipe, "scene": scene, "npcs": npcs,
                    "query": query}, indent=1))
    tv, vv = np.asarray(vr["t"]) - vr["t"][0], gaussian_filter1d(track[:, 3], 3)

    def pin(d: dict) -> dict:
        d = pin_ego_assertions(pin_ego_speed(infer_action_types(move_signs_off_the_road(d)), tv, vv), tv, vv)
        return pin_npcs(d, npcs)        # also without measurements: nobody starts behind the ego

    res: LiftResult = Lifter(llm).lift(query, wd / "lift", sid=_slug("video " + path.stem, 40), on_stage=on_stage,
                                       pin=pin)
    fid = speed_fidelity(res.final.report, ego)
    if npcs:
        fid |= npc_fidelity(res.final.report, npcs)
    stats = [w["stats"] for w in vr["windows"]]
    return {"result": res, "ego": ego, "scene": scene, "query": query, "fidelity": fid, "npc": npcs,
            "timing": {"vipe_s": round(t_vipe, 1), "scene_s": round(t_scene, 1), "vipe_runtime_s": vr["runtime_s"],
                       "npc_s": round(t_npc, 1), "dense_s": round(sum(s.get("dense_s", 0.0) for s in stats), 1),
                       "vipe_gpu_peak_gb": round(max(s["gpu_peak_reserved_gb"] for s in stats), 1),
                       "vipe_host_peak_gb": round(max(s["rss_peak_gb"] for s in stats), 1),
                       "npc_tracks": len(npcs), "ego_source": ego_src, "idm_s": round(t_idm, 1),
                       **({"npc_error": npc_err} if npc_err else {}), **({"ego_error": ego_err} if ego_err else {})},
            "npc_drops": drops}
