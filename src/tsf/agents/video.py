"""UC2: single dash-cam video (no ego data) -> ViPE ego motion + VLM scene description -> verified TSIR.

ego    : ViPE metric camera poses -> planar ego track -> speed profile, turns (integrated yaw) and lane
         changes (lateral offset from the low-passed path)
scene  : Qwen3.8 (vision) reads keyframes and writes a structured description (road, lanes, signals, road
         users relative to the ego, timed events, appearance)
lift   : both become one request for the UC1 Lifter; the ego profile is a hard constraint, checked afterwards
         by comparing the simulated ego speed with the ViPE speed (MAE) and the manoeuvres with the TSIR
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

from tsf.agents.lifter import Lifter, LiftResult, _slug
from tsf.agents.llm import LLM

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


def request_text(scene: dict, ego: dict) -> str:
    actors = "; ".join(f"a {a['category']} {a['relation'].replace('_', ' ')} at about {a['approx_distance_m']:.0f} m "
                       f"({a['behavior']})" for a in scene["actors"]) or "no other relevant road users"
    man = [f"turns {x['direction']} between t={x['t_start']} s and t={x['t_end']} s" for x in ego["turns"]]
    man += [f"changes lane to the {x['direction']} at t={x['t']} s" for x in ego["lane_changes"]]
    return (f"Re-create this real dash-cam scenario. {scene['summary']} Road: {scene['road_type'].replace('_', ' ')}, "
            f"{scene['lanes_same_direction']} lane(s) in the ego direction, {scene['lanes_oncoming']} oncoming, ego in "
            f"lane {scene['ego_lane_from_left']} from the left, {'with' if scene['traffic_lights'] else 'without'} traffic "
            f"lights, speed limit about {scene['speed_limit_kph_guess']:.0f} km/h; {scene['time_of_day']}, "
            f"{scene['weather']}, {scene['road_surface']} road. Ego (measured, must be reproduced): speed profile in "
            f"km/h at 1 s steps {ego['speed_kmh_per_s']}; the ego {', '.join(man) or 'keeps its lane'}. Other road "
            f"users: {actors}. Events: {'; '.join(scene['events']) or 'none'}. Duration {ego['duration_s']:.0f} s.")


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
    return {"speed_mae_kmh": round(float(np.abs(sim - ref).mean()), 2), "n": int(len(secs))}


def lift_video(video: str, workdir: Path, llm: LLM | None = None, max_seconds: float = 20.0, on_stage=None) -> dict:
    from tsf.ingest import vipe
    stage = on_stage or (lambda *a: None)
    llm = llm or LLM()
    wd = Path(workdir)
    wd.mkdir(parents=True, exist_ok=True)
    path = resolve(video)
    stage("vipe", "running", path.name)
    t0 = time.perf_counter()
    vr = vipe.run_vipe(path, wd / "vipe", max_seconds=max_seconds, fps=15)
    track = vipe.despike(vipe.ego_track(vr["poses"], vr["t"]), vr["t"])
    ego = ego_summary(track, np.asarray(vr["t"]))
    t_vipe = time.perf_counter() - t0
    stage("vipe", "ok", f"{vr['frames']} frames in {vr['runtime_s']:.0f} s; {ego['distance_m']} m, "
                        f"{len(ego['turns'])} turn(s), {len(ego['lane_changes'])} lane change(s)")
    stage("scene", "running", "")
    t1 = time.perf_counter()
    scene = describe_scene(keyframes(path, 6, max_seconds), ego, llm)
    stage("scene", "ok", scene["summary"][:200])
    t_scene = time.perf_counter() - t1
    query = request_text(scene, ego)
    (wd / "uc2_request.json").write_text(json.dumps({"ego": ego, "scene": scene, "query": query}, indent=1))
    res: LiftResult = Lifter(llm).lift(query, wd / "lift", sid=_slug("video " + path.stem, 40), on_stage=on_stage)
    fid = speed_fidelity(res.final.report, ego)
    return {"result": res, "ego": ego, "scene": scene, "query": query, "fidelity": fid,
            "timing": {"vipe_s": round(t_vipe, 1), "scene_s": round(t_scene, 1), "vipe_runtime_s": vr["runtime_s"]}}
