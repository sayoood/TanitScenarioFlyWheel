"""Sensor rendering and judging of augmented variants (camera + LiDAR).

render   : ``tsf.cosmos.pipeline.render_sensors`` (TanitSim -> WSM controls + LiDAR control -> Cosmos3 Sim-Auto joint
           camera+LiDAR) with ``appearance`` = the variant environment; the same seed for the parent and its variants
           so that differences come from the variation. Waits for the single Cosmos slot and for free memory.
judge    : (a) camera: Qwen3.8 as VLM on 3 generated frames with schema-constrained questions (weather, day/night,
           wet road, are the changed objects visible), agreement with the variant's environment; a failed judgement
           triggers ONE re-render with another seed. (b) LiDAR: the numbers of ``sensors.json`` (checks).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from tsf.agents.augment_plan import time_band
from tsf.agents.llm import LLM

MIN_FREE_GB = 35.0
FRAMES = 97


def render_available() -> bool:
    try:
        from tsf.cosmos import pipeline
        return hasattr(pipeline, "render_sensors")
    except Exception:   # noqa: BLE001
        return False


MIN_FREE_GB = 10.0     # headroom to run the export / point-cloud stages next to a Sim-Auto server that is already up


def free_gb() -> float:
    """MemAvailable of this host in GB (Thor: unified memory, shared by the LLM, Cosmos and the page cache)."""
    try:
        for ln in Path("/proc/meminfo").read_text().splitlines():
            if ln.startswith("MemAvailable:"):
                return int(ln.split()[1]) / 1024 ** 2
    except OSError:
        pass
    return 1e9        # not a Linux host: do not block


SIGN_NAME = {"speed_limit": "speed-limit sign", "stop": "stop sign", "yield": "yield sign", "no_entry": "no-entry sign",
             "pedestrian_crossing": "pedestrian-crossing sign", "road_works": "road-works sign"}
OBJECT_NAME = {"cone": ("traffic cone", "traffic cones"), "barrier": ("barrier", "barriers"),
               "parked_car": ("parked car", "parked cars"), "parked_truck": ("parked truck", "parked trucks"),
               "pole": ("pole", "poles"), "bollard": ("bollard", "bollards"),
               "construction_sign": ("construction warning board", "construction warning boards"),
               "traffic_sign": ("traffic sign", "traffic signs")}


def _object_name(o: dict) -> str:
    """Natural name of one static object: 'speed-limit sign 30', 'road-works sign', 'traffic cone', 'parked truck'."""
    if o.get("kind") == "traffic_sign" and o.get("sign_type"):
        v = o.get("sign_value")
        return SIGN_NAME.get(o["sign_type"], o["sign_type"].replace("_", " ") + " sign") + (f" {v:g}" if v is not None else "")
    return OBJECT_NAME.get(o["kind"], (o["kind"].replace("_", " "),) * 2)[0]


def object_names(objs: list[dict]) -> list[str]:
    """Distinct static objects with counts for the caption text: ['7 traffic cones', 'speed-limit sign 30', ...]."""
    counts: dict[str, int] = {}
    for o in objs:
        counts[_object_name(o)] = counts.get(_object_name(o), 0) + 1
    plural = {s: p for s, p in OBJECT_NAME.values()}
    return [f"{n} {plural.get(name, name + 's')}" if n > 1 else name for name, n in counts.items()]


def appearance_of(doc: dict) -> dict:
    """Variant environment as the appearance spec of the sensor renderer (+ a caption-ready text)."""
    e = doc["environment"]
    band = time_band(e["time_of_day"])
    bits = [{"day": "daytime", "dusk": "dusk or dawn twilight", "night": "night"}[band] + f" ({e['time_of_day']})",
            {"clear": "clear sky", "cloudy": "overcast sky", "rain": "light rain", "heavy_rain": "heavy rain",
             "fog": f"fog with {e['fog_visibility_m']:.0f} m visibility", "snow": "snowfall"}[e["weather"]],
            f"{e['road_surface']} road surface"]
    objs = [{k: o.get(k) for k in ("kind", "sign_type", "sign_value") if o.get(k) is not None}
            for o in doc.get("static_objects") or []]
    if e.get("setting"):
        bits.append(f"{e['setting']} surroundings")
    if objs:
        bits.append("static objects: " + ", ".join(object_names(doc.get("static_objects") or [])))
    return {"time_of_day": e["time_of_day"], "time_band": band, "weather": e["weather"],
            "precipitation": e["precipitation"], "fog_visibility_m": e["fog_visibility_m"],
            "cloud_cover": e["cloud_cover"], "road_surface": e["road_surface"], "setting": e.get("setting"),
            "static_objects": objs, "text": ", ".join(bits)}


def event_time(log) -> float:
    """Time of the closest approach between the ego and any other actor (centre distance) in a TanitSim log."""
    e = log.roles.index("ego")
    best_t, best = 0.0, 1e9
    for j in range(len(log.ids)):
        if j == e:
            continue
        d = np.hypot(log.states[:, j, 0] - log.states[:, e, 0], log.states[:, j, 1] - log.states[:, e, 1])
        k = int(np.argmin(d))
        if d[k] < best:
            best, best_t = float(d[k]), float(log.t[k])
    return best_t


def window_t0(log, duration: float, frames: int = FRAMES, fps: float = 30.0) -> float:
    """Start of the render window: 1.5 s before the closest approach, clipped into the scenario."""
    win = frames / fps
    return round(float(np.clip(event_time(log) - 1.5, 0.0, max(0.0, duration - win - 0.1))), 2)


# ----------------------------------------------------------------------------- render
def _running_cosmos() -> list[str]:
    try:
        from tsf.cosmos import generate
        return list(generate.running_servers())
    except Exception:   # noqa: BLE001 - docker not reachable: let ensure_server decide
        return []


def render(doc: dict, out_dir: Path, *, seed: int, t0: float | None, scenario: dict | None = None,
           geometry_dir: Path | None = None, cameras=None, frames: int = FRAMES, lidar: bool = True,
           wait_s: float = 7200.0, log=print, state: dict | None = None) -> dict:
    """Wait for memory and the Cosmos slot, then call ``render_sensors``.

    ``doc`` is the variant (its environment becomes the ``appearance``); ``scenario`` is what TanitSim replays: the
    parent TSIR for appearance-only variants (D1), so the scenario-only stages are reused through ``geometry_dir``."""
    from tsf.cosmos import pipeline
    out_dir.mkdir(parents=True, exist_ok=True)
    busy = getattr(pipeline, "CosmosBusyError", ())
    t_wait = time.perf_counter()
    while True:
        try:
            if not (state or {}).get("started_server"):          # a server of ours (earlier call of this job) is reused
                other = _running_cosmos()
                if other:                                      # never take over or restart someone else's server
                    raise (busy or RuntimeError)(f"a Cosmos server is running: {other}")
            info = pipeline.ensure_server(views=len(cameras) if cameras else 3, lidar=lidar)
            if state is not None and info.get("started"):
                state["started_server"] = True
            break
        except Exception as e:   # noqa: BLE001 - the Cosmos slot is shared: wait and retry while another job holds it
            msg = str(e).lower()
            retry = isinstance(e, busy) or any(w in msg for w in ("busy", "refus", "another", "occupied", "in use"))
            if retry and time.perf_counter() - t_wait < wait_s:
                log(f"render: Cosmos slot busy ({str(e)[:110]}), retry in 60 s")
                time.sleep(60)
                continue
            raise
    while free_gb() < MIN_FREE_GB and time.perf_counter() - t_wait < wait_s:     # unified memory is shared with the LLM
        log(f"render: only {free_gb():.0f} GB available (need {MIN_FREE_GB:.0f}), waiting")
        time.sleep(30)
    waited = time.perf_counter() - t_wait
    t = time.perf_counter()
    env = {k: v for k, v in doc["environment"].items() if v is not None}
    kw = dict(frames=frames, t0=t0, appearance=env, lidar=lidar, seed=seed)
    if cameras is not None:
        kw["cameras"] = tuple(cameras)
    if geometry_dir is not None:
        kw["geometry_dir"] = str(geometry_dir)
    res = pipeline.render_sensors(scenario or doc, out_dir, **kw)
    return {"result": res if isinstance(res, dict) else {}, "seed": seed, "t0": t0,
            "wait_s": round(waited, 1), "render_s": round(time.perf_counter() - t, 1)}


def find_outputs(out_dir: Path, result: dict) -> dict:
    """Locate mosaic / lidar BEV / sensors.json / front-camera video from the dict ``render_sensors`` returned."""
    def ok(v):
        return Path(str(v)) if isinstance(v, (str, Path)) and Path(str(v)).exists() else None
    lid = result.get("lidar") or {}
    mosaic = ok(result.get("mosaic")) or next(iter(sorted(out_dir.rglob("mosaic.mp4"))), None)
    bev = ok(lid.get("bev_video")) or next(iter(sorted(out_dir.rglob("lidar_bev.mp4"))), None)
    sj = ok(out_dir / "sensors.json")
    vids = result.get("videos") or {}
    front = next((ok(v) for k, v in vids.items() if "front" in k and ok(v)), None) or         next((ok(v) for v in vids.values() if ok(v)), None)
    return {"mosaic": mosaic, "lidar_bev": bev, "sensors_json": sj, "front_video": front or mosaic,
            "videos": {k: str(ok(v)) for k, v in vids.items() if ok(v)}}


# ----------------------------------------------------------------------------- judge
def camera_schema(objects: list[str]) -> dict:
    props = {
        "weather": {"enum": ["clear", "cloudy", "rain", "heavy_rain", "fog", "snow"]},
        "time_of_day": {"enum": ["day", "dusk_dawn", "night"]},
        "road_wet_or_snowy": {"type": "boolean"},
        "headlights_or_street_lights_on": {"type": "boolean"},
        "objects_visible": {"type": "object", "additionalProperties": False, "required": [f"o{i}" for i in range(len(objects))],
                            "properties": {f"o{i}": {"type": "boolean"} for i in range(len(objects))}},
        "note": {"type": "string", "maxLength": 240}}
    return {"type": "object", "additionalProperties": False, "required": list(props), "properties": props}


def judge_camera(video: Path | None, doc: dict, objects: list[str], llm: LLM, n: int = 3) -> dict:
    """VLM judgement of generated frames against the variant's environment (and the changed objects)."""
    if video is None or not Path(video).exists():
        return {"ok": None, "skipped": "no camera video"}
    from tsf.agents.video import keyframes
    frames = keyframes(Path(video), n, width=896)
    q = ("These are frames of a generated automotive camera video. Answer from what is visible only. "
         "weather: the weather you see (rain = rain streaks / wet lens, heavy_rain = very heavy rain, fog = reduced "
         "visibility, snow = falling or lying snow, cloudy = overcast, clear = bright and dry). time_of_day: day, "
         "dusk_dawn or night. ")
    if objects:
        q += "objects_visible: for each key say whether the object is clearly visible in at least one frame: " + \
             "; ".join(f"o{i} = {o}" for i, o in enumerate(objects)) + ". "
    content = [{"type": "text", "text": q}]
    for ts, b64 in frames:
        content += [{"type": "text", "text": f"frame at t = {ts} s"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]
    msgs = [{"role": "system", "content": "You are a strict visual quality inspector for synthetic driving videos."},
            {"role": "user", "content": content}]
    a = llm.chat(msgs, schema=camera_schema(objects), role="camera_judge", max_tokens=600, temperature=0.1)
    return {"answer": a, **agreement(a, doc, objects)}


_FAMILY = {"clear": "dry", "cloudy": "dry", "rain": "rain", "heavy_rain": "rain", "fog": "fog", "snow": "snow"}


def agreement(a: dict, doc: dict, objects: list[str]) -> dict:
    e = doc["environment"]
    w_ok = _FAMILY[a["weather"]] == _FAMILY[e["weather"]] or (e["weather"] == "fog" and a["weather"] in ("cloudy", "rain"))
    band = time_band(e["time_of_day"])
    got = {"day": "day", "dusk_dawn": "dusk", "night": "night"}[a["time_of_day"]]
    order = ["day", "dusk", "night"]
    dist = abs(order.index(band) - order.index(got))
    band_score = {0: 1.0, 1: 0.5, 2: 0.0}[dist]
    vis = a.get("objects_visible", {})
    obj_ratio = (sum(bool(vis.get(f"o{i}")) for i in range(len(objects))) / len(objects)) if objects else None
    ok = bool(w_ok and band_score >= 0.5 and (obj_ratio is None or obj_ratio >= 0.5))
    return {"ok": ok, "weather_agrees": bool(w_ok), "time_band_score": band_score,
            "objects_visible_ratio": None if obj_ratio is None else round(obj_ratio, 2),
            "expected": {"weather": e["weather"], "time_band": band, "objects": objects}}


def judge_lidar(result: dict | None, sensors_json: Path | None = None) -> dict:
    """LiDAR numbers = the checks ``render_sensors`` recorded (``result['lidar']['checks_summary']``): points per
    sweep, range error against the control, ground plane, object recall; ok = no warnings and no post-processing error."""
    lid = (result or {}).get("lidar")
    if lid is None and sensors_json is not None and Path(sensors_json).exists():
        try:
            lid = json.loads(Path(sensors_json).read_text()).get("lidar")
        except (OSError, json.JSONDecodeError):
            lid = None
    if not lid or not lid.get("enabled", True):
        return {"ok": None, "skipped": "no LiDAR in this render"}
    if lid.get("error"):
        return {"ok": False, "error": lid["error"]}
    cs = lid.get("checks_summary") or {}
    s, o = cs.get("summary") or {}, cs.get("objects") or {}
    keep = ("sweeps", "mean_points_per_sweep", "min_points_per_sweep", "mean_valid_fraction", "median_abs_range_err_m",
            "p90_abs_range_err_m", "control_agree_within_1m", "ground_present_sweeps", "ground_height_m_median")
    nums = {k: s.get(k) for k in keep if k in s}
    nums.update({k: o.get(k) for k in ("pairs", "recall_ge10", "recall_ge10_given_control", "median_points_generated")
                 if k in o})
    weather = (lid.get("weather") or {})
    return {"ok": not cs.get("warnings"), "warnings": cs.get("warnings", []), "numbers": nums,
            "weather_physics_applied": weather.get("physics_applied"), "checks_file": lid.get("checks")}


def changed_objects(doc: dict, parent: dict) -> list[str]:
    """Static objects added by the variant, as short names for the VLM question."""
    old = {o["id"] for o in parent.get("static_objects") or []}
    out = []
    for o in doc.get("static_objects") or []:
        if o["id"] in old:
            continue
        name = _object_name(o)
        if name not in out:
            out.append(name)
    return out[:6]
