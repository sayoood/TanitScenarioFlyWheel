"""Planning helpers of the scenario augmenter (code, no LLM):

* D1 weather & time-of-day draw with a seeded RNG, consistent precipitation / fog / road surface / cloud cover
* environment consistency check (a verifier-style finding fed back to the editor)
* scenario-specific candidate levers for D2 (behaviour) and D3 (environment) that the editor chooses from
* deterministic intent check: terms a variant description must contain for every changed attribute
"""
from __future__ import annotations

import random
import re

from tsf.agents.augment_patch import Change, diff

WEATHERS = ["clear", "cloudy", "rain", "heavy_rain", "fog", "snow"]
BANDS = ["day", "dusk", "night"]
STALE_TAGS = set(WEATHERS) | {"day", "night", "dusk", "dawn", "twilight", "daylight", "rainy", "foggy", "snowy"}

# ranges the draw picks from, per weather: (precipitation, fog_visibility_m, cloud_cover, road surfaces)
PRESET = {
    "clear": ((0.0, 0.0), (100000, 100000), (0.0, 0.25), ["dry"]),
    "cloudy": ((0.0, 0.0), (30000, 100000), (0.6, 0.95), ["dry"]),
    "rain": ((0.3, 0.5), (4000, 9000), (0.8, 1.0), ["wet"]),
    "heavy_rain": ((0.7, 0.95), (800, 2000), (0.9, 1.0), ["wet"]),
    "fog": ((0.0, 0.05), (60, 250), (0.6, 1.0), ["dry", "wet"]),
    "snow": ((0.35, 0.7), (600, 3000), (0.85, 1.0), ["snow"]),
}
# validity envelope per weather (looser than the draw): precipitation, fog visibility, cloud cover, surfaces
ENVELOPE = {
    "clear": ((0.0, 0.05), (10000, 1e9), (0.0, 0.5), {"dry", "ice"}),
    "cloudy": ((0.0, 0.05), (5000, 1e9), (0.4, 1.0), {"dry", "wet", "ice"}),
    "rain": ((0.15, 0.65), (1000, 1e9), (0.5, 1.0), {"wet"}),
    "heavy_rain": ((0.55, 1.0), (0.0, 5000), (0.7, 1.0), {"wet"}),
    "fog": ((0.0, 0.3), (1, 1000), (0.3, 1.0), {"dry", "wet", "ice"}),
    "snow": ((0.15, 1.0), (1, 10000), (0.5, 1.0), {"snow", "ice"}),
}
# time bands in minutes since midnight: day 08:00-17:30, dusk/dawn 06:00-08:00 and 17:30-19:30, night otherwise
_DRAW = {"day": [(9 * 60, 16 * 60 + 30)], "dusk": [(6 * 60 + 15, 7 * 60 + 30), (17 * 60 + 45, 19 * 60 + 15)],
         "night": [(20 * 60 + 30, 23 * 60 + 45), (15, 4 * 60 + 30)]}


def minutes(hhmm: str) -> int:
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


def hhmm(mins: int) -> str:
    return f"{mins // 60:02d}:{mins % 60:02d}"


def time_band(t: str) -> str:
    m = minutes(t)
    if 8 * 60 <= m < 17 * 60 + 30:
        return "day"
    if 6 * 60 <= m < 8 * 60 or 17 * 60 + 30 <= m < 19 * 60 + 30:
        return "dusk"
    return "night"


def draw_weather(parent_env: dict, seed: int) -> dict:
    """D1 target: a seeded random (weather, time band) different from the parent's, with consistent companions."""
    rng = random.Random(seed)
    w0, b0 = parent_env.get("weather", "clear"), time_band(parent_env.get("time_of_day", "12:00"))
    pool = [(w, b) for w in WEATHERS for b in BANDS if (w, b) != (w0, b0)]
    w, b = rng.choice(pool)
    lo, hi = rng.choice(_DRAW[b])
    t = round(rng.randint(lo, hi) / 5) * 5 % (24 * 60)
    prec, fog, cloud, surf = PRESET[w]
    env = {"weather": w, "time_of_day": hhmm(t),
           "precipitation": round(rng.uniform(*prec), 2),
           "fog_visibility_m": float(round(rng.uniform(*fog) / 10) * 10) if fog[0] != fog[1] else float(fog[0]),
           "cloud_cover": round(rng.uniform(*cloud), 2), "road_surface": rng.choice(surf)}
    return {"environment": env, "weather": w, "band": b, "from": {"weather": w0, "band": b0}}


def env_consistency(env: dict) -> list[str]:
    """Findings (strings) when weather, precipitation, visibility, cloud cover and road surface contradict each other."""
    w = env.get("weather", "clear")
    (p0, p1), (f0, f1), (c0, c1), surf = ENVELOPE[w]
    out = []
    p, f, c, s = (env.get("precipitation", 0.0), env.get("fog_visibility_m", 1e5), env.get("cloud_cover", 0.1),
                  env.get("road_surface", "dry"))
    if not p0 <= p <= p1:
        out.append(f"weather '{w}' needs precipitation in [{p0}, {p1}], got {p}")
    if not f0 <= f <= f1:
        out.append(f"weather '{w}' needs fog_visibility_m in [{f0:g}, {f1:g}], got {f:g}")
    if not c0 <= c <= c1:
        out.append(f"weather '{w}' needs cloud_cover in [{c0}, {c1}], got {c}")
    if s not in surf:
        out.append(f"weather '{w}' needs road_surface in {sorted(surf)}, got '{s}'")
    return out


# ----------------------------------------------------------------------------- scenario helpers
def kmh(v: float) -> int:
    return int(round(v * 3.6))


def _ego(d: dict) -> dict:
    return next(a for a in d["actors"] if a["role"] == "ego")


def _acts(d: dict, aid: str) -> list[dict]:
    return [a for a in d.get("actions", []) if a["actor"] == aid]


def _where(a: dict, d: dict) -> str:
    p = a["initial"]
    if d["road"]["type"] == "junction":
        return f"arm {p['road']} {p['direction']} lane {p['lane']}, {p['s']:.0f} m from the stop line"
    return f"{p['direction']} lane {p['lane']}, s={p['s']:.0f} m"


def behavior_levers(d: dict) -> list[tuple[str, str]]:
    """Scenario-specific candidate behaviour changes (id, text). Written from the parent's own actors and actions."""
    ego, L = _ego(d), []
    junction = d["road"]["type"] == "junction"
    v = ego["initial_speed"]
    L.append(("ego_faster", f"The ego approaches faster: raise ego initial_speed from {kmh(v)} km/h to about "
              f"{kmh(v * 1.3)} km/h and the targets of its speed_change actions by the same factor."))
    L.append(("ego_slower_late_brake", f"The ego is slower and hesitant: initial_speed {kmh(v)} -> {kmh(v * 0.7)} km/h "
              "and a later, harder speed reduction before the event."))
    npcs = [a for a in d["actors"] if a["role"] != "ego"]
    for a in npcs[:6]:
        kinds = {x["type"] for x in _acts(d, a["id"])}
        who = f"{a['id']} ({a['category']}, {_where(a, d)}, {kmh(a['initial_speed'])} km/h)"
        if "stop_at_line" in kinds and not d["road"].get("signalized"):
            L.append((f"skips_stop_{a['id']}", f"{who} does not stop at the stop line: delete its stop_at_line action "
                      "and keep its speed, so it enters the junction with priority conflicts."))
        if "cut_in" in kinds:
            L.append((f"cut_in_tighter_{a['id']}", f"{who} cuts in tighter and earlier: smaller cut_in gap, shorter "
                      "duration, earlier trigger."))
        if "follow" in kinds:
            L.append((f"tailgate_{a['id']}", f"{who} tailgates: smaller follow time_gap and a higher speed than the "
                      "target."))
        if not ("stop_at_line" in kinds and not d["road"].get("signalized")) and "cut_in" not in kinds:
            L.append((f"faster_{a['id']}", f"{who} drives about 30 % faster and brakes later (speed_change with a "
                      "higher target_speed or a later trigger)."))
        L.append((f"brake_{a['id']}", f"{who} brakes hard unexpectedly (a brake action, decel 5-7 m/s2) shortly "
                  "before it reaches the ego's path."))
    if junction:
        L.append(("add_pedestrian", "Add a pedestrian NPC (category pedestrian) standing at the kerb near the ego's exit "
                  "arm that crosses in front of the ego (a cross action, triggered by the ego's distance)."))
    else:
        ln = d["road"]["forward"]["driving_lanes"]
        L.append(("add_lead_brake", f"Add a lead NPC {max(25, int(v * 1.5))} m ahead of the ego in the ego's lane that "
                  "brakes hard (a brake action) after 3 s."))
        if ln >= 2:
            L.append(("add_cut_in", "Add an NPC in a neighbouring lane, slightly ahead and faster, that cuts into the "
                      "ego's lane with a 10 m gap."))
    return L


def environment_levers(d: dict, static_objects: bool) -> list[tuple[str, str]]:
    """Scenario-specific candidate environment changes (id, text) for the road, signs, static objects and setting."""
    road, ego, L = d["road"], _ego(d), []
    if road["type"] == "junction":
        arms = road["approaches"]
        ea = ego["initial"]["road"]
        n_in = arms[ea]["inbound"]["driving_lanes"]
        if n_in < 5:
            L.append(("more_lanes", f"Add an inbound lane on the ego's arm {ea}: road.approaches/{ea}/inbound/"
                      f"driving_lanes {n_in} -> {n_in + 1} (keep every actor on a valid lane)."))
        if n_in > 2:
            L.append(("fewer_lanes", f"Remove an inbound lane on the ego's arm {ea} ({n_in} -> {n_in - 1}); move actors "
                      "that sat in the removed lane."))
        L.append(("speed_limit", f"Change the speed limit of the ego's arm {ea} from {arms[ea]['speed_limit_kph']:.0f} "
                  "km/h to 30 km/h (urban calming) or 70 km/h."))
        L.append(("corner_radius", f"Tighten the junction corners: corner_radius {road['corner_radius']:.0f} m -> 6 m "
                  "(or widen to 25 m)."))
        L.append(("signal_plan", f"Change the signal plan: green_s {road['signal_plan']['green_s']:.0f} s -> 12 s or "
                  "enable/disable protected_left; adjust initial_phase_offset_s so the intended light state holds."))
        if road.get("signalized"):
            L.append(("unsignalized", "Make the junction unsignalized: replace /road/signalized with false (leave "
                      "signal_plan untouched, it is required and simply ignored) and keep all actors as they are; the "
                      "ego then has to yield to oncoming traffic when turning left."))
        L.append(("turn_lane", f"Add a dedicated right_turn_lane or left_turn_lane to arm {ea} inbound."))
    else:
        f = road["forward"]
        n = f["driving_lanes"]
        L.append(("more_lanes", f"Add a lane in the ego direction: forward.driving_lanes {n} -> {min(6, n + 1)}."))
        if n > 2:
            L.append(("fewer_lanes", f"Remove a lane in the ego direction ({n} -> {n - 1}); move actors that sat in the "
                      "removed lane."))
        L.append(("curve", "Make the road curve: replace the straight segment by line + arc (curvature 0.002-0.004 1/m, "
                  "left or right) + line, keeping the total length."))
        L.append(("speed_limit", f"Change the speed limit from {road['speed_limit_kph']:.0f} km/h to "
                  f"{80 if road['speed_limit_kph'] > 90 else 120} km/h."))
        L.append(("median_shoulder", "Add a median (median_width 3 m) or widen/remove the shoulder (shoulder_width 0.5 m)."))
    if static_objects:
        L.append(("road_works", "Road works: a row of 'cone' static objects closing one lane ahead of the ego, a "
                  "'construction_sign' (sign_type road_works) and a 'traffic_sign' speed_limit with sign_value 30 "
                  "placed before the closure (static_objects/-)."))
        L.append(("parked_cars", "Parked vehicles: two 'parked_car' static objects on the right edge (lane of the ego's "
                  "direction, t_offset about -1.2 m) ahead of the ego."))
        L.append(("barrier", "A 'barrier' static object narrowing the carriageway ahead of the ego plus a "
                  "'traffic_sign' of sign_type pedestrian_crossing / yield / stop in front of it."))
        L.append(("sign_only", "Traffic signs: a 'traffic_sign' speed_limit (sign_value 30) and a 'traffic_sign' "
                  "pedestrian_crossing before the conflict area."))
        L.append(("setting", "Change environment.setting (e.g. urban -> industrial or suburban) together with matching "
                  "static objects (poles, bollards, parked trucks)."))
    return L


# ----------------------------------------------------------------------------- intent check
_W_TERMS = {"clear": ["clear", "sunny", "fair", "good weather", "dry"], "cloudy": ["cloud", "overcast", "grey", "gray"],
            "rain": ["rain", "shower", "drizzle", "wet"],
            "heavy_rain": ["heavy rain", "downpour", "torrential", "heavy_rain", "storm", "pouring", "heavy"],
            "fog": ["fog", "mist", "haze", "hazy", "low visibility", "poor visibility"],
            "snow": ["snow", "wintry", "winter", "blizzard", "flurr"]}
_B_TERMS = {"day": ["day", "daylight", "daytime", "morning", "noon", "midday", "afternoon", "bright", "sun"],
            "dusk": ["dusk", "dawn", "twilight", "evening", "sunset", "sunrise", "low sun", "low-light", "low light"],
            "night": ["night", "dark", "after dark", "nighttime", "night-time", "midnight"]}
_S_TERMS = {"dry": ["dry"], "wet": ["wet", "damp", "slippery", "puddle", "rain-soaked", "glossy"],
            "snow": ["snow", "snow-covered", "slippery"], "ice": ["ice", "icy", "frozen", "black ice", "slippery"]}
_OBJ_TERMS = {"cone": ["cone"], "barrier": ["barrier", "barricade", "guardrail"], "parked_car": ["parked"],
              "parked_truck": ["parked", "truck"], "pole": ["pole", "post", "lamp"], "bollard": ["bollard"],
              "construction_sign": ["construction", "road works", "roadworks", "road work", "road-works"],
              "traffic_sign": ["sign"]}
_SIGN_TERMS = {"speed_limit": ["speed limit", "speed-limit", "speed sign", "km/h"], "stop": ["stop sign", "stop"],
               "yield": ["yield", "give way"], "no_entry": ["no entry", "no-entry", "do not enter"],
               "pedestrian_crossing": ["pedestrian crossing", "crosswalk", "zebra", "pedestrian"],
               "road_works": ["road works", "roadworks", "road work", "road-works", "construction"]}
_CAT_TERMS = {"car": ["car", "vehicle", "sedan"], "suv": ["suv", "vehicle", "car"], "van": ["van", "vehicle"],
              "truck": ["truck", "lorry", "heavy vehicle"], "bus": ["bus"], "motorcycle": ["motorcycle", "motorbike", "rider"],
              "bicycle": ["bicycle", "cyclist", "bike"], "pedestrian": ["pedestrian", "walker", "person"]}
_ACT_TERMS = {"speed_change": ["speed", "faster", "slower", "accelerat", "decelerat", "km/h", "slow", "fast"],
              "lane_change": ["lane"], "follow": ["follow", "gap", "headway", "tailgat"], "cut_in": ["cut", "merge"],
              "stop_at_line": ["stop", "red light", "signal", "line"], "brake": ["brak", "stop", "decelerat"],
              "cross": ["cross", "walk"], "lane_offset": ["offset", "nudg", "swerv", "lateral", "shift"]}
_ROAD_TERMS = [("driving_lanes", ["lane"]), ("lane_width", ["lane width", "wider", "narrow", "width"]),
               ("speed_limit_kph", ["speed limit", "km/h", "speed"]), ("curvature", ["curve", "bend", "curv", "arc"]),
               ("segments", ["curve", "bend", "curv", "arc", "segment", "length", "straight"]),
               ("signalized", ["signal", "traffic light", "unsignal", "priority", "uncontrolled"]),
               ("signal_plan", ["signal", "traffic light", "green", "phase", "cycle"]),
               ("median_width", ["median", "divid"]), ("shoulder_width", ["shoulder", "hard shoulder"]),
               ("corner_radius", ["corner", "radius", "turn"]), ("turn_lane", ["turn lane", "turning lane", "lane"]),
               ("heading_deg", ["arm", "angle", "approach", "junction"])]


def _has(text: str, terms: list[str]) -> bool:
    return any(t in text for t in terms)


def expected_attributes(parent: dict, child: dict, dims, changes: list[Change] | None = None) -> list[dict]:
    """Attributes a description of this variant must name: [{attr, any: [terms]}], derived from the diff."""
    ch = changes if changes is not None else diff(parent, child)
    paths = {c.path: c for c in ch}
    out: list[dict] = []
    pe, ce = parent["environment"], child["environment"]
    if "weather" in dims:
        if pe["weather"] != ce["weather"]:
            out.append({"attr": f"weather -> {ce['weather']}", "any": _W_TERMS[ce["weather"]]})
        if time_band(pe["time_of_day"]) != time_band(ce["time_of_day"]):
            b = time_band(ce["time_of_day"])
            out.append({"attr": f"time of day -> {b} ({ce['time_of_day']})", "any": _B_TERMS[b] + [ce["time_of_day"]]})
        elif pe["time_of_day"] != ce["time_of_day"]:
            out.append({"attr": f"time of day -> {ce['time_of_day']}", "any": [ce["time_of_day"]] + _B_TERMS[time_band(ce["time_of_day"])]})
        if pe["road_surface"] != ce["road_surface"] and pe["weather"] == ce["weather"]:
            out.append({"attr": f"road surface -> {ce['road_surface']}", "any": _S_TERMS[ce["road_surface"]]})
    if "behavior" in dims:
        seen: set[str] = set()
        by_id = {a["id"]: a for a in child["actors"]}
        for c in ch:
            m = re.match(r"/actors/(\d+)(?:/|$)", c.path)
            a = None
            if m:
                a = child["actors"][int(m.group(1))] if c.op != "remove" else None
            m2 = re.match(r"/actions/(\d+)", c.path)
            if m2 and c.op != "remove":
                act = child["actions"][int(m2.group(1))]
                a = by_id.get(act["actor"])
                t = act["type"]
                key = f"{act['actor']}:{t}"
                if key not in seen:
                    seen.add(key)
                    out.append({"attr": f"{act['actor']} action {t}", "any": _ACT_TERMS[t]})
            elif m2 and c.op == "remove":
                act = c.before or {}
                if act.get("type") == "stop_at_line":
                    out.append({"attr": f"{act.get('actor')} no longer stops", "any": ["red light", "run", "not stop", "ignore", "does not", "doesn't", "without stopping", "no longer"]})
                else:
                    key = f"{act.get('actor')}:rm:{act.get('type')}"
                    if key not in seen:
                        seen.add(key)
                        out.append({"attr": f"{act.get('actor')} action {act.get('type')} removed", "any": _ACT_TERMS.get(act.get("type"), [])})
            if a is not None and a["id"] not in seen:
                seen.add(a["id"])
                toks = [t for t in re.split(r"[_\W]+", a["id"].lower()) if len(t) >= 3 and not t.isdigit()]
                out.append({"attr": f"actor {a['id']}", "any": toks + _CAT_TERMS.get(a["category"], [a["category"]])})
    if "environment" in dims:
        seen = set()
        for c in ch:
            m = re.match(r"/static_objects/(\d+)", c.path)
            if m and c.op != "remove":
                o = child["static_objects"][int(m.group(1))]
                kind, st = o["kind"], o.get("sign_type")
                if kind in ("traffic_sign", "construction_sign") and st:
                    key = f"sign:{st}:{o.get('sign_value')}"
                    if key not in seen:
                        seen.add(key)
                        terms = list(_SIGN_TERMS.get(st, [st.replace("_", " ")]))
                        if o.get("sign_value") is not None:
                            v = o["sign_value"]
                            terms = [str(int(v)) if isinstance(v, (int, float)) and float(v).is_integer() else str(v)]
                        out.append({"attr": f"sign {st}{'' if o.get('sign_value') is None else ' ' + str(o['sign_value'])}",
                                    "any": terms})
                        if o.get("sign_value") is not None:
                            out.append({"attr": f"sign type {st}", "any": _SIGN_TERMS.get(st, [st]) + ["sign"]})
                elif kind not in seen:
                    seen.add(kind)
                    out.append({"attr": f"object {kind}", "any": _OBJ_TERMS.get(kind, [kind.replace("_", " ")])})
            elif m and c.op == "remove":
                continue
        if "/environment/setting" in paths:
            s = ce.get("setting")
            out.append({"attr": f"setting -> {s}", "any": [str(s)]})
        road_seen: set[str] = set()
        for c in ch:
            if not c.path.startswith("/road"):
                continue
            for key, terms in _ROAD_TERMS:
                if key in c.path and key not in road_seen:
                    road_seen.add(key)
                    out.append({"attr": f"road {key}", "any": terms})
                    break
            else:
                if "road" not in road_seen:
                    road_seen.add("road")
                    out.append({"attr": "road geometry", "any": ["road", "lane", "curve", "junction", "intersection"]})
    return out


def check_description(text: str, attrs: list[dict]) -> list[str]:
    """Attributes whose terms do not appear in the description (case-insensitive substring match)."""
    low = text.lower()
    return [a["attr"] for a in attrs if a["any"] and not _has(low, [t.lower() for t in a["any"]])]


def summarise(changes: list[Change], max_items: int = 40) -> list[str]:
    """Human-readable change list for the describe prompt."""
    out = []
    for c in changes[:max_items]:
        if c.op == "add":
            out.append(f"added {c.path}: {_compact(c.after)}")
        elif c.op == "remove":
            out.append(f"removed {c.path}: {_compact(c.before)}")
        else:
            out.append(f"{c.path}: {_compact(c.before)} -> {_compact(c.after)}")
    return out


def _compact(v, n: int = 260) -> str:
    import json
    s = json.dumps(v, separators=(",", ":"), default=str) if not isinstance(v, str) else v
    return s if len(s) <= n else s[:n] + "..."
