"""LLM caption bodies for Cosmos3-Nano Auto, written from the TSIR (+ appearance overrides), replacing the templates.

WSM controls carry geometry only; time of day, weather, road surface and the geographic setting reach the model
through the captions. One grammar-constrained Qwen call writes a shared setting and, for three time windows, what
is visible ahead, left, right and behind; each camera body is assembled in the training style
(``[0.0s-3.3s]: ...`` blocks, 300-400 words, no camera words; the runtime adds the camera prefix). Appearance
variants from augmentation only change ``appearance``, so one call per variant covers all 7 cameras.

    caps = llm_captions(tsir, frames=297, appearance={"weather": "rain", "time_of_day": "20:30"})
    write_sidecars(caps, out_dir / "captions", frames=297)
"""
from __future__ import annotations

import json
import re

from pydantic import BaseModel, Field

from tsf.agents.llm import LLM
from tsf.cosmos.captions import CAMERAS_7, _bounds, environment_notes, placements

VIEW = {"camera_front_wide_120fov": "front", "camera_front_tele_30fov": "front", "camera_cross_left_120fov": "left",
        "camera_cross_right_120fov": "right", "camera_rear_left_70fov": "rear", "camera_rear_right_70fov": "rear",
        "camera_rear_tele_30fov": "rear"}


class Window(BaseModel):
    front: str = Field(max_length=700, description="what is visible ahead of the vehicle, 60-80 words")
    left: str = Field(max_length=700, description="what is visible on the left side, 60-80 words")
    right: str = Field(max_length=700, description="what is visible on the right side, 60-80 words")
    rear: str = Field(max_length=700, description="what is visible behind the vehicle, 60-80 words")


class SceneCaption(BaseModel):
    setting: str = Field(max_length=900, description="time of day, light, sky, weather, road surface and geographic "
                                                     "setting, 70-100 words")
    windows: list[Window] = Field(..., min_length=3, max_length=3)


# What a caption must not say: Sim-Auto paints what the caption names over what the WSM control shows (turn arrows
# anywhere, a green light where the map had red), and the closed-loop ego does what the policy decides, not the script.
RULES = {
    "road marking": re.compile(r"\b(arrows?|lane (lines?|markings?)|stop ?lines?|crosswalks?|zebra|road markings?|"
                               r"painted (lines?|markings?|arrows?))\b", re.I),
    "signal colour": re.compile(r"\b(red|amber|yellow|green)\s+((left-turn|turn|traffic|signal)\s+)?"
                                r"(lights?|signals?|lamps?)\b|\b(traffic lights?|signals?)\b[^.]{0,30}\b(is|are|turns?|"
                                r"turned|shows?|showing|remains?|changes?)\b[^.]{0,20}\b(red|amber|yellow|green)\b", re.I),
    "ego action": re.compile(r"\bego\b|\b(the|our) vehicle\b[^.]{0,30}\b(stops|stopped|waits|waiting|slows|slowing|"
                             r"decelerat\w*|accelerat\w*|changes? lanes?|lane change|turns|turning|proceeds?|yields?)\b",
                             re.I),
    "others yield": re.compile(r"\b(stop\w*|wait\w*|yield\w*|allow\w*|let)\b[^.]{0,30}\b(for|to let|allowing|let)\b"
                               r"[^.]{0,20}\b(the vehicle|us)\b", re.I),
    "light mounting": re.compile(r"\b(gantr\w*|span[- ]wires?)\b", re.I),     # the map hangs them from mast arms
    "traffic guess": re.compile(r"\b(intersection|junction|road|lanes?|street)s?\b[^.]{0,25}\b(is|are|remains?|"
                                r"becomes?)\s+(now\s+)?(clear|empty|free)\b|\bno (other )?(traffic|vehicles|cars)\b|"
                                r"\b(have|has) (passed|cleared|left)\b|\bno longer (visible|ahead|there)\b", re.I),
}
# ... and, for a caller that describes the junction per chunk from the sim (``junction=False``), the junction itself:
# one caption for the whole drive put 'the intersection directly in front, traffic-light poles at its edges, 140 m
# away' into every chunk, and the generator drew signal poles beside the road 90 m before the junction (720l)
JUNCTION_RULES = {
    "junction": re.compile(r"\b(intersections?|junctions?|crossroads?|cross[- ]?streets?|traffic[- ]lights?|"
                           r"signal(s|ised|ized| heads?)?|approach(es)?|(left|right)[- ]turn lanes?)\b", re.I),
    "distance": re.compile(r"\b\d+(\.\d+)?\s*(m|metres?|meters?)\b", re.I),      # not km/h: speed signs stay
}


def caption_issues(sc: SceneCaption, rules: dict = RULES) -> list[dict]:
    """Sentences that break ``rules``, as ``{where, rule, sentence}``."""
    texts = {"setting": sc.setting, **{f"window{k}.{side}": getattr(w, side) for k, w in enumerate(sc.windows)
                                       for side in ("front", "left", "right", "rear")}}
    out = []
    for where, text in texts.items():
        for sent in re.split(r"(?<=[.!?])\s+", text):
            for rule, rx in rules.items():
                if rx.search(sent):
                    out.append({"where": where, "rule": rule, "sentence": sent})
                    break
    return out


def _strip(sc: SceneCaption, issues: list[dict]) -> SceneCaption:
    bad = {(i["where"], i["sentence"]) for i in issues}

    def clean(where: str, text: str) -> str:
        return " ".join(s for s in re.split(r"(?<=[.!?])\s+", text) if (where, s) not in bad).strip()
    return SceneCaption(setting=clean("setting", sc.setting),
                        windows=[Window(**{side: clean(f"window{k}.{side}", getattr(w, side))
                                           for side in ("front", "left", "right", "rear")})
                                 for k, w in enumerate(sc.windows)])


def _brief_road(road: dict) -> dict:
    """The road without the signal timing (the captioner turned it into guessed colours) and with how lights stand."""
    out = {k: v for k, v in road.items() if k != "signal_plan"}
    if road.get("signalized"):                     # positive wording only: a caption's "no gantry" paints a gantry
        out["traffic_lights"] = "signal heads hang from a mast arm beyond the junction, one above each lane"
    return out


def _brief_actor(a: dict) -> dict:
    d = {k: a.get(k) for k in ("id", "role", "category", "color", "initial", "initial_speed")}
    if d["role"] == "ego":                         # "ego" ends up in the caption as "the ego vehicle stops ..."
        d["role"] = "camera vehicle (never describe it)"
    return d


def llm_captions(tsir: dict, cameras=CAMERAS_7, *, frames: int = 297, fps: float = 30.0,
                 appearance: dict | None = None, llm: LLM | None = None, road_users: bool = True,
                 junction: bool = True, repeat_setting: bool = False, report: dict | None = None) -> dict[str, str]:
    """``report``: filled with the rule check ({issues_first, issues_final, repaired, stripped}). ``road_users=False``:
    the captions describe the road and its surroundings only, for a caller that names the vehicles in view per chunk
    from the map (the T1 loop, :func:`tsf.closedloop.neural.vehicle_cast`). ``junction=False``: they also leave the
    junction, its lights and all distances out (the T1 loop describes them per chunk from the sim,
    :func:`tsf.closedloop.neural.junction_caption`) and the setting describes the roadside concretely. ``repeat_setting``:
    every time block starts with the setting, not only the first (a caller that captions each chunk with its block)."""
    env = {**tsir.get("environment", {}), **(appearance or {})}
    bounds = _bounds(frames, fps, 3)
    rules = RULES if junction else {**RULES, **JUNCTION_RULES}
    road = _brief_road(tsir["road"]) if junction else \
        {k: v for k, v in tsir["road"].items() if k not in ("signal_plan", "signalized")}
    actors = [_brief_actor(a) for a in tsir["actors"] if road_users or a.get("role") == "ego"]
    if not junction:                               # the ego's start (s = 140) became "140 metres from the intersection"
        actors = [{k: v for k, v in a.items() if k not in ("initial", "initial_speed")} for a in actors]
    brief = {"title": tsir["meta"]["title"], "description": tsir["meta"].get("description", ""), "environment": env,
             "road": road, "actors": actors,
             "actions": [{k: a.get(k) for k in ("actor", "type", "start", "direction", "target_speed")}
                         for a in tsir["actions"]] if road_users else [],
             "windows_s": [list(b) for b in bounds]}
    if tsir.get("static_objects"):        # signs, cones, barriers, parked vehicles: where they stand, what they look like
        brief["static_objects"] = [
            {**{k: o.get(k) for k in ("id", "kind", "sign_type", "sign_value")},
             "metres_left_of_ego": round(o["left"], 1), "metres_ahead_of_ego": round(o["long"])}
            for o in placements(tsir) if o["same_way"]]
        brief["visible_environment"] = {k: v for k, v in environment_notes(tsir).items() if v and not k.endswith("_again")}
    msgs =[{"role": "system", "content": "You write scene captions for a photorealistic multi-camera driving-video "
             "generator. Concrete and visual, present tense; never mention cameras, sensors, lenses or the rig; "
             "describe only what can be seen; keep road users, colours and positions consistent across windows."},
            {"role": "user", "content": "Scenario (JSON): " + json.dumps(brief, separators=(",", ":")) +
             "\nRespect the environment exactly (time of day, weather, surface, environment.setting). Invent a "
             "plausible, consistent setting. Every sign and static object listed in the scenario must appear, with its "
             "colour and shape (a speed-limit sign with its number, cones, barriers, parked vehicles), in the window "
             "and the direction where the vehicle passes it; do not add road works or signs the scenario does not "
             "list. Do not describe painted road markings (turn arrows, lane lines, stop lines, crosswalks): the map "
             "draws them where they are, and a caption that names them makes the generator paint them anywhere. Never "
             "use the word 'arrow'. " + (
                 "Never state the colour or state of a traffic signal: the map shows each signal's state, and a guessed "
                 "colour makes the generator paint it; you may say that traffic lights stand at the junction. "
                 if junction else
                 "Leave the junction out: do not describe the intersection, its traffic lights or the crossing roads, "
                 "and give no distances; they are described separately for each moment. In the setting, describe the "
                 "roadside concretely (buildings, vegetation, kerbs and pavements, with their colours and materials) so "
                 "that every moment of the drive shows the same world. ") +
             "Describe the surroundings, not the vehicle's own driving: do not say that it changes lanes, "
             "slows, stops, waits or turns, and do not say that other road users stop, wait or yield for it (a driving "
             "policy decides that). " + (
                 "Describe other road users by type, colour and position, with movements only as the "
                 "scenario's actions state them, and add no road users, cyclists or pedestrians the scenario does not "
                 "list. The map places every road user at every moment: do not guess where they have moved, whether "
                 "they have passed, or whether the road or junction is clear. Describe only what is there, never what "
                 "is absent. For each of the three time windows describe what is visible ahead, left, right and behind "
                 "the vehicle: the surroundings, and the listed road users in that direction by their look."
                 if road_users else
                 "Leave all road users out (vehicles, cyclists, pedestrians): they are named separately for each "
                 "moment. Describe only what is there, never what is absent. For each of the three time windows "
                 "describe the surroundings visible ahead, left, right and behind the vehicle.")}]
    llm = llm or LLM()
    raw = llm.chat(msgs, schema=SceneCaption.model_json_schema(), role="captioner", max_tokens=3000)
    sc = SceneCaption.model_validate(raw)
    first = issues = caption_issues(sc, rules)
    if issues:                                      # one repair round, then drop whatever still breaks a rule
        fix = msgs + [{"role": "assistant", "content": json.dumps(raw, separators=(",", ":"))},
                      {"role": "user", "content": "These sentences break the rules above: " + json.dumps(
                          [{"rule": i["rule"], "sentence": i["sentence"]} for i in issues]) +
                       ". Return the same JSON with only those sentences reworded or removed."}]
        sc = SceneCaption.model_validate(llm.chat(fix, schema=SceneCaption.model_json_schema(), role="captioner",
                                                  max_tokens=3000))
        issues = caption_issues(sc, rules)
        if issues:
            sc = _strip(sc, issues)
    if report is not None:
        report.update(issues_first=first, issues_final=caption_issues(sc, rules), repaired=bool(first),
                      stripped=len(issues))
    out = {}
    for cam in cameras:
        side = VIEW.get(cam, "front")
        blocks = [f"[{a:.1f}s-{b:.1f}s]: " + " ".join(p for p in (sc.setting if k == 0 or repeat_setting else "",
                                                                   getattr(sc.windows[k], side)) if p)
                  for k, (a, b) in enumerate(bounds)]
        out[cam] = " ".join(blocks)
    return out
