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


def llm_captions(tsir: dict, cameras=CAMERAS_7, *, frames: int = 297, fps: float = 30.0,
                 appearance: dict | None = None, llm: LLM | None = None) -> dict[str, str]:
    env = {**tsir.get("environment", {}), **(appearance or {})}
    bounds = _bounds(frames, fps, 3)
    brief = {"title": tsir["meta"]["title"], "description": tsir["meta"].get("description", ""), "environment": env,
             "road": tsir["road"],
             "actors": [{k: a.get(k) for k in ("id", "role", "category", "color", "initial", "initial_speed")}
                        for a in tsir["actors"]],
             "actions": [{k: a.get(k) for k in ("actor", "type", "start", "direction", "target_speed")}
                         for a in tsir["actions"]],
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
             "list. For each of the three time windows describe what is visible ahead, left, right and behind "
             "the vehicle as the scenario unfolds."}]
    sc = SceneCaption.model_validate((llm or LLM()).chat(msgs, schema=SceneCaption.model_json_schema(),
                                                        role="captioner", max_tokens=3000))
    out = {}
    for cam in cameras:
        side = VIEW.get(cam, "front")
        blocks = [f"[{a:.1f}s-{b:.1f}s]: " + (sc.setting + " " if k == 0 else "") + getattr(sc.windows[k], side)
                  for k, (a, b) in enumerate(bounds)]
        out[cam] = " ".join(blocks)
    return out
