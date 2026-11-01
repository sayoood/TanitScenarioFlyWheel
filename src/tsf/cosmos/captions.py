"""Deterministic per-camera caption bodies for Cosmos3-Nano Auto (placeholder until the LLM captioner exists).

Transfer-Auto and Sim-Auto read one *scene-caption body* per camera (camera/FOV prefix and WSM task prompt are added by
the runtime). Bodies follow the training style: timestamped blocks ``[0.0s-3.3s]: ...`` of 300-400 words in total,
stating time of day, weather, road surface and geographic setting, and describing only what that camera can see.
The templates cover the two bridged scenarios (signalized intersection, three-lane highway), all dry / clear / day.

    caps = template_captions("uc1_left_turn_intersection", CAMERAS_7, frames=297)
    write_sidecars(caps, out_dir / "captions", frames=297)      # captions/<camera_key>.json, native + Videos API layout
"""
from __future__ import annotations

import json
import re
from pathlib import Path

CAMERAS_7 = (
    "camera_front_wide_120fov", "camera_cross_right_120fov", "camera_rear_right_70fov", "camera_rear_tele_30fov",
    "camera_rear_left_70fov", "camera_cross_left_120fov", "camera_front_tele_30fov",
)

# -- shared setting (time of day, weather, road surface, geography) -------------------------------------------------
_INTERSECTION = {
    "setting": (
        "It is late morning on a clear day under a mostly blue sky with a few thin high clouds, and the sunlight is bright "
        "and even with short crisp shadows. The pavement is dry medium-gray asphalt with sharp white and yellow paint. The "
        "scene is a suburban American-style signalized four-way intersection with wide lanes, concrete curbs, grass verges, "
        "low commercial buildings and mature deciduous trees."
    ),
    "remind": "The weather stays clear and the late-morning sunlight stays bright, with dry gray asphalt and short crisp shadows on the "
                  "ground.",
    "detail": {
        "front": (
            "Cars wear ordinary glossy paint with clear windshields and working brake lights, and every vehicle keeps a "
            "regular speed and a steady gap.",
            "Beyond the junction a low beige retail building with a flat roof, a gas station canopy and several tall "
            "street lamps line the far side of the road, with rounded green trees behind them.",
            "Utility poles with overhead wires run along the roadside, white lane arrows are painted on the dry asphalt and "
            "small shrubs and a concrete sidewalk edge the right curb.",
        ),
        "side": (
            "The shadows of poles and trees lie short and sharp on the pavement, and the vehicles on this side keep a steady, "
            "ordinary pace with clear windshields and glossy paint.",
            "Beyond the curb stand low brick buildings, a gray utility pole, a street sign on a metal post and a row of "
            "rounded green trees whose leaves move slightly.",
            "A concrete sidewalk with a red fire hydrant, a trimmed hedge and a bright blue sky above the rooftops complete "
            "the roadside.",
        ),
        "rear": (
            "The shadows of poles and trees lie short and sharp on the pavement, and the vehicle behind keeps an ordinary "
            "pace with a clear windshield, glossy paint and working brake lights.",
            "Behind the camera vehicle the pavement carries worn white lane lines, a concrete curb and a grass verge, and "
            "rounded green trees cast soft shadows across the road.",
            "A strip of low commercial buildings, a utility pole with wires and a bright blue sky above the distant rooftops "
            "frame the road behind.",
        ),
    },
    "view": {
        "camera_front_wide_120fov": (
            "The camera vehicle, a white sedan, drives forward on a two-lane approach toward the junction. A blue van travels "
            "ahead in the right lane. Mast-arm signal heads hang above the junction, a white stop line crosses the lanes and a "
            "crosswalk follows it. Oncoming traffic, a red car and a white truck, passes in the opposite lanes on the right of "
            "the median markings.",
            "The sedan eases into the left lane and slows toward the stop line. The blue van keeps its lane and its tail "
            "lights stay steady. The oncoming red car and white truck pass the junction on the right side of the view while "
            "the signal heads stay clearly visible overhead.",
            "The sedan holds the left lane near the white stop line behind the crosswalk, ready for a left turn. The blue van "
            "stays ahead in the right lane. Cross streets open to the left and right of the junction and the view stays level "
            "and sharp.",
        ),
        "camera_front_tele_30fov": (
            "A narrow distant view looks straight down the approach. The far side of the junction is compressed with small "
            "signal heads, the opposite arm of the road and low buildings behind it. The blue van is a compact shape in the "
            "right lane.",
            "The junction grows slowly larger as the camera vehicle slows. The blue van stays in the right lane and the "
            "oncoming red car and white truck appear small on the opposite side of the road.",
            "Signal heads, lane arrows and the far crosswalk become clearer while the sedan comes to a gentle stop. Trees and "
            "storefronts frame the distant road.",
        ),
        "camera_cross_left_120fov": (
            "The view looks left across the road edge toward the cross street. A grass verge, a concrete curb and the north "
            "arm of the junction are visible. A gray car waits near a white stop line on that arm.",
            "As the camera vehicle shifts into the left lane the curb and the cross street slide past. The gray car remains "
            "stopped at its line and the crosswalk markings and signal poles pass by.",
            "The wide open mouth of the cross street fills the view. The gray car still waits at its stop line and the "
            "pavement leading into the left turn is clear and dry.",
        ),
        "camera_cross_right_120fov": (
            "The view looks right across the road edge toward the south arm of the junction. A concrete curb, a grass verge "
            "and a crosswalk lie nearby. A yellow bus waits at a white stop line on the cross street.",
            "The curb and the cross street pass slowly as the camera vehicle decelerates. The yellow bus stays at its stop "
            "line and the right side of the junction remains open.",
            "The yellow bus and the right crosswalk stay on the right side as the sedan stops. Trees, a sidewalk and low "
            "buildings line the cross street.",
        ),
        "camera_rear_left_70fov": (
            "The view looks back and to the left along the approach. A black SUV follows in the left lane behind the camera "
            "vehicle. The lane markings and curb recede behind it.",
            "As the camera vehicle changes lanes the black SUV follows into the left lane and keeps a steady gap. The road "
            "behind it stays dry and empty beyond.",
            "The black SUV slows and stops behind the sedan. Lane lines, a curb and the shadows of trees fall across the "
            "dry pavement behind.",
        ),
        "camera_rear_right_70fov": (
            "The view looks back and to the right along the approach. The right lane and the curb recede behind the camera "
            "vehicle. A concrete sidewalk, a grass verge and trees line the roadside.",
            "The right lane behind stays open while the camera vehicle moves left. Lane markings and the curb drift across "
            "the view and trees pass by on the roadside.",
            "The camera vehicle halts near the stop line. The right lane behind is empty and dry, with the curb, the sidewalk "
            "and the shadows of trees visible.",
        ),
        "camera_rear_tele_30fov": (
            "A narrow distant view looks straight back along the approach. The two lanes converge toward a far vanishing "
            "point with trees and low buildings along the roadside. A black SUV is a compact shape in the left lane.",
            "The black SUV follows into the left lane and the receding road stays dry and clear in the distance behind.",
            "The black SUV slows with the camera vehicle and remains in the left lane while the long road behind stays clear.",
        ),
    },
}

_STEADY = "Colors stay natural and consistent, and the scene keeps the same lighting from the first frame to the last."
_INTERSECTION["closing"] = {
    "front": "The camera stays level and steady with no lens contamination, and distant objects fade only gently with distance.",
    "side": "The camera stays level and steady with no lens contamination, and the sunlit surfaces keep a natural neutral color balance "
            "with fine detail visible on the curb, pavement and foliage.",
    "rear": "The camera stays level and steady with no lens contamination, and the sunlit surfaces keep a natural neutral color balance "
            "with fine detail visible on the pavement, curb and foliage.",
}
_HIGHWAY = {
    "setting": (
        "It is early afternoon on a clear day under a bright blue sky with a few thin high clouds, and the sunlight is "
        "strong with short crisp shadows. The pavement is dry, smooth, medium-gray asphalt with sharp white dashed lane lines "
        "and solid edge lines. The scene is a three-lane divided highway in an open American landscape with grass shoulders, "
        "metal guardrails and distant low hills."
    ),
    "remind": "The weather stays clear and the afternoon sunlight stays bright, with dry gray asphalt and short crisp shadows on the "
                  "ground.",
    "detail": {
        "front": (
            "Both cars are ordinary sedans with glossy paint, clear rear windows and steady brake lights, and they keep a "
            "constant highway speed.",
            "Beyond the traffic a green overhead sign gantry spans the lanes in the distance, and low brown hills with sparse "
            "shrubs rise under the bright blue sky.",
            "Lane lines are white and crisp on the dry asphalt, concrete expansion joints cross the lanes at regular spacing "
            "and a metal guardrail runs along the roadside.",
        ),
        "side": (
            "Dark tire marks and thin patches of sealant cross the dry asphalt, and passing vehicles keep a constant highway "
            "speed with glossy paint and clear windows.",
            "Beyond the guardrail a grass strip, a line of small trees and a distant fence run parallel to the highway, with "
            "low hills under a bright blue sky.",
            "Reflective delineator posts stand beside the guardrail at regular spacing, and the shoulder carries a worn white "
            "edge line.",
        ),
        "rear": (
            "Dark tire marks and thin patches of sealant cross the dry asphalt, and the light on the pavement is even and "
            "bright with no glare or haze.",
            "Behind the camera vehicle the pavement carries long white dashed lane lines, a solid edge line and a metal "
            "guardrail, with low brown hills in the distance.",
            "A green overhead sign gantry spans the lanes far behind, and the straight road runs under a bright blue sky with "
            "a few thin high clouds.",
        ),
    },
    "view": {
        "camera_front_wide_120fov": (
            "The camera vehicle, a white sedan, travels in the middle lane at highway speed. A red car moves ahead in the right "
            "lane and a blue car moves ahead in the left lane, both slightly farther along than the sedan. White dashed lane "
            "lines run ahead and the road narrows toward the horizon.",
            "The red car drifts left across the dashed line and merges into the middle lane ahead of the sedan, closing the "
            "gap. The sedan keeps its lane and the blue car stays in the left lane ahead.",
            "The blue car then moves right across the dashed line and merges into the middle lane in front of the sedan. "
            "The red car is just ahead of it and the road beyond stays open.",
        ),
        "camera_front_tele_30fov": (
            "A narrow distant view shows the highway far ahead. The three lanes converge toward the horizon with small "
            "distant hills. A small red car and a small blue car are visible ahead.",
            "The red car shifts into the middle lane and appears larger as the gap closes. The blue car holds the left lane "
            "beyond it.",
            "The blue car joins the middle lane behind the red car. Both cars appear as a compact pair ahead in the middle "
            "of the road.",
        ),
        "camera_cross_left_120fov": (
            "The view looks left across the road edge. The left lane and the dashed line beside the camera vehicle are "
            "visible. A blue car travels in the left lane just ahead and a guardrail and grass verge run along the median.",
            "The blue car holds the left lane while the red car merges ahead in the middle lane. Guardrail posts pass quickly "
            "on the left.",
            "The blue car moves right across the dashed line and into the middle lane ahead of the sedan, leaving the left "
            "lane empty. The guardrail and the verge keep passing.",
        ),
        "camera_cross_right_120fov": (
            "The view looks right across the road edge. The right lane and the dashed line beside the camera vehicle are "
            "visible. A red car travels in the right lane just ahead and a grass shoulder and guardrail run along the edge.",
            "The red car drifts left across the dashed line and into the middle lane ahead of the sedan, leaving the right "
            "lane empty. The shoulder and the guardrail pass by.",
            "The right lane stays empty while the traffic ahead has merged into the middle lane. Posts, grass and distant "
            "hills pass on the right.",
        ),
        "camera_rear_left_70fov": (
            "The view looks back and to the left along the highway. The left lane and the dashed lane line recede behind the "
            "camera vehicle. The guardrail and the median run along the left edge.",
            "The left lane behind stays open and dry while the dashed line slides past. The guardrail continues along "
            "the edge.",
            "The left lane behind remains clear. The dashed line, the guardrail and the distant hills continue to recede.",
        ),
        "camera_rear_right_70fov": (
            "The view looks back and to the right along the highway. The right lane and the dashed lane line recede behind "
            "the camera vehicle. A grass shoulder and a guardrail run along the right edge.",
            "The right lane behind stays open and dry while the dashed line slides past and the shoulder continues.",
            "The right lane behind remains clear. The dashed line, the guardrail and the distant hills continue to recede.",
        ),
        "camera_rear_tele_30fov": (
            "A narrow distant view looks straight back along the highway. The three lanes converge toward a far vanishing "
            "point with distant hills and an open sky. The road behind is clear.",
            "The dashed lane lines stream toward the vanishing point and the road behind stays empty and dry.",
            "The long straight road behind remains clear under the bright sky while the dashed lines keep streaming away.",
        ),
    },
}

_HIGHWAY["closing"] = {
    "front": "The camera stays level and steady with no lens contamination, and distant objects fade only gently with distance.",
    "side": "The camera stays level and steady with no lens contamination, and the sunlit surfaces keep a natural neutral color balance "
            "with fine detail visible on the pavement, guardrail and grass.",
    "rear": "The camera stays level and steady with no lens contamination, and the sunlit surfaces keep a natural neutral color balance "
            "with fine detail visible on the pavement, guardrail and distant hills.",
}
TEMPLATES = {"intersection": _INTERSECTION, "highway": _HIGHWAY}


def scenario_kind(scenario: str) -> str:
    """Template family for a scenario id (``*intersection*`` / ``*junction*`` -> intersection, ``*highway*`` -> highway)."""
    s = scenario.lower()
    if "highway" in s:
        return "highway"
    if "intersection" in s or "junction" in s or "uc1" in s:
        return "intersection"
    raise KeyError(f"no caption template for scenario {scenario!r}; pass your own captions")


def _bounds(frames: int, fps: float, n: int = 3) -> list[tuple[float, float]]:
    """``n`` consecutive time windows covering ``frames / fps`` seconds, edges rounded to 0.1 s."""
    dur = frames / fps
    edges = [0.0] + [round(dur * i / n, 1) for i in range(1, n)] + [round(dur, 1)]
    return list(zip(edges[:-1], edges[1:]))


def template_captions(scenario: str, cameras=CAMERAS_7, *, frames: int = 297, fps: float = 30.0) -> dict[str, str]:
    """Timestamped caption body per camera (300-400 words) for ``scenario``; ``frames``/``fps`` set the block bounds."""
    t = TEMPLATES[scenario_kind(scenario)]
    out = {}
    for cam in cameras:
        if cam not in t["view"]:
            raise KeyError(f"no caption template for camera {cam!r}")
        group = "front" if "front" in cam else "side" if "cross" in cam else "rear"
        extra = t["detail"][group]
        blocks = []
        for i, ((a, b), text) in enumerate(zip(_bounds(frames, fps), t["view"][cam])):
            lead = t["setting"] if i == 0 else t["remind"]
            tail = f" {t['closing'][group]} {_STEADY}" if i == 2 else ""
            blocks.append(f"[{a:g}s-{b:g}s]: {lead} {text} {extra[i]}{tail}")
        out[cam] = "\n\n".join(blocks)
    return out


def word_count(caption: str) -> int:
    """Body words, ignoring the ``[a s-b s]:`` time labels."""
    return len(re.sub(r"\[[^\]]*\]:", "", caption).split())


def write_sidecars(captions: dict[str, str], out_dir, *, frames: int = 297) -> dict[str, Path]:
    """Write ``<out_dir>/<camera_key>.json`` in the native layout ``{"caption_structured": {"chunk_0_<frames>": {...}}}``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for cam, text in captions.items():
        doc = {"caption_structured": {f"chunk_0_{frames}": {"start_frame": 0, "end_frame": frames, "caption": text}}}
        paths[cam] = out_dir / f"{cam}.json"
        paths[cam].write_text(json.dumps(doc, indent=2))
    return paths


def caption_window(doc: dict, frames: int, fps: float = 30.0) -> str:
    """Caption text of the chunk starting at frame 0, with time labels clipped to the first ``frames`` frames."""
    covering = [c for c in doc["caption_structured"].values() if c["start_frame"] == 0 and c["end_frame"] >= frames]
    if len(covering) != 1:
        raise ValueError(f"need exactly one caption chunk covering frames 0..{frames}")
    text = covering[0]["caption"].strip()
    spans = list(re.finditer(r"\[([^\]\n]+?)s\s*-\s*([^\]\n]+?)s\]\s*:\s*", text))
    if not spans:
        return text
    parts = [text[: spans[0].start()].strip()] if text[: spans[0].start()].strip() else []
    dur = frames / fps
    for i, m in enumerate(spans):
        a, b = map(float, m.groups())
        stop = spans[i + 1].start() if i + 1 < len(spans) else len(text)
        if a < dur:
            parts.append(f"[{a:g}s-{min(b, dur):g}s]: {text[m.end():stop].strip()}")
    return "\n".join(parts)
