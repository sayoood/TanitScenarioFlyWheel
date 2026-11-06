from tsf.cosmos.llm_captions import SceneCaption, Window, _brief_road, _strip, caption_issues

# sentences the captioner actually wrote for the UC1 left turn before the rules (2026-11-02), and benign ones
BAD = {
    "signal colour": "The left-turn signal is green.",
    "ego action": "The ego vehicle has completed its lane change to the left lane.",
    "others yield": "The oncoming cars are stopped in their lanes, allowing us to proceed.",
    "light mounting": "Traffic lights are mounted on overhead gantries at the intersection ahead.",
    "road marking": "A white turn arrow is painted on the lane.",
    "traffic guess": "The red car and white truck have passed through the intersection.",
}
GOOD = ["A red car and a white truck approach from the opposite direction.", "The sun is high overhead.",
        "Brightly painted shopfronts line the road.", "Traffic lights stand on poles at the junction.",
        "The sky is clear and the road surface is dry."]


def _caption(front: str) -> SceneCaption:
    w = Window(front=front, left="Low buildings.", right="A pavement.", rear="An empty road.")
    return SceneCaption(setting="Clear midday sky over dry asphalt.", windows=[w, w, w])


def test_rules_flag_what_contradicts_the_map_and_spare_scene_text():
    for rule, sent in BAD.items():
        assert [i["rule"] for i in caption_issues(_caption(sent))] == [rule] * 3, sent
    assert caption_issues(_caption(" ".join(GOOD))) == []


def test_strip_removes_only_the_offending_sentences():
    sc = _caption(" ".join([GOOD[0], BAD["signal colour"], GOOD[3]]))
    clean = _strip(sc, caption_issues(sc))
    assert clean.windows[0].front == f"{GOOD[0]} {GOOD[3]}" and caption_issues(clean) == []


def test_brief_drops_the_signal_timing_and_states_the_mounting():
    road = {"type": "junction", "signalized": True, "signal_plan": {"green_s": 25}}
    b = _brief_road(road)
    assert "signal_plan" not in b and "mast arm" in b["traffic_lights"] and not caption_issues(_caption(b["traffic_lights"]))
    assert "traffic_lights" not in _brief_road({"type": "junction", "signalized": False})


class _LLM:
    def __init__(self, raw: dict):
        self.raw, self.msgs = raw, []

    def chat(self, msgs, **kw):
        self.msgs.append(msgs)
        return self.raw


def test_t1_captions_leave_the_junction_and_distances_out_and_repeat_the_setting():
    """Sentences 720l's captions carried into every chunk; the junction is described per chunk from the sim instead,
    and each chunk, getting its time block alone, gets the setting too."""
    import json
    import re

    from tsf.cosmos.llm_captions import JUNCTION_RULES, llm_captions
    w = Window(front="The intersection is directly in front. Oak trees line the pavement.", left="Low brick houses.",
               right="Traffic lights stand on short poles at the kerbside corners.", rear="A grass verge.")
    raw = SceneCaption(setting="Clear midday sky over a leafy suburb. The vehicle is 140 meters from the centre.",
                       windows=[w, w, w]).model_dump()
    tsir = {"meta": {"title": "t", "id": "t"}, "environment": {}, "actions": [],
            "road": {"type": "junction", "signalized": True, "signal_plan": {"green_s": 25}},
            "actors": [{"id": "ego", "role": "ego", "category": "car", "initial": {"s": 140}, "initial_speed": 12.5}]}
    llm = _LLM(raw)
    caps = llm_captions(tsir, ["camera_front_wide_120fov", "camera_cross_right_120fov"], frames=1201, llm=llm,
                        road_users=False, junction=False, repeat_setting=True)
    brief = json.loads(llm.msgs[0][1]["content"].split("Scenario (JSON): ")[1].split("\n")[0])
    assert "signalized" not in brief["road"] and "initial" not in brief["actors"][0]
    split = lambda c: re.split(r"\s*\[[0-9.]+s-[0-9.]+s\]:\s*", c)[1:]
    assert split(caps["camera_front_wide_120fov"]) == ["Clear midday sky over a leafy suburb. Oak trees line the pavement."] * 3
    assert split(caps["camera_cross_right_120fov"]) == ["Clear midday sky over a leafy suburb."] * 3
    assert not any(rx.search("A 50 km/h speed-limit sign stands on the right.") for rx in JUNCTION_RULES.values())
