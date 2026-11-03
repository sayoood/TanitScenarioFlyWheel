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
    assert "signal_plan" not in b and "poles" in b["traffic_lights"] and not caption_issues(_caption(b["traffic_lights"]))
    assert "traffic_lights" not in _brief_road({"type": "junction", "signalized": False})
