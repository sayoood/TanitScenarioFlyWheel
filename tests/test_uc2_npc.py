import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from tsf.agents.video import (
    drop_actor,
    infer_action_types,
    move_signs_off_the_road,
    npc_fidelity,
    npc_line,
    pin_ego_assertions,
    pin_ego_signal,
    pin_npcs,
    request_text,
    select_npcs,
)
from tsf.ir.scenario import Scenario
from tsf.sim.engine import SimLog

EXAMPLE = json.loads((Path(__file__).parents[1] / "examples" / "double_cut_in_highway.json").read_text())

SCENE = {"summary": "A dash-cam drive.", "road_type": "urban_street", "lanes_same_direction": 3, "lanes_oncoming": 0,
         "ego_lane_from_left": 2, "traffic_lights": False, "speed_limit_kph_guess": 50, "time_of_day": "day",
         "weather": "clear", "road_surface": "dry", "events": [],
         "actors": [{"category": "car", "relation": "behind", "approx_distance_m": 8, "behavior": "follows"},
                    {"category": "car", "relation": "ahead_same_lane", "approx_distance_m": 29, "behavior": "slows"},
                    {"category": "bus", "relation": "ahead_right_lane", "approx_distance_m": 60, "behavior": "moves"}]}
EGO = {"turns": [], "lane_changes": [], "speed_kmh_per_s": [40.0] * 10, "duration_s": 10.0}


def npc(**kw):
    base = {"id": 0, "category": "car", "relation": "ahead_same_lane", "confidence": 0.7, "t0": 0.0, "t1": 9.0,
            "gap0_m": 30.0, "lat0_m": 0.1, "lane0": 0, "speed0_kmh": 30.0, "speed_mean_kmh": 28.0, "moving": True,
            "events": [], "profile": {"t": [0.0, 2.0, 4.0, 6.0, 8.0], "v_kmh": [30.0, 30.0, 20.0, 10.0, 10.0]}}
    return base | kw


def test_request_uses_measurements_and_drops_actors_behind():
    cut = npc(id=1, category="truck", relation="ahead_right_lane", lane0=-1, lat0_m=-3.4, gap0_m=22.0, speed0_kmh=36.0,
              events=[{"type": "cut_in", "t": 4.9, "from_lane": -1, "gap_m": 14.0}])
    txt = request_text(SCENE, EGO, [npc(), cut])
    assert "car 28 m ahead in the ego lane, 30 km/h" in txt
    assert "truck 20 m ahead in the lane to the right" in txt and "cuts into the ego lane at t=4.9 s (gap 12 m)" in txt
    assert "no vehicle starts beside or behind the ego" in txt and "follows" not in txt
    assert "a bus ahead right lane" in txt                      # the scene analysis still adds what was not measured
    assert "a car ahead same lane at about 29 m" not in txt     # ... but not what the measurement covers
    plain = request_text(SCENE, EGO)
    assert "a car ahead same lane at about 29 m" in plain and "follows" not in plain
    assert [n["id"] for n in select_npcs([npc(confidence=0.1), npc(id=2, relation="behind"), cut])] == [1]
    assert "stationary" in npc_line(npc(speed0_kmh=0.5, relation="parked", lat0_m=-3.6, lane0=-1))


def with_followers(n=4):
    d = copy.deepcopy(EXAMPLE)
    d["actors"][0]["initial"]["s"] = 0.0
    d["actors"] = d["actors"][:1] + [
        {"id": f"behind{i}", "category": "car", "initial": {"direction": "forward", "lane": 2, "s": 0.0},
         "initial_speed": 10.0} for i in range(1, n + 1)] + [
        {"id": "lead", "category": "car", "initial": {"direction": "forward", "lane": 2, "s": 38.0},
         "initial_speed": 20.0}]
    d["actions"] = [{"id": "lead_v", "type": "speed_change", "actor": "lead", "target_speed": 15.0, "accel": 1.0,
                     "start": {"kind": "sim_time", "t": 2.0}}]
    d["assertions"] = [{"kind": "no_collision"}]
    d["parameters"] = [{"name": "lead_speed", "path": f"/actors/{n + 1}/initial_speed", "low": 15, "high": 25},
                       {"name": "lead_t", "path": "/actions/0/start/t", "low": 1, "high": 4}]
    if n:
        d["actions"] += [{"id": "b1_v", "type": "speed_change", "actor": "behind1", "target_speed": 12.0, "accel": 1.0,
                          "start": {"kind": "sim_time", "t": 1.0}},
                         {"id": "b1_after", "type": "speed_change", "actor": "lead", "target_speed": 12.0,
                          "start": {"kind": "after_action", "action_id": "b1_v", "on": "end"}}]
        d["assertions"].append({"kind": "min_ttc_above", "actor": "behind1", "value": 1.0})
        d["parameters"].append({"name": "b2_speed", "path": "/actors/2/initial_speed", "low": 5, "high": 12})
    return d


def test_drop_actor_removes_references_and_reindexes_pointers():
    d = drop_actor(with_followers(), "behind1")
    assert [a["id"] for a in d["actors"]] == ["ego", "behind2", "behind3", "behind4", "lead"]
    assert [a["id"] for a in d["actions"]] == ["lead_v"]                  # b1_v and the action waiting for it are gone
    assert [a["kind"] for a in d["assertions"]] == ["no_collision"]
    assert {p["name"]: p["path"] for p in d["parameters"]} == {
        "lead_speed": "/actors/4/initial_speed", "b2_speed": "/actors/1/initial_speed", "lead_t": "/actions/0/start/t"}
    assert drop_actor(with_followers(), "lead")["actions"][0]["actor"] == "behind1"
    Scenario.model_validate(d)


def test_pin_drops_followers_moves_ego_and_pins_the_matched_leader():
    d = pin_npcs(with_followers(), [npc()])
    ids = [a["id"] for a in d["actors"]]
    assert ids == ["ego", "lead"], ids
    ego, lead = d["actors"]
    assert ego["initial"]["s"] >= 30.0
    # camera gap 30 m -> bumper gap 28 m; centre distance adds half of both lengths (4.6 m default cars)
    assert lead["initial"]["s"] == pytest.approx(ego["initial"]["s"] + 28.0 + 4.6, abs=0.01)
    assert lead["initial"]["lane"] == ego["initial"]["lane"]
    assert lead["initial_speed"] == pytest.approx(30 / 3.6, abs=0.01)
    lead_acts = [a for a in d["actions"] if a["actor"] == "lead"]
    assert lead_acts and all(a["type"] == "speed_change" for a in lead_acts)
    assert lead_acts[-1]["target_speed"] == pytest.approx(10 / 3.6, abs=0.2)   # the measured slowdown replaces 15 m/s
    Scenario.model_validate(d)


def test_ego_start_shift_keeps_gaps_and_works_without_measurements():
    d = with_followers(0)                                   # ego s = 0, unmatched leader 38 m ahead of it
    out = pin_npcs(d, [])
    ego, lead = out["actors"]
    assert ego["initial"]["s"] == 30.0 and lead["initial"]["s"] - ego["initial"]["s"] == pytest.approx(38.0)
    Scenario.model_validate(out)
    assert [a["id"] for a in pin_npcs(with_followers(), [])["actors"]] == ["ego", "lead"]


def test_pin_leaves_unmatched_actors_alone_and_adds_a_cut_in():
    d = with_followers(0)
    far = npc(id=3, category="bus", gap0_m=90.0)                         # no bus in the proposal -> nothing to match
    cut = npc(id=4, relation="ahead_right_lane", lane0=-1, lat0_m=-3.5, gap0_m=36.0, speed0_kmh=70.0,
              events=[{"type": "cut_in", "t": 5.0, "from_lane": -1, "gap_m": 14.0}])
    d["actors"].append({"id": "right_car", "category": "car", "initial": {"direction": "forward", "lane": 3, "s": 45.0},
                        "initial_speed": 10.0})
    out = pin_npcs(d, [far, cut])
    rc = next(a for a in out["actors"] if a["id"] == "right_car")
    assert rc["initial"]["lane"] == 3 and rc["initial"]["s"] > out["actors"][0]["initial"]["s"] + 30
    ci = next(a for a in out["actions"] if a["type"] == "cut_in")
    assert ci["actor"] == "right_car" and ci["target"] == "ego" and ci["start"]["t"] == pytest.approx(3.5)
    assert next(a for a in out["actors"] if a["id"] == "lead")["initial_speed"] == 20.0
    Scenario.model_validate(out)


def fake_report(ego_x, npc_rows):
    """SimLog with ego at the origin driving +x at 10 m/s and NPCs at (x, y, v)."""
    t = np.arange(0, 10.01, 0.5)
    n = 1 + len(npc_rows)
    S = np.zeros((len(t), n, 6))
    S[:, 0, 0], S[:, 0, 3] = ego_x + 10 * t, 10.0
    for j, (x, y, v) in enumerate(npc_rows, 1):
        S[:, j, 0], S[:, j, 1], S[:, j, 3] = x + v * t, y, v
    log = SimLog(t, ["ego"] + [f"n{j}" for j in range(1, n)], S, [[None] * n] * len(t),
                 np.tile([4.6, 1.85, 1.5], (n, 1)),
                 ["car"] * n, ["ego"] + ["npc"] * (n - 1), {})
    return SimpleNamespace(_log=log)


def test_npc_fidelity_compares_gaps_speeds_lanes_and_flags_followers():
    # measured: leader 30 m ahead (bumper 28 m) at 30 km/h; right-lane car 20 m ahead (bumper 18 m) at 36 km/h
    npcs = [npc(), npc(id=1, relation="ahead_right_lane", lane0=-1, lat0_m=-3.5, gap0_m=20.0, speed0_kmh=36.0)]
    ok = fake_report(0.0, [(28 + 4.6, 0.0, 30 / 3.6), (18 + 4.6, -3.5, 10.0)])
    k = npc_fidelity(ok, npcs)
    assert k["npc_measured"] == 2 and k["npc_matched"] == 2 and k["npc_start_behind_ego"] == 0
    assert k["npc_gap_mae_m"] < 0.1 and k["npc_speed_mae_kmh"] < 0.1 and k["npc_lane_agree"] == 1.0
    bad = fake_report(0.0, [(34.0, 0.0, 12.0), (-2.0, 0.0, 10.0), (-4.0, 0.2, 10.0)])    # two followers at the ego
    k = npc_fidelity(bad, npcs)
    assert k["npc_start_behind_ego"] == 2 and k["npc_matched"] == 1 and k["npc_speed_mae_kmh"] == pytest.approx(
        abs(12 * 3.6 - 30), abs=0.1)
    assert npc_fidelity(SimpleNamespace(), npcs) == {}


def test_ego_speed_assertions_follow_the_measured_profile():
    d = copy.deepcopy(EXAMPLE)
    d["duration"] = 10.0
    d["assertions"] = [{"kind": "ego_speed_between", "actor": "ego", "value": 27.0, "value2": 28.5},
                       {"kind": "ego_reaches_s", "actor": "ego", "value": 200.0},
                       {"kind": "max_decel_below", "actor": "ego", "value": 1.0},
                       {"kind": "ego_stops_before_line", "actor": "ego"},
                       {"kind": "no_collision", "actor": "ego"}]
    t = np.arange(0.0, 10.5, 0.5)
    out = pin_ego_assertions(copy.deepcopy(d), t, np.zeros_like(t))               # parked camera
    ks = {a["kind"]: a for a in out["assertions"]}
    assert (ks["ego_speed_between"]["value"], ks["ego_speed_between"]["value2"]) == (0.0, 1.0)
    assert ks["ego_reaches_s"]["value"] == 0.0 and "no_collision" in ks and "ego_stops_before_line" in ks
    v = np.where(t < 5.0, 10.0, 6.0)                                                # slows, never stops
    out = pin_ego_assertions(copy.deepcopy(d), t, v)
    ks = {a["kind"]: a for a in out["assertions"]}
    assert (ks["ego_speed_between"]["value"], ks["ego_speed_between"]["value2"]) == (5.0, 11.0)
    assert 0 < ks["ego_reaches_s"]["value"] < 80.0 and ks["max_decel_below"]["value"] > 1.0
    assert "ego_stops_before_line" not in ks
    Scenario.model_validate(out)


def test_signal_phase_follows_the_measured_departure():
    from tsf.sim.engine import SignalPlan
    d = json.loads((Path(__file__).parents[1] / "examples" / "uc1_left_turn_intersection.json").read_text())
    ego = next(a for a in d["actors"] if a["role"] == "ego")
    arm, turn = ego["initial"]["road"], ego["route"][0]
    t = np.arange(0.0, 20.0, 0.5)
    sp = SignalPlan(Scenario.model_validate(pin_ego_signal(copy.deepcopy(d), t, np.where(t < 9.5, 0.2, 6.0))))
    assert sp.state(arm, turn, 8.5) != "green" and sp.state(arm, turn, 9.2) == "green"   # waits, then pulls away
    sp = SignalPlan(Scenario.model_validate(pin_ego_signal(copy.deepcopy(d), t, np.full_like(t, 12.0))))
    assert sp.state(arm, turn, 0.0) == "green"                                         # never stops


def test_road_without_oncoming_lanes_gets_no_oncoming_or_crossing_users():
    ns = [npc(id=1, relation="oncoming", lane0=-2, gap0_m=40.0), npc(id=2, relation="crossing_from_left", gap0_m=25.0),
          npc(id=3)]
    assert [n["id"] for n in select_npcs(ns, scene=SCENE)] == [3]                  # SCENE: 0 oncoming lanes
    two_way = SCENE | {"lanes_oncoming": 1, "road_type": "urban_junction"}
    assert len(select_npcs(ns, scene=two_way)) == 3
    assert "oncoming" not in request_text(SCENE, EGO, ns).split("Other road users")[1].split("The camera")[0]


def test_untyped_actions_get_their_discriminator_back():
    d = with_followers()
    d["actions"] += [{"id": "lead_cutin", "actor": "lead", "target": "ego", "gap": 20.0},
                     {"id": "lead_lc", "actor": "lead", "direction": "left"}]
    with pytest.raises(ValidationError):
        Scenario.model_validate(d)
    d = pin_npcs(infer_action_types(d), [npc()])
    Scenario.model_validate(d)
    assert [a["type"] for a in d["actions"] if a["id"] in ("lead_cutin", "lead_lc")] == ["cut_in", "lane_change"]


def test_far_parked_and_off_road_users_stay_out_of_the_request():
    junction = SCENE | {"traffic_lights": True, "lanes_oncoming": 2}
    ns = [npc(id=1, lat0_m=-46.0, relation="parked", gap0_m=60.0),
          npc(id=2, relation="crossing_from_right", lat0_m=-31.0),
          npc(id=3, relation="parked", gap0_m=10.0, lat0_m=-2.5, speed0_kmh=7.0, events=[
              {"type": "cut_in", "t": 3.0, "from_lane": -1, "gap_m": 9.0}]),
          npc(id=4, gap0_m=60.0, events=[{"type": "leaves_ego_lane", "t": 2.0, "from_lane": 0, "to_lane": 1}])]
    assert [n["id"] for n in select_npcs(ns, scene=junction)] == [3, 4]
    line = npc_line(ns[2])
    assert line.count("stationary") == 1 and "cuts" not in line and "7 km/h" not in line
    assert "leaves" not in npc_line(ns[3])


def test_junction_approach_drops_followers_counting_s_down_to_the_junction():
    d = with_followers()
    d["actors"][0]["initial"].update(direction="inbound", s=40.0)
    for a in d["actors"][1:]:
        a["initial"].update(direction="inbound", lane=d["actors"][0]["initial"]["lane"])
        a["initial"]["s"] = 40.0 + (10.0 if a["id"].startswith("behind") else -30.0)
    ids = [a["id"] for a in d["actors"]]
    out = pin_npcs(d, [npc()])
    assert out["actors"][0]["initial"]["s"] == 40.0
    assert [a["id"] for a in out["actors"]] == [i for i in ids if not i.startswith("behind")]


def test_late_first_seen_user_starts_where_its_track_extrapolates_to_at_t0():
    # first seen at t = 10 s, 21 m ahead at 45 km/h while the ego (50 km/h) travelled 139 m: 21 + 139 - 125 = 35 m
    late = npc(t0=10.0, gap0_m=21.0, speed0_kmh=45.0, gap_start_m=35.0)
    assert "car 33 m ahead in the ego lane, 45 km/h (from t=0, extrapolated; first visible at t=10.0 s, 19 m" in \
        npc_line(late)
    d = pin_npcs(with_followers(), [late])
    ego, lead = d["actors"]
    assert lead["initial"]["s"] == pytest.approx(ego["initial"]["s"] + 33.0 + 4.6, abs=0.01)   # camera -> bumper 2 m
    early = npc(t0=0.0, gap0_m=30.0)
    assert "extrapolated" not in npc_line(early)
    behind = npc(t0=12.0, gap0_m=18.0, gap_start_m=-40.0)                  # overtook the ego: no start behind it
    assert "car 6 m ahead" in npc_line(behind)


def test_a_panel_inside_a_travelling_lane_moves_to_the_verge():
    d = copy.deepcopy(EXAMPLE)
    d["road"] = {"type": "linear", "segments": [{"length": 1000.0}], "forward": {"driving_lanes": 5, "lane_width": 3.5}}
    pos = {"road": 0, "direction": "forward", "lane": 5, "s": 400.0, "t_offset": 10.0}   # 5.75 m right of the line
    d["static_objects"] = [{"id": "gantry", "kind": "traffic_sign", "position": dict(pos), "sign_type": "speed_limit",
                            "sign_value": 100.0},
                           {"id": "ok", "kind": "traffic_sign", "position": dict(pos, t_offset=-3.0),
                            "sign_type": "speed_limit", "sign_value": 100.0},
                           {"id": "cone", "kind": "cone", "position": dict(pos, lane=2, t_offset=0.0)}]
    out = move_signs_off_the_road(d)["static_objects"]
    assert out[0]["position"]["t_offset"] == pytest.approx(-3.75)          # 2 m beyond the 17.5 m carriageway edge
    assert out[1]["position"]["t_offset"] == -3.0 and out[2]["position"]["t_offset"] == 0.0
    Scenario.model_validate(d)
