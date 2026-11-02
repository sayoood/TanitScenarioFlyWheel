"""Static environment of a scenario (TSIR ``static_objects`` + ``environment.setting``): schema, compilers, TanitSim,
RDS-HQ export, captions. The road-works example is the end-to-end fixture."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from lxml import etree
from pydantic import ValidationError

from tsf.compile import osc2, pycode, xodr, xosc
from tsf.cosmos import captions, rds_hq
from tsf.ir import scenario as ir
from tsf.ir.scenario import DEFAULT_STATIC_DIMS, Scenario, json_schema
from tsf.sim import odr
from tsf.sim.engine import TanitSim
from tsf.verify import ladder

ROOT = Path(__file__).resolve().parents[1]
EX = ROOT / "examples"
ROADWORKS = EX / "uc1_roadworks_variant.json"


def _have_nvidia() -> bool:
    try:
        rds_hq.wsm_dir()
        return True
    except FileNotFoundError:
        return False


needs_nvidia = pytest.mark.skipif(not _have_nvidia(), reason="Cosmos WSM checkout not available")
needs_xsd = pytest.mark.skipif(not (ladder.XSD_DIR / "OpenDRIVE_Core.xsd").exists(), reason="XSDs not installed")


def highway(statics=(), *, v=20.0, s=0.0, lane=1, lanes=2, duration=12.0, actions=()) -> Scenario:
    """Straight 600 m road, one ego, optional static objects (dicts) and actions."""
    return Scenario.model_validate({
        "meta": {"id": "t_static", "title": "static objects"},
        "road": {"type": "linear", "segments": [{"kind": "line", "length": 600}], "forward": {"driving_lanes": lanes},
                 "backward": {"driving_lanes": 1}, "speed_limit_kph": 100},
        "actors": [{"id": "ego", "role": "ego", "initial": {"lane": lane, "s": s}, "initial_speed": v}],
        "static_objects": list(statics), "actions": list(actions), "duration": duration})


def obj(oid, kind, s, lane=1, t=0.0, **kw) -> dict:
    return {"id": oid, "kind": kind, "position": {"lane": lane, "s": s, "t_offset": t}, **kw}


def simulate(sc: Scenario, tmp_path):
    xp = xodr.write(sc, tmp_path / "t.xodr")
    sim = TanitSim(sc, odr.load(xp))
    return sim, sim.run()


@pytest.fixture(scope="module")
def roadworks(tmp_path_factory):
    rep = ladder.run((ROADWORKS).read_text(), tmp_path_factory.mktemp("rw"), qc=False)
    return rep


# ------------------------------------------------------------------------------ schema
def test_schema_contract_and_roundtrip():
    defs = json_schema()["$defs"]
    assert defs["StaticObjectKind"]["enum"] == ["cone", "barrier", "parked_car", "parked_truck", "traffic_sign", "pole",
                                                "bollard", "construction_sign"]
    assert defs["SignType"]["enum"] == ["speed_limit", "stop", "yield", "no_entry", "pedestrian_crossing", "road_works"]
    assert set(defs["StaticObject"]["properties"]) == {"id", "kind", "position", "heading_deg", "dimensions",
                                                       "sign_type", "sign_value"}
    assert defs["Environment"]["properties"]["setting"]["enum"] == ["urban", "suburban", "rural", "highway", "industrial"]
    sc = Scenario.model_validate_json(ROADWORKS.read_text())
    assert sc.environment.setting == "suburban" and len(sc.static_objects) > 10
    assert Scenario.model_validate_json(sc.model_dump_json()) == sc
    assert sc.static_objects[0].dims() == DEFAULT_STATIC_DIMS["traffic_sign"]
    assert ir.StaticObject(id="big", kind="cone", position={"s": 1}, dimensions=(1, 2, 3)).dims() == (1, 2, 3)


@pytest.mark.parametrize("name", ["uc1_left_turn_intersection.json", "double_cut_in_highway.json"])
def test_existing_examples_unchanged(name):
    sc = Scenario.model_validate_json((EX / name).read_text())
    assert sc.static_objects == [] and sc.environment.setting == "urban"
    assert "static_objects" not in json.loads((EX / name).read_text())


@pytest.mark.parametrize("bad, msg", [
    (obj("a", "traffic_sign", 10), "needs a sign_type"),
    (obj("a", "cone", 10, sign_type="stop"), "belong to kind traffic_sign"),
    (obj("a", "construction_sign", 10, sign_type="road_works"), "belong to kind traffic_sign"),
    (obj("a", "traffic_sign", 10, sign_type="speed_limit"), "sign_value"),
    (obj("a", "traffic_sign", 10, sign_type="stop", sign_value=30), "sign_value"),
    (obj("ego", "cone", 10), "not unique"),
    (obj("a", "cone", 10, lane=3), "lane 3 does not exist"),
    (obj("a", "cone", 700), "outside the road"),
    (obj("a", "cone", 10, dimensions=(0.0, 1.0, 1.0)), "positive"),
    ({**obj("a", "cone", 10), "position": {"road": 1, "lane": 1, "s": 10}}, "only road 0"),
    ({**obj("a", "cone", 10), "position": {"direction": "inbound", "lane": 1, "s": 10}}, "junction arms"),
])
def test_static_object_validation(bad, msg):
    with pytest.raises(ValidationError, match=msg):
        highway([bad])


def test_static_object_ids_unique_and_junction_checks():
    with pytest.raises(ValidationError, match="not unique"):
        highway([obj("a", "cone", 10), obj("a", "cone", 20)])
    base = json.loads((EX / "uc1_left_turn_intersection.json").read_text())
    base["static_objects"] = [{"id": "c", "kind": "cone", "position": {"road": 7, "direction": "inbound", "lane": 1, "s": 5}}]
    with pytest.raises(ValidationError, match="junction has 4"):
        Scenario.model_validate(base)
    base["static_objects"] = [{"id": "c", "kind": "cone", "position": {"road": 1, "direction": "forward", "lane": 1, "s": 5}}]
    with pytest.raises(ValidationError, match="linear roads"):
        Scenario.model_validate(base)
    base["static_objects"] = [{"id": "c", "kind": "cone", "position": {"road": 1, "direction": "inbound", "lane": 3, "s": 5}}]
    with pytest.raises(ValidationError, match="lane 3 does not exist"):
        Scenario.model_validate(base)


# ------------------------------------------------------------------------------ compilers
def test_xodr_objects_and_signals(tmp_path):
    sc = highway([
        obj("c1", "cone", 100, 2, -0.5), obj("bar", "barrier", 120, 2, 0.0, heading_deg=90),
        obj("pc", "parked_car", 80, 2, -3.2), obj("pt", "parked_truck", 60, 2, -3.4), obj("pl", "pole", 50, 2, -3.8),
        obj("bo", "bollard", 55, 2, -2.2), obj("lim", "traffic_sign", 40, 2, -2.6, sign_type="speed_limit", sign_value=30),
        obj("stp", "traffic_sign", 45, 2, -2.6, sign_type="stop"), obj("yl", "traffic_sign", 46, 2, -2.6, sign_type="yield"),
        obj("ne", "traffic_sign", 47, 2, -2.6, sign_type="no_entry"),
        obj("pxg", "traffic_sign", 48, 2, -2.6, sign_type="pedestrian_crossing"),
        obj("rw", "traffic_sign", 30, 2, -2.6, sign_type="road_works"), obj("cs", "construction_sign", 35, 2, -2.6),
        obj("opp", "cone", 100, 1, 0.5, ) | {"position": {"direction": "backward", "lane": 1, "s": 100, "t_offset": 0.5}},
    ])
    root = etree.parse(str(xodr.write(sc, tmp_path / "t.xodr"))).getroot()
    road = root.find("road")
    assert [c.tag for c in road] == ["type", "planView", "lanes", "objects", "signals"]
    ob = {o.get("id"): o for o in road.find("objects")}
    assert (ob["c1"].get("type"), ob["c1"].get("subtype")) == ("obstacle", "cone")
    assert ob["bar"].get("type") == "barrier" and float(ob["bar"].get("hdg")) == pytest.approx(math.pi / 2)
    assert ob["pc"].get("type") == "car" and ob["pt"].get("type") == "van" and ob["pt"].get("subtype") == "truck"
    assert ob["pl"].get("type") == "pole" and ob["bo"].get("subtype") == "bollard"
    assert float(ob["pt"].get("length")) == 12.0 and float(ob["pt"].get("height")) == 3.6
    # lane 2 of a 2-lane forward carriageway: centre t = -(2 - 0.5) * 3.5 = -5.25; t_offset -3.2 is further right
    assert float(ob["pc"].get("t")) == pytest.approx(-5.25 - 3.2) and float(ob["pc"].get("s")) == pytest.approx(80)
    # the opposite (backward) lane drives against s: + t_offset (left of its driving direction) is - t, hdg turns by pi
    opp = ob["opp"]
    assert float(opp.get("t")) == pytest.approx(1.75 - 0.5) and abs(float(opp.get("hdg"))) == pytest.approx(math.pi)
    assert float(opp.get("s")) == pytest.approx(600 - 100)
    sg = {s.get("id"): s for s in road.find("signals")}
    code = {k: (v.get("type"), v.get("subtype")) for k, v in sg.items()}
    assert code["lim"] == ("274", "30") and sg["lim"].get("value") == "30" and sg["lim"].get("unit") == "km/h"
    assert code["stp"][0] == "206" and code["yl"][0] == "205" and code["ne"][0] == "267" and code["pxg"][0] == "350"
    assert code["rw"][0] == "123" and code["cs"][0] == "123"
    assert all(s.get("dynamic") == "no" and s.get("country") == "DE" and s.get("orientation") == "+" for s in sg.values())


@needs_xsd
def test_every_kind_passes_the_xsds_on_a_linear_road(tmp_path):
    kinds = [obj("c1", "cone", 100, 2, -0.5), obj("bar", "barrier", 120, 2, 0.0, heading_deg=90),
             obj("pc", "parked_car", 80, 2, -3.2, heading_deg=15), obj("pt", "parked_truck", 60, 2, -3.4),
             obj("pl", "pole", 50, 2, -3.8), obj("bo", "bollard", 55, 2, -2.2), obj("cs", "construction_sign", 35, 2, -2.6),
             obj("opp", "traffic_sign", 100, 1, 2.0, sign_type="stop", heading_deg=20) | {
                 "position": {"direction": "backward", "lane": 1, "s": 100, "t_offset": -2.6}}]
    kinds += [obj(f"sg_{t}", "traffic_sign", 20 + 3 * i, 2, -2.6, sign_type=t, **({"sign_value": 80.0} if t == "speed_limit" else {}))
              for i, t in enumerate(["speed_limit", "stop", "yield", "no_entry", "pedestrian_crossing", "road_works"])]
    keep = [{"id": "hold", "actor": "ego", "type": "speed_change", "target_speed": 20.0,
             "start": {"kind": "sim_time", "t": 0.5}}]            # an empty Act is not valid OpenSCENARIO
    sc = highway(kinds, actions=keep)
    xp, sp = xodr.write(sc, tmp_path / "t.xodr"), xosc.write(sc, tmp_path / "t.xosc", "t.xodr")
    for f in (ladder.l1_xsd(xp, "OpenDRIVE_Core.xsd", "xodr"), ladder.l1_xsd(sp, "OpenSCENARIO.xsd", "xosc")):
        assert f.ok and not f.detail.get("skipped"), f.message
    text = osc2.emit(sc)
    assert "keep(it.sign_value == 80kph)" in text and "sign_type!no_entry" in text and "heading: 200deg" in text
    sim, log = simulate(sc, tmp_path)
    assert len(log.static) == len(kinds)
    assert ladder.l2_static_objects(sc, sim.m)[0].ok


@needs_xsd
def test_roadworks_example_compiles_and_validates(roadworks, tmp_path):
    assert roadworks.passed, roadworks.diagnostics()
    checks = {f.check: f for f in roadworks.findings}
    for name in ("opendrive_xsd", "openscenario_xsd", "static_objects"):
        assert checks[name].ok and not checks[name].detail.get("skipped"), checks[name]
    assert checks["no_collision"].ok
    xp = Path(roadworks.artifacts["xodr"])
    arm = next(r for r in etree.parse(str(xp)).getroot().findall("road") if r.get("id") == "2")
    sigs = {s.get("id"): s for s in arm.find("signals")}
    assert sigs["sign_limit_30"].get("type") == "274" and sigs["sign_road_works"].get("type") == "123"
    # the traffic light of the arm is still there next to the signs, with its numeric id
    assert any(s.get("dynamic") == "yes" for s in sigs.values())
    assert [c.tag for c in arm][-2:] == ["objects", "signals"]


def test_xosc_and_osc2_entities(roadworks):
    sc = roadworks._sim.sc
    root = etree.parse(roadworks.artifacts["xosc"]).getroot()
    ents = {e.get("name"): e for e in root.find("Entities")}
    assert {o.id for o in sc.static_objects} <= set(ents)
    assert ents["cone_1"].find("MiscObject") is not None and ents["cone_1"].find("MiscObject").get("miscObjectCategory") == "obstacle"
    assert ents["parked_truck"].find("Vehicle").get("vehicleCategory") == "truck"
    assert ents["barrier_end"].find("MiscObject").get("miscObjectCategory") == "barrier"
    props = {p.get("name"): p.get("value") for p in ents["sign_limit_30"].iter("Property")}
    assert props["tsf_sign_type"] == "speed_limit" and props["tsf_sign_value_kph"] == "30"
    priv = {p.get("entityRef"): p for p in root.iter("Private")}
    tp = priv["cone_3"].find(".//TeleportAction/Position/LanePosition")
    assert tp.get("roadId") == "2" and tp.get("laneId") == "2" and float(tp.get("s")) == pytest.approx(52.0)
    sign = priv["sign_limit_30"].find(".//LanePosition/Orientation")
    assert sign.get("type") == "relative" and float(sign.get("h")) == pytest.approx(math.pi)   # faces the traffic
    acting = {e.get("entityRef") for mg in root.iter("ManeuverGroup") for e in mg.iter("EntityRef")}
    assert not acting & {o.id for o in sc.static_objects}                                         # stationary: no maneuvers
    text = Path(roadworks.artifacts["osc2"]).read_text()
    assert "sign_limit_30: static_object with:" in text and "keep(it.sign_value == 30kph)" in text
    assert "parked_truck: vehicle with:" in text and "cone_1.place(road: 2, lane: 2" in text


def test_pycode_replay_lists_statics(roadworks):
    code = pycode.emit(roadworks._sim.sc, roadworks._log)
    assert '"cone_1", "cone"' in code and "STATIC = [[" in code
    compile(code, "replay.py", "exec")


# ------------------------------------------------------------------------------ TanitSim
def test_readback_map_matches_simulator(roadworks):
    sim, m = roadworks._sim, roadworks._sim.m
    ids = {b.id for b in sim.static}
    assert ids == {o.id for o in roadworks._sim.sc.static_objects}
    in_map = {o.id: (o.x, o.y) for o in m.objects()} | {s.id: (s.x, s.y) for s in m.signs()}
    for b in sim.static:
        assert np.hypot(in_map[b.id][0] - b.x, in_map[b.id][1] - b.y) < 0.02, b.id
    assert m.signals() and all(s.dynamic for s in m.signals())      # signs never show up as traffic lights
    sg = next(b for b in sim.static if b.id == "sign_limit_30")
    cone = next(b for b in sim.static if b.id == "cone_3")
    assert cone.lane[2] == 2 and abs(cone.t) < 1e-9 and cone.s == pytest.approx(200 - 52)
    assert sg.is_panel and not cone.is_panel
    assert abs(math.remainder(sg.yaw - math.pi, 2 * math.pi)) < 1e-6    # arm 2 inbound drives east; the panel faces west, at the traffic


def test_ego_hits_cone_when_it_cannot_stop(tmp_path):
    sc = highway([obj("cone_a", "cone", 25)], v=25.0, duration=6.0)
    sim, log = simulate(sc, tmp_path)
    assert log.metrics["collision"] and log.metrics["collisions"][0]["with"] == "cone_a"
    assert log.metrics["min_ttc_actor"] == "cone_a" and log.metrics["min_ttc"] < 1.5
    assert [b.id for b in log.static] == ["cone_a"]
    msg = ladder._collision_hint(log, log.metrics["collisions"])
    assert "static cone" in msg and "lane_change" in msg
    sc.assertions = [ir.Assertion(kind="no_collision")]
    assert not ladder.l3_assertions(sc, log)[0].ok


def test_ego_stops_behind_a_cone_and_a_parked_truck(tmp_path):
    for kind, dx in (("cone", 0.2), ("parked_truck", 6.0)):
        sc = highway([obj("blocker", kind, 260)], v=15.0, duration=40.0)
        sim, log = simulate(sc, tmp_path)
        x_end, v_end = log.states[-1, 0, 0], log.states[-1, 0, 3]
        assert not log.metrics["collision"], kind
        assert v_end < 0.1 and 260 - dx - 5.0 < x_end + 2.3 < 260 - dx, (kind, x_end)   # standstill ~2 m behind the object
        assert log.metrics["ego_max_decel"] < 5.0                                             # a normal, early braking


def test_lane_change_clears_the_obstacle_and_other_lane_is_no_obstacle(tmp_path):
    cones = [obj(f"c{i}", "cone", 150 + 6 * i) for i in range(4)]
    lc = [{"id": "lc", "actor": "ego", "type": "lane_change", "direction": "right", "start": {"kind": "sim_time", "t": 0.5}}]
    sim, log = simulate(highway(cones, v=15.0, duration=14.0, actions=lc), tmp_path)
    assert not log.metrics["collision"] and log.metrics["ego_lane_changes_left"] == -1
    assert log.states[:, 0, 3].min() > 14.0                                    # never slows down for them
    sim, log = simulate(highway([obj("far", "cone", 100, lane=2, t=0.0)], v=15.0, duration=10.0), tmp_path)
    assert not log.metrics["collision"] and log.states[:, 0, 3].min() > 14.9
    assert log.metrics["min_ttc"] is None


def test_roadside_signs_and_poles_do_not_brake_the_ego(tmp_path):
    sc = highway([obj("lim", "traffic_sign", 100, 2, -2.6, sign_type="speed_limit", sign_value=30),
                  obj("pl", "pole", 140, 2, -3.8), obj("pk", "parked_car", 160, 2, -3.2)], v=15.0, lane=2, duration=14.0)
    sim, log = simulate(sc, tmp_path)
    assert not log.metrics["collision"] and log.states[:, 0, 3].min() > 14.9


def test_roadworks_zone_the_ego_leaves_the_closed_lane(roadworks):
    m = roadworks.metrics
    assert not m["collision"] and m["ego_lane_changes_left"] == 1 and m["ego_turns"] == ["left"]
    # the cone taper and the stationary work van are obstacles for anyone staying in lane 2 of the arm
    sim = roadworks._sim
    van = sim.agents["lead_right"]
    assert van.v == 0.0 and van.x < -19 - 20


# ------------------------------------------------------------------------------ RDS-HQ / WSM export
@needs_nvidia
def test_rds_hq_export_has_objects_signs_and_poles(roadworks, tmp_path):
    sim, log = roadworks._sim, roadworks._log
    root = rds_hq.export_clip(sim, log, tmp_path / "rds", "rw_test", t0=3.0, duration=2.0)
    rep = rds_hq.export_report(root, "rw_test")
    nv = rds_hq.nvidia_io()
    v = nv.validate_clip(root, "rw_test", "ftheta", require_planar_polygons=True)
    objs = nv.read_archive(root / "all_object_info" / "rw_test.tar", "rw_test")
    frame0 = objs["000000.all_object_info.json"]
    assert frame0["cone_3"]["object_type"] == "Others" and frame0["cone_3"]["object_is_moving"] is False
    assert frame0["parked_truck"]["object_type"] == "Truck" and frame0["barrier_end"]["object_type"] == "Others"
    assert frame0["parked_truck"]["object_lwh"] == [12.0, 2.55, 3.6]
    assert frame0["parked_truck"]["object_to_world"] == objs["000050.all_object_info.json"]["parked_truck"]["object_to_world"]
    b = next(b for b in log.static if b.id == "cone_3")
    wf = rds_hq.WorldFrame(np.array(rep["ego"]["sim_rear_axle0"]), rep["ego"]["sim_yaw0"])
    T = np.array(frame0["cone_3"]["object_to_world"])
    assert np.allclose(T[:2, 3], wf.xy([b.x, b.y]), atol=1e-3) and T[2, 3] == pytest.approx(b.lwh[2] / 2)
    signs = nv.read_archive(root / "3d_traffic_signs" / "rw_test.tar", "rw_test")["traffic_signs.json"]["labels"]
    cls = [a["text"] for lab in signs for a in lab["labelData"]["shape3d"]["attributes"] if a["name"] == "sign_class"]
    assert cls.count("TRAFFIC_SIGN_REGULATORY_R2_SPEED_LIMIT") >= 1 and "TRAFFIC_SIGN_WARNING_W20_ROAD_WORK" in cls
    assert rep["static_objects"]["as_signs"]["sign_limit_30"] == "TRAFFIC_SIGN_REGULATORY_R2_SPEED_LIMIT"
    assert rep["static_objects"]["as_poles"] == ["lamp_post"]
    assert v["map_labels"]["3d_poles"] >= 3 and rep["static_objects"]["n"] == len(sim.static)
    assert rep["objects"]["class_mapping"]["cone_1"] == "cone -> Others"
    # road 2 carries an explicit speed sign, so only the three other roads get their two map-derived ones
    assert sum(1 for c in cls if c.endswith("R2_SPEED_LIMIT")) == 1 + 2 * 3 and len(cls) == 1 + 2 * 3 + 2


def test_default_t0_prefers_the_work_zone(roadworks):
    t0, info = rds_hq.choose_t0(roadworks._sim, roadworks._log)
    assert info["parts"]["static"] >= 0.0 and 0.0 <= t0 <= roadworks._sim.sc.duration


# ------------------------------------------------------------------------------ captions
def test_captions_mention_signs_objects_and_setting():
    tsir = json.loads(ROADWORKS.read_text())
    notes = captions.environment_notes(tsir)
    front = " ".join(notes["front"])
    assert "speed-limit sign" in front and "30" in front and "road-works" in front and "cones" in front
    assert any("parked truck" in s for s in notes["right"]) and not notes["left"]
    assert "suburban" in notes["setting"][0]
    caps = captions.template_captions("uc1_left_turn_intersection", tsir=tsir)
    assert "traffic cones" in caps["camera_front_wide_120fov"] and "parked truck" in caps["camera_cross_right_120fov"]
    for cam, text in caps.items():
        assert 250 <= captions.word_count(text) <= 400, (cam, captions.word_count(text))
    base = captions.template_captions("uc1_left_turn_intersection")
    assert captions.template_captions("uc1_left_turn_intersection", tsir=json.loads((EX / "uc1_left_turn_intersection.json").read_text())) == base
    hw = json.loads((EX / "double_cut_in_highway.json").read_text())
    hw["environment"]["setting"] = "industrial"
    assert "industrial district" in captions.template_captions("double_cut_in_highway", tsir=hw)["camera_front_wide_120fov"]


def test_llm_caption_brief_carries_the_environment():
    from tsf.cosmos import llm_captions

    sent = {}

    class Fake:
        def chat(self, msgs, schema=None, role=None, max_tokens=None):
            sent["user"] = msgs[-1]["content"]
            win = {k: "w" for k in ("front", "left", "right", "rear")}
            return {"setting": "s", "windows": [win] * 3}

    llm_captions.llm_captions(json.loads(ROADWORKS.read_text()), llm=Fake())
    assert '"static_objects"' in sent["user"] and '"sign_value":30' in sent["user"] and '"setting":"suburban"' in sent["user"]
    assert "visible_environment" in sent["user"]


# ------------------------------------------------------------------------------ BEV
def test_bev_renders_the_static_environment(tmp_path):
    """One frame of a straight road (cheap map layer) with every static kind; the markers must change the picture."""
    import cv2

    from tsf.sim.render2d import MapArt, Renderer

    statics = [obj("c1", "cone", 60, 1, 0.0), obj("b1", "barrier", 70, 1, 0.0, heading_deg=90),
               obj("pc", "parked_car", 80, 2, -3.2), obj("pt", "parked_truck", 100, 2, -3.4), obj("pl", "pole", 50, 2, -3.8),
               obj("bo", "bollard", 55, 2, -2.2), obj("lim", "traffic_sign", 40, 2, -2.6, sign_type="speed_limit", sign_value=30),
               obj("stp", "traffic_sign", 42, 2, -2.6, sign_type="stop"), obj("yl", "traffic_sign", 44, 2, -2.6, sign_type="yield"),
               obj("ne", "traffic_sign", 46, 2, -2.6, sign_type="no_entry"),
               obj("px", "traffic_sign", 48, 2, -2.6, sign_type="pedestrian_crossing"),
               obj("rw", "traffic_sign", 30, 2, -2.6, sign_type="road_works"), obj("cs", "construction_sign", 35, 2, -2.6)]
    sc = highway(statics, v=10.0, duration=5.0)
    sim, log = simulate(sc, tmp_path)
    r = Renderer(sim, log, (640, 360))
    with_objects = r.frame(int(2.0 / sim.dt))
    plain_sim, plain_log = simulate(highway([], v=10.0, duration=5.0), tmp_path)
    plain = Renderer(plain_sim, plain_log, (640, 360)).frame(int(2.0 / sim.dt))
    assert with_objects.shape == (360, 640, 3) and np.abs(with_objects.astype(int) - plain.astype(int)).sum() > 50_000
    cv2.imwrite(str(tmp_path / "bev.png"), cv2.cvtColor(with_objects, cv2.COLOR_RGB2BGR))
    # the explicit speed-limit sign replaces the map-derived signs of its road
    assert MapArt(sim.m, 4.0, skip_speed_roads={"0"}).skip_speed_roads == {"0"}
    from tsf.sim.static import explicit_speed_roads
    assert explicit_speed_roads(sim.static) == {"0"}


# --- augmentation: environment variants stay ahead of the ego and do not move the scene -------------------------------
def _uc1() -> dict:
    return json.loads((ROOT / "examples" / "uc1_left_turn_intersection.json").read_text())


def _with(doc: dict, *objs) -> dict:
    out = json.loads(json.dumps(doc))
    out["static_objects"] = [dict(o) for o in objs]
    return out


def _sign(oid, s, **kw) -> dict:
    return {"id": oid, "kind": "traffic_sign", "sign_type": "road_works",
            "position": {"road": 2, "direction": "inbound", "lane": 2, "s": s, "t_offset": -2.6}, **kw}


def test_scene_sanity_flags_objects_behind_the_ego():
    from tsf.agents import augment_patch as P

    parent = _uc1()                                              # ego on arm 2, inbound, s=140 (s shrinks while driving)
    ahead, behind = _with(parent, _sign("a", 100.0)), _with(parent, _sign("b", 170.0))
    assert not [v for v in P.isolation(parent, ahead, ["environment"]).violations if "behind the ego" in v["why"]]
    bad = [v for v in P.isolation(parent, behind, ["environment"]).violations if "behind the ego" in v["why"]]
    assert len(bad) == 1 and "30 m behind" in bad[0]["why"] and "s < 140" in bad[0]["why"]
    other_arm = _with(parent, {**_sign("c", 170.0), "position": {"road": 1, "direction": "inbound", "lane": 1, "s": 170.0}})
    assert not [v for v in P.isolation(parent, other_arm, ["environment"]).violations if "behind the ego" in v["why"]]


def test_scene_sanity_rejects_moving_agents_to_make_room():
    from tsf.agents import augment_patch as P

    parent = _uc1()
    child = _with(parent, _sign("a", 100.0))
    child["actors"][0]["initial"]["s"] = 185.0                   # the ego start moved back: a behaviour change
    child["actors"][1]["initial"]["lane"] = 1                    # lane 2 still exists, so no reason to move the van
    why = [v["why"] for v in P.isolation(parent, child, ["environment"]).violations]
    assert sum("behaviour change" in w for w in why) == 2
    child2 = _with(parent, _sign("a", 100.0))
    child2["road"]["approaches"][2]["inbound"]["driving_lanes"] = 1       # lane 2 vanishes: the actors must move
    child2["actors"][0]["initial"]["lane"] = 1
    assert not [v for v in P.isolation(parent, child2, ["environment"]).violations if "behaviour change" in v["why"]]


def test_index_table_names_the_ego_start():
    from tsf.agents import augment_patch as P

    row = [r for r in P.index_table(_uc1()).splitlines() if r.startswith("ego start")][0]
    assert "s=140" in row and "s < 140" in row and "Do not move the ego" in row
