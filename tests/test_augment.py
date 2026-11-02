"""Scenario augmentation: patch / isolation / planning units (no LLM) and end-to-end runs with a scripted editor."""
import copy
import json
import re
import time
from pathlib import Path

import pytest

from tsf.agents import augment_patch as P
from tsf.agents import augment_plan as PL
from tsf.agents.llm import CallRecord

ROOT = Path(__file__).resolve().parents[1]
UC1 = json.loads((ROOT / "examples" / "uc1_left_turn_intersection.json").read_text())
HWY = json.loads((ROOT / "examples" / "double_cut_in_highway.json").read_text())


def norm(d):
    from tsf.agents.augment import normalise
    return normalise(d)


# ----------------------------------------------------------------------------- patch + diff
def test_apply_patch_ops():
    d = {"a": {"x": 1}, "l": [{"id": "p"}, {"id": "q"}]}
    out = P.apply_patch(d, [{"op": "replace", "path": "/a/x", "value": 2}, {"op": "add", "path": "/a/y", "value": 3},
                            {"op": "add", "path": "/l/-", "value": {"id": "r"}}, {"op": "remove", "path": "/l/0"},
                            {"op": "add", "path": "/l/0", "value": {"id": "s"}}])
    assert out["a"] == {"x": 2, "y": 3} and [x["id"] for x in out["l"]] == ["s", "q", "r"]
    assert d["a"]["x"] == 1                                   # the input is not mutated


@pytest.mark.parametrize("op", [{"op": "replace", "path": "/nope", "value": 1}, {"op": "remove", "path": "/a/zz"},
                                {"op": "add", "path": "/l/9", "value": 1}, {"op": "add", "path": "/q/r", "value": 1},
                                {"op": "move", "path": "/a", "value": 1}, {"op": "replace", "path": "/a/x"}])
def test_apply_patch_errors(op):
    with pytest.raises(P.PatchError):
        P.apply_patch({"a": {"x": 1}, "l": [1]}, [op])


def test_diff_aligns_lists_by_id():
    a = {"actions": [{"id": "a1", "v": 1}, {"id": "a2", "v": 2}, {"id": "a3", "v": 3}]}
    b = {"actions": [{"id": "a1", "v": 1}, {"id": "a3", "v": 4}]}
    ch = {c.path: c.op for c in P.diff(a, b)}
    assert ch == {"/actions/1": "remove", "/actions/1/v": "replace"}       # a2 removed, a3 (now index 1) changed
    assert any(c.op == "remove" and c.before["id"] == "a2" for c in P.diff(a, b))


def test_write_pattern_grammar():
    w = re.compile(P.write_pattern(["weather"]))
    assert w.match("/environment/weather") and not w.match("/environment/setting") and not w.match("/meta/id")
    b = re.compile(P.write_pattern(["behavior"]))
    for ok in ("/actors/3/initial_speed", "/actors/-", "/actions/-", "/actions/2/start/t", "/actors/0/initial/s", "/duration"):
        assert b.match(ok), ok
    for bad in ("/actors/0/color", "/actors/0/category", "/road/forward", "/environment/weather", "/meta/id"):
        assert not b.match(bad), bad
    e = re.compile(P.write_pattern(["environment"]))
    assert e.match("/road/approaches/2/inbound/driving_lanes") and e.match("/environment/setting") and not e.match("/actions/-")


# ----------------------------------------------------------------------------- isolation
def test_isolation_weather_only():
    p = norm(UC1)
    c = P.apply_patch(p, [{"op": "replace", "path": "/environment/weather", "value": "rain"}])
    assert P.isolation(p, c, ["weather"]).ok
    c2 = P.apply_patch(c, [{"op": "replace", "path": "/actors/1/color", "value": "red"}])
    iso = P.isolation(p, c2, ["weather"])
    assert not iso.ok and iso.violations[0]["path"] == "/actors/1/color"


def test_isolation_dimension_must_change_and_union():
    p = norm(UC1)
    same = copy.deepcopy(p)
    assert P.isolation(p, same, ["behavior"]).untouched == ["behavior"]
    c = P.apply_patch(p, [{"op": "replace", "path": "/environment/weather", "value": "fog"},
                          {"op": "replace", "path": "/actors/0/initial_speed", "value": 9.0}])
    assert not P.isolation(p, c, ["weather"]).ok                        # speed is D2
    assert P.isolation(p, c, ["weather", "behavior"]).ok                # union in manual combined mode
    assert P.isolation(p, c, ["weather", "behavior", "environment"]).untouched == ["environment"]


def test_isolation_added_npc_and_ego():
    p = norm(UC1)
    npc = {"id": "ped1", "role": "npc", "category": "pedestrian", "initial": {"road": 1, "direction": "outbound", "lane": 1, "s": 5.0}}
    c = P.apply_patch(p, [{"op": "add", "path": "/actors/-", "value": npc}])
    assert P.isolation(p, c, ["behavior"]).ok
    c = P.apply_patch(p, [{"op": "add", "path": "/actors/-", "value": {**npc, "role": "ego"}}])
    assert not P.isolation(p, c, ["behavior"]).ok


def test_housekeeping_is_exempt():
    p = norm(UC1)
    c = copy.deepcopy(p)
    c["meta"]["id"] = "x"
    c["meta"]["tags"] = ["augmentation"]
    c["environment"]["weather"] = "snow"
    iso = P.isolation(p, c, ["weather"])
    assert iso.ok and iso.housekeeping


# ----------------------------------------------------------------------------- planning
def test_weather_draw_is_seeded_different_and_consistent():
    env = norm(UC1)["environment"]
    seen = set()
    for seed in range(120):
        t = PL.draw_weather(env, seed)
        assert t == PL.draw_weather(env, seed)
        assert (t["weather"], t["band"]) != ("clear", "day")           # never equal to the parent's
        assert PL.time_band(t["environment"]["time_of_day"]) == t["band"]
        assert PL.env_consistency(t["environment"]) == [], (seed, t)
        seen.add((t["weather"], t["band"]))
    assert len(seen) >= 12                                               # 17 combinations exist, the draw covers them


def test_env_consistency_flags_contradictions():
    bad = {"weather": "heavy_rain", "precipitation": 0.0, "fog_visibility_m": 100000.0, "cloud_cover": 0.1, "road_surface": "dry"}
    assert len(PL.env_consistency(bad)) >= 3
    assert PL.env_consistency({"weather": "clear", "precipitation": 0.0, "fog_visibility_m": 100000.0, "cloud_cover": 0.1,
                               "road_surface": "dry"}) == []


def test_levers_are_scenario_specific():
    p = norm(UC1)
    ids = [i for i, _ in PL.behavior_levers(p)]
    assert "ego_faster" in ids and any(i.startswith("brake_oncoming_1") for i in ids) and "add_pedestrian" in ids
    assert not any(i.startswith("runs_red") for i in ids)               # TanitSim traffic always obeys the signals
    ids3 = [i for i, _ in PL.environment_levers(p, static_objects=False)]
    assert "more_lanes" in ids3 and "road_works" not in ids3
    assert "road_works" in [i for i, _ in PL.environment_levers(p, static_objects=True)]
    hw = [i for i, _ in PL.behavior_levers(norm(HWY))]
    assert "add_lead_brake" in hw and "add_cut_in" in hw


def test_description_check_names_changed_attributes():
    p = norm(UC1)
    c = copy.deepcopy(p)
    c["environment"].update(weather="heavy_rain", time_of_day="22:10", road_surface="wet")
    attrs = PL.expected_attributes(p, c, ["weather"])
    assert [a["attr"].split(" ->")[0] for a in attrs] == ["weather", "time of day"]
    assert PL.check_description("A night drive in heavy rain.", attrs) == []
    assert PL.check_description("The ego turns left.", attrs) == [a["attr"] for a in attrs]


def test_numeric_levers_and_retune():
    from tsf.agents.augment import numeric_levers, retune_assertions
    p = norm(UC1)
    c = P.apply_patch(p, [{"op": "replace", "path": "/actors/0/initial_speed", "value": 16.0},
                          {"op": "replace", "path": "/actions/1/start/distance", "value": 30.0},
                          {"op": "replace", "path": "/environment/weather", "value": "fog"}])
    ps = {x["path"]: x for x in numeric_levers(p, c)}
    assert set(ps) == {"/actors/0/initial_speed", "/actions/1/start/distance"}
    assert 9.0 < ps["/actors/0/initial_speed"]["low"] < 16 < ps["/actors/0/initial_speed"]["high"]
    d = copy.deepcopy(c)
    ch = retune_assertions(d, {"min_ttc": 1.2, "ego_max_decel": 5.2})
    assert {x["path"] for x in ch} == {"/assertions/3/value", "/assertions/5/value"}
    assert d["assertions"][3]["value"] <= 1.1 and d["assertions"][5]["value"] >= 5.2


# ----------------------------------------------------------------------------- end to end with a scripted editor
class FakeLLM:
    """Scripted stand-in for the Qwen client: patches per requested dimension, descriptions echo the required terms."""

    def __init__(self, *a, **k):
        self.calls: list[CallRecord] = []

    def chat(self, messages, schema=None, role="agent", **kw):
        self.calls.append(CallRecord(role, 0.01, 100, 50, True))
        props = (schema or {}).get("properties", {})
        user = messages[1]["content"]
        if "ops" in props:
            return self.edit(user, props)
        if "description" in props:
            attrs = re.search(r"plain words: (.*)\.$", user.strip().splitlines()[-1]) or re.search(r"plain words: (.*)\.", user)
            return {"title": "Variant: " + (attrs.group(1)[:60] if attrs else "changed"),
                    "description": "This variant changes " + (attrs.group(1) if attrs else "the scenario") + "."}
        raise AssertionError(f"unexpected schema {list(props)}")

    def edit(self, user, props):
        parent = json.loads(re.search(r"## Parent scenario[^\n]*\n(\{.*\})\n", user).group(1))
        lever = props["lever"]["enum"][0]
        ops = []
        if "D1 weather" in user:
            ops += [{"op": "replace", "path": "/environment/weather", "value": "rain"},
                    {"op": "replace", "path": "/environment/precipitation", "value": 0.4},
                    {"op": "replace", "path": "/environment/cloud_cover", "value": 0.9},
                    {"op": "replace", "path": "/environment/fog_visibility_m", "value": 6000},
                    {"op": "replace", "path": "/environment/road_surface", "value": "wet"},
                    {"op": "replace", "path": "/environment/time_of_day", "value": "22:30"}]
        if "D2 behaviour" in user:
            ops += [{"op": "replace", "path": "/actors/0/initial_speed", "value": round(parent["actors"][0]["initial_speed"] * 1.1, 2)}]
        if "D3 environment" in user:
            if parent["road"]["type"] == "junction":
                ops += [{"op": "replace", "path": "/road/approaches/0/speed_limit_kph", "value": 30}]
            else:
                ops += [{"op": "replace", "path": "/road/speed_limit_kph", "value": 80}]
        return {"lever": lever, "rationale": "scripted", "limitations": "", "ops": ops}

    def stats(self):
        return {"calls": len(self.calls), "failed": 0, "latency_s": 0.1, "prompt_tokens": 100 * len(self.calls),
                "completion_tokens": 50 * len(self.calls), "tok_per_s": 500.0}


def test_auto_three_variants_one_per_dimension(tmp_path):
    from tsf.agents.augment import augment
    res = augment(ROOT / "examples" / "uc1_left_turn_intersection.json", tmp_path, mode="auto", seed=3, render=False,
                  criticality=False, package=False, llm=FakeLLM, parent_qc=False)
    assert [r["dimensions"] for r in res] == [["weather"], ["behavior"], ["environment"]]
    assert all(r["status"] == "ok" and r["isolation"]["ok"] and r["verifier"]["passed"] for r in res)
    w = res[0]
    assert w["lever"] == "plan" and w["tokens"]["calls"] == 1 and w["first_pass_valid"]       # D1 patch comes from the plan
    final = json.loads(Path(w["paths"]["tsir"]).read_text())
    assert final["meta"]["parent_id"] == "uc1_left_turn_intersection" and "augmentation" in final["meta"]["tags"]
    assert (final["environment"]["weather"], PL.time_band(final["environment"]["time_of_day"])) != ("clear", "day")
    assert all(r["description_check"]["ok"] for r in res)
    assert json.loads((tmp_path / "summary.json").read_text())["variants"][1]["id"] == res[1]["id"]
    assert Path(w["paths"]["xodr"]).exists() and Path(w["paths"]["xosc"]).exists() and Path(w["paths"]["osc2"]).exists()


def test_manual_combined_and_separate(tmp_path):
    from tsf.agents.augment import augment
    req = {"weather": "rain at night", "behavior": "ego faster", "environment": "lower the speed limit"}
    one = augment(HWY, tmp_path / "c", mode="manual", requests=req, render=False, criticality=False, package=False,
                  llm=FakeLLM, parent_qc=False)
    assert len(one) == 1 and one[0]["dimensions"] == ["weather", "behavior", "environment"] and one[0]["status"] == "ok"
    assert {c["path"] for c in one[0]["diff"]} >= {"/environment/weather", "/actors/0/initial_speed", "/road/speed_limit_kph"}
    sep = augment(HWY, tmp_path / "s", mode="manual", requests=req, separate=True, render=False, criticality=False,
                  package=False, llm=FakeLLM, parent_qc=False)
    assert [r["dimensions"] for r in sep] == [["weather"], ["behavior"], ["environment"]]
    assert all(r["status"] == "ok" for r in sep)
    with pytest.raises(ValueError):
        augment(HWY, tmp_path / "e", mode="manual", requests={}, render=False, llm=FakeLLM, parent_qc=False)


def test_isolation_violation_is_repaired(tmp_path):
    """The editor first touches a field outside its whitelist; the isolation message drives the repair."""
    from tsf.agents.augment import augment

    class Sloppy(FakeLLM):
        n = 0

        def edit(self, user, props):
            Sloppy.n += 1
            out = super().edit(user, props)
            if Sloppy.n == 1:
                out["ops"].append({"op": "replace", "path": "/actors/1/initial_speed", "value": 5.0})   # D2 field in a D1 request
            return out

    res = augment(HWY, tmp_path, mode="manual", requests={"weather": "rain at night"}, render=False, criticality=False,
                  package=False, llm=Sloppy, parent_qc=False)[0]
    assert res["status"] == "ok" and res["first_pass_valid"] is False and res["repairs"] == 1
    assert res["attempts"][0]["stage"] == "isolation" and "/actors/1/initial_speed" in res["attempts"][0]["issues"][0]


def test_augment_endpoint(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    monkeypatch.setenv("TSF_RUNS", str(tmp_path))
    import importlib
    import tsf.server.runs as runs
    importlib.reload(runs)
    import tsf.server.app as appmod
    importlib.reload(appmod)
    import tsf.agents.augment as A
    monkeypatch.setattr(A, "LLM", FakeLLM)
    with TestClient(appmod.app) as c:
        r = c.post("/api/runs", json={"uc": "uc1", "input": {"example": "double_cut_in_highway"},
                                      "options": {"video": False, "package": False}}).json()
        for _ in range(240):
            d = c.get(f"/api/runs/{r['id']}").json()
            if d["status"] in ("ok", "failed"):
                break
            time.sleep(0.5)
        assert d["status"] == "ok", d.get("error")
        assert c.post(f"/api/runs/{r['id']}/augment", json={"mode": "manual", "requests": {}}).status_code == 422
        assert c.post("/api/runs/run_nope/augment", json={}).status_code == 404
        k = c.post(f"/api/runs/{r['id']}/augment", json={"mode": "auto", "seed": 1, "render": False, "criticality": False,
                                                          "package": False}).json()
        assert k["uc"] == "augment"
        for _ in range(480):
            j = c.get(f"/api/runs/{k['id']}").json()
            if j["status"] in ("ok", "failed"):
                break
            time.sleep(0.5)
        assert j["status"] == "ok", j.get("error")
        vs = j["augment"]["variants"]
        assert [v["dimensions"] for v in vs] == [["weather"], ["behavior"], ["environment"]]
        assert all(v["status"] == "ok" and v["isolation"]["ok"] and v["media"]["tsir"] for v in vs)
        assert c.get(f"/api/runs/{k['id']}/artifacts/{vs[0]['media']['xodr']}").status_code == 200
        parent = c.get(f"/api/runs/{r['id']}").json()
        assert [a["id"] for a in parent["augmentations"]] == [k["id"]]


# ----------------------------------------------------------------------------- Harness² probes and scoring
def test_probe_set_and_intent_checks():
    from tsf.agents import augment_harness as H
    probes = H.load_probes()
    assert 6 <= len(probes) <= 8 and len({p["id"] for p in probes}) == len(probes)
    assert {p["parent"] for p in probes} == {"examples/uc1_left_turn_intersection.json", "examples/double_cut_in_highway.json"}
    child = copy.deepcopy(HWY)
    child["environment"].update(weather="heavy_rain", time_of_day="22:00")
    child["road"]["speed_limit_kph"] = 70
    child["actors"][1]["initial_speed"] += 3
    checks = dict(H.intent_checks({"env": {"weather": ["heavy_rain"], "band": "night"},
                                   "road": [{"path": "/road/speed_limit_kph", "max": 80}],
                                   "actors_touched": [child["actors"][1]["id"]]}, HWY, child))
    assert checks and all(checks.values()), checks
    assert not all(ok for _, ok in H.intent_checks({"env": {"weather": ["fog"]}}, HWY, child))
    assert not all(ok for _, ok in H.intent_checks({"new_actors_min": 1}, HWY, child))


def test_probe_evaluation_scores_first_attempts(tmp_path):
    from tsf.agents.augment_harness import AugmentEvaluator
    from tsf.agents.harness import ROOT as HROOT
    probes = [{"id": "hwy_rain", "parent": "examples/double_cut_in_highway.json", "requests": {"weather": "light rain at night"},
               "expect": {"env": {"weather": ["rain"], "band": "night"}}},
              {"id": "hwy_snow", "parent": "examples/double_cut_in_highway.json", "requests": {"weather": "snow"},
               "expect": {"env": {"weather": ["snow"]}}}]
    ev = AugmentEvaluator(FakeLLM(), probes, tmp_path)
    r = ev.evaluate(HROOT, "t")
    assert set(r) >= {"tag", "score", "valid_rate", "rows"} and len(r["rows"]) == 2
    ok, bad = r["rows"]
    assert ok["passed"] and ok["score"] == 1.0                    # scripted editor writes rain at 22:30
    assert not bad["passed"] and 0.7 <= bad["score"] < 1.0 and "intent not met" in bad["diag"]
    assert r["valid_rate"] == 0.5


class PkgLLM(FakeLLM):
    """Adds the Lifter's analysis and documentation steps to the scripted editor."""

    def chat(self, messages, schema=None, role="agent", **kw):
        props = (schema or {}).get("properties", {})
        if "analysisOfUserQuery" in props:
            self.calls.append(CallRecord(role, 0.01, 100, 50, True))
            return {"analysisOfUserQuery": {"extractedInformation": {"scenarioGoal": "goal", "actors": [], "road": {}, "environment": {}},
                                            "variables": [], "semanticMeaning": "parent meaning"},
                    "planToDefineScenario": ["a", "b", "c"],
                    "requiredAssumptionsAndMissingValues": {"assumptions": ["x"], "missingValues": []},
                    "mathematicalCorpus": ["TTC = d / v", "a = v^2 / 2d"]}
        if "functionalRequirementsSummary" in props:
            self.calls.append(CallRecord(role, 0.01, 100, 50, True))
            return {"functionalRequirementsSummary": "s", "improvementsSummary": ["none"],
                    "codeBeamerScenarioDescription": {
                        "shortScenarioDescription": {"title": "t", "keywords": ["k"], "tags": ["t"]}, "scenarioCategory": "c",
                        "initialStates": [], "finalStates": [], "environmentConditions": {}, "roadDescription": {},
                        "staticEnvironment": [], "dynamicEnvironment": [], "events": [],
                        "behaviorSequence": {"ego": ["drive"], "agents": []}},
                    "derivedFunctionalRequirements": [
                        {"id": "REQ-EGO-A-01", "requirement": "r", "passCriteria": ["p"]},
                        {"id": "REQ-EGO-A-02", "requirement": "r", "passCriteria": ["p"]}],
                    "scenarioVariations": {"description": "d", "variableIntervals": []}}
        return super().chat(messages, schema, role, **kw)


def test_package_and_imdad_record_of_variants(tmp_path):
    """The analysis is computed once and shared; every variant gets a Scenario_Structure package + IMDAD v2 record."""
    from tsf.agents.augment import augment
    made = []

    def factory():
        made.append(PkgLLM())
        return made[-1]

    res = augment(HWY, tmp_path, mode="auto", seed=2, render=False, criticality=False, package=True, llm=factory,
                  parent_qc=False)
    assert [r["status"] for r in res] == ["ok"] * 3
    for r in res:
        assert r["package"]["imdad_record_valid"], r["package"]
        assert r["package"]["scenario_structure_valid"], r["package"]
        rec = json.loads(Path(r["paths"]["imdad_record"]).read_text())
        assert rec["lineage"]["origin_type"] == "SYNTHETIC_AUGMENTATION" and rec["lineage"]["is_synthetic_augmentation"]
        assert rec["lineage"]["parent_incident_id"] == "double_cut_in_highway"
        pkg = json.loads(Path(r["paths"]["package"]).read_text())
        assert pkg["groundTruthAnswer"]["assumptionsAndFilledMissedValues"]["assumptions"][-1].startswith("Augmentation (")
    analyses = [c for m in made for c in m.calls if c.role == "analyst"]
    assert len(analyses) == 1                                                       # shared by the three variants
    assert json.loads((tmp_path / "summary.json").read_text())["shared_analysis_tokens"]["calls"] == 1


class JudgeLLM(FakeLLM):
    """The camera judge always sees a clear day: agrees with a parent-like variant, disagrees with a weather variant."""

    def chat(self, messages, schema=None, role="agent", **kw):
        props = (schema or {}).get("properties", {})
        if "headlights_or_street_lights_on" in props:
            self.calls.append(CallRecord(role, 0.01, 100, 50, True))
            return {"weather": "clear", "time_of_day": "day", "road_wet_or_snowy": False,
                    "headlights_or_street_lights_on": False, "objects_visible": {}, "note": "fake"}
        return super().chat(messages, schema, role, **kw)


def test_render_wiring_with_fake_cosmos(tmp_path, monkeypatch):
    """Parent rendered once, same seed everywhere, D1 reuses the parent geometry, a failed camera judgement re-renders once."""
    import tsf.agents.video as video
    import tsf.cosmos.pipeline as pipe
    from tsf.agents import augment_render as R
    from tsf.agents.augment import augment
    calls = []

    def fake_render(scenario, out_dir, **kw):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        calls.append({"scenario": scenario["meta"]["id"], "out": out.name, **kw})
        for n in ("mosaic.mp4", "front.mp4", "lidar_bev.mp4"):
            (out / n).write_bytes(b"x")
        (out / "sensors.json").write_text("{}")
        return {"mosaic": str(out / "mosaic.mp4"), "videos": {"front_wide": str(out / "front.mp4")},
                "lidar": {"enabled": True, "bev_video": str(out / "lidar_bev.mp4"),
                          "checks_summary": {"summary": {"sweeps": 3, "mean_points_per_sweep": 90000}, "objects": {}, "warnings": []}}}

    monkeypatch.setattr(pipe, "ensure_server", lambda *a, **k: {"started": False, "server": "tsf-cosmos-sim"})
    monkeypatch.setattr(pipe, "render_sensors", fake_render)
    monkeypatch.setattr(R, "free_gb", lambda: 100.0)
    monkeypatch.setattr(R, "_running_cosmos", lambda: [])
    monkeypatch.setattr(video, "keyframes", lambda path, n, width=896: [(1.0, "AAAA"), (2.0, "AAAA")])
    res = augment(HWY, tmp_path, mode="auto", seed=4, render=True, criticality=False, package=False, llm=JudgeLLM,
                  parent_qc=False)
    assert [r["status"] for r in res] == ["ok"] * 3
    assert calls[0]["scenario"] == "double_cut_in_highway" and calls[0]["out"] == "render"             # the parent, first
    assert sum(c["scenario"] == "double_cut_in_highway" and "geometry_dir" not in c for c in calls) == 1   # ... and only once
    assert {c["seed"] for c in calls} == {4, 4 + 101}
    w, b, e = res
    assert w["render"]["geometry_reused"] is True and w["judge"]["camera"]["ok"] is False
    assert w["judge"]["rerender"] and w["judge"]["ok"] is False and w["timings"]["render_s"] >= 0
    assert [c["seed"] for c in calls if c["out"] == "render_retry"] == [4 + 101]
    assert b["judge"]["camera"]["ok"] and b["judge"]["lidar"]["ok"] and b["judge"]["ok"] and not b["judge"]["rerender"]
    assert b["render"]["geometry_reused"] is False and e["render"]["geometry_reused"] is False   # D2 / D3 change the geometry
    assert b["render"]["files"]["mosaic"].endswith("mosaic.mp4") and b["judge"]["lidar"]["numbers"]["sweeps"] == 3


def test_package_failure_does_not_lose_the_verified_variant(tmp_path):
    """The LLM dying during the package step leaves the verified scenario files as the variant's result."""
    from tsf.agents.augment import augment

    class Dying(PkgLLM):
        def chat(self, messages, schema=None, role="agent", **kw):
            if "functionalRequirementsSummary" in (schema or {}).get("properties", {}):
                raise RuntimeError("EngineCore encountered an issue")
            return super().chat(messages, schema, role, **kw)

    res = augment(HWY, tmp_path, mode="manual", requests={"weather": "rain at night"}, render=False, criticality=False,
                  package=True, llm=Dying, parent_qc=False)[0]
    assert res["status"] == "ok" and res["verifier"]["passed"] and Path(res["paths"]["xodr"]).exists()
    assert res["package"]["imdad_record_valid"] is False and "EngineCore" in res["package"]["error"]
    assert "package" not in res["paths"]


def test_render_never_takes_over_a_foreign_cosmos_server(tmp_path, monkeypatch):
    import tsf.cosmos.pipeline as pipe
    from tsf.agents import augment_render as R
    called = []
    monkeypatch.setattr(pipe, "ensure_server", lambda *a, **k: called.append("ensure") or {"started": False})
    monkeypatch.setattr(pipe, "render_sensors", lambda *a, **k: called.append("render") or {})
    monkeypatch.setattr(R, "_running_cosmos", lambda: ["tsf-cosmos-sim"])
    with pytest.raises(Exception, match="Cosmos server is running"):
        R.render(HWY, tmp_path, seed=0, t0=0.0, wait_s=0.0, log=lambda m: None)
    assert called == []                                            # neither ensure_server nor the render ran
    # a server this job started itself (state flag) is reused for the next render of the same job
    state = {"started_server": True}
    monkeypatch.setattr(R, "free_gb", lambda: 100.0)
    R.render(HWY, tmp_path, seed=0, t0=0.0, wait_s=0.0, log=lambda m: None, state=state)
    assert called == ["ensure", "render"]


def test_ablation_drops_the_op_that_breaks_the_scenario_intent(tmp_path):
    """The editor insists on closing a lane the ego needs for its lane change; ablation keeps the speed limit only."""
    from tsf.agents.augment import augment

    class Stubborn(FakeLLM):
        def edit(self, user, props):
            return {"lever": "custom", "rationale": "close the right lane and slow down", "limitations": "",
                    "ops": [{"op": "replace", "path": "/road/approaches/2/inbound/driving_lanes", "value": 1},
                            {"op": "replace", "path": "/road/approaches/2/speed_limit_kph", "value": 30}]}

    res = augment(ROOT / "examples" / "uc1_left_turn_intersection.json", tmp_path, mode="manual",
                  requests={"environment": "road works close the right lane, 30 km/h"}, render=False, criticality=False,
                  package=False, max_repairs=1, llm=Stubborn, parent_qc=False)[0]
    assert res["status"] == "ok" and res["verifier"]["passed"] and res["isolation"]["ok"]
    assert res["lever"] == "ablated" and res["first_pass_valid"] is False and res["repairs"] == 2
    assert [c["path"] for c in res["diff"]] == ["/road/approaches/2/speed_limit_kph"]
    assert "driving_lanes" in res["limitations"] and "Dropped" in res["limitations"]
