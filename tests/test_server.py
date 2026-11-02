import time

from fastapi.testclient import TestClient


def test_example_run(tmp_path, monkeypatch):
    monkeypatch.setenv("TSF_RUNS", str(tmp_path))
    import importlib
    import tsf.server.runs as runs
    importlib.reload(runs)
    import tsf.server.app as appmod
    importlib.reload(appmod)
    with TestClient(appmod.app) as c:
        assert c.get("/api/health").status_code == 200
        r = c.post("/api/runs", json={"uc": "uc1", "input": {"example": "double_cut_in_highway"},
                                      "options": {"video": False, "package": False, "augment": {"n_seed": 4, "n_iter": 1, "n_per_iter": 4},
                                                  "closed_loop": {"policy": "const_vel"}}}).json()
        for _ in range(240):
            d = c.get(f"/api/runs/{r['id']}").json()
            if d["status"] in ("ok", "failed"):
                break
            time.sleep(0.5)
        assert d["status"] == "ok", d.get("error")
        assert {s["name"]: s["status"] for s in d["stages"]}["closed_loop"] == "ok"
        k = d["kpis"]
        assert 0.0 <= k["epdms_scenario"] <= 1.0 and k["epdms_policy"] == k["epdms"]       # scenario ego vs test policy
        sc = c.get(f"/api/runs/{r['id']}/scene").json()
        assert sc["actors"][0]["role"] in ("ego", "npc") and len(sc["t"]) == len(sc["tracks"]["ego"])
        assert c.get(f"/api/runs/{r['id']}/artifacts/scenario.xodr").status_code == 200
        assert c.get("/api/kpis").json()["latest"]["epdms"] > 0
        assert c.post("/api/runs", json={"uc": "uc9"}).status_code == 501


def test_example_run_builds_package_and_skips_sensors(tmp_path, monkeypatch):
    """A run from an example TSIR also yields the Scenario_Structure package; the sensors stage is opt-in."""
    monkeypatch.setenv("TSF_RUNS", str(tmp_path))
    import importlib
    import tsf.agents.lifter as lifter
    import tsf.server.runs as runs
    importlib.reload(runs)
    import tsf.server.app as appmod
    importlib.reload(appmod)

    import threading
    gate = threading.Event()

    class FakeLLM:
        def stats(self):
            return {"calls": 2, "tok_per_s": 10.0}

    class FakeLifter:
        def __init__(self, *a, **k):
            self.llm = FakeLLM()

        def analyse(self, q):
            gate.wait(60)                       # the verified TSIR must be downloadable while the package is written
            return {"analysisOfUserQuery": {"extractedInformation": {"scenarioGoal": "g", "actors": [], "road": {}, "environment": {}},
                                            "variables": [], "semanticMeaning": "m"},
                    "planToDefineScenario": ["a", "b", "c"],
                    "requiredAssumptionsAndMissingValues": {"assumptions": ["x"], "missingValues": []},
                    "mathematicalCorpus": ["TTC", "a"]}

        def document(self, q, analysis, attempts, final):
            return {"functionalRequirementsSummary": "s", "improvementsSummary": [],
                    "codeBeamerScenarioDescription": {
                        "shortScenarioDescription": {"title": "t", "keywords": ["k"], "tags": ["t"]}, "scenarioCategory": "c",
                        "initialStates": [], "finalStates": [], "environmentConditions": {}, "roadDescription": {},
                        "staticEnvironment": [], "dynamicEnvironment": [], "events": [],
                        "behaviorSequence": {"ego": ["drive"], "agents": []}},
                    "derivedFunctionalRequirements": [{"id": "REQ-EGO-A-01", "requirement": "r", "passCriteria": ["p"]},
                                                      {"id": "REQ-EGO-A-02", "requirement": "r", "passCriteria": ["p"]}],
                    "scenarioVariations": {"description": "d", "variableIntervals": []}}

    monkeypatch.setattr(lifter, "Lifter", FakeLifter)
    with TestClient(appmod.app) as c:
        r = c.post("/api/runs", json={"uc": "uc1", "input": {"example": "double_cut_in_highway"},
                                      "options": {"video": False, "augment": {"n_seed": 4, "n_iter": 1, "n_per_iter": 4},
                                                  "closed_loop": {"policy": "const_vel"}}}).json()
        # 409 = no verified TSIR yet; 422 (empty manual request) = past that check although the package is still being written
        for _ in range(240):
            code = c.post(f"/api/runs/{r['id']}/augment", json={"mode": "manual", "requests": {}}).status_code
            if code != 409:
                break
            time.sleep(0.25)
        assert code == 422 and c.get(f"/api/runs/{r['id']}").json()["status"] == "running", code
        gate.set()
        for _ in range(240):
            d = c.get(f"/api/runs/{r['id']}").json()
            if d["status"] in ("ok", "failed"):
                break
            time.sleep(0.5)
        assert d["status"] == "ok", d.get("error")
        st = {s["name"]: s["status"] for s in d["stages"]}
        assert st["analyse"] == "ok" and st["document"] == "ok" and st["sensors"] == "skipped" and st["closed_loop"] == "ok"
        assert d["kpis"]["package_valid"] is True
        assert c.get(f"/api/runs/{r['id']}/artifacts/package.json").status_code == 200


def test_studio_render_options_start_the_sensors_stage():
    """Studio's Rendering panel sends options.render (docs/API.md); the server maps it onto the sensors stage."""
    from tsf.server.runs import _sensors_from_render
    assert _sensors_from_render(None) is None and _sensors_from_render({}) is None
    so = _sensors_from_render({"views": 3, "res": "720p", "frames": 121, "lidar": False})
    assert so == {"cameras": ["camera_front_wide_120fov", "camera_cross_left_120fov", "camera_cross_right_120fov"],
                  "frames": 121, "lidar": False, "res": "720"}
    assert len(_sensors_from_render({"views": 99})["cameras"]) == 7
