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
                                      "options": {"video": False, "augment": {"n_seed": 4, "n_iter": 1, "n_per_iter": 4},
                                                  "closed_loop": {"policy": "const_vel"}}}).json()
        for _ in range(240):
            d = c.get(f"/api/runs/{r['id']}").json()
            if d["status"] in ("ok", "failed"):
                break
            time.sleep(0.5)
        assert d["status"] == "ok", d.get("error")
        assert {s["name"]: s["status"] for s in d["stages"]}["closed_loop"] == "ok"
        sc = c.get(f"/api/runs/{r['id']}/scene").json()
        assert sc["actors"][0]["role"] in ("ego", "npc") and len(sc["t"]) == len(sc["tracks"]["ego"])
        assert c.get(f"/api/runs/{r['id']}/artifacts/scenario.xodr").status_code == 200
        assert c.get("/api/kpis").json()["latest"]["epdms"] > 0
        assert c.post("/api/runs", json={"uc": "uc9"}).status_code == 501
