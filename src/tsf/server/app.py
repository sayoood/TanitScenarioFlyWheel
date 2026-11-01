"""Flywheel API (docs/API.md) on thor3:8080. Run: `tsf serve` or `uvicorn tsf.server.app:app --host 0.0.0.0 --port 8080`."""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import httpx
from fastapi import Body, FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from tsf.agents.harness import ROOT as HARNESS_ROOT, Harness
from tsf.server.runs import ROOT, RUNS, Store

LLM_URL = os.environ.get("TSF_LLM_URL", "http://127.0.0.1:8000/v1")
COSMOS = {"cosmos_sim": os.environ.get("TSF_COSMOS_SIM_URL", "http://127.0.0.1:8001"),
          "cosmos_transfer": os.environ.get("TSF_COSMOS_TRANSFER_URL", "http://127.0.0.1:8002")}
CONFIG_DEFAULT = ROOT / "config" / "pipeline.json"
CONFIG = RUNS / "config.json"


class Hub:
    """Fan-out of events from worker threads to WebSocket clients."""

    def __init__(self):
        self.subs: set[asyncio.Queue] = set()
        self.loop: asyncio.AbstractEventLoop | None = None

    def publish(self, ev: dict):
        if self.loop is None:
            return
        for q in list(self.subs):
            self.loop.call_soon_threadsafe(q.put_nowait, ev)


hub = Hub()
store = Store(publish=hub.publish)
app = FastAPI(title="Tanit Scenario Flywheel", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.on_event("startup")
async def _startup():
    hub.loop = asyncio.get_running_loop()
    asyncio.create_task(_health_ticker())


# ----------------------------------------------------------------------------- health
_cache: dict = {}


def _meminfo() -> dict:
    kv = {}
    for ln in Path("/proc/meminfo").read_text().splitlines():
        k, v = ln.split(":", 1)
        kv[k] = int(v.split()[0]) / 1024 ** 2
    return {"mem_total_gb": round(kv["MemTotal"], 1), "mem_used_gb": round(kv["MemTotal"] - kv["MemAvailable"], 1)}


def _gpu() -> dict:
    if time.time() - _cache.get("gpu_t", 0) < 5:
        return _cache["gpu"]
    out = {"gpu_util": None, "power_w": None}
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,power.draw", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=3)
        u, p = (x.strip() for x in r.stdout.splitlines()[0].split(","))
        out = {"gpu_util": float(u) / 100 if u.replace(".", "").isdigit() else None,
               "power_w": float(p) if p.replace(".", "").isdigit() else None}
    except (OSError, IndexError, ValueError, subprocess.TimeoutExpired):
        pass
    _cache.update(gpu=out, gpu_t=time.time())
    return out


async def _up(url: str) -> bool:
    try:
        async with httpx.AsyncClient(timeout=1.5) as c:
            return (await c.get(url)).status_code < 500
    except httpx.HTTPError:
        return False


def _last_tok_s() -> float | None:
    for r in store.list(20):
        v = (r.get("kpis") or {}).get("llm_tok_per_s")
        if v:
            return v
    return None


@app.get("/api/health")
async def health():
    llm_up, sim_up, tr_up = await asyncio.gather(_up(f"{LLM_URL}/models"), _up(COSMOS["cosmos_sim"] + "/health"),
                                                 _up(COSMOS["cosmos_transfer"] + "/health"))
    mem = _meminfo() if Path("/proc/meminfo").exists() else {}
    return {"thor": {"host": os.uname().nodename if hasattr(os, "uname") else "local", **mem,
                     **await asyncio.to_thread(_gpu),
                     "disk_free_gb": round(shutil.disk_usage(Path.home()).free / 1e9)},
            "services": {"llm": {"up": llm_up, "model": "qwen3.8-27b", "tok_s": _last_tok_s()},
                         "cosmos_sim": {"up": sim_up}, "cosmos_transfer": {"up": tr_up},
                         "scheduler": {"queue": len(store.queue), "running": store.running}}}


async def _health_ticker():
    while True:
        await asyncio.sleep(10)
        if hub.subs:
            try:
                hub.publish({"type": "health", **await health()})
            except Exception:  # noqa: BLE001 - never let the ticker die
                pass


# ----------------------------------------------------------------------------- runs
@app.get("/api/runs")
def list_runs(limit: int = Query(50, ge=1, le=500)):
    return store.list(limit)


@app.post("/api/runs")
def create_run(body: dict = Body(...)):
    try:
        return store.create(body.get("uc", "uc1"), body.get("input") or {}, body.get("options") or {})
    except NotImplementedError as e:
        raise HTTPException(501, f"use case {e} is not wired into the API yet") from e


def _run(rid: str) -> dict:
    try:
        return store.get(rid)
    except KeyError as e:
        raise HTTPException(404, f"run {rid} not found") from e


@app.get("/api/runs/{rid}")
def get_run(rid: str):
    run = _run(rid)
    pkg = store.path(rid) / "package.json"
    if pkg.exists():
        run["package"] = json.loads(pkg.read_text())
    videos = [a["name"] for a in run.get("artifacts", []) if a["kind"] == "video" and a["name"].startswith("cam_")]
    if videos:
        run["cameras"] = [{"name": v[4:-4], "artifact": v} for v in videos]
    return run


@app.get("/api/runs/{rid}/artifacts/{name}")
def get_artifact(rid: str, name: str):
    _run(rid)
    p = (store.path(rid) / name).resolve()
    if p.parent != store.path(rid) or not p.is_file():
        raise HTTPException(404, name)
    return FileResponse(p)


@app.get("/api/runs/{rid}/scene")
def get_scene(rid: str):
    _run(rid)
    p = store.path(rid) / "scene.json"
    if not p.exists():
        raise HTTPException(404, "scene not available (run has not simulated yet)")
    return FileResponse(p, media_type="application/json")


@app.get("/api/examples")
def examples():
    return [{"name": p.stem, "title": json.loads(p.read_text())["meta"]["title"]}
            for p in sorted((ROOT / "examples").glob("*.json"))]


# ----------------------------------------------------------------------------- events
@app.websocket("/api/events")
async def events(ws: WebSocket):
    await ws.accept()
    q: asyncio.Queue = asyncio.Queue(maxsize=1000)
    hub.subs.add(q)
    try:
        while True:
            await ws.send_json(await q.get())
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        hub.subs.discard(q)


# ----------------------------------------------------------------------------- harness
def _jsonl(p: Path) -> list[dict]:
    return [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()] if p.exists() else []


@app.get("/api/harness")
def harness():
    agents = sorted(p.name for p in HARNESS_ROOT.iterdir() if (p / "system_prompt.md").exists())
    return {"agents": agents, "knobs": {a: Harness(a).knobs() for a in agents},
            "lessons": _jsonl(HARNESS_ROOT / "EXPERIENCE.jsonl"), "revisions": _jsonl(HARNESS_ROOT / "revisions.jsonl"),
            "generations": _jsonl(HARNESS_ROOT / "generations.jsonl")}


@app.put("/api/harness/knobs")
def put_knobs(body: dict = Body(...)):
    h = Harness(body.get("agent", "lifter"))
    if not h.dir.exists():
        raise HTTPException(404, "unknown agent")
    knobs = {**h.knobs(), **(body.get("knobs") or {})}
    (h.dir / "knobs.json").write_text(json.dumps(knobs, indent=2))
    h.log_revision("HUMAN", "knobs.json", json.dumps(body.get("knobs")), "applied", {})
    return knobs


@app.post("/api/harness/lessons/{lid}")
def set_lesson(lid: str, body: dict = Body(...)):
    if body.get("status") not in ("active", "retired", "candidate"):
        raise HTTPException(422, "status must be active|retired|candidate")
    if not any(r["id"] == lid for r in _jsonl(HARNESS_ROOT / "EXPERIENCE.jsonl")):
        raise HTTPException(404, lid)
    Harness("lifter").set_lesson_status(lid, body["status"], {"human_gate": body["status"], "t": time.time()})
    return next(r for r in _jsonl(HARNESS_ROOT / "EXPERIENCE.jsonl") if r["id"] == lid)


# ----------------------------------------------------------------------------- KPIs + config
@app.get("/api/kpis")
def kpis():
    runs = sorted(store.list(500), key=lambda r: r["created"])
    series = {k: [] for k in ("first_pass_valid_rate", "time_to_valid_s", "tok_s", "cosmos_s_per_frame", "sim_rtf",
                              "epdms")}
    n = ok = 0
    for r in runs:
        k, t = r.get("kpis") or {}, r["created"]
        if "first_pass_valid" in k:
            n, ok = n + 1, ok + bool(k["first_pass_valid"])
            series["first_pass_valid_rate"].append([t, round(ok / n, 3)])
        for src, dst in (("time_to_valid_s", "time_to_valid_s"), ("llm_tok_per_s", "tok_s"), ("sim_rtf", "sim_rtf"),
                         ("cosmos_s_per_frame", "cosmos_s_per_frame"), ("epdms", "epdms")):
            if k.get(src) is not None:
                series[dst].append([t, k[src]])
    return {"series": series, "latest": {k: v[-1][1] for k, v in series.items() if v}}


@app.get("/api/config")
def get_config():
    return json.loads((CONFIG if CONFIG.exists() else CONFIG_DEFAULT).read_text())


@app.put("/api/config")
def put_config(cfg: dict = Body(...)):
    CONFIG.write_text(json.dumps(cfg, indent=2))
    return cfg


# The built operator UI (ui/dist) is served at / when present; the UI also runs standalone on the PC.
_dist = ROOT / "ui" / "dist"
if _dist.exists():
    app.mount("/", StaticFiles(directory=_dist, html=True), name="ui")
