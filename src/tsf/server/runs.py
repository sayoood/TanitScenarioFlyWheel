"""Run store + single-worker executor. A run is one flywheel execution of a use case (docs/API.md).

Runs execute one at a time on thor3 (LLM and GPU are shared); state is persisted as runs/api/<id>/run.json
so the UI can reload history after a restart. Every state change is published as an event.
"""
from __future__ import annotations

import json
import os
import shutil
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[3]
RUNS = Path(os.environ.get("TSF_RUNS", ROOT / "runs" / "api"))
EXAMPLES = ROOT / "examples"
KIND = {".xodr": "xodr", ".xosc": "xosc", ".osc": "osc2", ".mp4": "video", ".png": "image", ".json": "json"}
_LIFT = ["analyse", "propose", "verify", "repair", "document"]
_DOWN = ["simulate", "augment", "closed_loop"]
STAGES = {"uc1": _LIFT + _DOWN, "uc3": ["brief"] + _LIFT + ["twin"] + _DOWN}


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    def __init__(self, root: Path = RUNS, publish: Callable[[dict], None] = lambda ev: None):
        self.root, self.publish = root, publish
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tsf-run")
        self.queue: list[str] = []
        self.running: str | None = None
        for p in self.root.glob("*/run.json"):  # runs interrupted by a restart are marked failed
            d = json.loads(p.read_text())
            if d["status"] in ("queued", "running"):
                d["status"] = "failed"
                d.setdefault("error", "interrupted by server restart")
                for st in d.get("stages", []):
                    st["status"] = {"running": "failed", "pending": "skipped"}.get(st["status"], st["status"])
                p.write_text(json.dumps(d, indent=1))

    # ------------------------------------------------------------------ persistence
    def path(self, rid: str) -> Path:
        p = (self.root / rid).resolve()
        if p.parent != self.root.resolve():
            raise KeyError(rid)
        return p

    def get(self, rid: str) -> dict:
        p = self.path(rid) / "run.json"
        if not p.exists():
            raise KeyError(rid)
        return json.loads(p.read_text())

    def save(self, run: dict, event: bool = True):
        p = self.path(run["id"]) / "run.json"
        with self.lock:  # atomic replace: readers never see a half-written file
            tmp = p.with_suffix(".tmp")
            tmp.write_text(json.dumps(run, indent=1, default=str))
            os.replace(tmp, p)
        if event:
            self.publish({"type": "run", "run_id": run["id"], "run": summary(run)})

    def list(self, limit: int = 50) -> list[dict]:
        rows = []
        for p in self.root.glob("*/run.json"):
            try:
                rows.append(summary(json.loads(p.read_text())))
            except (json.JSONDecodeError, KeyError):
                continue
        return sorted(rows, key=lambda r: r["created"], reverse=True)[:limit]

    # ------------------------------------------------------------------ execution
    def create(self, uc: str, inp: dict, options: dict) -> dict:
        if uc not in PIPELINES:
            raise NotImplementedError(uc)
        rid = f"run_{time.strftime('%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}"
        self.path(rid).mkdir(parents=True)
        run = {"id": rid, "uc": uc, "title": _title(uc, inp), "status": "queued", "created": now(), "input": inp,
               "options": options, "stages": [{"name": s, "status": "pending"} for s in STAGES[uc]],
               "attempts": [], "metrics": {}, "artifacts": [], "kpis": {}}
        self.save(run)
        self.queue.append(rid)
        self.pool.submit(self._execute, rid)
        return summary(run)

    def _execute(self, rid: str):
        self.queue.remove(rid)
        self.running = rid
        run = self.get(rid)
        run["status"], t0 = "running", time.perf_counter()
        self.save(run)
        ctx = Ctx(self, run)
        try:
            PIPELINES[run["uc"]](ctx)
            run["status"] = "ok"
        except Exception as e:  # noqa: BLE001 - the run records the failure, the server keeps going
            run["status"], run["error"] = "failed", f"{type(e).__name__}: {e}"
            run["traceback"] = traceback.format_exc()[-4000:]
            for s in run["stages"]:
                if s["status"] == "running":
                    s["status"], s["detail"] = "failed", run["error"][:300]
        run["seconds"] = round(time.perf_counter() - t0, 1)
        for s in run["stages"]:
            if s["status"] == "pending":
                s["status"] = "skipped"
        run["artifacts"] = artifacts(self.path(rid))
        self.save(run)
        self.running = None


class Ctx:
    """Handle a pipeline uses to report progress."""

    def __init__(self, store: Store, run: dict):
        self.store, self.run = store, run
        self.dir = store.path(run["id"])
        self._t: dict[str, float] = {}

    def stage(self, name: str, status: str, detail: str = ""):
        st = next((s for s in self.run["stages"] if s["name"] == name), None)
        if st is None:  # optional stage not declared up front (e.g. replay_negative)
            st = {"name": name, "status": "pending"}
            self.run["stages"].append(st)
        if status == "running":
            self._t[name] = time.perf_counter()
            st["started"] = now()
        elif name in self._t:
            st["seconds"] = round(st.get("seconds", 0) + time.perf_counter() - self._t.pop(name), 2)
        st["status"], st["detail"] = status, detail
        self.store.save(self.run, event=False)
        self.store.publish({"type": "stage", "run_id": self.run["id"], "stage": st})

    def log(self, msg: str):
        self.store.publish({"type": "log", "run_id": self.run["id"], "t": now(), "msg": msg})

    def save(self):
        self.run["artifacts"] = artifacts(self.dir)
        self.store.save(self.run)


def summary(run: dict) -> dict:
    return {k: run[k] for k in ("id", "uc", "title", "status", "created", "seconds", "score", "kpis")
            if run.get(k) is not None}  # optional fields are omitted, never null (API.md `?:`)


def artifacts(d: Path) -> list[dict]:
    return [{"name": p.name, "kind": KIND.get(p.suffix, "json") if p.name != "package.json" else "package",
             "size": p.stat().st_size} for p in sorted(d.iterdir()) if p.is_file() and p.name != "run.json"]


def _title(uc: str, inp: dict) -> str:
    t = inp.get("query") or inp.get("example") or inp.get("incident_id") or inp.get("video") or inp.get("snippet") or ""
    if not t and isinstance(inp.get("tsir"), dict):
        t = inp["tsir"].get("meta", {}).get("title", "")
    return f"{uc.upper()} · {str(t)[:90]}"


# ============================================================================ pipelines
def _attempt(i: int, rep, seconds: float) -> dict:
    d = rep.to_dict()
    return {"index": i, "passed": d["passed"], "score": d["score"], "seconds": round(seconds, 2),
            "findings": [{k: f[k] for k in ("level", "check", "ok", "message")} for f in d["findings"]]}


def _publish_files(src: Path, dst: Path):
    for p in src.glob("*"):
        if p.suffix in (".xodr", ".xosc", ".osc") or p.name == "report.json":
            shutil.copy2(p, dst / ("scenario" + p.suffix if p.suffix != ".json" else p.name))


def uc1(ctx: Ctx):
    """Text (or a given TSIR) -> verified scenario package -> simulation -> augmentation -> closed loop."""
    from tsf.verify import ladder

    run, inp, wd = ctx.run, ctx.run["input"], ctx.dir
    if inp.get("query") and not (inp.get("tsir") or inp.get("example")):
        from tsf.agents.lifter import Lifter
        res = Lifter().lift(inp["query"], wd / "lift", on_stage=ctx.stage)
        tsir, rep = _adopt_lift(ctx, res, wd / "lift")
    else:
        for s in ("analyse", "propose", "repair", "document"):
            ctx.stage(s, "skipped", "TSIR given")
        tsir = inp.get("tsir") or json.loads((EXAMPLES / f"{Path(inp['example']).stem}.json").read_text())
        ctx.stage("verify", "running")
        t0 = time.perf_counter()
        rep = ladder.run(tsir, wd / "attempt0")
        run["attempts"] = [_attempt(0, rep, time.perf_counter() - t0)]
        ctx.stage("verify", "ok" if rep.passed else "failed", rep.diagnostics()[:300])
        run["kpis"] = {"first_pass_valid": rep.passed, "time_to_valid_s": round(time.perf_counter() - t0, 2)}
        _publish_files(wd / "attempt0", wd)
    downstream(ctx, tsir, rep)


def _adopt_lift(ctx: Ctx, res, lift_dir: Path):
    """Record a LiftResult (attempts, package, LLM stats, KPIs) on the run; returns the final (tsir, report)."""
    run = ctx.run
    run["attempts"] = [_attempt(i, a.report, a.seconds) for i, a in enumerate(res.attempts)]
    run["package_valid"], run["package_errors"] = res.package_valid, res.package_errors
    run["llm"] = {"calls": res.llm_stats.get("calls", 0), "tok_per_s": res.llm_stats.get("tok_per_s", 0.0)}
    run["kpis"] = res.kpis()
    shutil.copy2(lift_dir / "package.json", ctx.dir / "package.json")
    _publish_files(lift_dir / f"attempt{len(res.attempts) - 1}", ctx.dir)
    return res.final.tsir, res.final.report


def downstream(ctx: Ctx, tsir: dict, rep):
    """Shared by all use cases once a TSIR passed the ladder: simulate (+scene, video) -> augment -> closed loop."""
    run, opt, wd = ctx.run, ctx.run["options"] or {}, ctx.dir
    (wd / "tsir.json").write_text(json.dumps(tsir, indent=1))
    (wd / "report.json").write_text(json.dumps(rep.to_dict(), indent=1))
    run["score"] = rep.score()
    run["metrics"] = dict(rep.metrics)
    ctx.save()
    if not rep.passed or not hasattr(rep, "_log"):
        raise RuntimeError("scenario did not pass the verifier ladder: " + rep.diagnostics()[:300])

    from tsf.server.scene import scene
    from tsf.sim.render2d import render_video
    sim, log = rep._sim, rep._log
    ctx.stage("simulate", "running")
    (wd / "scene.json").write_text(json.dumps(scene(sim, log)))
    t0 = time.perf_counter()
    if opt.get("video", True):
        render_video(sim, log, wd / "bev.mp4", title=sim.sc.meta.title[:70])
    run["kpis"]["render_s"] = round(time.perf_counter() - t0, 1)
    run["kpis"]["sim_rtf"] = round(sim.sc.duration / max(rep.timing.get("l3", 1) - rep.timing.get("l2", 0), 1e-3), 1)
    ctx.stage("simulate", "ok", f"{len(log.ids)} actors, {sim.sc.duration:.0f} s")
    ctx.save()

    if opt.get("augment"):
        from tsf.augment.search import Augmenter, plot_coverage
        ctx.stage("augment", "running")
        a = opt["augment"] if isinstance(opt["augment"], dict) else {}
        aug = Augmenter(sim.sc, sim.m)
        res = aug.run(a.get("n_seed", 16), a.get("n_iter", 2), a.get("n_per_iter", 8))
        (wd / "variants").mkdir(exist_ok=True)
        for v in res.kept[: a.get("n", 8)]:
            (wd / "variants" / f"{v.tsir['meta']['id']}.json").write_text(json.dumps(v.tsir, indent=1))
        plot_coverage(res, aug, wd / "augment_coverage.png")
        run["metrics"]["augment"] = res.summary()
        ctx.stage("augment", "ok", f"{res.sims} sims, {len(res.kept)} kept, coverage {res.coverage:.0%}")
        ctx.save()

    pol = (opt.get("closed_loop") or {}).get("policy")
    if pol:
        from tsf.closedloop import runner
        policies = {"const_vel": runner.ConstantVelocity, "accel_decel": runner.AccelDecel}
        if pol not in policies:
            ctx.stage("closed_loop", "skipped", f"policy {pol} needs tier T1 (Cosmos sensor stage)")
        else:
            ctx.stage("closed_loop", "running")
            cl = runner.run(sim.sc, sim.m, policies[pol]())
            if opt.get("video", True):
                render_video(cl.sim, cl.log, wd / f"closed_loop_{pol}.mp4",
                             title=f"Closed loop [{pol}]  {sim.sc.meta.title}"[:80])
            run["metrics"]["closed_loop"] = {"policy": pol, **cl.score, "metrics": cl.log.metrics}
            run["kpis"]["epdms"] = cl.score["EPDMS"]
            ctx.stage("closed_loop", "ok", f"EPDMS {cl.score['EPDMS']:.3f}")
    ctx.save()


def uc3(ctx: Ctx):
    """NEGATIVE incident -> brief -> lifted COUNTERFACTUAL_POSITIVE twin (incident v2) -> UC1 downstream."""
    from tsf.agents.counterfactual import Counterfactual
    out = Counterfactual().run(ctx.run["input"]["incident_id"], ctx.dir / "uc3", on_stage=ctx.stage,
                               replay_negative=bool((ctx.run["options"] or {}).get("replay_negative")))
    tsir, rep = _adopt_lift(ctx, out["positive"], ctx.dir / "uc3" / "positive")
    shutil.copy2(ctx.dir / "uc3" / "incident_cf_positive.json", ctx.dir / "incident_cf_positive.json")
    shutil.copy2(ctx.dir / "uc3" / "brief.json", ctx.dir / "incident_brief.json")
    ctx.run["kpis"].update(twin_valid=out["twin_valid"])
    ctx.run["twin_errors"] = out["twin_errors"]
    downstream(ctx, tsir, rep)


PIPELINES = {"uc1": uc1, "uc3": uc3}
