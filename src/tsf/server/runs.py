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
_DOWN = ["simulate", "sensors", "augment", "closed_loop"]
# scenario augmentation job (tsf.agents.augment): one child job per request, started from a finished run's TSIR
_AUG = ["parent", "plan", "edit", "verify", "criticality", "describe", "package", "render", "judge"]
STAGES = {"uc1": _LIFT + _DOWN, "uc3": ["brief"] + _LIFT + ["replay_negative", "twin"] + _DOWN,
          "augment": _AUG,
          "uc2": ["vipe", "scene"] + _LIFT + _DOWN,
          "uc4": ["lift", "verify", "analyse", "document"] + _DOWN,
          "uc5": ["lift", "verify", "twin", "analyse", "document"] + _DOWN}


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
        if uc == "augment":
            inp = {**inp, "title": _augment_parent_title(self, inp)}
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
    pkg = ("package.json", "_package.json", "_imdad_record.json")
    return [{"name": p.name, "kind": KIND.get(p.suffix, "json") if not p.name.endswith(pkg) else "package",
             "size": p.stat().st_size} for p in sorted(d.iterdir()) if p.is_file() and p.name != "run.json"]


def _title(uc: str, inp: dict) -> str:
    if uc == "augment":
        return f"AUGMENT · {inp.get('mode', 'auto')} · {str(inp.get('title') or inp.get('parent_run') or inp.get('example') or '')[:80]}"
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
        for s in ("propose", "repair"):
            ctx.stage(s, "skipped", "TSIR given")
        tsir = inp.get("tsir") or json.loads((EXAMPLES / f"{Path(inp['example']).stem}.json").read_text())
        ctx.stage("verify", "running")
        t0 = time.perf_counter()
        rep = ladder.run(tsir, wd / "attempt0")
        run["attempts"] = [_attempt(0, rep, time.perf_counter() - t0)]
        ctx.stage("verify", "ok" if rep.passed else "failed", rep.diagnostics()[:300])
        run["kpis"] = {"first_pass_valid": rep.passed, "time_to_valid_s": round(time.perf_counter() - t0, 2)}
        _publish_files(wd / "attempt0", wd)
        if rep.passed:                    # the verified TSIR is usable (augment, replay) while the LLM still writes the package
            (wd / "tsir.json").write_text(json.dumps(tsir, indent=1))
        if rep.passed and (run["options"] or {}).get("package", True):
            _package_given(ctx, tsir, rep)
        else:
            for s in ("analyse", "document"):
                ctx.stage(s, "skipped", "TSIR given" if rep.passed else "TSIR failed verification")
    downstream(ctx, tsir, rep)


def _package_given(ctx: Ctx, tsir: dict, rep, query: str | None = None) -> bool:
    """Scenario_Structure (IMDAD) package for a TSIR that did not come from the Lifter (example, given TSIR, variant):
    the Lifter's analyst + documenter on the verified scenario, assembled and validated like a lifted one. A missing
    LLM skips the stage with the reason instead of failing the run."""
    from tsf.agents.lifter import Attempt, Lifter, assemble_package, validate_package
    run = ctx.run
    meta = tsir["meta"]
    q = query or (meta.get("description") or meta["title"])
    try:
        lif = Lifter()
        ctx.stage("analyse", "running")
        analysis = lif.analyse(q)
        ctx.stage("analyse", "ok")
        att = Attempt(tsir, rep, 0.0)
        ctx.stage("document", "running")
        nar = lif.document(q, analysis, [att], att)
        pkg = assemble_package(q, analysis, [att], att, nar)
        valid, errs = validate_package(pkg)
    except Exception as e:  # noqa: BLE001 - LLM down / schema failure: the scenario itself is still a valid run
        for st in ("analyse", "document"):
            cur = next((x for x in run["stages"] if x["name"] == st), {})
            if cur.get("status") in (None, "pending", "running"):
                ctx.stage(st, "skipped", f"package not built: {type(e).__name__}: {str(e)[:160]}")
        run["package_error"] = f"{type(e).__name__}: {str(e)[:300]}"
        return False
    (ctx.dir / "package.json").write_text(json.dumps(pkg, indent=2), encoding="utf-8")
    st = lif.llm.stats()
    run["package_valid"], run["package_errors"] = valid, errs
    run["llm"] = {"calls": st.get("calls", 0), "tok_per_s": st.get("tok_per_s", 0.0)}
    run.setdefault("kpis", {}).update(package_valid=valid, llm_tok_per_s=st.get("tok_per_s"))
    ctx.stage("document", "ok", "IMDAD package valid" if valid else "; ".join(errs[:2]))
    return True


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


def _epdms_of(rep) -> dict | None:
    """EPDMS of the scenario's own simulated ego (the verifier's sim + log), None when there is no simulation."""
    try:
        from tsf.closedloop import runner
        return runner.epdms(rep._sim, rep._log)
    except Exception:  # noqa: BLE001 - a score is a KPI, never a reason to fail a run
        return None


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
    sc_doc = scene(sim, log)
    if log.static:                         # signs, cones, barriers, parked vehicles: boxes for the viewer (Scene.static)
        sc_doc["static"] = [b.to_dict() for b in log.static]
    (wd / "scene.json").write_text(json.dumps(sc_doc))
    t0 = time.perf_counter()
    if opt.get("video", True):
        render_video(sim, log, wd / "bev.mp4", title=sim.sc.meta.title[:70])
    run["kpis"]["render_s"] = round(time.perf_counter() - t0, 1)
    run["kpis"]["sim_rtf"] = round(sim.sc.duration / max(rep.timing.get("l3", 1) - rep.timing.get("l2", 0), 1e-3), 1)
    sc_score = _epdms_of(rep)
    if sc_score is not None:             # the scenario's own ego (scripted / lifted), next to the closed-loop test policy's
        run["kpis"]["epdms_scenario"] = sc_score["EPDMS"]
        run["metrics"]["epdms_scenario"] = sc_score
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

    if opt.get("sensors"):
        _sensors(ctx, tsir, opt["sensors"] if isinstance(opt["sensors"], dict) else {})
    else:
        ctx.stage("sensors", "skipped", "not requested (options.sensors)")

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
            run["kpis"]["epdms_policy"] = run["kpis"]["epdms"] = cl.score["EPDMS"]    # epdms: backward-compatible alias
            ctx.stage("closed_loop", "ok", f"EPDMS {cl.score['EPDMS']:.3f}")
    ctx.save()


def _running_cosmos() -> list[str]:
    try:
        from tsf.cosmos import generate
        return list(generate.running_servers())
    except Exception:  # noqa: BLE001 - docker not reachable: let ensure_server decide
        return []


def _sensors(ctx: Ctx, tsir: dict, so: dict):
    """Camera + LiDAR (Cosmos3 Sim-Auto joint mode) of the verified scenario: ``tsf.cosmos.pipeline.render_sensors``.

    Options ``{"cameras": [...], "frames": 97, "lidar": true, "t0": null, "seed": 0, "appearance": {...},
    "wait_s": 0, "keep_server": false}``. The Cosmos slot and the unified memory are shared with other jobs: when the
    server is busy or missing the stage is *skipped* with the reason (``wait_s`` > 0 waits and retries) and the run goes on."""
    run, wd = ctx.run, ctx.dir
    ctx.stage("sensors", "running")
    try:
        from tsf.cosmos import pipeline
        render_sensors = pipeline.render_sensors
    except Exception as e:  # noqa: BLE001
        ctx.stage("sensors", "skipped", f"skipped: Cosmos sensor pipeline not available ({type(e).__name__}: {str(e)[:120]})")
        return
    cams = tuple(so["cameras"]) if so.get("cameras") else None
    frames, lidar, wait_s = int(so.get("frames", 97)), bool(so.get("lidar", True)), float(so.get("wait_s", 0))
    t_wait, info = time.perf_counter(), {}
    busy = getattr(pipeline, "CosmosBusyError", ())
    while True:
        try:
            other = _running_cosmos()        # never take over or restart a server that is not ours (its owner may be mid-job)
            if other:
                raise (busy or RuntimeError)(f"a Cosmos server is running: {other}")
            info = pipeline.ensure_server(views=len(cams or pipeline.SENSOR_CAMERAS), lidar=lidar)
            break
        except Exception as e:  # noqa: BLE001
            if isinstance(e, busy) and time.perf_counter() - t_wait < wait_s:
                ctx.log(f"sensors: Cosmos busy, retrying: {str(e)[:100]}")
                time.sleep(30)
                continue
            why = "Cosmos server busy" if isinstance(e, busy) else "Cosmos server not available"
            ctx.stage("sensors", "skipped", f"skipped: {why} ({str(e)[:160]})")
            return
    try:
        kw = {"cameras": cams} if cams else {}
        res = render_sensors(tsir, wd / "sensors", frames=frames, lidar=lidar, seed=int(so.get("seed", 0)),
                             t0=so.get("t0"), appearance=so.get("appearance"), **kw)
    except Exception as e:  # noqa: BLE001 - a failed render is reported, the verified scenario stays a valid run
        ctx.stage("sensors", "failed", f"{type(e).__name__}: {str(e)[:240]}")
        run["sensors"] = {"error": f"{type(e).__name__}: {str(e)[:400]}"}
        return
    finally:
        if info.get("started") and not so.get("keep_server"):
            try:
                from tsf.cosmos import generate
                generate.stop("sim")
            except Exception:  # noqa: BLE001
                pass
    lid = res.get("lidar") or {}
    for src, dst in ((res.get("mosaic"), "mosaic.mp4"), (lid.get("bev_video"), "lidar_bev.mp4"),
                     (wd / "sensors" / "sensors.json", "sensors.json")):
        if src and Path(src).exists():
            shutil.copy2(src, wd / dst)
    warn = (lid.get("checks_summary") or {}).get("warnings", [])
    run["sensors"] = {"cameras": res["cameras"], "frames": res["frames"], "t0_s": res["t0_s"], "timings": res["timings"],
                      "lidar_warnings": warn, "errors": res.get("errors") or {}, "captions": res["captions"]["source"]}
    run.setdefault("kpis", {}).update(cosmos_s_per_frame=round(res["timings"]["cosmos_s"] / frames, 2),
                                      lidar_checks_ok=not warn)
    ctx.stage("sensors", "ok", f"{len(res['cameras'])} cameras{' + LiDAR' if lid.get('enabled') else ''}, "
              f"{res['timings']['cosmos_s']:.0f} s Cosmos" + (f"; {len(warn)} LiDAR warning(s)" if warn else ""))
    ctx.save()


def uc3(ctx: Ctx):
    """NEGATIVE incident -> brief -> lifted COUNTERFACTUAL_POSITIVE twin (incident v2) -> UC1 downstream."""
    from tsf.agents.counterfactual import Counterfactual
    out = Counterfactual().run(ctx.run["input"]["incident_id"], ctx.dir / "uc3", on_stage=ctx.stage,
                               replay_negative=bool((ctx.run["options"] or {}).get("replay_negative", True)))
    tsir, rep = _adopt_lift(ctx, out["positive"], ctx.dir / "uc3" / "positive")
    shutil.copy2(ctx.dir / "uc3" / "incident_cf_positive.json", ctx.dir / "incident_cf_positive.json")
    shutil.copy2(ctx.dir / "uc3" / "brief.json", ctx.dir / "incident_brief.json")
    ctx.run["kpis"].update(twin_valid=out["twin_valid"])
    ctx.run["twin_errors"] = out["twin_errors"]
    neg = out.get("negative")        # the NEGATIVE TSIR that reproduces the incident is a required artifact
    if neg is not None:
        (ctx.dir / "negative_tsir.json").write_text(json.dumps(neg.final.tsir, indent=1))
        if (neg.workdir / "package.json").exists():
            shutil.copy2(neg.workdir / "package.json", ctx.dir / "negative_package.json")
        ctx.run["kpis"].update(negative_valid=neg.final.report.passed)
        ng = _epdms_of(neg.final.report)
        if ng is not None:
            ctx.run["kpis"]["epdms_negative"] = ng["EPDMS"]
        ctx.run["negative"] = {"valid": neg.final.report.passed, "metrics": neg.final.report.metrics,
                               "attempts": len(neg.attempts)}
    downstream(ctx, tsir, rep)


def _lift_snippet(ctx: Ctx, polarity: str) -> dict:
    from tsf.agents import snippet
    ctx.stage("lift", "running")
    lifted = snippet.lift_snippet(ctx.run["input"]["snippet"], ctx.dir, polarity=polarity)
    fid = lifted["fidelity"]
    ctx.stage("lift", "ok", f"{len(lifted['tsir']['actors'])} actors, {len(lifted['tsir']['actions'])} actions; ego ADE "
              f"{(fid.get('ego') or {}).get('ade_s')} m, NPC ADE {fid.get('npc_mean_ade_s')} m")
    rep = lifted["report"]
    ctx.stage("verify", "ok" if rep.passed else "failed", rep.diagnostics()[:300])
    ctx.run["lift"] = {"event": lifted["event"], "t0": lifted["t0"], "dropped": lifted["dropped"], "fidelity": fid}
    ctx.run["kpis"].update(lift_s=round(lifted["seconds"], 2), lift_consistent=rep.passed,
                           lift_ego_ade_s=(fid.get("ego") or {}).get("ade_s"), lift_npc_ade_s=fid.get("npc_mean_ade_s"))
    ctx.save()
    return lifted


def uc4(ctx: Ctx):
    """Sensor snippet with GT ego -> lifted, verified TSIR -> LLM analysis/documentation -> UC1 downstream."""
    from tsf.agents import snippet
    from tsf.agents.llm import LLM
    lifted = _lift_snippet(ctx, "positive")
    if not lifted["report"].passed:
        raise RuntimeError("lifted snippet is not consistent: " + lifted["report"].diagnostics()[:300])
    ctx.stage("analyse", "running")
    llm = LLM()
    res = snippet.document(lifted, llm, ctx.dir / "doc")
    ctx.stage("analyse", "ok")
    ctx.stage("document", "ok", "IMDAD package valid" if res.package_valid else "; ".join(res.package_errors[:2]))
    tsir, rep = lifted["tsir"], lifted["report"]
    ctx.run["attempts"] = [_attempt(0, rep, lifted["seconds"])]
    ctx.run["package_valid"], ctx.run["package_errors"] = res.package_valid, res.package_errors
    ctx.run["llm"] = {"calls": res.llm_stats.get("calls", 0), "tok_per_s": res.llm_stats.get("tok_per_s", 0.0)}
    shutil.copy2(ctx.dir / "doc" / "package.json", ctx.dir / "package.json")
    _publish_files(ctx.dir / "lift", ctx.dir)
    ctx.run["kpis"].update(first_pass_valid=rep.passed, package_valid=res.package_valid,
                           llm_tok_per_s=res.llm_stats.get("tok_per_s"))
    downstream(ctx, tsir, rep)


def uc5(ctx: Ctx):
    """Negative snippet -> lifted NEGATIVE -> LLM ego-plan twin (ego edits only) -> verified POSITIVE -> downstream."""
    from tsf.agents import snippet
    from tsf.agents.llm import LLM
    lifted = _lift_snippet(ctx, "negative")
    neg = lifted["report"]
    if not hasattr(neg, "_log"):
        raise RuntimeError("negative snippet could not be simulated: " + neg.diagnostics()[:300])
    (ctx.dir / "negative_tsir.json").write_text(json.dumps(lifted["tsir"], indent=1))
    ctx.stage("twin", "running")
    llm = LLM()
    res = snippet.twin(lifted, llm, ctx.dir / "twin")
    fin = res.final.report
    ctx.stage("twin", "ok" if fin.passed else "failed", res.extra["plan"]["strategy"][:200])
    ctx.stage("analyse", "ok")
    ctx.stage("document", "ok", "IMDAD package valid" if res.package_valid else "; ".join(res.package_errors[:2]))
    kpis = dict(ctx.run["kpis"])
    tsir, rep = _adopt_lift(ctx, res, ctx.dir / "twin")      # resets kpis to the twin's lift KPIs
    nm = res.extra["negative_metrics"]
    ctx.run["kpis"] = {**kpis, **ctx.run["kpis"], "neg_peak_decel": nm.get("ego_max_decel"),
                       "pos_peak_decel": fin.metrics.get("ego_max_decel"), "neg_peak_jerk": nm.get("ego_max_abs_jerk"),
                       "pos_peak_jerk": fin.metrics.get("ego_max_abs_jerk")}
    ctx.run["twin"] = res.extra
    ng = _epdms_of(neg)                  # the lifted NEGATIVE snippet, next to the twin's score (epdms_scenario)
    if ng is not None:
        ctx.run["kpis"]["epdms_negative"] = ng["EPDMS"]
    downstream(ctx, tsir, rep)


def uc2(ctx: Ctx):
    """Dash-cam video -> ViPE ego motion + VLM scene description -> Lifter -> UC1 downstream."""
    from tsf.agents.video import lift_video
    opt = ctx.run["options"] or {}
    out = lift_video(ctx.run["input"]["video"], ctx.dir / "uc2", on_stage=ctx.stage,
                     ego_speed=opt.get("ego_speed", "auto"), vipe_dir=opt.get("vipe_dir"))
    tsir, rep = _adopt_lift(ctx, out["result"], ctx.dir / "uc2" / "lift")
    ctx.run["kpis"].update(**{f"uc2_{k}": v for k, v in out["fidelity"].items()}, **out["timing"])
    ctx.run["uc2"] = {"ego": out["ego"], "scene": out["scene"], "query": out["query"], "npc": out.get("npc", []),
                      "npc_drops": out.get("npc_drops", {}), "ego_source": out["timing"].get("ego_source")}
    downstream(ctx, tsir, rep)


# ----------------------------------------------------------------------------- augmentation job
def _augment_parent_title(store: "Store", inp: dict) -> str:
    try:
        if inp.get("parent_run"):
            return store.get(inp["parent_run"])["title"]
    except KeyError:
        pass
    if isinstance(inp.get("tsir"), dict):
        return inp["tsir"].get("meta", {}).get("title", "")
    return inp.get("example", "")


class _AugProgress:
    """Maps the augmenter's per-variant events onto the run's stage list and a per-variant progress table."""

    NAMES = {"final": "describe"}

    def __init__(self, ctx: Ctx):
        self.ctx = ctx
        self.table = ctx.run["augment"]["progress"]
        self.started: set[str] = set()

    def event(self, ev: dict):
        st = self.NAMES.get(ev["stage"], ev["stage"])
        if ev["variant"] and ev["stage"] not in ("log", "done"):
            self.table[ev["variant"]] = {"stage": ev["stage"], "status": ev["status"], "detail": str(ev["detail"])[:200]}
        if st in _AUG and ev["status"] in ("running", "info"):
            self.started.add(st)
            self.ctx.stage(st, "running", f"{(ev['variant'] or 'parent')[-24:]}: {ev['detail']}"[:200])
        elif st == "plan" and ev["status"] == "ok":
            self.ctx.stage("plan", "ok", f"{(ev['variant'] or '')[-24:]}: {ev['detail']}"[:200])
        elif st == "render" and ev["status"] == "skipped":
            self.ctx.stage("render", "skipped", ev["detail"])
        elif ev["stage"] == "parent" and ev["status"] in ("ok", "failed"):
            self.ctx.stage("parent", ev["status"], ev["detail"])
        self.ctx.log(f"{(ev['variant'] or 'parent')[-30:]} {ev['stage']} {ev['status']} {str(ev['detail'])[:120]}")

    def finish(self, failed: bool):
        for name in self.started:
            cur = next((x for x in self.ctx.run["stages"] if x["name"] == name), {})
            if cur.get("status") == "running":
                self.ctx.stage(name, "failed" if failed else "ok", cur.get("detail", ""))


def _trim_variant(r: dict, n: int, media: dict) -> dict:
    """The part of a variant result the UI shows (full detail stays in augment/<id>/result.json)."""
    keep = ("id", "parent_id", "mode", "seed", "dimensions", "requests", "status", "error", "title", "description",
            "lever", "rationale", "limitations", "first_pass_valid", "repairs", "isolation", "description_check",
            "criticality", "effect", "plan", "package", "timings", "tokens", "tokens_judge", "patch", "diff",
            "housekeeping")
    out = {k: r[k] for k in keep if r.get(k) is not None}
    out["index"] = n
    out["attempts"] = [{k: a.get(k) for k in ("index", "lever", "stage", "passed", "issues", "seconds", "tokens", "limitations")}
                       for a in r.get("attempts", [])]
    v = r.get("verifier")
    if v:
        out["verifier"] = {"passed": v["passed"], "score": v["score"], "metrics": v.get("metrics"), "qc_run": v.get("qc_run"),
                           "findings": [{k: f[k] for k in ("level", "check", "ok", "message")} for f in v["findings"]]}
    rd, jd = r.get("render") or {}, r.get("judge") or {}
    out["render"] = {k: rd.get(k) for k in ("skipped", "error", "seed", "t0", "render_s", "wait_s", "geometry_reused") if rd.get(k) is not None}
    out["judge"] = jd
    out["media"] = media
    return out


def augment_job(ctx: Ctx):
    """Child job: variants of a finished run's verified TSIR (weather / behaviour / environment), see tsf.agents.augment.

    input  {"parent_run": rid | "tsir": {...} | "example": name, "mode": "auto"|"manual", "requests": {"weather",
            "behavior", "environment"}, "seed": 0, "separate": false}
    options {"render": false, "criticality": true, "package": true, "cameras": [...], "frames": 97, "lidar": true,
            "wait_s": 7200, "workers": 3}"""
    from tsf.agents import augment as A
    run, inp, opt, wd = ctx.run, ctx.run["input"], ctx.run["options"] or {}, ctx.dir
    parent_record = None
    if inp.get("parent_run"):
        src = ctx.store.path(inp["parent_run"])
        if not (src / "tsir.json").exists():
            raise RuntimeError(f"run {inp['parent_run']} has no verified TSIR (tsir.json): it did not pass the ladder")
        tsir = json.loads((src / "tsir.json").read_text())
        if (src / "incident_cf_positive.json").exists():
            parent_record = json.loads((src / "incident_cf_positive.json").read_text())
    elif isinstance(inp.get("tsir"), dict):
        tsir = inp["tsir"]
    else:
        tsir = json.loads((EXAMPLES / f"{Path(inp['example']).stem}.json").read_text())
    mode = inp.get("mode", "auto")
    run["augment"] = {"mode": mode, "seed": int(inp.get("seed", 0)), "separate": bool(inp.get("separate")),
                      "requests": inp.get("requests"), "render": bool(opt.get("render")), "progress": {}, "variants": []}
    ctx.save()
    prog = _AugProgress(ctx)
    t0 = time.perf_counter()
    try:
        results = A.augment(
            tsir, wd / "augment", mode=mode, requests=inp.get("requests"), seed=int(inp.get("seed", 0)),
            separate=bool(inp.get("separate")), render=bool(opt.get("render")), criticality=opt.get("criticality", True),
            package=opt.get("package", True), workers=int(opt.get("workers", 3)), on_event=prog.event,
            parent_record=parent_record, cameras=tuple(opt["cameras"]) if opt.get("cameras") else None,
            frames=int(opt.get("frames", 97)), lidar=bool(opt.get("lidar", True)),
            render_wait_s=float(opt.get("wait_s", 7200)))
    except Exception:
        prog.finish(True)
        raise
    prog.finish(all(r["status"] != "ok" for r in results))
    _publish_augment(ctx, results, time.perf_counter() - t0)
    if all(r["status"] != "ok" for r in results):
        raise RuntimeError("no variant passed: " + "; ".join(f"{r['id']}: {str(r.get('error'))[:120]}" for r in results))


def _publish_augment(ctx: Ctx, results: list[dict], wall_s: float):
    """Register the variants' files as run artifacts (v<n>_*), summarise KPIs, keep the UI-facing records."""
    run, wd = ctx.run, ctx.dir
    variants, first_pkg = [], None
    parent_media: dict = {}
    for n, r in enumerate(results, 1):
        pre, media = f"v{n}_", {}

        def cp(src, name, key=None):
            if src and Path(src).exists():
                shutil.copy2(src, wd / (pre + name))
                media[key or name] = pre + name

        paths = r.get("paths", {})
        cp(paths.get("tsir"), "tsir.json", "tsir")
        cp(paths.get("xodr"), "scenario.xodr", "xodr")
        cp(paths.get("xosc"), "scenario.xosc", "xosc")
        cp(paths.get("osc2"), "scenario.osc", "osc2")
        cp(paths.get("package"), "package.json", "package")
        cp(paths.get("imdad_record"), "imdad_record.json", "imdad_record")
        files = (r.get("render") or {}).get("files") or {}
        cp(files.get("mosaic"), "mosaic.mp4", "mosaic")
        cp(files.get("lidar_bev"), "lidar_bev.mp4", "lidar_bev")
        cp(files.get("sensors_json"), "sensors.json", "sensors")
        for cam, v in (files.get("videos") or {}).items():
            cp(v, f"{cam}.mp4", f"camera:{cam}")
        pf = ((r.get("parent_render") or {}).get("files")) or {}
        if pf and not parent_media:
            for key, name in (("mosaic", "parent_mosaic.mp4"), ("lidar_bev", "parent_lidar_bev.mp4")):
                if pf.get(key) and Path(pf[key]).exists():
                    shutil.copy2(pf[key], wd / name)
                    parent_media[key] = name
        if first_pkg is None and r["status"] == "ok" and (wd / (pre + "package.json")).exists():
            first_pkg = wd / (pre + "package.json")
        variants.append(_trim_variant(r, n, media))
    ok = [r for r in results if r["status"] == "ok"]
    n = len(results)

    def rate(f):
        return round(sum(1 for r in results if f(r)) / n, 3)

    aug = run["augment"]
    aug.update(variants=variants, parent_media=parent_media, wall_s=round(wall_s, 1))
    toks = sum((r.get("tokens") or {}).get("completion_tokens", 0) + (r.get("tokens") or {}).get("prompt_tokens", 0) for r in results)
    run["kpis"] = {"aug_variants": n, "aug_ok": len(ok), "aug_first_pass_rate": rate(lambda r: r.get("first_pass_valid")),
                   "aug_repairs_mean": round(sum(r.get("repairs", 0) for r in results) / n, 2),
                   "aug_isolation_rate": rate(lambda r: (r.get("isolation") or {}).get("ok")),
                   "aug_description_rate": rate(lambda r: (r.get("description_check") or {}).get("ok")),
                   "aug_llm_tokens": toks, "aug_wall_s": round(wall_s, 1)}
    if ok:
        run["score"] = round(sum((r.get("verifier") or {}).get("score", 0) for r in ok) / len(ok), 4)
    run["metrics"] = {"augment": {"variants": n, "ok": len(ok)}}
    if first_pkg:
        shutil.copy2(first_pkg, wd / "package.json")   # run-level package = the first verified variant's package
        pj = next((v.get("package") for v in variants if v.get("status") == "ok" and v.get("package")), None) or {}
        run["package_valid"] = bool(pj.get("scenario_structure_valid"))
        run["package_errors"] = pj.get("scenario_structure_errors", [])
    ctx.save()


PIPELINES = {"uc1": uc1, "uc2": uc2, "uc3": uc3, "uc4": uc4, "uc5": uc5, "augment": augment_job}
