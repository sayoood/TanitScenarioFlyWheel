"""`tsf` command line: compile, simulate and render TSIR scenarios."""
from __future__ import annotations

import json
import time
from pathlib import Path

import typer
from rich import print

from tsf.ir.scenario import Scenario

app = typer.Typer(no_args_is_help=True, add_completion=False)


def load_scenario(path: Path) -> Scenario:
    return Scenario.model_validate_json(Path(path).read_text(encoding="utf-8"))


@app.command()
def sim(scenario: Path, out: Path = Path("runs/sim"), video: bool = True):
    """Compile TSIR -> .xodr, load it with TanitSim's own parser, simulate, write metrics (+ BEV video)."""
    from tsf.compile import xodr
    from tsf.sim import odr
    from tsf.sim.engine import TanitSim

    sc = load_scenario(scenario)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    xp = xodr.write(sc, out / f"{sc.meta.id}.xodr")
    t1 = time.perf_counter()
    m = odr.load(xp)
    t2 = time.perf_counter()
    s = TanitSim(sc, m)
    log = s.run()
    t3 = time.perf_counter()
    res = {"scenario": sc.meta.id, "lanes": len(m.lanes), "events": log.events, "metrics": log.metrics,
           "timing_s": {"compile_xodr": round(t1 - t0, 3), "load_map": round(t2 - t1, 3),
                        "simulate": round(t3 - t2, 3), "realtime_factor": round(sc.duration / (t3 - t2), 1)}}
    if video:
        from tsf.sim.render2d import render_video
        render_video(s, log, out / f"{sc.meta.id}_bev.mp4", title=sc.meta.title[:70])
        res["timing_s"]["render"] = round(time.perf_counter() - t3, 2)
    (out / "result.json").write_text(json.dumps(res, indent=2))
    print(res)


@app.command()
def verify(scenario: Path, out: Path = Path("runs/verify"), qc: bool = True, video: bool = True):
    """Run the verifier ladder L0-L3 on a TSIR file; write report (+ BEV video of the simulation)."""
    from tsf.verify import ladder

    rep = ladder.run(Path(scenario).read_text(encoding="utf-8"), out, qc=qc)
    (out / "report.json").write_text(json.dumps(rep.to_dict(), indent=2))
    if video and hasattr(rep, "_log"):
        from tsf.sim.render2d import render_video
        render_video(rep._sim, rep._log, out / "bev.mp4", title=rep._sim.sc.meta.title[:70])
    print({"passed": rep.passed, "score": rep.score(), "diagnostics": rep.diagnostics(), "metrics": rep.metrics,
           "timing": rep.timing})


@app.command()
def closedloop(scenario: Path, policy: str = "const_vel", out: Path = Path("runs/cl"), video: bool = True):
    """Tier-T0 closed loop: TanitSim world + policy-driven ego (const_vel | accel_decel)."""
    from tsf.closedloop import runner
    from tsf.compile import xodr
    from tsf.sim import odr

    sc = load_scenario(scenario)
    out.mkdir(parents=True, exist_ok=True)
    m = odr.load(xodr.write(sc, out / f"{sc.meta.id}.xodr"))
    pol = {"const_vel": runner.ConstantVelocity, "accel_decel": runner.AccelDecel}[policy]()
    t0 = time.perf_counter()
    res = runner.run(sc, m, pol)
    dt = time.perf_counter() - t0
    if video:
        from tsf.sim.render2d import render_video
        render_video(res.sim, res.log, out / f"{policy}.mp4", title=f"Closed loop [{policy}]  {sc.meta.title}"[:80])
    summary = {"policy": policy, "score": res.score, "metrics": res.log.metrics, "sim_s": round(dt, 2),
               "realtime_factor": round(sc.duration / dt, 1)}
    (out / f"{policy}.json").write_text(json.dumps(summary, indent=2))
    print(summary)


@app.command()
def augment(scenario: Path, out: Path = Path("runs/aug"), n_seed: int = 32, n_iter: int = 4, n_per_iter: int = 16):
    """Coverage + criticality-guided behavioural augmentation over the TSIR logical parameters."""
    from tsf.augment.search import Augmenter, plot_coverage
    from tsf.compile import xodr
    from tsf.sim import odr

    sc = load_scenario(scenario)
    out.mkdir(parents=True, exist_ok=True)
    m = odr.load(xodr.write(sc, out / f"{sc.meta.id}.xodr"))
    aug = Augmenter(sc, m)
    t0 = time.perf_counter()
    res = aug.run(n_seed, n_iter, n_per_iter)
    dt = time.perf_counter() - t0
    for v in res.kept:
        (out / f"{v.tsir['meta']['id']}.json").write_text(json.dumps(v.tsir, indent=1))
    plot_coverage(res, aug, out / "coverage.png")
    table = [{**v.values, "criticality": round(v.criticality, 3), "min_ttc": v.metrics.get("min_ttc"),
              "max_decel": v.metrics.get("ego_max_decel")} for v in sorted(res.kept, key=lambda v: -v.criticality)]
    summ = {**res.summary(), "seconds": round(dt, 1), "sims_per_s": round(res.sims / dt, 1), "top": table[:8]}
    (out / "augment.json").write_text(json.dumps(summ, indent=2))
    print(summ)


@app.command()
def serve(host: str = "0.0.0.0", port: int = 8080):
    """Flywheel API + event stream for the operator UI (docs/API.md)."""
    import uvicorn
    uvicorn.run("tsf.server.app:app", host=host, port=port, log_level="info")


if __name__ == "__main__":
    app()
