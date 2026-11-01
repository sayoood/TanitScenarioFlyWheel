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


if __name__ == "__main__":
    app()
