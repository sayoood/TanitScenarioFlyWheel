"""T1 neural closed loop: TanitSim world + Cosmos3-Nano-Sim-Auto sensors + a camera policy (REFe SFT-4).

Sim-time-decoupled loop (the world waits while the sensors are generated):

    every chunk (16 new frames @ 30 fps = 0.533 s of world time, i.e. the policy's 2 Hz replan):
      1. export the last 17 frames of the *closed-loop* world (ego as driven by the policy) to RDS-HQ
      2. WSM v4 renders the control videos of the cameras the policy needs
      3. Sim-Auto generates the chunk, conditioned on the previous chunk's last frames (I2V chaining),
         captions written once from the TSIR
      4. the newest generated frames (resized back to the 16:9 rig resolution) go to the policy
    the first chunk has no frames yet: the policy's const_vel fallback drives the first 0.53 s.

Usage (thor3; Sim-Auto server up with kv_views >= len(cams), REFe server up):
    python -m tsf.closedloop.neural examples/uc1_left_turn_intersection.json --seconds 8 --out runs/t1/uc1
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from tsf.closedloop.runner import ClosedLoopResult, epdms
from tsf.ir.scenario import Junction, Scenario
from tsf.sim.engine import SimLog, TanitSim, metrics

REFE_CAMS = ["camera_front_wide_120fov", "camera_cross_left_120fov", "camera_cross_right_120fov",
             "camera_rear_left_70fov", "camera_rear_right_70fov"]
CHUNK = 17                                           # frames per Sim-Auto request (1 conditioning + 16 new)
FPS = 30


class Recorder:
    """Incremental version of TanitSim.run()'s logging, so the world can be exported mid-run."""

    def __init__(self, sim: TanitSim):
        self.sim, self.ids = sim, list(sim.agents)
        self.arms = range(len(sim.sc.road.approaches)) if isinstance(sim.sc.road, Junction) else []
        self.t, self.S, self.lanes, self.sig = [], [], [], {a: [] for a in self.arms}

    def record(self) -> None:
        sim = self.sim
        self.t.append(sim.t)
        self.S.append([(ag.x, ag.y, ag.yaw, ag.v, ag.a, ag.d) for ag in (sim.agents[a] for a in self.ids)])
        self.lanes.append([ag.path.keys[ag.path.lane_index(ag.p)] if ag.path else None
                           for ag in (sim.agents[a] for a in self.ids)])
        for arm in self.arms:
            self.sig[arm].append(sim.signals.arm_state(arm, sim.t))

    def log(self, with_metrics: bool = False) -> SimLog:
        sim = self.sim
        log = SimLog(np.asarray(self.t), self.ids, np.asarray(self.S, float), list(self.lanes),
                     np.array([sim.agents[a].lwh for a in self.ids]), [sim.agents[a].category for a in self.ids],
                     [sim.agents[a].role for a in self.ids], {k: list(v) for k, v in self.sig.items()}, sim.events)
        if with_metrics:
            log.metrics = metrics(sim, log)
        return log


class CosmosSensors:
    def __init__(self, tsir: dict, out: Path, cams: list[str], res: str = "480", captions: dict | None = None,
                 dry: bool = False):
        from tsf.cosmos.llm_captions import llm_captions
        self.tsir, self.out, self.cams, self.res, self.dry = tsir, out, cams, res, dry
        self.caps = captions or ({} if dry else llm_captions(tsir, cams, frames=CHUNK, fps=FPS))
        (out / "gen").mkdir(parents=True, exist_ok=True)
        (out / "captions.json").write_text(json.dumps(self.caps, indent=1))
        self.last: dict[str, str] | None = None
        self.k, self.timing = 0, []

    def render(self, sim: TanitSim, log: SimLog) -> dict[str, np.ndarray]:
        from tsf.cosmos import generate, rds_hq, wsm
        k, t_end = self.k, float(log.t[-1])
        cid = f"t1_{k:03d}"
        t0 = t_end - (CHUNK - 1) / FPS
        tic = time.perf_counter()
        root = rds_hq.export_clip(sim, log, self.out / "rds", cid, t0=t0, duration=CHUNK / FPS, cameras=self.cams,
                                  overwrite=True)
        t_exp = time.perf_counter() - tic
        ctrl = wsm.render_wsm(root, cid, self.out / "wsm" / cid, self.cams)
        t_wsm = time.perf_counter() - tic - t_exp
        if self.dry:                                 # plumbing test: the policy sees the WSM controls themselves
            self.k += 1
            self.timing.append({"chunk": k, "t_world": round(t_end, 3), "export_s": round(t_exp, 1),
                                "wsm_s": round(t_wsm, 1)})
            return {c: cv2.cvtColor(cv2.imread(str(generate.extract_frame(p, self.out / "gen" / f"{cid}_{c}.png", -1))),
                                    cv2.COLOR_BGR2RGB) for c, p in ctrl.items()}
        r = generate.sim_chunk(self.out / "wsm" / cid, self.out / "gen" / cid, self.cams, frames=CHUNK, res=self.res,
                               captions=self.caps, first_frames=self.last, seed=k, scenario=self.tsir["meta"]["id"])
        self.last = r["last_frames"]
        self.k += 1
        self.timing.append({"chunk": k, "t_world": round(t_end, 3), "export_s": round(t_exp, 1),
                            "wsm_s": round(t_wsm, 1), "cosmos_s": round(r["request_s"], 1)})
        # Sim-Auto's 480p preset (832x480) is a non-uniform rescale of the 16:9 control geometry: undo it for the rig
        return {c: cv2.resize(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB), (1280, 720), interpolation=cv2.INTER_CUBIC)
                for c, p in self.last.items()}


def run_t1(tsir: dict, policy, out: str | Path, seconds: float | None = None, cams: list[str] = REFE_CAMS,
           res: str = "480", dry: bool = False) -> ClosedLoopResult:
    from tsf.compile import xodr
    from tsf.sim import odr
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    sc = Scenario.model_validate(tsir)
    if seconds:
        sc = sc.model_copy(update={"duration": min(sc.duration, seconds)})
    xodr_path = out / f"{sc.meta.id}.xodr"
    xodr.write(sc, xodr_path)
    sim = TanitSim(sc, odr.load(xodr_path), ego_policy=policy)
    sensors = CosmosSensors(tsir, out, cams, res, dry=dry)
    rec = Recorder(sim)
    n = int(round(sc.duration / sim.dt)) + 1
    chunk_s = (CHUNK - 1) / FPS
    next_frames = chunk_s
    tic = time.perf_counter()
    for i in range(n):
        rec.record()
        if sim.t + 1e-9 >= next_frames:
            policy.set_frames(sensors.render(sim, rec.log()))
            next_frames += chunk_s
            print(f"t={sim.t:.2f}s chunk {sensors.k} {sensors.timing[-1]} plan k={getattr(policy, 'last', {}).get('k')}",
                  flush=True)
        if i < n - 1:
            sim.step()
    log = rec.log(with_metrics=True)
    res_ = ClosedLoopResult(getattr(policy, "name", type(policy).__name__), log, epdms(sim, log))
    res_.sim = sim
    summary = {"policy": res_.policy, "scenario": sc.meta.id, "seconds": sc.duration, "cams": cams, "res": res,
               "chunks": sensors.k, "wall_s": round(time.perf_counter() - tic, 1), "score": res_.score,
               "metrics": {k: v for k, v in log.metrics.items() if k != "collisions"},
               "policy_calls": getattr(policy, "n_calls", None), "fallback_ticks": getattr(policy, "n_fallback", None),
               "timing": sensors.timing}
    (out / "t1_summary.json").write_text(json.dumps(summary, indent=1, default=float))
    return res_


def main():
    from tsf.closedloop.refe import RefePolicy
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario", help="TSIR json (e.g. examples/uc1_left_turn_intersection.json)")
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--out", default="runs/t1")
    ap.add_argument("--res", default="480")
    ap.add_argument("--cams", default=",".join(REFE_CAMS))
    ap.add_argument("--dry", action="store_true", help="no Cosmos: feed the WSM controls (plumbing test)")
    a = ap.parse_args()
    tsir = json.loads(Path(a.scenario).read_text())
    out = Path(a.out)
    pol = RefePolicy(fallback="const_vel")
    r = run_t1(tsir, pol, out, a.seconds, a.cams.split(","), a.res, dry=a.dry)
    from tsf.sim.render2d import render_video
    render_video(r.sim, r.log, out / "t1_bev.mp4", title=f"T1 closed loop: REFe SFT-4 on Cosmos3-Nano-Sim-Auto, "
                                                           f"EPDMS {r.score['EPDMS']:.2f}")
    print(json.dumps(r.score))


if __name__ == "__main__":
    main()
