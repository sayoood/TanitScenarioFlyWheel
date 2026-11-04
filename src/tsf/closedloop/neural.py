"""T1 neural closed loop: TanitSim world + Cosmos3-Nano-Sim-Auto sensors + a camera policy (REFe SFT-4).

Sim-time-decoupled loop (the world waits while the sensors are generated):

    every chunk (16 new frames @ 30 fps = 0.533 s of world time, i.e. the policy's 2 Hz replan):
      1. export the last 17 frames of the *closed-loop* world (ego as driven by the policy) to RDS-HQ
      2. WSM v4 renders the control videos of the cameras the policy needs
      3. Sim-Auto generates the chunk, conditioned on the previous chunk's last frames (I2V chaining),
         captions: the chunk's span of one drive-long caption plus, for the front cameras, the ego's signal state
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
FRONT_CAMS = ("camera_front_wide_120fov", "camera_front_tele_30fov")      # the cameras that see the ego's signal
NUMBER = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"}
CHUNK = 17                                           # frames per Sim-Auto request (1 conditioning + 16 new)
FPS = 30


def signal_caption(sim: TanitSim, t0: float, t1: float, max_dist: float = 90.0) -> str:
    """The ego approach's signal state over ``[t0, t1]`` (sim time) as one caption sentence, while the ego is within
    ``max_dist`` m of its stop line ('' otherwise). The LLM captions leave signal colours out: written once before the
    drive, they guessed them from the plan and contradicted it (a green left-turn signal while the map showed red)."""
    plan = getattr(sim, "signals", None)
    ego = next(a for a in sim.agents.values() if a.role == "ego")
    if plan is None or not plan.enabled or ego.path is None:
        return ""
    nj, arm = sim._next_junction_lane(ego), sim._arm_of(ego)
    if nj is None or arm is None or ego.path.starts[nj[0]] - ego.p > max_dist:
        return ""

    def say(what: str, turn: str) -> str:
        a, b = plan.state(arm, turn, t0), plan.state(arm, turn, t1)
        return f"{what} is {a}" if a == b else f"{what} turns from {a} to {b}"
    if plan.protected:
        return (f"At the junction ahead, {say('the light for straight-ahead traffic', 'straight')} and "
                f"{say('the left-turn signal', 'left')}.")
    return f"At the junction ahead, {say('the traffic light', 'straight')}."


def _vehicle_name(a: dict) -> str:
    words = " ".join(w for w in (a.get("color"), {"suv": "SUV"}.get(a.get("category"), a.get("category") or "vehicle")) if w)
    return ("an " if words[0].lower() in "aeiou" or words.startswith("SUV") else "a ") + words


def _visible(objects: list[dict], size, scale: int = 4) -> dict[str, float]:
    """Share of each object's box that no nearer object's box covers (on a grid of ``scale`` pixels)."""
    w, h = int(size[0] // scale) + 1, int(size[1] // scale) + 1
    seen, out = np.zeros((h, w), bool), {}
    for o in sorted(objects, key=lambda o: o["depth"]):
        u0, v0, u1, v1 = (int(c // scale) for c in o["box"])
        cell = np.s_[max(v0, 0):min(v1 + 1, h), max(u0, 0):min(u1 + 1, w)]
        out[o["id"]] = 1.0 - float(seen[cell].mean()) if seen[cell].size else 0.0
        seen[cell] = True
    return out


def vehicle_cast(tsir: dict, projs: list[dict | None], cams: list[str], max_depth: float = 120.0,
                 min_px: float = 6.0, min_visible: float = 0.3) -> dict[str, str]:
    """One caption sentence per camera naming the vehicles the map shows in it, nearest first, by the scenario's colour
    and category. ``projs``: :func:`tsf.cosmos.adherence.project_clip` of the frames the chunk spans (its conditioning
    frame and its last frame); no projection, no sentences. The drive-long captions named road users per direction from
    the plan: the front caption left out the van ahead (generated white, blue after a re-anchor), and a truck in a block
    that also named a yellow bus came back as a bus. Named per chunk from the map, a re-drawn chunk gets the same looks.
    Listed in one sentence the colours still leaked ("a grey car, a yellow bus, a red car and a white truck" drew the
    truck yellow), so each vehicle gets its own sentence with where it is in the view: the third of the width its box
    centre falls in, far away beyond 50 m, close within 10 m (from the projection where it is nearest). A vehicle that
    nearer ones hide is not named: under ``min_visible`` of its box uncovered in every projection. The yellow bus
    behind the van beside the ego was named "on the right, close" for 6 s, and its yellow went onto the truck."""
    if not any(projs):
        return {}
    look = {a["id"]: _vehicle_name(a) for a in tsir["actors"] if a.get("role") != "ego"}
    out = {}
    for cam in cams:
        near: dict[str, tuple[float, float]] = {}       # id -> (depth, box centre as a share of the width)
        for p in projs:
            r = (p or {}).get(cam) or {}
            seen = _visible(r.get("objects", []), r.get("size", (1, 1)))
            for o in r.get("objects", []):
                if (o["id"] in look and o["depth"] <= max_depth and o["box"][3] - o["box"][1] >= min_px
                        and seen[o["id"]] >= min_visible):
                    if o["depth"] < near.get(o["id"], (max_depth + 1,))[0]:
                        near[o["id"]] = (o["depth"], (o["box"][0] + o["box"][2]) / 2 / r["size"][0])
        if not near:
            out[cam] = "The road in view is empty."
            continue
        n = len(near)
        lines = [f"Exactly {NUMBER.get(n, str(n))} vehicle{'s are' if n > 1 else ' is'} in view."]
        for i in sorted(near, key=lambda i: near[i][0]):
            depth, u = near[i]
            where = "on the left" if u < 1 / 3 else "on the right" if u > 2 / 3 else "in the middle"
            where += ", far away" if depth > 50 else ", close" if depth < 10 else ""
            lines.append(f"{look[i][0].upper()}{look[i][1:]} is {where}.")
        out[cam] = " ".join(lines)
    return out


def _boxes(proj: dict | None) -> dict | None:
    """The projected objects of :func:`tsf.cosmos.adherence.project_clip` in the drift guard's box format."""
    return {c: {"size": r["size"], "boxes": [o["box"] for o in r["objects"]]} for c, r in proj.items()} if proj else None


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
    """``duration``: captions are written once for the whole drive and each chunk gets its time span of them (without it,
    one chunk's caption is reused for every chunk: the drive's opening described for its whole length). ``guard``: the
    conditioning frames pass through :class:`tsf.cosmos.drift.DriftGuard` (colour, sharpness, re-anchor)."""

    def __init__(self, tsir: dict, out: Path, cams: list[str], res: str = "480", captions: dict | None = None,
                 dry: bool = False, resume: bool = False, duration: float | None = None, guard: bool = True,
                 chunk: int = CHUNK, judge: bool = False):
        from tsf.cosmos.drift import DriftGuard
        from tsf.cosmos.llm_captions import llm_captions
        self.tsir, self.out, self.cams, self.res, self.dry, self.resume = tsir, out, cams, res, dry, resume
        self.chunk = chunk                                  # frames per request: 4k+1, 1 conditioning + chunk-1 new
        self.judge = judge                                  # check each chunk's signals, one re-roll if a colour is wrong
        cap_file, span_file = out / "captions.json", out / "captions_span.json"
        self.span: float | None = None                      # seconds the captions' time blocks cover
        if not captions and resume and cap_file.exists():   # the generated chunks were conditioned on these
            captions = json.loads(cap_file.read_text())
            self.span = json.loads(span_file.read_text())["span_s"] if span_file.exists() else None
        elif not captions and not dry:
            self.span, lint = duration, {}
            captions = llm_captions(tsir, cams, frames=int(round(duration * FPS)) + 1 if duration else chunk, fps=FPS,
                                    road_users=False, report=lint)       # vehicles: vehicle_cast per chunk
            (out / "captions_lint.json").write_text(json.dumps(lint, indent=1))
        self.caps = captions or {}
        (out / "gen").mkdir(parents=True, exist_ok=True)
        cap_file.write_text(json.dumps(self.caps, indent=1))
        if self.span:
            span_file.write_text(json.dumps({"span_s": self.span}))
        self.guard = DriftGuard(veil_cams=FRONT_CAMS, detail_below=0.65, marks_below=0.6) if guard and not dry else None
        self.last: dict[str, str] | None = None
        self.ctrl: dict[str, Path] | None = None            # the last chunk's control frames (front cameras) at its end
        self.proj: dict | None = None                       # the sim projected into the last chunk's last frame
        self.k, self.timing = 0, []

    def _captions(self, t0: float, signal: str = "", cast: dict | None = None) -> dict:
        """The chunk's span of the drive-long captions plus each camera's ``cast`` (:func:`vehicle_cast`); ``signal``
        (:func:`signal_caption`) is appended for the front cameras only, since a caption that names a signal makes the
        generator paint one wherever it looks."""
        from tsf.cosmos.generate import window_caption
        cast = cast or {}
        return {c: (window_caption(txt, t0, t0 + self.chunk / FPS) if self.span else txt)
                + (f" {cast[c]}" if c in cast else "") + (f" {signal}" if signal and c in FRONT_CAMS else "")
                for c, txt in self.caps.items()}

    def _project(self, cid: str) -> dict | None:
        """The sim's objects and signal heads in every camera at chunk ``cid``'s last frame (None if that fails: the
        guard then judges the veil over the whole band and the captions name no vehicles)."""
        from tsf.cosmos import adherence
        try:
            return adherence.project_clip(self.out, cid)
        except Exception as e:  # noqa: BLE001 - WSM venv missing or the export unreadable
            print(f"projection of {cid} failed: {e}", flush=True)
            return None

    def _verdicts(self, cid: str, r: dict) -> list[str]:
        """Signal verdicts (:func:`tsf.cosmos.adherence.score_signals`) of the front camera's newest frame of chunk ``cid``."""
        from tsf.cosmos import adherence
        cam = next((c for c in FRONT_CAMS if c in self.cams), None)
        rec = adherence.project_clip(self.out, cid).get(cam) if cam else None
        img = cv2.imread(r["last_frames"][cam]) if rec else None
        return [s["verdict"] for s in adherence.score_signals(img, rec)] if img is not None else []

    def _judged(self, cid: str, r: dict, regenerate) -> tuple[dict, dict]:
        """Keep chunk ``cid`` unless the front camera shows a signal colour the sim does not; then generate it once more
        with another seed (``regenerate(out_dir, seed_offset)``) and keep whichever attempt has fewer wrong colours. A
        promoted re-roll takes the chunk's directory; the first attempt moves to ``<cid>_rejected``."""
        import shutil
        first = self._verdicts(cid, r)
        info = {"first": first}
        if "wrong" not in first:
            return r, info
        gen = self.out / "gen"
        alt = regenerate(gen / f"{cid}_retry", 1000)
        info["retry"] = self._verdicts(cid, alt)
        if info["retry"].count("wrong") < first.count("wrong"):
            shutil.rmtree(gen / f"{cid}_rejected", ignore_errors=True)
            (gen / cid).rename(gen / f"{cid}_rejected")
            (gen / f"{cid}_retry").rename(gen / cid)
            res_p = gen / cid / "result.json"
            res_p.write_text(res_p.read_text().replace(f"{cid}_retry", cid))
            alt = json.loads(json.dumps(alt).replace(f"{cid}_retry", cid))
            info["kept"] = "retry"
            return alt, info
        info["kept"] = "first"
        return r, info

    def _cached(self, cid: str) -> dict | None:
        """A chunk an interrupted run already generated (``--resume``): its result plus the last frames."""
        d = self.out / "gen" / cid
        last = {c: d / f"last_{c}.png" for c in self.cams}
        if not (self.resume and (d / "result.json").exists() and all(p.exists() for p in last.values())):
            return None
        r = json.loads((d / "result.json").read_text())
        r["last_frames"] = {c: str(p) for c, p in last.items()}
        return r

    def render(self, sim: TanitSim, log: SimLog) -> dict[str, np.ndarray]:
        from tsf.cosmos import generate, rds_hq, wsm
        k, t_end = self.k, float(log.t[-1])
        cid = f"t1_{k:03d}"
        t0 = t_end - (self.chunk - 1) / FPS
        tic = time.perf_counter()
        t_exp = t_wsm = 0.0
        drift = judged = None
        signal = signal_caption(sim, t0, t_end) if self.span else ""
        r = None if self.dry else self._cached(cid)
        if r is None:
            root = rds_hq.export_clip(sim, log, self.out / "rds", cid, t0=t0, duration=self.chunk / FPS,
                                      cameras=self.cams, overwrite=True)
            t_exp = time.perf_counter() - tic
            ctrl = wsm.render_wsm(root, cid, self.out / "wsm" / cid, self.cams)
            t_wsm = time.perf_counter() - tic - t_exp
            if self.dry:                             # plumbing test: the policy sees the WSM controls themselves
                self.k += 1
                self.timing.append({"chunk": k, "t_world": round(t_end, 3), "export_s": round(t_exp, 1),
                                    "wsm_s": round(t_wsm, 1)})
                return {c: cv2.cvtColor(cv2.imread(str(generate.extract_frame(p, self.out / "gen" / f"{cid}_{c}.png",
                                                                              -1))), cv2.COLOR_BGR2RGB)
                        for c, p in ctrl.items()}
            proj = self._project(cid)
            ctrl_last = {c: generate.extract_frames(p, {self.chunk - 1: self.out / "wsm" / cid / f"last_{c}.png"})[
                self.chunk - 1] for c, p in ctrl.items() if c in FRONT_CAMS}
            first = self.last
            if self.guard is not None and first:                     # None: re-anchor on the controls
                first, drift = self.guard(first, self.out / "gen" / cid / "cond", boxes=_boxes(self.proj),
                                          controls=self.ctrl)
            cast = vehicle_cast(self.tsir, [self.proj, proj], self.cams)

            def gen(out_dir: Path, seed_offset: int = 0) -> dict:
                return generate.sim_chunk(self.out / "wsm" / cid, out_dir, self.cams, frames=self.chunk, res=self.res,
                                          captions=self._captions(t0, signal, cast), first_frames=first,
                                          seed=k + seed_offset, scenario=self.tsir["meta"]["id"])
            r = gen(self.out / "gen" / cid)
            if self.judge:
                r, judged = self._judged(cid, r, gen)
        else:
            proj = self._project(cid)                                # resumed: the chunk's export is still on disk
            ctrl_last = {c: p for c in FRONT_CAMS if (p := self.out / "wsm" / cid / f"last_{c}.png").exists()}
        if self.guard is not None and k == 0:
            self.guard.reference(r["last_frames"], _boxes(proj), ctrl_last)
        self.last, self.proj, self.ctrl = r["last_frames"], proj, ctrl_last
        self.k += 1
        self.timing.append({"chunk": k, "t_world": round(t_end, 3), "export_s": round(t_exp, 1),
                            "wsm_s": round(t_wsm, 1), "cosmos_s": round(r["request_s"], 1),
                            **({"signal": signal} if signal else {}), **({"drift": drift} if drift else {}),
                            **({"judge": judged} if judged else {})})
        # Sim-Auto's 480p preset (832x480) is a non-uniform rescale of the 16:9 control geometry: undo it for the rig
        return {c: cv2.resize(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB), (1280, 720), interpolation=cv2.INTER_CUBIC)
                for c, p in self.last.items()}


def run_t1(tsir: dict, policy, out: str | Path, seconds: float | None = None, cams: list[str] = REFE_CAMS,
           res: str = "480", dry: bool = False, resume: bool = False, guard: bool = True,
           chunk: int = CHUNK, judge: bool = False) -> ClosedLoopResult:
    """``policy=None``: the scenario's scripted ego drives and the sensors are still generated every chunk, so sensor
    variants (resolution, chunk length, guard, captions) can be compared on one identical world."""
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
    sensors = CosmosSensors(tsir, out, cams, res, dry=dry, resume=resume, duration=sc.duration, guard=guard, chunk=chunk,
                            judge=judge)
    rec = Recorder(sim)
    n = int(round(sc.duration / sim.dt)) + 1
    chunk_s = (chunk - 1) / FPS
    next_frames = chunk_s
    tic = time.perf_counter()
    for i in range(n):
        rec.record()
        if sim.t + 1e-9 >= next_frames:
            frames = sensors.render(sim, rec.log())
            if policy is not None:
                policy.set_frames(frames)
            next_frames += chunk_s
            print(f"t={sim.t:.2f}s chunk {sensors.k} {sensors.timing[-1]} plan k={getattr(policy, 'last', {}).get('k')}",
                  flush=True)
        if i < n - 1:
            sim.step()
    log = rec.log(with_metrics=True)
    res_ = ClosedLoopResult("scripted" if policy is None else getattr(policy, "name", type(policy).__name__), log,
                            epdms(sim, log))
    res_.sim = sim
    summary = {"policy": res_.policy, "scenario": sc.meta.id, "seconds": sc.duration, "cams": cams, "res": res,
               "chunk": chunk, "guard": guard, "judge": judge, "chunks": sensors.k,
               "wall_s": round(time.perf_counter() - tic, 1),
               "score": res_.score,
               "metrics": log.metrics,   # incl. the first collisions (time, actor): the score alone hides who was hit
               "policy_calls": getattr(policy, "n_calls", None), "fallback_ticks": getattr(policy, "n_fallback", None),
               "timing": sensors.timing}
    (out / "t1_summary.json").write_text(json.dumps(summary, indent=1, default=float))
    if getattr(policy, "calls", None):
        (out / "refe_calls.json").write_text(json.dumps(policy.calls, default=float))
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
    ap.add_argument("--resume", action="store_true", help="reuse the chunks an interrupted run left in --out")
    ap.add_argument("--no-video", action="store_true",
                    help="skip the BEV video: next to a resident Cosmos server it renders on one process; replay with "
                         "--resume after stopping Cosmos instead")
    ap.add_argument("--no-drift-guard", action="store_true", help="condition each chunk on the raw last frames (A/B)")
    ap.add_argument("--scripted", action="store_true",
                    help="no policy: the scenario's scripted ego drives (sensor benchmark on an identical world)")
    ap.add_argument("--chunk", type=int, default=CHUNK, help="frames per Sim-Auto request, 4k+1 (17: 0.53 s, 25: 0.8 s)")
    ap.add_argument("--judge", action="store_true",
                    help="check each chunk's front-camera signal colours against the sim; one re-roll when one is wrong")
    a = ap.parse_args()
    if (a.chunk - 1) % 4:
        ap.error("--chunk must be 4k+1")
    tsir = json.loads(Path(a.scenario).read_text())
    out = Path(a.out)
    pol = None if a.scripted else RefePolicy(fallback="const_vel")
    r = run_t1(tsir, pol, out, a.seconds, a.cams.split(","), a.res, dry=a.dry, resume=a.resume, guard=not a.no_drift_guard,
               chunk=a.chunk, judge=a.judge)
    if a.no_video:
        print(json.dumps(r.score))
        return
    from tsf.sim.render2d import render_video
    who = "scripted ego" if a.scripted else "REFe SFT-4"
    render_video(r.sim, r.log, out / "t1_bev.mp4", title=f"T1 closed loop: {who} on Cosmos3-Nano-Sim-Auto, "
                                                           f"EPDMS {r.score['EPDMS']:.2f}")
    print(json.dumps(r.score))


if __name__ == "__main__":
    main()
