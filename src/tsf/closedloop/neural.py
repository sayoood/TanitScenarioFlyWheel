"""T1 neural closed loop: TanitSim world + Cosmos3-Nano-Sim-Auto sensors + a camera policy (REFe SFT-4).

Sim-time-decoupled loop (the world waits while the sensors are generated):

    every chunk (16 new frames @ 30 fps = 0.533 s of world time, i.e. the policy's 2 Hz replan):
      1. export the last 17 frames of the *closed-loop* world (ego as driven by the policy) to RDS-HQ
      2. WSM v4 renders the control videos of the cameras the policy needs
      3. Sim-Auto generates the chunk, conditioned on the previous chunk's last frames (I2V chaining),
         captions: the chunk's span of one drive-long caption plus, for the front cameras, the junction ahead and its lights
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
ACHROMATIC = ("black", "dark grey", "grey", "silver", "white")


def junction_caption(sim: TanitSim, t0: float, t1: float, max_dist: float = 90.0) -> str:
    """The junction ahead of the ego over ``[t0, t1]`` (sim time) as caption sentences, while the ego is within
    ``max_dist`` m of its stop line ('' otherwise): how far it is and, at a signalised one, the lights on the mast arm
    beyond it (:meth:`tsf.cosmos.rds_hq.MapLayers._signals`: one above each inbound lane, the left lane's showing the
    left-turn state under a protected left) with their colours. The drive-long captions leave the junction out: written
    once before the drive, they guessed signal colours from the plan (a green left-turn signal while the map showed red),
    and their 'intersection directly in front, traffic-light poles at its edges, 140 m away' in every chunk drew signal
    poles beside the road 90 m before it (720l)."""
    ego = next(a for a in sim.agents.values() if a.role == "ego")
    if ego.path is None:
        return ""
    nj, arm = sim._next_junction_lane(ego), sim._arm_of(ego)
    if nj is None or arm is None or (d := ego.path.starts[nj[0]] - ego.p) > max_dist:
        return ""
    out = f"A junction is about {max(5, 5 * round(d / 5))} metres ahead."
    plan = getattr(sim, "signals", None)
    if plan is None or not plan.enabled:
        return out
    lanes = sim.sc.road.approaches[arm].inbound
    n = lanes.driving_lanes + lanes.left_turn_lane + lanes.right_turn_lane

    def say(what: str, turn: str, many: bool = False) -> str:
        a, b = plan.state(arm, turn, t0), plan.state(arm, turn, t1)
        return f"{what} {'are' if many else 'is'} {a}" if a == b else f"{what} turn{'' if many else 's'} from {a} to {b}"
    if plan.protected and n == 1:
        return (f"{out} Beyond it, a mast arm holds two traffic lights above the lane: {say('the left one', 'left')} "
                f"and {say('the right one', 'straight')}.")
    if plan.protected:
        rest = say("the one above the right lane", "straight") if n == 2 else \
            say("those above the other lanes", "straight", many=True)
        return (f"{out} Beyond it, a mast arm holds one traffic light above each lane: "
                f"{say('the one above the left lane', 'left')} and {rest}.")
    if n == 1:
        return f"{out} Beyond it, {say('the traffic light on a mast arm above the lane', 'straight')}."
    return f"{out} Beyond it, a mast arm holds one traffic light above each lane, and {say('they', 'straight', many=True)}."


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


def seen_colours(last_frames: dict[str, str], proj: dict | None, min_px: float = 64.0,
                 min_visible: float = 0.8) -> dict[str, str | None]:
    """How each vehicle looks in a chunk's last frames, from the camera where its box is widest (at least ``min_px``
    generated pixels, ``min_visible`` of it uncovered, inside the frame): None where it shows a colour, else 'black',
    'dark grey', 'grey' or 'white' by the median lightness of the box's inner part. A colour shows where at least 12 %
    of those pixels are lit (L over 30 of 255) and more than 12 off grey in a/b, once the frame's colour cast (the
    road and ground outside the boxes) is taken out in proportion to their lightness: the warm cast on the white truck
    read as a colour, and taken out in full it made black cars blue. The chain loses colours as vehicles come close:
    the blue van ahead (30-63 % coloured) was black from 19 s on (2-7 %; the black SUV 0 %, the white truck 1-2 %)
    and the red car crossing on the left dark grey, so captions that named them by the scenario recoloured them; a
    yellow bus seen from the front, its windscreen filling the box, still has 18-50 %."""
    best: dict[str, tuple[float, str | None]] = {}
    for cam, r in (proj or {}).items():
        img = cv2.imread(str(last_frames[cam])) if cam in last_frames else None
        if img is None:
            continue
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        (h, w_img), (sx, sy) = img.shape[:2], (img.shape[1] / r["size"][0], img.shape[0] / r["size"][1])
        free = np.zeros((h, w_img), bool)
        free[h // 2:] = True
        for o in r["objects"]:
            u0, v0, u1, v1 = o["box"]
            free[max(int(v0 * sy), 0):max(int(v1 * sy), 0), max(int(u0 * sx), 0):max(int(u1 * sx), 0)] = False
        ground = lab[free] if free.mean() > 0.05 else lab[h // 2:].reshape(-1, 3)
        light, cast = max(float(np.median(ground[:, 0])), 1.0), np.median(ground[:, 1:], 0) - 128
        uncovered = _visible(r["objects"], r["size"])
        for o in r["objects"]:
            u0, v0, u1, v1 = o["box"]
            w = (u1 - u0) * sx
            if (uncovered[o["id"]] < min_visible or w < min_px or w <= best.get(o["id"], (0.0,))[0]
                    or min(u0, v0) < 0 or u1 > r["size"][0] or v1 > r["size"][1]):
                continue
            du, dv = (u1 - u0) / 5, (v1 - v0) / 5
            px = lab[int((v0 + dv) * sy):int((v1 - dv) * sy), int((u0 + du) * sx):int((u1 - du) * sx)].reshape(-1, 3)
            ab = px[:, 1:] - 128 - cast * (px[:, :1] / light)
            if ((px[:, 0] > 30) & (np.hypot(*ab.T) > 12)).mean() >= 0.12:
                best[o["id"]] = (w, None)
                continue
            lum = float(np.median(px[:, 0])) * 100 / 255
            best[o["id"]] = (w, "black" if lum < 20 else "dark grey" if lum < 40 else "grey" if lum < 60 else "white")
    return {i: word for i, (_, word) in best.items()}


def settle_looks(seen: dict[str, str | None], history: dict[str, list], verdicts: dict[str, str | None]) -> None:
    """Update ``seen`` (each vehicle's look) from a chunk's :func:`seen_colours` ``verdicts``: a look changes between a
    colour (None) and black, grey or white only when the vehicle's last two verdicts (kept in ``history``) agree. Single
    verdicts flip: the van, dark for 14 chunks, read as coloured once at 25.1 s and the redraw next drew it blue again,
    and the yellow bus read black once at 34.7 s."""
    for i, v in verdicts.items():
        last = history[i] = (history.get(i, []) + [v])[-2:]
        if len(last) == 2 and (last[0] is None) == (last[1] is None):
            seen[i] = v


def vehicle_cast(tsir: dict, projs: list[dict | None], cams: list[str], max_depth: float = 120.0,
                 min_px: float = 6.0, min_visible: float = 0.3, looks: dict[str, str] | None = None) -> dict[str, str]:
    """One caption sentence per camera naming the vehicles the map shows in it, nearest first, by the scenario's colour
    and category. ``projs``: :func:`tsf.cosmos.adherence.project_clip` of the frames the chunk spans (its conditioning
    frame and its last frame); no projection, no sentences. The drive-long captions named road users per direction from
    the plan: the front caption left out the van ahead (generated white, blue after a re-anchor), and a truck in a block
    that also named a yellow bus came back as a bus. Named per chunk from the map, a re-drawn chunk gets the same looks.
    Listed in one sentence the colours still leaked ("a grey car, a yellow bus, a red car and a white truck" drew the
    truck yellow), so each vehicle gets its own sentence with where it is in the view: the third of the width its box
    centre falls in, far away beyond 50 m, close within 10 m (from the projection where it is nearest). A vehicle that
    nearer ones hide is not named: under ``min_visible`` of its box uncovered in every projection. The yellow bus
    behind the van beside the ego was named "on the right, close" for 6 s, and its yellow went onto the truck.
    ``looks``: colours that replace the scenario's (vehicles named as the chain shows them)."""
    if not any(projs):
        return {}
    looks = looks or {}
    look = {a["id"]: _vehicle_name({**a, "color": looks.get(a["id"], a.get("color"))})
            for a in tsir["actors"] if a.get("role") != "ego"}
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
        from tsf.cosmos.hood import Hood
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
                                    road_users=False, junction=False, repeat_setting=True, report=lint)
            # vehicles: vehicle_cast per chunk; the junction: junction_caption; the setting in every time block, since
            # each chunk gets its block alone and a redraw draws the world from it
            (out / "captions_lint.json").write_text(json.dumps(lint, indent=1))
        self.caps = captions or {}
        (out / "gen").mkdir(parents=True, exist_ok=True)
        cap_file.write_text(json.dumps(self.caps, indent=1))
        if self.span:
            span_file.write_text(json.dumps({"span_s": self.span}))
        self.guard = DriftGuard(veil_cams=FRONT_CAMS, detail_below=0.65, marks_below=0.6) if guard and not dry else None
        self.hood = Hood() if not dry else None              # chunk 0's hood on every later front frame
        self.last: dict[str, str] | None = None
        self.ctrl: dict[str, Path] | None = None            # the last chunk's control frames (front cameras) at its end
        self.proj: dict | None = None                       # the sim projected into the last chunk's last frame
        self.seen: dict[str, str | None] = {}               # each vehicle's look (settle_looks), for the cast
        self.verdicts: dict[str, list] = {}                 # its last two seen_colours verdicts
        self.k, self.timing = 0, []

    def _captions(self, t0: float, signal: str = "", cast: dict | None = None) -> dict:
        """The chunk's span of the drive-long captions plus each camera's ``cast`` (:func:`vehicle_cast`); ``signal``
        (:func:`junction_caption`) is appended for the front cameras only, since a caption that names a signal makes the
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
        first = self._verdicts(cid, r)
        info = {"first": first}
        if "wrong" not in first:
            return r, info
        alt = regenerate(self.out / "gen" / f"{cid}_retry", 1000)
        info["retry"] = self._verdicts(cid, alt)
        if info["retry"].count("wrong") < first.count("wrong"):
            info["kept"] = "retry"
            return self._promote(cid, "retry", alt), info
        info["kept"] = "first"
        return r, info

    def _promote(self, cid: str, tag: str, alt: dict) -> dict:
        """Re-roll ``<cid>_<tag>`` takes the chunk's directory; the attempt there moves to ``<cid>_rejected``."""
        import shutil
        gen = self.out / "gen"
        shutil.rmtree(gen / f"{cid}_rejected", ignore_errors=True)
        (gen / cid).rename(gen / f"{cid}_rejected")
        (gen / f"{cid}_{tag}").rename(gen / cid)
        res_p = gen / cid / "result.json"
        res_p.write_text(res_p.read_text().replace(f"{cid}_{tag}", cid))
        return json.loads(json.dumps(alt).replace(f"{cid}_{tag}", cid))

    def _hooded(self, cid: str, r: dict, regenerate, tries: int = 2) -> tuple[dict, dict]:
        """A redraw (a chunk generated without conditioning frames) draws its own hood; when that shows around the
        pasted one (:meth:`Hood.leak` above :data:`LEAK_ABOVE`) generate it again with other seeds, up to ``tries``
        times, and keep the attempt that leaks least, before the paste."""
        from tsf.cosmos.hood import LEAK_ABOVE
        cam = next(c for c in self.hood.cams if c in r["last_frames"])
        score = lambda x: round(self.hood.leak(cv2.imread(str(x["last_frames"][cam])), cam), 3)
        best, scores = (0, r), [score(r)]
        for i in range(1, tries + 1):
            if scores[best[0]] <= LEAK_ABOVE:
                break
            alt = regenerate(self.out / "gen" / f"{cid}_hood{i}", 1000 * (i + 1))
            scores.append(score(alt))
            if scores[-1] < scores[best[0]]:
                best = (i, alt)
        info = {"leak": scores, "kept": best[0]}
        return (self._promote(cid, f"hood{best[0]}", best[1]) if best[0] else r), info

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
        drift = judged = hooded = looks = None
        signal = junction_caption(sim, t0, t_end) if self.span else ""
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
            # the vehicles the chain lost a colour to are named as it shows them: a redraw named by the scenario
            # recoloured the van (720i), and chunks that kept naming it blue pulled the black van back to navy (720j)
            colour = {a["id"]: a.get("color") for a in self.tsir["actors"]}
            looks = {i: c for i, c in self.seen.items() if c and colour.get(i) not in ACHROMATIC}
            cast = vehicle_cast(self.tsir, [self.proj, proj], self.cams, looks=looks)

            def gen(out_dir: Path, seed_offset: int = 0) -> dict:
                return generate.sim_chunk(self.out / "wsm" / cid, out_dir, self.cams, frames=self.chunk, res=self.res,
                                          captions=self._captions(t0, signal, cast), first_frames=first,
                                          seed=k + seed_offset, scenario=self.tsir["meta"]["id"])
            r = gen(self.out / "gen" / cid)
            if self.judge:
                r, judged = self._judged(cid, r, gen)
            if self.hood is not None and k and first is None:
                r, hooded = self._hooded(cid, r, gen)
            if self.hood is not None and k:
                self.hood(r["last_frames"], r["videos"])
        else:
            proj = self._project(cid)                                # resumed: the chunk's export is still on disk
            ctrl_last = {c: p for c in FRONT_CAMS if (p := self.out / "wsm" / cid / f"last_{c}.png").exists()}
            if self.guard is not None and self.last:        # the guard's re-anchors and references as if generated
                drift = self.guard(self.last, self.out / "gen" / cid / "cond", boxes=_boxes(self.proj),
                                   controls=self.ctrl)[1]
        if self.guard is not None and k == 0:
            self.guard.reference(r["last_frames"], _boxes(proj), ctrl_last)
        if self.hood is not None and k == 0:
            self.hood.reference(r["last_frames"])
        self.last, self.proj, self.ctrl = r["last_frames"], proj, ctrl_last
        settle_looks(self.seen, self.verdicts, seen_colours(r["last_frames"], proj))
        self.k += 1
        self.timing.append({"chunk": k, "t_world": round(t_end, 3), "export_s": round(t_exp, 1),
                            "wsm_s": round(t_wsm, 1), "cosmos_s": round(r["request_s"], 1),
                            **({"signal": signal} if signal else {}), **({"drift": drift} if drift else {}),
                            **({"judge": judged} if judged else {}), **({"hood": hooded} if hooded else {}),
                            **({"looks": looks} if looks else {})})
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
    pol = None if a.scripted else RefePolicy(fallback="const_vel", replan_on_frames=True)
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
