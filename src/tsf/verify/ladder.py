"""Verifier ladder L0-L3 (L4 LLM-judge and L5 plausibility live in tsf.agents / tsf.verify.plausibility).

Each check yields a `Finding`; the first failing level stops the ladder and its diagnostics are
fed back to the generating agent (repair loop) and logged as harness reward.
"""
from __future__ import annotations

import functools

import json
import math
import os
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from lxml import etree
from pydantic import ValidationError

from tsf.ir.scenario import Junction, Scenario

XSD_DIR = Path(os.environ.get("TSF_XSD_DIR", Path(__file__).resolve().parents[3] / "third_party" / "xsd"))


@dataclass
class Finding:
    level: str
    check: str
    ok: bool
    message: str = ""
    detail: dict = field(default_factory=dict)


@dataclass
class Report:
    findings: list[Finding] = field(default_factory=list)
    artifacts: dict[str, str] = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    timing: dict = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return all(f.ok for f in self.findings)

    @property
    def first_failure(self) -> Finding | None:
        return next((f for f in self.findings if not f.ok), None)

    def score(self) -> float:
        """Fraction of passed checks, levels weighted L0..L3 = 1,2,3,4 (reward for the harness)."""
        w = {"L0": 1, "L1": 2, "L2": 3, "L3": 4}
        tot = sum(w.get(f.level, 1) for f in self.findings) or 1
        return round(sum(w.get(f.level, 1) for f in self.findings if f.ok) / tot, 4)

    def diagnostics(self) -> str:
        """Compact text for the repair prompt."""
        bad = [f for f in self.findings if not f.ok]
        return "\n".join(f"[{f.level}:{f.check}] {f.message}" for f in bad) or "all checks passed"

    def to_dict(self) -> dict:
        return {"passed": self.passed, "score": self.score(), "findings": [asdict(f) for f in self.findings],
                "artifacts": self.artifacts, "metrics": self.metrics, "timing": self.timing}


# ----------------------------------------------------------------------------- L0
def l0_parse(raw: str | dict) -> tuple[Scenario | None, list[Finding]]:
    try:
        sc = Scenario.model_validate_json(raw) if isinstance(raw, str) else Scenario.model_validate(raw)
        return sc, [Finding("L0", "tsir_schema", True)]
    except ValidationError as e:
        errs = [{"loc": "/".join(map(str, x["loc"])), "msg": x["msg"]} for x in e.errors()[:12]]
        msg = "; ".join(f"{x['loc']}: {x['msg']}" for x in errs)
        return None, [Finding("L0", "tsir_schema", False, msg, {"errors": errs})]
    except json.JSONDecodeError as e:
        return None, [Finding("L0", "json", False, f"invalid JSON: {e}")]


# ----------------------------------------------------------------------------- L1
@functools.cache
def _xsd(name: str):
    """lxml for XSD 1.0 (OpenSCENARIO); xmlschema for XSD 1.1 (OpenDRIVE >= 1.8, as ASAM QC does)."""
    p = XSD_DIR / name
    if not p.exists():
        return None
    if name.startswith("OpenDRIVE"):
        import xmlschema
        return xmlschema.XMLSchema11(str(p))
    return etree.XMLSchema(etree.parse(str(p)))


def l1_xsd(path: Path, xsd_name: str, check: str) -> Finding:
    try:
        schema = _xsd(xsd_name)
    except Exception as e:  # unparseable schema is a tooling problem, not a scenario defect
        return Finding("L1", check, True, f"skipped: {type(e).__name__}: {str(e)[:120]}", {"skipped": True})
    if schema is None:
        return Finding("L1", check, True, f"skipped: {xsd_name} not installed", {"skipped": True})
    if isinstance(schema, etree.XMLSchema):
        ok = schema.validate(etree.parse(str(path)))
        errs = [f"line {e.line}: {e.message}" for e in list(schema.error_log)[:8]]
    else:
        errs = [f"{e.path}: {e.reason}" for e in schema.iter_errors(str(path))][:8]
        ok = not errs
    return Finding("L1", check, ok, "; ".join(errs), {"errors": errs})


def l1_asam_qc_opendrive(xodr: Path, outdir: Path) -> Finding:
    """ASAM Quality Checker for OpenDRIVE, run as an isolated uvx tool (it pins numpy<2)."""
    uvx = shutil.which("uvx") or str(Path.home() / ".local/bin/uvx")
    cfg = outdir / "qc_config.xml"
    cfg.write_text(f'<?xml version="1.0" encoding="UTF-8"?>\n<Config>\n  <Param name="InputFile" value="{xodr}"/>\n'
                   f'  <CheckerBundle application="xodrBundle">\n'
                   f'    <Param name="resultFile" value="{outdir / "xodr_bundle_report.xqar"}"/>\n'
                   f'  </CheckerBundle>\n</Config>\n')
    try:
        r = subprocess.run([uvx, "--from", "asam-qc-opendrive", "qc_opendrive", "-c", str(cfg)],
                           capture_output=True, text=True, timeout=300)
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        return Finding("L1", "asam_qc_opendrive", True, f"skipped: {e}", {"skipped": True})
    rep = outdir / "xodr_bundle_report.xqar"
    if not rep.exists():
        return Finding("L1", "asam_qc_opendrive", True, "skipped: no report", {"stderr": r.stderr[-800:],
                                                                             "skipped": True})
    root = etree.parse(str(rep)).getroot()
    issues = [(i.get("level"), i.get("description")) for i in root.iter("Issue")]
    errors = [d for lvl, d in issues if lvl in ("1", "ERROR", "error")]
    return Finding("L1", "asam_qc_opendrive", not errors, "; ".join(errors[:6]),
                   {"n_issues": len(issues), "n_errors": len(errors), "issues": issues[:20]})


# ----------------------------------------------------------------------------- L2
def l2_lane_graph(sc: Scenario, m) -> list[Finding]:
    """The map TanitSim reads back must realise every movement and lane the TSIR asked for."""
    from tsf.compile.xodr import lane_movements
    from tsf.sim.engine import lane_turn

    out = []
    gaps = []
    for k, ln in m.lanes.items():
        for s in ln.successors:
            g = float(np.hypot(*(ln.xy[-1] - m.lanes[s].xy[0])))
            if g > 0.1:
                gaps.append(f"{k}->{s}: {g:.2f} m")
    out.append(Finding("L2", "link_continuity", not gaps,
                       f"{len(gaps)} discontinuous lane link(s): " + "; ".join(gaps[:6]) if gaps else "",
                       {"gaps": gaps[:50]}))
    if isinstance(sc.road, Junction):
        missing = []
        for fi, fl, ti, tl, turn in lane_movements(sc):
            src = [k for k in m.lanes if k[0] == str(fi) and k[2] == fl]
            ok = False
            for k in src:
                for j in m.lanes[k].successors:
                    if m.lanes[j].junction == "-1" or lane_turn(m.lanes[j]) != turn:
                        continue
                    if any(n[0] == str(ti) and n[2] == -tl for n in m.lanes[j].successors):
                        ok = True
            if not ok:
                missing.append(f"arm{fi} lane {fl} -{turn}-> arm{ti} lane {tl}")
        out.append(Finding("L2", "junction_movements", not missing,
                           f"{len(missing)} movement(s) not realised: " + "; ".join(missing[:6]) if missing else "",
                           {"missing": missing}))
        bad = []
        for i, ap in enumerate(sc.road.approaches):
            n_in = sum(1 for k in m.lanes if k[0] == str(i) and k[2] > 0 and k[1] == 0)
            n_out = sum(1 for k in m.lanes if k[0] == str(i) and k[2] < 0 and k[1] == 0)
            if (n_in, n_out) != (ap.inbound.driving_lanes, ap.outbound.driving_lanes):
                bad.append(f"arm{i}: lanes in/out {n_in}/{n_out}, expected "
                           f"{ap.inbound.driving_lanes}/{ap.outbound.driving_lanes}")
        out.append(Finding("L2", "arm_lanes", not bad, "; ".join(bad)))
    else:
        r = sc.road
        n_f = sum(1 for k in m.lanes if k[2] < 0 and k[1] == 0)
        ok = n_f == r.forward.driving_lanes
        out.append(Finding("L2", "lanes", ok, "" if ok else f"forward lanes {n_f} != {r.forward.driving_lanes}"))
    return out


def l2_static_objects(sc: Scenario, m) -> list[Finding]:
    """Every static object must be in the compiled map (as <object> or <signal>) where TanitSim puts it: a second,
    independent implementation of the LanePosition -> world transform (xodr.py writes, odr.py reads it back)."""
    if not sc.static_objects:
        return []
    from tsf.sim.static import build_static

    try:
        bodies = build_static(sc, m)
    except Exception as e:  # position that does not resolve on the map: actionable for the agent
        return [Finding("L2", "static_objects", False, f"{type(e).__name__}: {e}")]
    in_map = {o.id: (o.x, o.y) for o in m.objects()}
    in_map.update({sg.id: (sg.x, sg.y) for sg in m.signs()})
    bad = []
    for b in bodies:
        if b.id not in in_map:
            bad.append(f"{b.id}: missing in the compiled map")
        elif math.hypot(in_map[b.id][0] - b.x, in_map[b.id][1] - b.y) > 0.1:
            bad.append(f"{b.id}: map position differs from TanitSim by "
                       f"{math.hypot(in_map[b.id][0] - b.x, in_map[b.id][1] - b.y):.2f} m")
    return [Finding("L2", "static_objects", not bad, "; ".join(bad[:6]), {"n": len(bodies), "bad": bad})]


# ----------------------------------------------------------------------------- L3
def l3_assertions(sc: Scenario, log) -> list[Finding]:
    """Simulation oracle. Failure messages carry quantitative repair hints (the repairer's main signal)."""
    mt, out = log.metrics, []
    e = log.roles.index("ego")
    v = log.states[:, e, 3]
    ego = next(a for a in sc.actors if a.role == "ego")
    v0 = ego.initial_speed
    for a in sc.assertions:
        k, ok, msg = a.kind, True, ""
        if k == "no_collision":
            ok = not mt["collision"]
            msg = "" if ok else _collision_hint(log, mt["collisions"])
        elif k == "min_ttc_above":
            ok = mt["min_ttc"] is None or mt["min_ttc"] >= a.value
            msg = "" if ok else f"min TTC {mt['min_ttc']} s with {mt['min_ttc_actor']} < {a.value} s"
        elif k == "ego_lane_change":
            want = int(a.value) if a.value is not None else 1
            ok = mt["ego_lane_changes_left"] == want
            msg = "" if ok else f"ego net lane changes to the left = {mt['ego_lane_changes_left']}, expected {want}"
        elif k == "ego_turn":
            ok = (a.label or "left") in mt["ego_turns"]
            msg = "" if ok else (f"ego turns executed {mt['ego_turns'] or 'none'}, expected '{a.label}'. "
                                 + _reach_hint(sc, ego, mt["ego_distance"], float(v[-1])))
        elif k == "red_light_respected":
            ok = not mt["red_light_violations"]
            msg = "" if ok else f"ego entered junction on red at t={mt['red_light_violations']}"
        elif k == "max_decel_below":
            ok = mt["ego_max_decel"] <= a.value
            ti = float(log.t[int(np.argmin(log.states[:, e, 4]))])
            msg = "" if ok else (f"ego peak decel {mt['ego_max_decel']} m/s2 > {a.value} at t={ti:.1f}s. Give the ego "
                                 "more distance/time before the braking event, or relax the threshold (normal urban "
                                 "braking is 2.5-4 m/s2) unless the request demands gentle braking")
        elif k == "ego_speed_between":
            ok = bool((v >= (a.value or 0) - 0.3).all() and (v <= (a.value2 or 1e9) + 0.3).all())
            msg = "" if ok else f"ego speed range [{v.min():.1f}, {v.max():.1f}] m/s outside [{a.value}, {a.value2}]"
            if not ok and not (a.value or 0) <= v0 <= (a.value2 or 1e9):
                msg += (f". The assertion contradicts the scenario itself: ego initial_speed {v0} m/s is outside the "
                        "range; widen the range or change initial_speed/speed actions")
        elif k == "ego_reaches_s":
            ok = mt["ego_distance"] >= a.value
            msg = "" if ok else f"ego travelled {mt['ego_distance']} m < {a.value} m"
        elif k == "ego_stops_before_line":
            ok = bool(v.min() < 0.3)
            msg = "" if ok else f"ego never stopped (min speed {v.min():.2f} m/s)"
        out.append(Finding("L3", f"{k}{':' + a.label if a.label else ''}", ok, msg))
    return out


def _reach_hint(sc: Scenario, ego, travelled: float, v_end: float) -> str:
    from tsf.ir.scenario import Junction
    if not isinstance(sc.road, Junction) or ego.initial.direction != "inbound":
        return f"Ego travelled {travelled} m in {sc.duration} s."
    need = ego.initial.s + 25.0  # reach the stop line and clear the junction
    if travelled >= need:
        return (f"Ego travelled {travelled} m, enough to reach the junction {ego.initial.s} m ahead: check the route "
                "(turn list), the inbound lane (left turns only from lane 1) and the signal phase timing.")
    t_more = (need - travelled) / max(v_end, 3.0)
    return (f"Ego started {ego.initial.s} m before the junction and travelled only {travelled} m in {sc.duration} s. "
            f"Increase duration to >= {sc.duration + t_more + 3:.0f} s or start the ego closer (initial.s <= "
            f"{max(5.0, travelled - 25.0):.0f} m); then re-check signal timing so it arrives on green.")


def _collision_hint(log, cols: list) -> str:
    c = cols[0]
    e = log.roles.index("ego")
    i = int(round(c["t"] / (log.t[1] - log.t[0])))
    msg = f"first contact t={c['t']:.1f}s with {c['with']}"
    sb = next((b for b in getattr(log, "static", None) or [] if b.id == c["with"]), None)
    if sb is not None:
        msg += (f" (static {sb.kind} on lane {sb.lane}, {sb.t:+.1f} m from the lane centre; ego lane "
                f"{log.lanes[i][e]}, ego {log.states[i, e, 3]:.1f} m/s)")
        return (msg + ". The ego drives into a fixed object: move the object off the ego's path (other lane or "
                "t_offset beyond the lane edge), or give the ego a lane_change / lane_offset / speed_change that "
                "clears it with room to react; do not delete the assertion.")
    if c["with"] in log.ids:
        j = log.ids.index(c["with"])
        dx, dy = log.states[i, j, :2] - log.states[i, e, :2]
        msg += (f" (ego lane {log.lanes[i][e]}, other lane {log.lanes[i][j]}, distance {math.hypot(dx, dy):.1f} m, "
                f"ego {log.states[i, e, 3]:.1f} m/s, other {log.states[i, j, 3]:.1f} m/s)")
    return (msg + ". Fix the root cause (NPC initial gaps/speeds, or the timing of the ego manoeuvre such as a lane "
            "change into an occupied lane); do not delete the assertion.")


# ----------------------------------------------------------------------------- ladder
def run(raw: str | dict | Scenario, workdir: str | Path, qc: bool = True, simulate: bool = True) -> Report:
    from tsf.compile import osc2, xodr, xosc
    from tsf.sim import odr
    from tsf.sim.engine import TanitSim

    wd = Path(workdir)
    wd.mkdir(parents=True, exist_ok=True)
    rep = Report()
    t0 = time.perf_counter()
    if isinstance(raw, Scenario):
        sc, f = raw, [Finding("L0", "tsir_schema", True)]
    else:
        sc, f = l0_parse(raw)
    rep.findings += f
    if sc is None:
        return rep
    try:
        xp = xodr.write(sc, wd / f"{sc.meta.id}.xodr")
        sp = xosc.write(sc, wd / f"{sc.meta.id}.xosc", xp.name)
        op = osc2.write(sc, wd / f"{sc.meta.id}.osc", xp.name)
        rep.artifacts.update(xodr=str(xp), xosc=str(sp), osc2=str(op))
        rep.findings.append(Finding("L1", "compile", True))
    except Exception as e:  # compiler rejected the IR: actionable for the agent
        rep.findings.append(Finding("L1", "compile", False, f"{type(e).__name__}: {e}"))
        return rep
    rep.timing["compile"] = round(time.perf_counter() - t0, 3)
    rep.findings.append(l1_xsd(xp, "OpenDRIVE_Core.xsd", "opendrive_xsd"))
    rep.findings.append(l1_xsd(sp, "OpenSCENARIO.xsd", "openscenario_xsd"))
    if qc:
        rep.findings.append(l1_asam_qc_opendrive(xp, wd))
    rep.timing["l1"] = round(time.perf_counter() - t0, 3)
    if not rep.passed:
        return rep
    try:
        m = odr.load(xp)
    except Exception as e:
        rep.findings.append(Finding("L2", "tanitsim_parse", False, f"{type(e).__name__}: {e}"))
        return rep
    rep.findings += l2_lane_graph(sc, m)
    rep.findings += l2_static_objects(sc, m)
    rep.timing["l2"] = round(time.perf_counter() - t0, 3)
    if not rep.passed or not simulate:
        return rep
    try:
        sim = TanitSim(sc, m)
        log = sim.run()
    except Exception as e:
        rep.findings.append(Finding("L3", "simulate", False, f"{type(e).__name__}: {e}"))
        return rep
    rep.metrics = log.metrics
    rep.findings += l3_assertions(sc, log)
    rep.timing["l3"] = round(time.perf_counter() - t0, 3)
    rep._sim, rep._log = sim, log  # for renderers (not serialised)
    return rep
