"""Verifier ladder L0-L3 (L4 LLM-judge and L5 plausibility live in tsf.agents / tsf.verify.plausibility).

Each check yields a `Finding`; the first failing level stops the ladder and its diagnostics are
fed back to the generating agent (repair loop) and logged as harness reward.
"""
from __future__ import annotations

import json
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
def _xsd(name: str) -> etree.XMLSchema | None:
    p = XSD_DIR / name
    return etree.XMLSchema(etree.parse(str(p))) if p.exists() else None


def l1_xsd(path: Path, xsd_name: str, check: str) -> Finding:
    schema = _xsd(xsd_name)
    if schema is None:
        return Finding("L1", check, True, f"skipped: {xsd_name} not installed", {"skipped": True})
    doc = etree.parse(str(path))
    ok = schema.validate(doc)
    errs = [f"line {e.line}: {e.message}" for e in list(schema.error_log)[:8]]
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


# ----------------------------------------------------------------------------- L3
def l3_assertions(sc: Scenario, log) -> list[Finding]:
    mt, out = log.metrics, []
    e = log.roles.index("ego")
    v = log.states[:, e, 3]
    for a in sc.assertions:
        k, ok, msg = a.kind, True, ""
        if k == "no_collision":
            ok = not mt["collision"]
            msg = "" if ok else f"collision(s): {mt['collisions']}"
        elif k == "min_ttc_above":
            ok = mt["min_ttc"] is None or mt["min_ttc"] >= a.value
            msg = "" if ok else f"min TTC {mt['min_ttc']} s with {mt['min_ttc_actor']} < {a.value} s"
        elif k == "ego_lane_change":
            want = int(a.value) if a.value is not None else 1
            ok = mt["ego_lane_changes_left"] == want
            msg = "" if ok else f"ego net lane changes to the left = {mt['ego_lane_changes_left']}, expected {want}"
        elif k == "ego_turn":
            ok = (a.label or "left") in mt["ego_turns"]
            msg = "" if ok else f"ego turns executed {mt['ego_turns'] or 'none'}, expected '{a.label}'" \
                                f" (did ego reach the junction within {sc.duration}s? distance {mt['ego_distance']} m)"
        elif k == "red_light_respected":
            ok = not mt["red_light_violations"]
            msg = "" if ok else f"ego entered junction on red at t={mt['red_light_violations']}"
        elif k == "max_decel_below":
            ok = mt["ego_max_decel"] <= a.value
            msg = "" if ok else f"ego peak decel {mt['ego_max_decel']} m/s2 > {a.value}"
        elif k == "ego_speed_between":
            ok = bool((v >= (a.value or 0) - 0.3).all() and (v <= (a.value2 or 1e9) + 0.3).all())
            msg = "" if ok else f"ego speed range [{v.min():.1f}, {v.max():.1f}] m/s outside [{a.value}, {a.value2}]"
        elif k == "ego_reaches_s":
            ok = mt["ego_distance"] >= a.value
            msg = "" if ok else f"ego travelled {mt['ego_distance']} m < {a.value} m"
        elif k == "ego_stops_before_line":
            ok = bool(v.min() < 0.3)
            msg = "" if ok else f"ego never stopped (min speed {v.min():.2f} m/s)"
        out.append(Finding("L3", f"{k}{':' + a.label if a.label else ''}", ok, msg))
    return out


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
