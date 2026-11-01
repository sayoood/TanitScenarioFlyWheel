"""UC4/UC5 agents on lifted sensor snippets (PhysicalAI-AV).

UC4  lift the snippet deterministically (tsf.ingest.pai_av), verify, then let the LLM analyse and document it
     (IMDAD package) so the scenario can be augmented along its logical parameters.
UC5  the lifted snippet is the NEGATIVE (critical ego behaviour, e.g. harsh late braking); the twin editor writes
     only an ego plan (defensive / anticipatory behaviour), spliced into a copy of the negative, so every NPC
     keeps its logged behaviour by construction. Positive assertions are relative to the negative's metrics.
"""
from __future__ import annotations

import copy
import json
import os
import time
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

from tsf.agents.lifter import Attempt, Lifter, LiftResult, assemble_package, validate_package
from tsf.agents.llm import LLM
from tsf.ingest import pai_av
from tsf.verify import ladder

REASONING = Path(os.environ.get("TSF_PAI_AV_META", Path.home() / "data" / "pai_av")) / "reasoning" / "ood_reasoning.parquet"


def event_text(clip_id: str) -> str:
    import pandas as pd
    if not REASONING.exists():
        return ""
    r = pd.read_parquet(REASONING)
    if clip_id not in r.index:
        return ""
    row = r.loc[clip_id]
    ev = json.loads(row.events) if isinstance(row.events, str) else row.events
    return f"{row.event_cluster}: " + "; ".join(f"t={e['event_start_timestamp'] / 1e6:.1f}s {e['coc']}" for e in ev)


def lift_snippet(spec: str, workdir: Path, polarity: str = "positive") -> dict:
    """spec = 'pai_av:<clip_id>[@t0]' -> verified, de-conflicted TSIR + fidelity vs the log."""
    src, _, rest = spec.partition(":")
    if src != "pai_av":
        raise ValueError(f"unsupported snippet source {src!r}")
    cid, _, t0s = rest.partition("@")
    t0 = time.perf_counter()
    clip = pai_av.load_clip(cid)
    ev = event_text(cid)
    te = float(ev.split("t=")[1].split("s")[0]) if "t=" in ev else 8.0
    start = float(t0s) if t0s else max(0.0, te - 6.0)
    d = pai_av.lift(clip, t0=start, duration=min(15.0, clip.t[-1] - start), polarity=polarity,
                    title=f"PhysicalAI-AV {cid[:8]}: {ev.split(': ', 1)[-1][:80]}" if ev else "",
                    description=f"Lifted from a 20 s PhysicalAI-AV clip (GT egomotion + auto-labelled tracks). Event: {ev}")
    gt = d.pop("_log")
    closest = min((a for a in d["actors"] if a["role"] != "ego"), default=None,
                  key=lambda a: abs(a["initial"]["s"] - d["actors"][0]["initial"]["s"]))
    d, rep, dropped = pai_av.deconflict(d, workdir / "lift", protect=(closest["id"],) if closest else ())
    pai_av.add_parameters(d)
    rep = ladder.run(d, workdir / "lift", qc=True)
    fid = pai_av.fidelity(rep._sim, rep._log, gt["gt"]) if hasattr(rep, "_sim") else {}
    return {"clip_id": cid, "t0": start, "event": ev, "tsir": d, "report": rep, "dropped": dropped,
            "fidelity": {k: v for k, v in fid.items() if k != "per_actor"}, "seconds": time.perf_counter() - t0,
            "gt_ego_speed": [round(float(v), 2) for v in np.interp(np.arange(0, 15.01, 1.0), clip.t - start, clip.v)]}


def describe(lifted: dict) -> str:
    d = lifted["tsir"]
    ego = d["actors"][0]
    npcs = [f"{a['id']} ({a['category']}, {a['initial']['direction']} lane {a['initial']['lane']}, "
            f"{a['initial']['s'] - ego['initial']['s']:+.0f} m, {a['initial_speed'] * 3.6:.0f} km/h)" for a in d["actors"][1:]]
    lanes = d["road"]["forward"]["driving_lanes"], (d["road"]["backward"] or {}).get("driving_lanes", 0)
    return (f"Real-world snippet: {lifted['event'] or 'no event annotation'}. Road: {lanes[0]} lane(s) in the ego "
            f"direction, {lanes[1]} oncoming. Ego starts in lane {ego['initial']['lane']} at "
            f"{ego['initial_speed'] * 3.6:.0f} km/h; logged ego speed profile (1 s steps, m/s): "
            f"{lifted['gt_ego_speed']}. Other road users: {'; '.join(npcs) or 'none'}.")


def document(lifted: dict, llm: LLM, workdir: Path) -> LiftResult:
    """LLM analysis + codeBeamer documentation of a verified lifted scenario -> IMDAD package."""
    lif = Lifter(llm)
    t0 = time.perf_counter()
    query = describe(lifted)
    analysis = lif.analyse(query)
    att = Attempt(lifted["tsir"], lifted["report"], lifted["seconds"])
    nar = lif.document(query, analysis, [att], att)
    pkg = assemble_package(query, analysis, [att], att, nar)
    ok, errs = validate_package(pkg)
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "package.json").write_text(json.dumps(pkg, indent=2), encoding="utf-8")
    return LiftResult(query, analysis, [att], nar, pkg, ok, errs, llm.stats(), time.perf_counter() - t0, workdir)


# ----------------------------------------------------------------------------- UC5 twin
class EgoAction(BaseModel):
    type: Literal["speed_change", "lane_offset", "lane_change", "brake"]
    start_t: float = Field(..., ge=0, description="simulation time in s")
    target_speed: float | None = Field(None, ge=0, description="speed_change: m/s")
    accel: float | None = Field(None, gt=0, le=6, description="speed_change: |a| in m/s2, comfortable 0.5-2.5")
    offset: float | None = Field(None, ge=-2.0, le=2.0, description="lane_offset: m from lane centre, + = left")
    direction: Literal["left", "right"] | None = None
    duration: float | None = Field(None, gt=0.5, le=8)
    decel: float | None = Field(None, gt=0, le=8, description="brake: m/s2")


class EgoPlan(BaseModel):
    diagnosis: str = Field(max_length=500, description="Why the logged ego behaviour is critical (cause, timing, numbers), at most 3 sentences")
    strategy: str = Field(max_length=500, description="The anticipatory/defensive behaviour of the positive twin, at most 3 sentences")
    initial_speed: float = Field(..., ge=0, description="m/s")
    actions: list[EgoAction] = Field(..., min_length=1, max_length=8)


def _splice(neg: dict, plan: EgoPlan, assertions: list[dict]) -> dict:
    pos = copy.deepcopy(neg)
    ego = pos["actors"][0]
    ego["initial_speed"] = round(plan.initial_speed, 2)
    pos["actions"] = [a for a in pos["actions"] if a["actor"] != "ego"]
    for i, a in enumerate(plan.actions):
        base = {"id": f"ego_p{i}", "actor": "ego", "start": {"kind": "sim_time", "t": round(a.start_t, 2)}}
        if a.type == "speed_change":
            base.update(type="speed_change", target_speed=a.target_speed if a.target_speed is not None else plan.initial_speed,
                        accel=a.accel or 1.5)
        elif a.type == "lane_offset":
            base.update(type="lane_offset", offset=a.offset or 0.0, duration=a.duration or 2.0)
        elif a.type == "lane_change":
            base.update(type="lane_change", direction=a.direction or "left", lanes=1, duration=a.duration or 4.0)
        else:
            base.update(type="brake", decel=a.decel or 3.0)
        pos["actions"].append(base)
    pos["assertions"] = assertions
    m = pos["meta"]
    m.update(id=m["id"] + "_pos", polarity="positive", parent_id=m["id"], source="negative_snippet",
             title=("[Positive twin] " + m["title"])[:120], tags=list(dict.fromkeys(m["tags"] + ["COUNTERFACTUAL_POSITIVE"])))
    return pos


def _twin_feedback(rep) -> str:
    """Time-resolved facts for the repairer: ego speed per second and, at the braking peak, the vehicle ahead
    (gap, closing speed). Quantitative hints are what makes verifier-guided repair work (ablation, round 1)."""
    if not hasattr(rep, "_log"):
        return ""
    log = rep._log
    e = log.roles.index("ego")
    S, t = log.states, log.t
    secs = np.arange(0.0, float(t[-1]) + 1e-6, 1.0)
    v = np.interp(secs, t, S[:, e, 3])
    k = int(np.argmin(S[:, e, 4]))
    ex, ey, eyaw = S[k, e, :3]
    best = None
    for j, aid in enumerate(log.ids):
        if j == e or log.lanes[k][j] != log.lanes[k][e]:
            continue
        dx, dy = S[k, j, 0] - ex, S[k, j, 1] - ey
        lon = dx * np.cos(eyaw) + dy * np.sin(eyaw)
        if lon > 0 and (best is None or lon < best[1]):
            best = (aid, lon, S[k, j, 3])
    lead = (f"at the braking peak (t={t[k]:.1f} s, ego {S[k, e, 3]:.1f} m/s) the vehicle ahead in the ego lane is "
            f"{best[0]}: {best[1] - (log.lwh[e, 0] + log.lwh[log.ids.index(best[0]), 0]) / 2:.1f} m bumper gap, "
            f"{best[2]:.1f} m/s. " if best else f"at the braking peak (t={t[k]:.1f} s) no vehicle is ahead in the ego lane. ")
    return f"Twin ego speed per second (m/s): {np.round(v, 1).tolist()}; {lead}\n"


def twin(lifted: dict, llm: LLM, workdir: Path, max_repairs: int = 3) -> LiftResult:
    neg, nrep = lifted["tsir"], lifted["report"]
    nm = nrep.metrics
    target_decel = round(max(1.5, min(3.0, 0.75 * nm.get("ego_max_decel", 4.0))), 2)
    assertions = [{"kind": "no_collision"}, {"kind": "max_decel_below", "value": target_decel},
                  {"kind": "min_ttc_above", "value": 2.0}]
    ctx = {"situation": describe(lifted), "negative_metrics": {k: nm.get(k) for k in
           ("ego_max_decel", "ego_max_abs_jerk", "min_ttc", "min_ttc_actor", "ego_mean_speed", "collision")},
           "npc_actions": [a for a in neg["actions"] if a["actor"] != "ego"][:30],
           "logged_ego_actions": [a for a in neg["actions"] if a["actor"] == "ego"], "duration_s": neg["duration"],
           "pass_criteria": {"no_collision": True, "peak_decel_below_mps2": target_decel, "min_ttc_above_s": 2.0}}
    msgs = [{"role": "system", "content": "You are an automated-driving safety engineer. Given a critical real-world "
             "snippet (the NEGATIVE), write the ego plan of its POSITIVE twin: same road users and their behaviour, "
             "but an ego that anticipates (earlier, gentler speed reduction, larger gap, lateral clearance) and meets "
             "the pass criteria while keeping progress reasonable. Only the ego plan is yours to change."},
            {"role": "user", "content": json.dumps(ctx, separators=(",", ":"))}]
    attempts, t0 = [], time.perf_counter()
    for i in range(max_repairs + 1):
        ta = time.perf_counter()
        plan = EgoPlan.model_validate(llm.chat(msgs, schema=EgoPlan.model_json_schema(), role="twin_editor",
                                               max_tokens=4096))
        pos = _splice(neg, plan, assertions)
        rep = ladder.run(pos, workdir / f"attempt{i}", qc=True)
        attempts.append(Attempt(pos, rep, time.perf_counter() - ta))
        (workdir / f"plan{i}.json").write_text(plan.model_dump_json(indent=1), encoding="utf-8")
        if rep.passed:
            break
        msgs = msgs + [{"role": "assistant", "content": plan.model_dump_json()},
                       {"role": "user", "content": "The verifier rejected the twin:\n" + rep.diagnostics() +
                        f"\nTwin metrics: {json.dumps({k: rep.metrics.get(k) for k in ctx['negative_metrics']})}\n"
                        + _twin_feedback(rep) +
                        "Revise the ego plan (start the speed reduction earlier or lower, add clearance)."}]
    final = next((a for a in reversed(attempts) if a.report.passed), attempts[-1])
    lif = Lifter(llm)
    query = (f"Positive twin of: {describe(lifted)} Diagnosis: {plan.diagnosis} Strategy: {plan.strategy}")
    analysis = lif.analyse(query)
    nar = lif.document(query, analysis, attempts, final)
    pkg = assemble_package(query, analysis, attempts, final, nar)
    ok, errs = validate_package(pkg)
    (workdir / "package.json").write_text(json.dumps(pkg, indent=2), encoding="utf-8")
    return LiftResult(query, analysis, attempts, nar, pkg, ok, errs, llm.stats(), time.perf_counter() - t0, workdir,
                      extra={"negative_metrics": ctx["negative_metrics"], "plan": plan.model_dump()})
