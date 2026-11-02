"""Scenario augmentation agent: a verified parent TSIR -> variants along three dimensions, each with a new scenario
description, verified scenario files (OpenDRIVE, OpenSCENARIO XML + DSL, IMDAD v2 record) and generated sensor data.

    D1 weather & time of day   (``weather``)      /environment/*   (random draw in auto mode, never equal to the parent)
    D2 behaviour               (``behavior``)     ego / other agents: speeds, positions, routes, actions, extra NPCs
    D3 environment             (``environment``)  road geometry, lanes, signs, static objects, ``environment.setting``

AUTO mode makes exactly three variants, one per dimension (D1 drawn with a seeded RNG, D2 / D3 chosen by the LLM from
scenario-specific levers). MANUAL mode takes free text per dimension and makes ONE variant applying all of them
(``separate=True``: one variant per given dimension).

Workflow per variant (every step is timed and the LLM tokens are counted):
  1 plan (code)  2 edit (Qwen3.8, restricted RFC 6902 patch, whitelist per dimension)  3 apply + isolation check
  4 verifier ladder L0-L3 with up to 3 repairs  5 criticality targeting for D2 (Sobol + cross-entropy, TanitSim)
  6 describe (Qwen3.8) + deterministic intent check  7 package (OpenDRIVE / OSC / IMDAD v2)
  8 render camera + LiDAR (``tsf.cosmos.pipeline.render_sensors``)  9 judge (VLM on frames, LiDAR checks)
The editor's rules live in ``harness/augmenter`` (Harness²), probes in ``harness/augmenter/probes.jsonl``.

    python -m tsf.agents.augment parent.json out_dir --mode auto [--seed N] [--no-render]
    python -m tsf.agents.augment parent.json out_dir --mode manual --weather "night with heavy rain" --behavior "..."
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from pydantic import BaseModel, Field

from tsf.agents import augment_patch as P
from tsf.agents import augment_plan as PL
from tsf.agents.harness import ROOT as HARNESS_ROOT, Harness
from tsf.agents.llm import LLM
from tsf.ir.scenario import Scenario, json_schema
from tsf.verify import ladder

AGENT = "augmenter"


def harness_root() -> Path:
    """Where the augmenter's harness lives: the live (evolving) state dir when it has one, else the versioned repo copy.
    The live API runs with TSF_HARNESS=<state dir> that only carries agents that were evolved."""
    if (HARNESS_ROOT / AGENT / "system_prompt.md").exists():
        return HARNESS_ROOT
    return Path(__file__).resolve().parents[3] / "harness"


_LADDER_LOCK = threading.Lock()      # lxml / xmlschema validators and the ASAM QC call are not worth racing
BAND = (1.0, 2.5)                    # target min-TTC band of the criticality search (s)
DIM_TAG = {"weather": "weather", "behavior": "behavior", "environment": "environment"}


# ----------------------------------------------------------------------------- helpers
def normalise(doc: dict) -> dict:
    """Schema-validated TSIR with every default filled in (so JSON-Patch can address any field)."""
    return Scenario.model_validate(doc).model_dump(mode="json")


def load_parent(parent) -> dict:
    d = json.loads(Path(parent).read_text(encoding="utf-8")) if isinstance(parent, (str, Path)) else copy.deepcopy(parent)
    if "meta" not in d:
        d = (d.get("tanitFlywheel") or {}).get("tsir") or d.get("tsir") or d
    return normalise(d)


def _compact(x) -> str:
    return json.dumps(x, separators=(",", ":"), ensure_ascii=False)


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def _write(p: Path, obj) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=str) if not isinstance(obj, str) else obj,
                 encoding="utf-8")
    return p


def _findings(rep) -> list[dict]:
    return [{"level": f.level, "check": f.check, "ok": f.ok, "message": f.message} for f in rep.findings]


def _report(rep) -> dict:
    return {"passed": rep.passed, "score": rep.score(), "findings": _findings(rep), "metrics": rep.metrics,
            "timing": rep.timing, "artifacts": rep.artifacts}


def _set_ptr(doc, ptr: str, value):
    toks = P.tokens(ptr)
    cur = doc
    for t in toks[:-1]:
        cur = cur[int(t)] if isinstance(cur, list) else cur[t]
    if isinstance(cur, list):
        cur[int(toks[-1])] = value
    else:
        cur[toks[-1]] = value


def _metrics_line(m: dict) -> str:
    return _compact({k: m.get(k) for k in ("collision", "min_ttc", "min_ttc_actor", "ego_max_decel", "ego_turns",
                                           "ego_lane_changes_left", "red_light_violations", "ego_distance")})


def blame(errors: list[dict], ops: list[dict]) -> str:
    """Name the ops that touched the paths a schema error points at (so the repair knows what to undo)."""
    out = []
    for e in errors[:8]:
        loc = "/" + str(e.get("loc", "")).strip("/")
        for k, op in enumerate(ops):
            pth = op.get("path", "")
            if pth and (loc.startswith(pth) or pth.startswith(loc)) and loc != "/":
                out.append(f"{e['loc']} <- op {k} ({op['op']} {pth})")
                break
    return (" | caused by: " + "; ".join(out)) if out else ""


def sim_effect(plog, vlog, eps: float = 1.0) -> dict:
    """How different the variant's replay is from the parent's: ego path deviation, moved agents, metric deltas."""
    import numpy as np
    n = min(len(plog.t), len(vlog.t))
    e0, e1 = plog.roles.index("ego"), vlog.roles.index("ego")

    def dev(i0, i1):
        d = np.hypot(plog.states[:n, i0, 0] - vlog.states[:n, i1, 0], plog.states[:n, i0, 1] - vlog.states[:n, i1, 1])
        return float(d.max())
    ego_dev = dev(e0, e1)
    moved = [a for a in vlog.ids if a in plog.ids and a != "ego" and dev(plog.ids.index(a), vlog.ids.index(a)) > eps]
    new = [a for a in vlog.ids if a not in plog.ids]
    pm, vm = plog.metrics, vlog.metrics
    d_ttc = (abs((vm["min_ttc"] or 0) - (pm["min_ttc"] or 0)) if (vm["min_ttc"] is not None or pm["min_ttc"] is not None) else 0.0)
    d_dec = abs(vm["ego_max_decel"] - pm["ego_max_decel"])
    rel = (ego_dev >= eps or d_ttc >= 0.25 or d_dec >= 0.25 or vm["collision"] != pm["collision"]
           or vm["min_ttc_actor"] != pm["min_ttc_actor"])
    return {"ego_dev_m": round(ego_dev, 2), "actors_changed": moved, "new_actors": new,
            "min_ttc": [pm["min_ttc"], vm["min_ttc"]], "ego_max_decel": [pm["ego_max_decel"], vm["ego_max_decel"]],
            "ego_relevant": bool(rel)}


# ----------------------------------------------------------------------------- editor
class Described(BaseModel):
    title: str = Field(..., max_length=110, description="short scenario title naming what is different")
    description: str = Field(..., max_length=900, description="what changed vs the parent and why it matters for testing")


def edit_schema(dims, lever_ids: list[str], use_pattern: bool = True) -> dict:
    path: dict = {"type": "string"}
    if use_pattern:
        path["pattern"] = P.write_pattern(dims)
    op = {"type": "object", "additionalProperties": False, "required": ["op", "path"],
          "properties": {"op": {"enum": ["add", "replace", "remove"]}, "path": path, "value": {}}}
    return {"type": "object", "additionalProperties": False, "required": ["lever", "rationale", "limitations", "ops"],
            "properties": {"lever": {"enum": list(lever_ids) + ["custom"]},
                           "rationale": {"type": "string", "maxLength": 400},
                           "limitations": {"type": "string", "maxLength": 300},
                           "ops": {"type": "array", "minItems": 1, "maxItems": 12, "items": op}}}


@dataclass
class Spec:
    idx: int
    vid: str
    dims: list[str]
    mode: str
    requests: dict = field(default_factory=dict)      # manual: dimension -> free text
    target: dict | None = None                         # auto D1: planned environment
    levers: dict = field(default_factory=dict)         # auto D2 / D3: dimension -> [(id, text)]
    facts: dict = field(default_factory=dict)          # manual D3: dimension -> placement facts for static objects
    seed: int = 0


@dataclass
class Cand:
    """One editor proposal evaluated against the parent."""
    ops: list
    lever: str = ""
    rationale: str = ""
    doc: dict | None = None
    report: object = None
    iso: P.Isolation | None = None
    issues: list = field(default_factory=list)
    stage: str = "ok"          # first failing stage: patch | schema | isolation | environment | verify | effect | ok
                               # ("critical" = valid but the minimum TTC missed the target band)
    limitations: str = ""
    crit: dict | None = None
    effect: dict | None = None
    seconds: float = 0.0
    edit_s: float = 0.0
    verify_s: float = 0.0
    tokens: dict = field(default_factory=dict)
    qc: bool = False

    @property
    def passed(self) -> bool:
        return self.stage == "ok" and self.report is not None and self.report.passed

    @property
    def valid(self) -> bool:
        return self.stage in ("ok", "critical") and self.report is not None and self.report.passed


class ScenarioAugmenter:
    def __init__(self, llm: LLM | Callable[[], LLM] | None = None, harness: Harness | None = None,
                 max_repairs: int | None = None, log: Callable[[str], None] | None = None, band=BAND):
        self._llm = llm
        self.band = tuple(band)
        self.h = harness or Harness(AGENT, harness_root())
        self.knobs = self.h.knobs()
        self.max_repairs = self.knobs["max_repairs"] if max_repairs is None else max_repairs
        self.log = log or (lambda m: None)
        self._schema = json.dumps(json_schema(), separators=(",", ":"))
        self._pattern_ok = True

    def new_llm(self) -> LLM:
        return self._llm() if callable(self._llm) else (self._llm or LLM())

    # ------------------------------------------------------------------ prompts
    def system(self, dims) -> str:
        return (self.h.system(" ".join(dims)) + "\n\n## TSIR JSON schema (every value you write must conform)\n" +
                self._schema)

    def user(self, parent: dict, pmetrics: dict, spec: Spec, aim_critical: bool = False) -> str:
        dims = spec.dims
        blocks = []
        for d in dims:
            head = f"### {P.DIM_ID[d]} {P.DIM_TITLE[d]}"
            if spec.mode == "manual":
                blocks.append(f'{head}\nUser request (realise it faithfully, with plausible numbers): "{spec.requests[d]}"')
                if d == "environment" and "static_objects" not in Scenario.model_fields:
                    blocks[-1] += (" NOTE: this TSIR version has NO static objects, cones, barriers or signs (there is no "
                                   "/static_objects field and none can be added). Realise what the request implies with "
                                   "fields that exist in the parent (lane counts, speed limits where the parent has "
                                   "speed_limit_kph, geometry) and name everything that cannot be represented in "
                                   "the limitations field. Lowering driving_lanes only closes a lane when the ego and the "
                                   "other agents can still do their parent manoeuvres in the remaining lanes (a lane change "
                                   "or turn needs its lanes): otherwise keep the lane count and apply only the safe part "
                                   "(e.g. the speed limit), listing the rest under limitations.")
                if d in spec.facts:
                    blocks[-1] += "\n" + spec.facts[d]
            elif d == "weather":
                blocks.append(f"{head}\nThe planner drew this target environment (seeded random, different from the "
                              "parent). Realise it exactly: replace every differing /environment field with these "
                              f"values.\n{_compact(spec.target['environment'])}")
            else:
                ls = "\n".join(f"- {i}: {t}" for i, t in spec.levers.get(d, []))
                blocks.append(f"{head}\nCandidate levers written for this scenario. Choose EXACTLY ONE that makes the "
                              "most meaningful, testable change (put its id into \"lever\"), or \"custom\" when none "
                              f"fits:\n{ls}")
        return (f"## Parent scenario (verified TSIR, defaults filled in)\n{_compact(parent)}\n\n"
                f"## Array indices for JSON pointers\n{P.index_table(parent)}\n\n"
                f"## Parent simulation metrics (TanitSim)\n{_compact({k: pmetrics.get(k) for k in ('min_ttc', 'min_ttc_actor', 'ego_max_decel', 'ego_mean_speed', 'collision', 'ego_turns', 'red_light_violations')})}\n\n"
                "## Task\nDerive ONE variant of the parent by changing only: " +
                ", ".join(f"{P.DIM_ID[d]} {P.DIM_TITLE[d]}" for d in dims) + ".\n\n" + "\n\n".join(blocks) +
                "\n\n## Allowed JSON pointers (anything else is rejected)\n" +
                "\n".join("- " + x for x in P.whitelist_text(dims)) +
                "\n\n## Output\nJSON {\"lever\", \"rationale\" (one line: why this change matters for testing), "
                "\"limitations\" (what part of the request the simulator cannot represent, else an empty string), "
                "\"ops\"}. The ops are RFC 6902 add / replace / remove operations applied IN ORDER to the parent "
                "above. Change nothing that is not needed; keep the scenario consistent and verifiable.")

    # ------------------------------------------------------------------ editor call
    def propose(self, llm: LLM, msgs: list, dims, lever_ids: list[str]) -> dict:
        for use_pattern in ([True, False] if self._pattern_ok else [False]):
            try:
                out = llm.chat(msgs, schema=edit_schema(dims, lever_ids, use_pattern), role="augment_editor",
                               thinking=self.knobs["thinking"], temperature=self.knobs["temperature"],
                               max_tokens=6144)
                return out
            except Exception as e:   # noqa: BLE001 - decoder rejected the path grammar: retry once without it
                if use_pattern and any(w in str(e).lower() for w in ("pattern", "regex", "grammar", "400")):
                    self.log(f"path grammar rejected by the decoder ({str(e)[:100]}); continuing without it")
                    self._pattern_ok = False
                    continue
                raise
        raise RuntimeError("unreachable")

    # ------------------------------------------------------------------ candidate evaluation
    def evaluate(self, parent: dict, spec: Spec, ops: list, wd: Path, meta: dict, prep=None) -> Cand:
        c = Cand(ops=ops)
        t0 = time.perf_counter()
        try:
            doc = P.apply_patch(parent, ops)
        except P.PatchError as e:
            c.issues, c.stage = [f"patch could not be applied: {e}"], "patch"
            return c
        doc["meta"] = {**parent["meta"], **meta}
        sc, f = ladder.l0_parse(doc)
        if sc is None:
            c.issues, c.stage = [f"[L0:{f[0].check}] {f[0].message}" + blame(f[0].detail.get("errors", []), ops)], "schema"
            return c
        doc = sc.model_dump(mode="json")
        c.doc = doc
        c.iso = P.isolation(parent, doc, spec.dims)
        if not c.iso.ok:
            c.issues, c.stage = ["isolation: " + m for m in c.iso.messages()], "isolation"
            return c
        if doc["environment"] != parent["environment"]:
            bad = PL.env_consistency(doc["environment"])
            if bad:
                c.issues, c.stage = ["environment inconsistent: " + b for b in bad], "environment"
                return c
        c.qc = doc["road"] != parent["road"]            # QC only judges the map: skip when the road is unchanged
        tv = time.perf_counter()
        with _LADDER_LOCK:
            c.report = ladder.run(sc, wd, qc=c.qc)
        c.verify_s = time.perf_counter() - tv
        if not c.report.passed:
            c.issues = [c.report.diagnostics(), "simulation metrics: " + _metrics_line(c.report.metrics)]
            c.stage = "verify"
        elif prep is not None and hasattr(c.report, "_log") and hasattr(prep, "_log"):
            c.effect = sim_effect(prep._log, c.report._log)
            if spec.mode == "auto" and "behavior" in spec.dims and not c.effect["ego_relevant"]:
                c.stage = "effect"
                c.issues = [f"the change has no effect on the ego in the simulation (min TTC {c.effect['min_ttc'][1]} s, "
                            f"peak decel {c.effect['ego_max_decel'][1]} m/s2, ego path deviation "
                            f"{c.effect['ego_dev_m']} m: all identical to the parent). Pick another agent or a stronger "
                            "change that interacts with the ego's path. Simulated traffic always obeys traffic lights and "
                            "stop lines; agents that never meet the ego do not matter."]
        c.seconds = time.perf_counter() - t0
        return c

    # ------------------------------------------------------------------ criticality targeting inside the loop
    def tune(self, parent: dict, c: Cand, spec: Spec, wd: Path, prep) -> None:
        """Land the minimum TTC in the band with the numeric levers of the patch; replaces c.doc / c.report."""
        tc = time.perf_counter()
        new, info = criticality_target(parent, c.doc, c.report, wd / "crit", spec.seed, band=self.band)
        if new is not c.doc:
            with _LADDER_LOCK:
                r2 = ladder.run(Scenario.model_validate(new), wd / "crit" / "final", qc=False)
            iso2 = P.isolation(parent, new, spec.dims)
            if r2.passed and iso2.ok:
                c.doc, c.report, c.iso = new, r2, iso2
                if hasattr(prep, "_log") and hasattr(r2, "_log"):
                    c.effect = sim_effect(prep._log, r2._log)
                info["applied"] = True
            else:
                info.update(applied=False, landed=False,
                            reason=("search result violates isolation: " + "; ".join(iso2.messages())[:160]) if not iso2.ok
                            else "search result failed the ladder: " + r2.diagnostics()[:160])
        info["seconds"] = round(time.perf_counter() - tc, 1)
        c.crit = info
        t = c.report.metrics.get("min_ttc")
        if not info.get("landed") and spec.mode == "auto":
            c.stage = "critical"
            c.issues = [f"the variant is valid but too benign: minimum TTC {t} s (with {c.report.metrics.get('min_ttc_actor')}) "
                        f"even after tuning the numeric values you set; the target is {self.band[0]}-{self.band[1]} s "
                        "without a collision. Make another agent's behaviour interact with the ego's path more directly "
                        "(earlier, closer, faster, an agent that really meets the ego)."]

    def band_dist(self, c: Cand) -> float:
        m = c.report.metrics
        t = m.get("min_ttc")
        if t is None or m.get("collision"):
            return 1e9
        return 0.0 if self.band[0] <= t <= self.band[1] else min(abs(t - self.band[0]), abs(t - self.band[1]))

    # ------------------------------------------------------------------ edit + verify with repairs
    def edit_verify(self, llm: LLM, parent: dict, prep, spec: Spec, wd: Path, meta: dict, emit: Callable,
                    criticality: bool = True) -> list[Cand]:
        cands: list[Cand] = []
        crit_on = criticality and "behavior" in spec.dims
        # D1 in auto mode is fully determined by the plan: the patch is derived in code (no LLM, no tokens)
        if spec.mode == "auto" and spec.dims == ["weather"]:
            ops = [{"op": "replace", "path": f"/environment/{k}", "value": v}
                   for k, v in spec.target["environment"].items() if parent["environment"].get(k) != v]
            emit("edit", "ok", "patch derived from the weather plan")
            c = self.evaluate(parent, spec, ops, wd / "attempt0", meta, prep)
            c.lever, c.rationale = "plan", (f"Weather and time of day drawn with seed {spec.seed}: "
                                            f"{spec.target['from']['weather']}/{spec.target['from']['band']} -> "
                                            f"{spec.target['weather']}/{spec.target['band']}")
            cands.append(c)
            emit("verify", "ok" if c.passed else "failed", "; ".join(c.issues)[:200])
            return cands
        lever_ids = [i for d in spec.dims for i, _ in spec.levers.get(d, [])] if spec.mode == "auto" else []
        msgs = [{"role": "system", "content": self.system(spec.dims)},
                {"role": "user", "content": self.user(parent, prep.metrics, spec, crit_on and spec.mode == "auto")}]
        for i in range(self.max_repairs + 1):
            n0 = len(llm.calls)
            emit("edit", "running", f"attempt {i}")
            te = time.perf_counter()
            out = self.propose(llm, msgs, spec.dims, lever_ids)
            edit_s = time.perf_counter() - te
            emit("edit", "ok", f"attempt {i}: {len(out['ops'])} ops")
            emit("verify", "running", f"attempt {i}")
            c = self.evaluate(parent, spec, out["ops"], wd / f"attempt{i}", meta, prep)
            c.lever, c.rationale, c.edit_s = out.get("lever", "custom"), out.get("rationale", ""), edit_s
            c.limitations = out.get("limitations", "")
            new = llm.calls[n0:]
            c.tokens = {"prompt": sum(x.prompt_tokens for x in new), "completion": sum(x.completion_tokens for x in new)}
            cands.append(c)
            if c.passed and crit_on:
                emit("criticality", "running", f"attempt {i}")
                self.tune(parent, c, spec, wd / f"attempt{i}", prep)
                emit("criticality", "ok", f"min TTC {c.report.metrics.get('min_ttc')} s, landed={c.crit.get('landed')}")
            emit("verify", "ok" if c.valid else "failed", "; ".join(c.issues)[:200])
            if c.passed or i == self.max_repairs:
                break
            msgs = msgs + [{"role": "assistant", "content": _compact(out)},
                           {"role": "user", "content": "The variant built from your patch was not accepted:\n" +
                            "\n".join("- " + x for x in c.issues) +
                            "\nFix the root cause with the MINIMAL change. Output the complete corrected JSON; its "
                            "ops are applied to the ORIGINAL parent (not to your previous result), so repeat every "
                            "change you still want."}]
        if not any(c.valid for c in cands):
            ab = self.ablate(parent, spec, cands, wd, meta, prep, emit)
            if ab is not None:
                cands.append(ab)
                if ab.passed and crit_on:
                    emit("criticality", "running", "ablated variant")
                    self.tune(parent, ab, spec, wd / "ablate", prep)
        return cands

    def ablate(self, parent: dict, spec: Spec, cands: list[Cand], wd: Path, meta: dict, prep, emit: Callable,
               rounds: int = 3) -> Cand | None:
        """Last resort when every proposal failed the simulator-level checks: drop the ops that break the scenario's verified
        intent (greedy, fewest failing findings first; every requested dimension must stay changed) and keep the rest. The
        dropped ops are reported in ``limitations`` so the description says what could not be realised."""
        failed = [c for c in cands if c.stage == "verify" and c.report is not None and len(c.ops) > 1]
        if not failed:
            return None
        base = min(failed, key=lambda c: sum(not f.ok for f in c.report.findings))
        t0 = time.perf_counter()
        emit("verify", "running", "ablation: dropping the ops that break the verified intent")
        cur, dropped, why = list(base.ops), [], ""
        for rnd in range(rounds):
            best = None
            for k in range(len(cur)):
                trial = cur[:k] + cur[k + 1:]
                c = self.evaluate(parent, spec, trial, wd / f"ablate{rnd}_{k}", meta, prep)
                if c.passed:
                    best = (k, c, 0)
                    break
                if c.stage == "verify" and c.report is not None:
                    n = sum(not f.ok for f in c.report.findings)
                    if best is None or n < best[2]:
                        best = (k, c, n)
            if best is None:
                return None
            k, c, n = best
            o = cur[k]
            dropped.append(f"{o['op']} {o['path']}")
            if not why:
                why = "; ".join(f"{f.check}: {f.message[:110]}" for f in base.report.findings if not f.ok)[:260]
            cur = cur[:k] + cur[k + 1:]
            if c.passed:
                c.lever = "ablated"
                c.rationale = base.rationale
                c.limitations = (base.limitations + " " if base.limitations.strip() else "") + (
                    "Dropped because they broke the scenario's verified intent in the simulator (" + why + "): " +
                    "; ".join(dropped) + ".")
                c.issues, c.edit_s, c.seconds = [], 0.0, time.perf_counter() - t0
                emit("verify", "ok", f"ablation: dropped {len(dropped)} op(s)")
                return c
        emit("verify", "failed", "ablation found no passing subset")
        return None

    def choose(self, cands: list[Cand]) -> Cand | None:
        """First accepted candidate, else the valid one closest to the criticality band."""
        good = next((c for c in cands if c.passed), None)
        if good:
            return good
        soft = [c for c in cands if c.valid]
        return min(soft, key=self.band_dist) if soft else None

    # ------------------------------------------------------------------ description
    def describe(self, llm: LLM, parent: dict, variant: dict, spec: Spec, cand: Cand, changes, prep) -> dict:
        attrs = PL.expected_attributes(parent, variant, spec.dims, changes)
        m, pm = cand.report.metrics, prep.metrics
        facts = (f"Simulation facts (TanitSim replay): variant min TTC {m.get('min_ttc')} s with {m.get('min_ttc_actor')}, "
                 f"ego peak deceleration {m.get('ego_max_decel')} m/s2, collision {m.get('collision')}; parent min TTC "
                 f"{pm.get('min_ttc')} s, peak deceleration {pm.get('ego_max_decel')} m/s2.")
        if cand.effect:
            facts += (f" Agents whose motion differs from the parent: {cand.effect['actors_changed'] or 'none'}; new agents: "
                      f"{cand.effect['new_actors'] or 'none'}.")
        base = [{"role": "system", "content": "You write precise test-scenario descriptions for automated-driving "
                 "validation engineers: concrete, factual, no marketing language. Describe only what the scenario "
                 "file really contains and the simulation shows; never claim behaviour the changes do not implement."},
                {"role": "user", "content":
                 f"Parent scenario: {parent['meta']['title']}\n{parent['meta']['description'][:600]}\n\n"
                 f"Variant dimensions: {', '.join(P.DIM_TITLE[d] for d in spec.dims)}\n"
                 + (f"User request: {_compact(spec.requests)}\n" if spec.mode == "manual" else "")
                 + f"Editor rationale: {cand.rationale}\n"
                 + ("Parts of the request the simulator cannot represent (say so, describe the closest behaviour): "
                    f"{cand.limitations}\n" if cand.limitations.strip() else "")
                 + "\nChanges vs the parent (JSON pointers into the TSIR):\n"
                 + "\n".join("- " + s for s in PL.summarise(changes)) + "\n\n" + facts + "\n\n"
                 "Write the variant's title (what is different, max 100 chars) and description (3-5 sentences: the "
                 "situation as a whole, what is changed against the parent, and why it matters for testing). The "
                 "description MUST explicitly name every changed attribute in plain words: "
                 + "; ".join(a["attr"] for a in attrs) + "."}]
        res = llm.chat(base, schema=Described.model_json_schema(), role="augment_describer", max_tokens=900)
        missing = PL.check_description(res["title"] + " " + res["description"], attrs)
        rec = {"attempts": 1, "first_pass": not missing, "missing_first": missing}
        if missing:
            base += [{"role": "assistant", "content": _compact(res)},
                     {"role": "user", "content": "The description does not name: " + "; ".join(missing) +
                      ". Rewrite title and description so that each is named explicitly in plain words."}]
            res = llm.chat(base, schema=Described.model_json_schema(), role="augment_describer", max_tokens=900)
            missing = PL.check_description(res["title"] + " " + res["description"], attrs)
            rec["attempts"] = 2
        rec.update(ok=not missing, missing=missing, attributes=[a["attr"] for a in attrs])
        return {"title": res["title"].strip(), "description": res["description"].strip(), "check": rec}

    def build(self, spec: Spec, pdoc: dict, prep, vdir: Path, emit: Callable, *, criticality: bool,
               package: bool, plan_s: float, parent_record: dict | None) -> dict:
        t_all = time.perf_counter()
        vdir.mkdir(parents=True, exist_ok=True)
        llm = self.new_llm()
        tm = {"plan_s": round(plan_s, 2)}
        pid = pdoc["meta"]["id"]
        env_tags = spec.target["environment"] if spec.target else None
        meta = {"id": spec.vid, "parent_id": pid, "title": f"{pdoc['meta']['title']} [{'+'.join(spec.dims)} variant]"[:140],
                "tags": _variant_tags(pdoc, spec.dims, env_tags), "source": "agent"}
        res: dict = {"id": spec.vid, "parent_id": pid, "mode": spec.mode, "seed": spec.seed, "dimensions": spec.dims,
                     "requests": spec.requests or None, "status": "failed", "paths": {"dir": str(vdir)}, "timings": tm}
        if spec.target:
            res["plan"] = {"weather": spec.target}
        if spec.levers:
            res["plan"] = {**res.get("plan", {}), "levers": {d: [{"id": i, "text": t} for i, t in ls]
                                                              for d, ls in spec.levers.items()}}
        try:
            cands = self.edit_verify(llm, pdoc, prep, spec, vdir, meta, emit, criticality=criticality)
            tm["edit_s"] = round(sum(c.edit_s for c in cands), 1)
            tm["verify_s"] = round(sum(c.verify_s for c in cands), 1)
            tm["criticality_s"] = round(sum((c.crit or {}).get("seconds", 0.0) for c in cands), 1)
            good = self.choose(cands)
            res["attempts"] = [{"index": i, "lever": c.lever, "rationale": c.rationale, "ops": c.ops, "stage": c.stage,
                                "passed": c.valid, "issues": c.issues, "seconds": round(c.seconds + c.edit_s, 1),
                                "tokens": c.tokens, "criticality": c.crit, "limitations": c.limitations or None}
                               for i, c in enumerate(cands)]
            res["first_pass_valid"] = bool(cands and cands[0].valid)
            res["repairs"] = (cands.index(good) if good else len(cands) - 1) if cands else 0
            last = good or cands[-1]
            res.update(lever=last.lever, rationale=last.rationale, patch=last.ops,
                       limitations=last.limitations or None, effect=last.effect)
            if last.iso is not None:
                res["isolation"] = last.iso.to_dict()
            if last.report is not None:
                res["verifier"] = _report(last.report)
            if good is None:
                res["error"] = "no attempt passed: " + "; ".join(last.issues)[:400]
                if getattr(last, "doc", None):
                    _write(vdir / "candidate.json", last.doc)      # last rejected edit, for diagnosis and Harness² scoring
                    res["paths"]["candidate"] = str(vdir / "candidate.json")
                res["tokens"] = llm.stats()
                tm["build_s"] = round(time.perf_counter() - t_all, 1)
                _write(vdir / "result.json", res)
                emit("done", "failed", res["error"][:200])
                return res
            doc, rep = good.doc, good.report
            res["criticality"] = good.crit
            res["isolation"] = good.iso.to_dict()
            allc = P.diff(pdoc, doc)
            changes = [c for c in allc if not P.is_housekeeping(c.path)]
            res["diff"] = [c.to_dict() for c in changes]
            res["housekeeping"] = [c.path for c in allc if P.is_housekeeping(c.path)]
            # --- describe
            emit("describe", "running", "")
            td = time.perf_counter()
            d = self.describe(llm, pdoc, doc, spec, good, changes, prep)
            tm["describe_s"] = round(time.perf_counter() - td, 1)
            res.update(title=d["title"], description=d["description"], description_check=d["check"])
            emit("describe", "ok" if d["check"]["ok"] else "failed", d["title"][:120] +
                 ("" if d["check"]["ok"] else " | missing: " + ", ".join(d["check"]["missing"])))
            doc["meta"].update(title=d["title"][:140], description=d["description"],
                           tags=_variant_tags(pdoc, spec.dims, doc["environment"]))
            # --- final compile + verify of the described variant (files that ship)
            emit("final", "running", "")
            tf = time.perf_counter()
            qc = doc["road"] != pdoc["road"]
            with _LADDER_LOCK:
                frep = ladder.run(Scenario.model_validate(doc), vdir / "final", qc=qc)
            tm["final_verify_s"] = round(time.perf_counter() - tf, 1)
            res["verifier"] = _report(frep)
            res["verifier"]["qc_run"] = qc
            res["final_doc"] = doc
            _write(vdir / "tsir.json", doc)
            _write(vdir / "patch.json", {"lever": last.lever, "rationale": last.rationale, "ops": last.ops,
                                         "criticality": res["criticality"], "attempts": res["attempts"]})
            _write(vdir / "diff.json", res["diff"])
            _write(vdir / "report.json", res["verifier"])
            for k, p in frep.artifacts.items():
                res["paths"][k] = str(p)
            res["paths"].update(tsir=str(vdir / "tsir.json"), diff=str(vdir / "diff.json"), patch=str(vdir / "patch.json"))
            emit("final", "ok" if frep.passed else "failed", frep.diagnostics()[:200])
            # --- package: OpenDRIVE / OSC come from the ladder; the IMDAD v2 record wraps the Scenario_Structure package
            pkg = None
            if package:
                emit("package", "running", "")
                tp = time.perf_counter()
                try:
                    pkg = self.package(llm, pdoc, doc, frep, d, good, changes)
                    tm["package_s"] = round(time.perf_counter() - tp, 1)
                    from tsf.agents import augment_pkg
                    from tsf.agents.counterfactual import validate_incident
                    rec = augment_pkg.make_record(
                        pdoc, doc, pkg["package"], variant_id=spec.vid, mode=spec.mode, dims=spec.dims, seed=spec.seed,
                        patch=last.ops, rationale=last.rationale, report=frep, diff_summary=PL.summarise(changes),
                        parent_record=parent_record, attempts=len(cands), timings={"total_s": time.perf_counter() - t_all})
                    ok, errs = validate_incident(rec)
                    _write(vdir / "package.json", pkg["package"])
                    _write(vdir / "imdad_record.json", rec)
                    res["package"] = {"scenario_structure_valid": pkg["valid"], "scenario_structure_errors": pkg["errors"],
                                      "imdad_record_valid": ok, "imdad_record_errors": errs}
                    res["paths"].update(package=str(vdir / "package.json"), imdad_record=str(vdir / "imdad_record.json"))
                    emit("package", "ok" if ok else "failed", "IMDAD v2 record " + ("valid" if ok else "; ".join(errs[:2])))
                except Exception as e:   # noqa: BLE001 - LLM down: the verified scenario files are still the variant's result
                    tm["package_s"] = round(time.perf_counter() - tp, 1)
                    msg = f"{type(e).__name__}: {e}"[:300]
                    res["package"] = {"error": msg, "scenario_structure_valid": False, "scenario_structure_errors": [msg],
                                      "imdad_record_valid": False, "imdad_record_errors": ["package not built"]}
                    emit("package", "failed", msg[:160])
            res["status"] = "ok" if frep.passed else "failed"
            if not frep.passed:
                res["error"] = "final verification failed: " + frep.diagnostics()[:300]
            res["event_t0"] = None
            try:
                from tsf.agents import augment_render as R
                old = {o["id"] for o in pdoc.get("static_objects") or []}
                new = [o["id"] for o in doc.get("static_objects") or [] if o["id"] not in old]
                res["event_t0"] = R.window_t0(frep._log, doc["duration"], focus=new) if hasattr(frep, "_log") else None
                if new and res["event_t0"] is not None:      # what the camera judge may ask for: objects in view
                    res["judge_objects"] = R.changed_objects(doc, pdoc, frep._log, res["event_t0"])
            except Exception:   # noqa: BLE001
                pass
        except Exception as e:   # noqa: BLE001 - one variant failing must not kill the others
            import traceback
            res["status"], res["error"] = "failed", f"{type(e).__name__}: {e}"
            res["traceback"] = traceback.format_exc()[-1500:]
            res.setdefault("first_pass_valid", False)
            res.setdefault("repairs", 0)
            emit("done", "failed", res["error"][:200])
        res["tokens"] = llm.stats()
        tm["build_s"] = round(time.perf_counter() - t_all, 1)
        _write(vdir / "result.json", {k: v for k, v in res.items() if k != "final_doc"})
        return res


    def start_analysis(self, pdoc: dict):
        """Engineering analysis of the PARENT (Lifter step 1), once per call and in the background while the variants are
        edited: that LLM call is the longest of the package and the same for every variant, so they share it."""
        from tsf.agents.lifter import Lifter
        llm = self.new_llm()
        lif = Lifter(llm, Harness("lifter"))
        q = pdoc["meta"].get("description") or pdoc["meta"]["title"]
        self._ex = ThreadPoolExecutor(1, thread_name_prefix="aug-analysis")
        self._analysis_llm = llm
        self._analysis = self._ex.submit(lif.analyse, q)

    def shared_analysis_stats(self) -> dict | None:
        return self._analysis_llm.stats() if getattr(self, "_analysis_llm", None) else None

    def package(self, llm: LLM, pdoc: dict, doc: dict, frep, d: dict, good: Cand, changes: list | None = None) -> dict:
        """Scenario_Structure package (analysis + codeBeamer documentation) of the variant, validated against the schema.
        The analysis is the parent's, extended by the variant's changes; only the documentation is written per variant."""
        from tsf.agents.lifter import Attempt, Lifter, assemble_package, validate_package
        lif = Lifter(llm, Harness("lifter"))
        dims = ", ".join(P.DIM_TITLE[x] for x in good.iso.by_dim) if good.iso else ""
        query = (d["description"] + chr(10) + "(Variant of the verified scenario " + chr(34) + pdoc["meta"]["title"] +
                 chr(34) + "; changed dimension(s): " + dims + ".)")
        fut = getattr(self, "_analysis", None)
        try:
            analysis = copy.deepcopy(fut.result()) if fut is not None else None
        except Exception:   # noqa: BLE001 - the shared call failed (LLM restart...): analyse this variant on its own
            analysis = None
        if analysis is None:
            analysis = lif.analyse(query)
        else:
            a = analysis["analysisOfUserQuery"]
            a["semanticMeaning"] = ("Variant of the verified scenario: " + d["description"] + " Parent scenario: " +
                                    a["semanticMeaning"])[:800]
            note = ("Augmentation (" + dims + "): " + ("; ".join(PL.summarise(changes)) if changes else "no TSIR change"))[:400]
            ra = analysis["requiredAssumptionsAndMissingValues"]
            ra["assumptions"] = (ra["assumptions"] + [note])[:12]
        att = Attempt(doc, frep, 0.0)
        for k in range(2):                  # a documentation that ends mid-string (sampling) is simply asked for again
            try:
                nar = lif.document(query, analysis, [att], att)
                break
            except json.JSONDecodeError:
                if k:
                    raise
        pkg = assemble_package(query, analysis, [att], att, nar)
        valid, errs = validate_package(pkg)
        return {"package": pkg, "valid": valid, "errors": errs}


    def render_all(self, pdoc: dict, prep, results: list[dict], out: Path, seed: int, emit_for,
                   cameras, frames: int, lidar: bool, wait_s: float, stop_server: bool = True):
        """Render the parent once, then every verified variant (same seed); judge camera + LiDAR; stop a Cosmos server
        this call started."""
        from tsf.agents import augment_render as R
        llm = self.new_llm()
        e0 = emit_for(None)
        judged_parent: dict = {}
        todo = [r for r in results if r["status"] == "ok"]
        for r in results:
            if r["status"] != "ok":
                r["render"], r["judge"] = {"skipped": "variant failed verification"}, {"skipped": "no render"}
        if not todo:
            return
        state: dict = {}
        pgeo = out / "parent" / "render"
        try:
            e0("render", "running", "parent")
            tp = time.perf_counter()
            t0p = R.window_t0(prep._log, pdoc["duration"], frames)
            try:
                pr = R.render(pdoc, pgeo, seed=seed, t0=t0p, cameras=cameras, frames=frames, lidar=lidar, wait_s=wait_s,
                              log=lambda m: e0("render", "running", m), state=state)
                files = R.find_outputs(pgeo, pr["result"])
                pr["files"] = {k: v if isinstance(v, dict) else str(v) for k, v in files.items() if v}
                judged_parent = {"render": {k: v for k, v in pr.items() if k != "result"},
                                 "camera": R.judge_camera(files["front_video"], pdoc, [], llm),
                                 "lidar": R.judge_lidar(pr["result"], files["sensors_json"])}
                _write(out / "parent" / "render.json", judged_parent)
                e0("render", "running", f"parent rendered in {time.perf_counter() - tp:.0f} s")
            except Exception as e:   # noqa: BLE001
                judged_parent = {"error": f"{type(e).__name__}: {e}"[:400]}
                e0("render", "failed", judged_parent["error"])
            for r in todo:
                emit = emit_for(Spec(0, r["id"], r["dimensions"], r["mode"]))
                vdir = Path(r["paths"]["dir"])
                doc = r.pop("final_doc", None) or json.loads(Path(r["paths"]["tsir"]).read_text())
                same_geom = ({k: v for k, v in doc.items() if k not in ("environment", "meta")}
                             == {k: v for k, v in pdoc.items() if k not in ("environment", "meta")})
                objs = r["judge_objects"] if "judge_objects" in r else R.changed_objects(doc, pdoc)
                kw = dict(cameras=cameras, frames=frames, lidar=lidar, wait_s=wait_s, state=state,
                          t0=t0p if same_geom else r.get("event_t0"),
                          scenario=pdoc if same_geom else None, geometry_dir=pgeo if same_geom else None,
                          log=lambda m: emit("render", "running", m))
                try:
                    emit("render", "running", "appearance-only variant: scenario geometry of the parent is reused"
                         if same_geom else "")
                    tr = time.perf_counter()
                    rr = R.render(doc, vdir / "render", seed=seed, **kw)
                    files = R.find_outputs(vdir / "render", rr["result"])
                    rr["files"] = {k: v if isinstance(v, dict) else str(v) for k, v in files.items() if v}
                    r["timings"]["render_s"] = round(time.perf_counter() - tr, 1)
                    emit("render", "ok", f"{rr['render_s']} s")
                    emit("judge", "running", "")
                    tj = time.perf_counter()
                    nj = len(llm.calls)
                    cam = R.judge_camera(files["front_video"], doc, objs, llm)
                    rerender = None
                    if cam.get("ok") is False:           # one re-render with another seed
                        emit("render", "running", "camera judgement failed: re-render with another seed")
                        rr2 = R.render(doc, vdir / "render_retry", seed=seed + 101, **kw)
                        files2 = R.find_outputs(vdir / "render_retry", rr2["result"])
                        rr2["files"] = {k: v if isinstance(v, dict) else str(v) for k, v in files2.items() if v}
                        cam2 = R.judge_camera(files2["front_video"], doc, objs, llm)
                        rerender = {"render": {k: v for k, v in rr2.items() if k != "result"}, "camera": cam2}
                        r["timings"]["render_s"] = round(r["timings"]["render_s"] + rr2["render_s"], 1)
                        if cam2.get("ok"):
                            rr, files, cam = rr2, files2, cam2
                    r["render"] = {k: v for k, v in rr.items() if k != "result"}
                    r["render"]["geometry_reused"] = same_geom
                    r["judge"] = {"camera": cam, "lidar": R.judge_lidar(rr["result"], files["sensors_json"]),
                                  "rerender": rerender, "parent": {k: judged_parent.get(k) for k in ("camera", "lidar")}}
                    r["judge"]["ok"] = bool(cam.get("ok") is not False and r["judge"]["lidar"].get("ok") is not False)
                    r["timings"]["judge_s"] = round(time.perf_counter() - tj, 1)
                    jc = llm.calls[nj:]
                    r["tokens_judge"] = {"calls": len(jc), "prompt_tokens": sum(c.prompt_tokens for c in jc),
                                         "completion_tokens": sum(c.completion_tokens for c in jc)}
                    emit("judge", "ok" if r["judge"]["ok"] else "failed", "")
                except Exception as e:   # noqa: BLE001
                    r["render"] = {"error": f"{type(e).__name__}: {e}"[:400]}
                    r["judge"] = {"skipped": "render failed"}
                    emit("render", "failed", r["render"]["error"][:200])
                r["parent_render"] = judged_parent.get("render")
            e0("render", "ok", "")
        finally:
            if state.get("started_server") and stop_server:
                try:
                    from tsf.cosmos import generate
                    generate.stop("sim")
                    e0("render", "info", "stopped the Cosmos server this job started")
                except Exception as e:   # noqa: BLE001
                    e0("render", "info", f"could not stop the Cosmos server: {e}")


# ----------------------------------------------------------------------------- criticality targeting (D2)
_TUNE = {"initial_speed": ("rel", 0.6, 1.4, 0.0, 45.0), "target_speed": ("rel", 0.6, 1.4, 0.0, 45.0),
         "t": ("add", -2.0, 3.0, 0.0, 600.0), "distance": ("rel", 0.6, 1.5, 3.0, 150.0),
         "gap": ("abs", 5.0, 25.0, 0, 0), "time_gap": ("abs", 0.8, 3.0, 0, 0), "delay": ("abs", 0.0, 3.0, 0, 0),
         "s": ("add", -12.0, 12.0, 1.0, 1900.0), "decel": ("rel", 0.6, 1.4, 1.0, 9.0), "accel": ("rel", 0.6, 1.6, 0.3, 8.0),
         "ttc": ("rel", 0.6, 1.5, 0.5, 8.0), "headway": ("abs", 0.6, 2.4, 0, 0), "speed": ("abs", 0.8, 2.5, 0, 0)}
_ORDER = list(_TUNE)


def numeric_levers(parent: dict, child: dict, limit: int = 5) -> list[dict]:
    """ParameterRange dicts for the numeric leaves the patch touched (changed values, fields of added items)."""
    found: list[tuple[int, str, float]] = []

    def walk(x, base: str):
        if isinstance(x, dict):
            for k, v in x.items():
                walk(v, f"{base}/{k}")
        elif isinstance(x, list):
            for i, v in enumerate(x):
                walk(v, f"{base}/{i}")
        elif isinstance(x, (int, float)) and not isinstance(x, bool):
            key = base.rsplit("/", 1)[-1]
            if key in _TUNE and base.startswith(("/actors/", "/actions/")):
                found.append((_ORDER.index(key), base, float(x)))

    for c in P.diff(parent, child):
        if c.path.startswith(("/actors", "/actions")) and c.op in ("add", "replace") and c.after is not None:
            walk(c.after, c.path)
    out, seen = [], set()
    for _, path, v in sorted(found, key=lambda x: (x[0], x[1])):
        if path in seen:
            continue
        seen.add(path)
        kind, a, b, lo, hi = _TUNE[path.rsplit("/", 1)[-1]]
        if kind == "rel":
            low, high = max(lo, v * a), min(hi, v * b)
        elif kind == "add":
            low, high = max(lo, v + a), min(hi, v + b)
        else:
            low, high = a, b
        if high > low:
            out.append({"name": "p" + _slug(path), "path": path, "distribution": "uniform", "low": round(low, 3),
                        "high": round(high, 3)})
    return out[:limit]


def retune_assertions(doc: dict, metrics: dict) -> list[dict]:
    """Safety thresholds follow the variant's achieved metrics (the intent assertions stay untouched)."""
    changes = []
    for i, a in enumerate(doc["assertions"]):
        if a["kind"] == "min_ttc_above" and metrics.get("min_ttc") is not None and a["value"] > metrics["min_ttc"] * 0.95:
            new = max(0.5, math.floor(metrics["min_ttc"] * 0.9 * 10) / 10)
            changes.append({"path": f"/assertions/{i}/value", "before": a["value"], "after": new})
            a["value"] = new
        if a["kind"] == "max_decel_below" and a["value"] < metrics.get("ego_max_decel", 0) * 1.05:
            new = min(12.0, math.ceil(metrics["ego_max_decel"] * 1.1 * 10) / 10)
            changes.append({"path": f"/assertions/{i}/value", "before": a["value"], "after": new})
            a["value"] = new
    return changes


def criticality_target(parent: dict, doc: dict, rep, wd: Path, seed: int, band=BAND, n_seed: int = 16, n_iter: int = 2,
                       n_per_iter: int = 8) -> tuple[dict, dict]:
    """Land the minimum TTC in ``band`` by searching only the numeric levers the patch exposed (Sobol + cross-entropy
    with TanitSim). Keeps the variant closest to the editor's proposal among those that land in the band."""
    from tsf.augment.search import Augmenter
    info: dict = {"band": list(band), "searched": False, "landed": False}
    allow_col = parent["meta"].get("polarity") == "negative"
    m0 = rep.metrics
    ok0 = m0.get("min_ttc") is not None and band[0] <= m0["min_ttc"] <= band[1] and (allow_col or not m0["collision"])
    info.update(initial_min_ttc=m0.get("min_ttc"))
    if ok0:
        info.update(landed=True, reason="the edited variant already lands in the band", min_ttc=m0["min_ttc"])
        return doc, info
    own = numeric_levers(parent, doc)
    extra: list[dict] = []
    try:                                  # generic levers of the scenario (NPC speeds / gaps, ego speed, action targets)
        from tsf.augment.search import auto_parameters
        taken = {p["path"] for p in own}
        extra = [q.model_dump() for q in auto_parameters(doc) if q.path not in taken][:6]
    except Exception:   # noqa: BLE001 - older search module without auto_parameters
        pass
    if not own and not extra:
        info["reason"] = "the patch exposes no numeric levers"
        return doc, info
    mid = 0.5 * (band[0] + band[1])
    best = params = None
    sims = 0
    for stage, cand_params in (("patch levers", own), ("patch levers + scenario levers", (own + extra)[:9])):
        if not cand_params or (stage != "patch levers" and not extra):
            continue
        sc = Scenario.model_validate({**doc, "parameters": cand_params})
        res = Augmenter(sc, rep._sim.m, seed=seed).run(n_seed, n_iter, n_per_iter)
        sims += res.sims
        info.update(searched=True, params=[p["name"] for p in cand_params], sims=sims, stage=stage)

        def dist(v, cp=cand_params):          # normalised distance to the editor's proposal
            return sum(abs(v.values[p["name"]] - _get(doc, p["path"])) / (p["high"] - p["low"]) for p in cp)

        def eligible(v, strict):
            t = v.metrics.get("min_ttc") if v.metrics else None
            if not v.valid or t is None or (v.metrics["collision"] and not allow_col):
                return False
            return band[0] <= t <= band[1] if strict else True

        inb = [v for v in res.variants if eligible(v, True)]
        if inb:
            best, params = min(inb, key=dist), cand_params
            info["landed"] = True
            info.pop("reason", None)
            break
        cand = [v for v in res.variants if eligible(v, False)]
        if cand:
            b = min(cand, key=lambda v: min(abs(v.metrics["min_ttc"] - band[0]), abs(v.metrics["min_ttc"] - band[1])))
            if best is None or abs(b.metrics["min_ttc"] - mid) < abs(best.metrics["min_ttc"] - mid):
                best, params = b, cand_params
            info["reason"] = "no sample inside the band; closest kept"
    if best is None:
        info["reason"] = "no valid sample"
        return doc, info
    new = copy.deepcopy(doc)
    for p in params:
        _set_ptr(new, p["path"], best.values[p["name"]])
    info.update(min_ttc=best.metrics["min_ttc"], values=best.values, parameters=params)
    new["assertions"] = copy.deepcopy(doc["assertions"])
    info["retuned"] = retune_assertions(new, best.metrics)
    return new, info


def _get(doc, ptr: str):
    cur = doc
    for t in P.tokens(ptr):
        cur = cur[int(t)] if isinstance(cur, list) else cur[t]
    return cur


# ----------------------------------------------------------------------------- the agent API
def _plan(parent: dict, mode: str, requests: dict | None, seed: int, separate: bool, pid: str,
          static_objects: bool, reach: float | None = None) -> list[Spec]:
    specs: list[Spec] = []
    if mode == "auto":
        for i, d in enumerate(P.DIMS):
            rng = random.Random(f"{seed}/{d}")
            sp = Spec(i, f"{pid}_aug_{d[0]}{seed}", [d], "auto", seed=seed)
            if d == "weather":
                sp.target = PL.draw_weather(parent["environment"], seed)
            else:
                ls = (PL.behavior_levers(parent) if d == "behavior" else
                      PL.environment_levers(parent, static_objects, reach))
                rng.shuffle(ls)
                sp.levers[d] = ls[:6]
            specs.append(sp)
    elif mode == "manual":
        req = {k: v.strip() for k, v in (requests or {}).items() if k in P.DIMS and v and v.strip()}
        if not req:
            raise ValueError("manual mode needs at least one request: weather, behavior or environment")
        h = hashlib.sha1(json.dumps([req, seed], sort_keys=True).encode()).hexdigest()[:6]
        if separate:
            for i, d in enumerate(req):
                specs.append(Spec(i, f"{pid}_aug_m{h}_{d[0]}", [d], "manual", {d: req[d]}, seed=seed))
        else:
            specs.append(Spec(0, f"{pid}_aug_m{h}", list(req), "manual", req, seed=seed))
        if static_objects and "environment" in req:
            for sp in specs:
                if "environment" in sp.dims:
                    sp.facts["environment"] = PL.placement_facts(parent, reach)
    else:
        raise ValueError(f"mode must be 'auto' or 'manual', got {mode!r}")
    return specs


def _variant_tags(parent: dict, dims: list[str], env: dict | None) -> list[str]:
    tags = [t for t in parent["meta"]["tags"] if t not in PL.STALE_TAGS or "weather" not in dims]
    if "weather" in dims and env:
        tags += [env["weather"], PL.time_band(env["time_of_day"])]
    return list(dict.fromkeys(["augmentation", *[DIM_TAG[d] for d in dims], "SYNTHETIC_AUGMENTATION", *tags]))


def augment(parent, out_dir, *, mode: str = "auto", requests: dict | None = None, seed: int = 0, separate: bool = False,
            render: bool = True, criticality: bool = True, package: bool = True, max_repairs: int | None = None,
            workers: int = 3, llm=None, harness: Harness | None = None, on_event: Callable[[dict], None] | None = None,
            parent_record: dict | None = None, cameras=None, frames: int = 97, lidar: bool = True,
            render_wait_s: float = 7200.0, stop_server: bool = True,
            parent_qc: bool = True) -> list[dict]:
    """Create variants of a verified parent scenario. Returns one dict per variant (ids, paths, patch, verifier report,
    repairs, isolation, description check, criticality, render / judge results, timings, tokens)."""
    t_all = time.perf_counter()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pdoc = load_parent(parent)
    pid = pdoc["meta"]["id"]
    evlock = threading.Lock()

    def emit_for(spec: Spec | None):
        def emit(stage: str, status: str, detail: str = ""):
            ev = {"variant": spec.vid if spec else None, "index": spec.idx if spec else None, "stage": stage,
                  "status": status, "detail": detail, "t": time.time()}
            with evlock:
                if on_event:
                    on_event(ev)
        return emit

    emit0 = emit_for(None)
    ag = ScenarioAugmenter(llm, harness, max_repairs, log=lambda m: emit0("log", "info", m))
    ag._analysis = None
    # --- parent: verified once (the same ladder as every run)
    emit0("parent", "running", "verifying the parent")
    t0 = time.perf_counter()
    with _LADDER_LOCK:
        prep = ladder.run(pdoc, out / "parent" / "verify", qc=parent_qc)
    _write(out / "parent" / "tsir.json", pdoc)
    emit0("parent", "ok" if prep.passed else "failed", prep.diagnostics()[:200])
    if not prep.passed or not hasattr(prep, "_log"):
        raise ValueError("the parent scenario does not pass the verifier ladder: " + prep.diagnostics()[:300])
    # --- plan
    t_plan = time.perf_counter()
    has_static = "static_objects" in Scenario.model_fields
    specs = _plan(pdoc, mode, requests, seed, separate, pid, has_static, PL.ego_exit_reach(pdoc, prep))
    plan_s = time.perf_counter() - t_plan
    for sp in specs:
        emit_for(sp)("plan", "ok", "+".join(sp.dims))

    if package:
        ag.start_analysis(pdoc)            # runs while the editors work; every variant's package reuses it

    def build(spec: Spec) -> dict:
        emit = emit_for(spec)
        return ag.build(spec, pdoc, prep, out / spec.vid, emit, criticality=criticality, package=package,
                        plan_s=plan_s / len(specs), parent_record=parent_record)

    nw = max(1, min(workers, len(specs)))
    with ThreadPoolExecutor(nw, thread_name_prefix="aug") as ex:
        results = list(ex.map(build, specs))
    # --- sensors: the parent once, then every verified variant (single Cosmos slot, sequential)
    if render:
        from tsf.agents import augment_render as R
        if not R.render_available():
            for r in results:
                r["render"] = {"skipped": "tsf.cosmos.pipeline.render_sensors is not available"}
                r["judge"] = {"skipped": "no render"}
            emit0("render", "skipped", "render_sensors not available")
        else:
            ag.render_all(pdoc, prep, results, out, seed, emit_for, cameras, frames, lidar, render_wait_s, stop_server)
    else:
        for r in results:
            r["render"] = {"skipped": "render=False"}
    for r in results:
        r.pop("final_doc", None)
        tmr = r["timings"]
        tmr["total_s"] = round(tmr.get("build_s", 0.0) + tmr.get("render_s", 0.0) + tmr.get("judge_s", 0.0), 1)
        _write(Path(r["paths"]["dir"]) / "result.json", r)
    summary = {"parent": pid, "mode": mode, "seed": seed, "separate": separate, "requests": requests,
               "parent_verified": prep.passed, "wall_s": round(time.perf_counter() - t_all, 1),
               "shared_analysis_tokens": ag.shared_analysis_stats(),
               "variants": [_row(r) for r in results]}
    if getattr(ag, "_ex", None):
        ag._ex.shutdown(wait=False)
    _write(out / "summary.json", summary)
    return results


def _row(r: dict) -> dict:
    return {"id": r["id"], "dimensions": r["dimensions"], "lever": r.get("lever"), "status": r["status"],
            "first_pass_valid": r["first_pass_valid"], "repairs": r["repairs"],
            "isolation_ok": (r.get("isolation") or {}).get("ok"),
            "description_ok": ((r.get("description_check") or {}).get("ok")),
            "verified": (r.get("verifier") or {}).get("passed"), "timings": r["timings"], "tokens": r["tokens"]}


# ----------------------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("parent", help="parent TSIR JSON")
    ap.add_argument("out_dir")
    ap.add_argument("--mode", choices=["auto", "manual"], default="auto")
    ap.add_argument("--weather", help="manual D1: weather and time of day")
    ap.add_argument("--behavior", help="manual D2: ego / other agents' behaviour")
    ap.add_argument("--environment", help="manual D3: road, lanes, signs, static objects, setting")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--separate", action="store_true", help="manual: one variant per given dimension")
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--no-criticality", action="store_true")
    ap.add_argument("--no-package", action="store_true",
                    help="skip the Scenario_Structure package and the IMDAD record (saves ~2k LLM tokens per variant)")
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args(argv)
    req = {k: v for k, v in (("weather", a.weather), ("behavior", a.behavior), ("environment", a.environment)) if v}

    def ev(e):
        if e["stage"] not in ("log",):
            print(f"[{e['variant'] or '-':40s}] {e['stage']:11s} {e['status']:8s} {e['detail'][:110]}", flush=True)

    res = augment(a.parent, a.out_dir, mode=a.mode, requests=req or None, seed=a.seed, separate=a.separate,
                  render=not a.no_render, criticality=not a.no_criticality, package=not a.no_package,
                  workers=a.workers, on_event=ev)
    for r in res:
        print(json.dumps(_row(r), ensure_ascii=False))


if __name__ == "__main__":
    main()
