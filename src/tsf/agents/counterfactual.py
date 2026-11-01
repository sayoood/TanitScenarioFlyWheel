"""UC3: NEGATIVE incident (IMDAD v2) -> verified COUNTERFACTUAL_POSITIVE scenario twin.

1. brief    : the LLM reads the incident (event, kinematics, parties, ODD, tri-pillar counterfactual extraction)
              and writes the failure mode, the normative ego behaviour and two TanitSim-expressible scenario
              requests (negative replay, positive twin with the same initial conditions)
2. lift     : the UC1 Lifter turns the positive request (and optionally the negative replay) into verified TSIR
3. twin     : an incident_schema_v2 record (origin COUNTERFACTUAL_POSITIVE, parent = the incident) whose
              scenario_specification is the verified package and whose kinematics come from the simulation
Incident records are internal data (thor3 only); only aggregate results leave the machine.
"""
from __future__ import annotations

import copy
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from tsf.agents.lifter import Lifter, LiftResult
from tsf.agents.llm import LLM

ROOT = Path(__file__).resolve().parents[3]
INCIDENTS = Path(os.environ.get("TSF_IMDAD_INCIDENTS", Path.home() / "data" / "imdad" / "incidents.jsonl"))
SPEC = ROOT / "spec" / "imdad"


class Brief(BaseModel):
    failure_mode: str = Field(description="What the automated system did wrong, one sentence")
    root_cause: str = Field(description="Technical root cause (perception, prediction, planning, ops)")
    normative_behavior: str = Field(description="What a correct AD system does instead, with numbers")
    negative_request: str = Field(description="Scenario request that replays the incident in a 2D traffic "
                                              "simulator (road layout, actors, speeds, timing)")
    positive_request: str = Field(description="Same road, actors and initial conditions, but the ego performs "
                                              "the normative behavior; state measurable pass criteria")
    simplifications: list[str] = Field(description="Aspects that cannot be represented (sensors, ops, weather "
                                                   "physics) and how they were abstracted")


def load_incident(incident_id: str, path: Path = INCIDENTS) -> dict:
    for ln in path.read_text(encoding="utf-8").splitlines():
        if ln.strip():
            d = json.loads(ln)
            if d["incident_id"] == incident_id:
                return d
    raise KeyError(incident_id)


def list_incidents(path: Path = INCIDENTS, polarity: str = "NEGATIVE") -> list[dict]:
    out = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        d = json.loads(ln) if ln.strip() else None
        if d and d["incident_polarity"] == polarity and not d["lineage"].get("is_synthetic_augmentation"):
            out.append({"incident_id": d["incident_id"], "title": d["meta_information"]["title"],
                        "has_cf_positive": bool(d["lineage"].get("counterfactual_positive_id"))})
    return out


def brief_text(doc: dict) -> str:
    ai, mi = doc["accident_information"], doc["meta_information"]
    cfx = (doc.get("tri_pillar_analysis") or {}).get("counterfactual_positive_extraction") or {}
    keep = {"title": mi["title"], "summary": mi["summary"], "involved_system": mi.get("involved_system"),
            "event_classification": ai.get("event_classification"), "environment_type": ai.get("environment_type"),
            "road_conditions": ai.get("road_conditions"), "traffic_signs_present": ai.get("traffic_signs_present"),
            "precise_location": ai.get("precise_location"), "kinematics_summary": ai.get("kinematics_summary"),
            "involved_parties": ai.get("involved_parties"), "event_description": ai.get("event_description"),
            "counterfactual_extraction": {k: cfx.get(k) for k in ("normative_ad_system_behavior",
                                                                 "kinematic_avoidance_proof") if cfx.get(k)}}
    return json.dumps(keep, ensure_ascii=False, separators=(",", ":"))[:9000]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_twin(doc: dict, brief: dict, res: LiftResult) -> dict:
    """incident_schema_v2 record of the verified counterfactual positive."""
    nid, final = doc["incident_id"], res.final
    m = final.report.metrics
    tsir = final.tsir or {}
    ego = next((a for a in tsir.get("actors", []) if a.get("role") == "ego"), {})
    twin = copy.deepcopy(doc)
    twin["incident_id"] = f"{nid}-CF-POS-TSF"
    twin["incident_polarity"] = "POSITIVE"
    twin["lineage"] = {"origin_type": "COUNTERFACTUAL_POSITIVE", "is_synthetic_augmentation": False,
                       "parent_incident_id": nid, "counterfactual_positive_id": None, "augmented_variant_ids": [],
                       "augmentation_metadata": None, "backlog_item_id": doc["lineage"].get("backlog_item_id"),
                       "created_at": _now(), "updated_at": _now()}
    twin["workflow_state"] = {"stage": "ANALYZED", "analysis_loops_completed": len(res.attempts),
                              "quality_score": round(final.report.score(), 3),
                              "overall_confidence_percent": 90 if final.report.passed else 40,
                              "flagged_for_human_review": not final.report.passed}
    mi = twin["meta_information"]
    mi["title"] = f"[Counterfactual Positive Twin · Tanit Flywheel] {doc['meta_information']['title']}"[:300]
    mi["summary"] = brief["normative_behavior"]
    mi["analysis_date"] = _now()[:10]
    if isinstance(mi.get("analysts"), list):
        mi["analysts"] = mi["analysts"] + ["Tanit Scenario Flywheel (Qwen3.8-27B + TanitSim verifier ladder)"]
    ai = twin["accident_information"]
    ai["event_description"] = (f"Counterfactual positive of {nid}: {brief['normative_behavior']} "
                               f"Failure avoided: {brief['failure_mode']}")
    ks = ai.get("kinematics_summary")
    if isinstance(ks, dict):
        ks.update({"ego_initial_speed_kmh": round(float(ego.get("initial_speed", 0.0)) * 3.6, 1),
                   "min_ttc_seconds": m.get("min_ttc") if m.get("min_ttc") is not None else ks.get("min_ttc_seconds"),
                   "peak_deceleration_mps2": m.get("ego_max_decel", ks.get("peak_deceleration_mps2"))})
    pkg = {k: v for k, v in (res.package or {}).items() if k in ("userInput", "groundTruthAnswer")}
    twin["scenario_specification"] = pkg
    for k in ("tri_pillar_analysis", "causal_analysis", "visualization_artifacts", "derived_ad_system_specification",
              "processing_telemetry"):
        if k in twin:
            twin[k] = None
    return twin


def validate_incident(doc: dict) -> tuple[bool, list[str]]:
    """Validate against incident_schema_v2; its `scenario_structure_v1.json` $ref is the vendored scenario schema."""
    from urllib.parse import urljoin

    import jsonschema
    from referencing import Registry, Resource
    from referencing.jsonschema import DRAFT7
    inc = json.loads((SPEC / "incident_schema_v2.json").read_text(encoding="utf-8"))
    scen = Resource.from_contents(json.loads((SPEC / "scenario_schema.json").read_text(encoding="utf-8")),
                                  default_specification=DRAFT7)
    uris = {"scenario_structure_v1.json", urljoin(inc.get("$id", ""), "scenario_structure_v1.json")}
    reg = Registry().with_resources([(u, scen) for u in uris])
    cls = jsonschema.validators.validator_for(inc)
    errs = sorted(cls(inc, registry=reg).iter_errors(doc), key=lambda e: list(e.path))
    return not errs, [f"{'/'.join(map(str, e.path))}: {e.message[:200]}" for e in errs[:20]]


class Counterfactual:
    def __init__(self, llm: LLM | None = None):
        self.llm = llm or LLM()
        self.lifter = Lifter(self.llm)

    def brief(self, doc: dict) -> dict:
        msgs = [{"role": "system", "content": "You are a senior automated-driving safety engineer. You turn real "
                 "incidents into simulation test cases. TanitSim supports: straight/curved multi-lane roads and "
                 "3-5 arm junctions (signalized or not), vehicles, trucks, buses, bicycles, pedestrians, speed "
                 "changes, lane changes, cut-ins, following, emergency braking, stopping at lines, pedestrians "
                 "crossing; it does not model sensors, teleoperation, smoke, fire or vehicle damage."},
                {"role": "user", "content": "Incident record (JSON):\n" + brief_text(doc) + "\n\nWrite the brief. "
                 "The positive request must keep the road, actors and initial conditions of the incident and "
                 "only change the ego behaviour to the normative one, with numbers (speeds in km/h, distances "
                 "in m, deceleration in m/s2, minimum TTC)."}]
        return self.llm.chat(msgs, schema=Brief.model_json_schema(), role="incident_analyst", max_tokens=3000)

    def run(self, incident_id: str, workdir: str | Path, on_stage=None, replay_negative: bool = False) -> dict:
        stage = on_stage or (lambda *a: None)
        wd = Path(workdir)
        wd.mkdir(parents=True, exist_ok=True)
        doc = load_incident(incident_id)
        if doc["incident_polarity"] != "NEGATIVE":
            raise ValueError(f"{incident_id} is not a NEGATIVE incident")
        stage("brief", "running", "")
        br = self.brief(doc)
        (wd / "brief.json").write_text(json.dumps(br, indent=1, ensure_ascii=False), encoding="utf-8")
        stage("brief", "ok", br["failure_mode"][:200])
        sid = re.sub(r"[^a-z0-9]+", "_", incident_id.lower()) + "_cf_pos"
        pos = self.lifter.lift(br["positive_request"], wd / "positive", sid=sid, on_stage=on_stage)
        out = {"incident_id": incident_id, "brief": br, "positive": pos}
        if replay_negative:
            stage("replay_negative", "running", "")
            out["negative"] = self.lifter.lift(br["negative_request"], wd / "negative",
                                               sid=sid.replace("_cf_pos", "_neg"))
            stage("replay_negative", "ok", "")
        stage("twin", "running", "")
        twin = make_twin(doc, br, pos)
        ok, errs = validate_incident(twin)
        (wd / "incident_cf_positive.json").write_text(json.dumps(twin, indent=1, ensure_ascii=False), encoding="utf-8")
        stage("twin", "ok" if ok else "failed", "incident_schema_v2 valid" if ok else "; ".join(errs[:3]))
        out.update(twin=twin, twin_valid=ok, twin_errors=errs)
        return out
