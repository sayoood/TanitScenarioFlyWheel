"""IMDAD v2 record (``spec/imdad/incident_schema_v2.json``) of an augmented scenario variant.

The record carries the lineage ``origin_type = SYNTHETIC_AUGMENTATION`` / ``is_synthetic_augmentation = true`` with the
parent id and the augmentation metadata (mode, dimension(s), seed, JSON patch, rationale) and embeds the verified
Scenario_Structure package as ``scenario_specification``. When the parent run already has an IMDAD record (UC3 twin)
it is the base, otherwise a minimal schema-valid record is synthesised from the TSIR.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone

from tsf.agents.augment_plan import time_band

_WEATHER = {"clear": "clear", "cloudy": "overcast", "rain": "rain", "heavy_rain": "heavy_rain", "fog": "fog",
            "snow": "snow"}
_SURFACE = {"dry": "dry", "wet": "wet", "snow": "snow_covered", "ice": "icy"}
_PARTY = {"pedestrian": "pedestrian", "bicycle": "cyclist", "motorcycle": "motorcyclist"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _lighting(env: dict, setting: str | None, junction: bool) -> str:
    b = time_band(env["time_of_day"])
    if b == "day":
        return "daylight"
    if b == "dusk":
        return "dusk_dawn"
    return "night_lit" if junction or setting in ("urban", "suburban", "industrial") else "night_unlit"


def _env_type(d: dict) -> str:
    """IMDAD environment_type: a work zone (cones, construction board or road-works sign) wins over the setting."""
    objs = d.get("static_objects") or []
    if any(o.get("kind") in ("cone", "construction_sign") or o.get("sign_type") == "road_works" for o in objs):
        return "construction_zone"
    st = d["environment"].get("setting")
    if st in ("highway", "rural", "suburban", "urban"):
        return st
    if st == "industrial":
        return "urban"
    road = d["road"]
    if road["type"] == "junction":
        return "intersection"
    return "highway" if road.get("speed_limit_kph", 50) > 90 else "urban"


def _signs_present(d: dict, old: list | None) -> list[str]:
    """`traffic_signs_present` of the IMDAD record: the parent's entries plus the sign objects of the variant."""
    from tsf.agents.augment_render import SIGN_NAME

    names = []
    for o in d.get("static_objects") or []:
        if o.get("kind") == "traffic_sign" and o.get("sign_type"):
            v = o.get("sign_value")
            names.append(SIGN_NAME.get(o["sign_type"], o["sign_type"]) + (f" {v:g} km/h" if v is not None else ""))
        elif o.get("kind") == "construction_sign":
            names.append("construction warning board")
    return list(dict.fromkeys([*(old or []), *names]))


def _parties(d: dict) -> list[dict]:
    out = []
    for i, a in enumerate(d["actors"]):
        ego = a["role"] == "ego"
        t = "SDV" if ego else _PARTY.get(a["category"], "human_driven_vehicle")
        out.append({"id": i, "role_id": a["id"], "type": t, "is_vru": t in ("pedestrian", "cyclist", "motorcyclist"),
                    "initial_speed_kmh": round(a["initial_speed"] * 3.6, 1),
                    "maneuver": ", ".join(x["type"] for x in d.get("actions", []) if x["actor"] == a["id"]) or "none",
                    "description": f"{a['category']} {a['color']}" + (" (ego, system under test)" if ego else "")})
    return out


def make_record(parent: dict, variant: dict, package: dict | None, *, variant_id: str, mode: str, dims: list[str],
                seed: int, patch: list[dict], rationale: str, report, diff_summary: list[str],
                parent_record: dict | None = None, attempts: int = 1, timings: dict | None = None) -> dict:
    """Schema-valid IMDAD v2 record of a verified variant (validate with ``counterfactual.validate_incident``)."""
    env, m = variant["environment"], report.metrics
    pol = variant["meta"].get("polarity", "positive").upper()
    rec = copy.deepcopy(parent_record) if parent_record else {}
    pid = (parent_record or {}).get("incident_id") or parent["meta"]["id"]
    lineage_old = (rec.get("lineage") or {})
    ttc = m.get("min_ttc")
    crit = ttc is not None and ttc < 2.5 or m.get("collision")
    rec.update(schema_version="2.0.0", incident_id=variant_id, incident_polarity=pol)
    rec["lineage"] = {
        "origin_type": "SYNTHETIC_AUGMENTATION", "is_synthetic_augmentation": True, "parent_incident_id": pid,
        "counterfactual_positive_id": None, "augmented_variant_ids": [], "backlog_item_id": lineage_old.get("backlog_item_id"),
        "augmentation_metadata": {
            "varied_dimensions": list(dims), "transformation_rationale": rationale,
            "odd_delta": {k: [parent["environment"].get(k), env.get(k)] for k in env
                          if parent["environment"].get(k) != env.get(k)},
            "behavior_delta": {"changes": diff_summary[:30]},
            "mode": mode, "dimensions": list(dims), "seed": seed, "patch": patch, "rationale": rationale},
        "created_at": _now(), "updated_at": _now()}
    rec["workflow_state"] = {"stage": "AUGMENTED", "analysis_loops_completed": attempts,
                             "quality_score": round(report.score() * 100, 1),
                             "overall_confidence_percent": 90 if report.passed else 40,
                             "flagged_for_human_review": not report.passed}
    old_meta = rec.get("meta_information") or {}
    rec["meta_information"] = {
        **old_meta, "report_id": variant_id, "title": variant["meta"]["title"][:300],
        "incident_date": old_meta.get("incident_date") or _now()[:10], "source_url": old_meta.get("source_url") or "",
        "sources": old_meta.get("sources") or [], "analysis_date": _now()[:10],
        "analysts": list(old_meta.get("analysts") or []) + ["Tanit Scenario Flywheel augmenter (Qwen3.8-27B + TanitSim verifier ladder)"],
        "location": old_meta.get("location") or {"city": "n/a (synthetic)", "state_province": "n/a", "country": "n/a",
                                                 "market_jurisdiction": "n/a"},
        "summary": variant["meta"]["description"][:2000] or variant["meta"]["title"],
        "involved_system": old_meta.get("involved_system") or {
            "developer_oem": "n/a (synthetic scenario)", "vehicle_make_model": "n/a", "system_name": "system under test",
            "sae_level": 4, "system_description": "Generic automated driving system under test",
            "monitoring_and_operation_system": {
                "dms_type": "n/a (synthetic scenario)", "hmi_tor_capability": "n/a",
                "remote_operations_architecture": "n/a", "monitoring_behavior_observed": "n/a (simulation only)"}}}
    old_ai = rec.get("accident_information") or {}
    junction = variant["road"]["type"] == "junction"
    rec["accident_information"] = {
        **old_ai,
        "event_classification": "critical_challenging_situation" if crit else "exemplary_edge_case_mitigation",
        "sotif_quadrant": old_ai.get("sotif_quadrant") or ("AREA_3_UNKNOWN_UNSAFE" if crit else "AREA_4_UNKNOWN_SAFE"),
        "risk_delta": {"direction": "INCREASED_RISK" if crit else "NEUTRAL_CONTROLLED_RISK",
                       "magnitude_score_minus10_to_plus10": 3.0 if crit else 0.0,
                       "rationale": f"Synthetic variant; minimum TTC {ttc} s, peak ego deceleration "
                                    f"{m.get('ego_max_decel')} m/s2 in TanitSim."},
        "environment_type": _env_type(variant),
        "road_conditions": {"weather": _WEATHER[env["weather"]],
                            "lighting": _lighting(env, env.get("setting"), junction),
                            "surface": _SURFACE[env["road_surface"]],
                            **({"speed_limit_kmh": min(o["sign_value"] for o in variant["static_objects"]
                                                       if o.get("sign_type") == "speed_limit")}
                               if any(o.get("sign_type") == "speed_limit" for o in variant.get("static_objects") or [])
                               else {})},
        **({"traffic_signs_present": _signs_present(variant, old_ai.get("traffic_signs_present"))}
           if variant.get("static_objects") or old_ai.get("traffic_signs_present") else {}),
        "involved_parties": _parties(variant),
        "event_description": variant["meta"]["description"] or variant["meta"]["title"],
        "damage_assessment": {"severity": "S0", "kabco_scale": "O_NO_INJURY",
                              "description": "Simulated scenario: no real-world damage."},
        "taxonomy_labels": {"odd_tags": list(dict.fromkeys(variant["meta"]["tags"] + list(dims))),
                            "scenario_tags": list(dims), "subsystem_tags": [], "agent_tags": [],
                            "regulatory_tags": []}}
    if package:
        rec["scenario_specification"] = {k: v for k, v in package.items() if k in ("userInput", "groundTruthAnswer")}
    else:
        rec["scenario_specification"] = {"userInput": {"userInput": variant["meta"]["description"]},
                                         "groundTruthAnswer": {}}
    for k in ("tri_pillar_analysis", "causal_analysis", "visualization_artifacts", "derived_ad_system_specification",
              "processing_telemetry"):
        rec[k] = None
    if timings:
        rec["processing_telemetry"] = {"augmentation_duration_seconds": round(timings.get("total_s", 0.0), 1)}
    return rec
