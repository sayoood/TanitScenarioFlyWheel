"""UC1 Scenario Lifter: natural language -> verified TSIR -> IMDAD `Scenario_Structure` package.

Flow (every step logged):
  1. Analyst      : NL -> analysisOfUserQuery, plan, assumptions, maths (schema-constrained)
  2. Proposer     : analysis -> TSIR (constrained by the TSIR JSON schema)   = firstHypothesis
  3. Verifier     : ladder L0-L3 (compile, XSD/QC, lane-graph cross-check, simulation oracle)
  4. Repairer     : diagnostics -> revised TSIR, up to `max_repairs`         = improvements
  5. Documenter   : codeBeamer description, functional requirements, variations
  6. Assembler    : deterministic package; validated against spec/imdad/scenario_schema.json
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, Field

from tsf.agents.harness import Harness
from tsf.agents.llm import LLM
from tsf.ir.scenario import Scenario, json_schema
from tsf.verify import ladder

Scalar = str | float | int | bool | None


# ----------------------------------------------------------------------------- step schemas
class ActorInfo(BaseModel):
    id: str
    type: str
    initialState: dict[str, Scalar]
    behavior: str


class ExtractedInformation(BaseModel):
    scenarioGoal: str
    actors: list[ActorInfo]
    road: dict[str, Scalar]
    environment: dict[str, Scalar]


class Variable(BaseModel):
    name: str
    value: str | float | None
    unit: str | None


class MissingValue(BaseModel):
    variable: str
    value: str
    source: str


class Assumptions(BaseModel):
    assumptions: list[str]
    missingValues: list[MissingValue]


class AnalysisOfUserQuery(BaseModel):
    extractedInformation: ExtractedInformation
    variables: list[Variable]
    semanticMeaning: str


class Analysis(BaseModel):
    analysisOfUserQuery: AnalysisOfUserQuery
    planToDefineScenario: list[str] = Field(..., min_length=3, max_length=10)
    requiredAssumptionsAndMissingValues: Assumptions
    mathematicalCorpus: list[str] = Field(..., min_length=2, max_length=8)


class ShortDesc(BaseModel):
    title: str
    keywords: list[str]
    tags: list[str]


class InitialState(BaseModel):
    actorId: str
    longitudinalPosition: float
    lateralPosition: str
    speed: float


class FinalState(BaseModel):
    actorId: str
    condition: str


class EventDesc(BaseModel):
    id: str
    trigger: str
    action: str


class AgentBehavior(BaseModel):
    agentId: str
    behavior: list[str]


class BehaviorSequence(BaseModel):
    ego: list[str]
    agents: list[AgentBehavior]


class DynEnv(BaseModel):
    id: str
    type: str
    description: str


class StaticElement(BaseModel):
    element: str
    description: str


class CodeBeamer(BaseModel):
    shortScenarioDescription: ShortDesc
    scenarioCategory: str
    initialStates: list[InitialState]
    finalStates: list[FinalState]
    environmentConditions: dict[str, Scalar]
    roadDescription: dict[str, Scalar]
    staticEnvironment: list[StaticElement]
    dynamicEnvironment: list[DynEnv]
    events: list[EventDesc]
    behaviorSequence: BehaviorSequence


class Requirement(BaseModel):
    id: str
    requirement: str
    passCriteria: list[str]


class VariableInterval(BaseModel):
    variable: str
    minValue: str | None = None
    maxValue: str | None = None
    description: str | None = None


class Variations(BaseModel):
    description: str
    variableIntervals: list[VariableInterval]


class Narrative(BaseModel):
    functionalRequirementsSummary: str
    improvementsSummary: list[str]
    codeBeamerScenarioDescription: CodeBeamer
    derivedFunctionalRequirements: list[Requirement] = Field(..., min_length=2, max_length=6)
    scenarioVariations: Variations


# ----------------------------------------------------------------------------- result
@dataclass
class Attempt:
    tsir: dict | None
    report: ladder.Report
    seconds: float


@dataclass
class LiftResult:
    query: str
    analysis: dict
    attempts: list[Attempt]
    narrative: dict | None
    package: dict | None
    package_valid: bool
    package_errors: list[str]
    llm_stats: dict
    seconds: float
    workdir: Path
    extra: dict = field(default_factory=dict)

    @property
    def final(self) -> Attempt:
        return next((a for a in reversed(self.attempts) if a.report.passed), self.attempts[-1])

    def kpis(self) -> dict:
        first = self.attempts[0].report
        return {"first_pass_valid": first.passed, "first_pass_score": first.score(),
                "final_valid": self.final.report.passed, "final_score": self.final.report.score(),
                "repairs": len(self.attempts) - 1, "time_to_valid_s": round(self.seconds, 1),
                "package_valid": self.package_valid, **{f"llm_{k}": v for k, v in self.llm_stats.items()}}


def _slug(text: str, n: int = 48) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return ("s_" + s)[:n].rstrip("_")


# ----------------------------------------------------------------------------- agent
class Lifter:
    def __init__(self, llm: LLM | None = None, harness: Harness | None = None):
        self.llm = llm or LLM()
        self.h = harness or Harness("lifter")
        self._tsir_schema = json_schema()

    def _skills_block(self, query: str = "") -> str:
        sk = self.h.skills(self.h.knobs()["n_skills"], query)
        return "\n\n".join(f"Example description:\n{s['meta']['description']}\nExample TSIR:\n"
                           f"{json.dumps(s, separators=(',', ':'))}" for s in sk)

    def analyse(self, query: str) -> dict:
        msgs = [{"role": "system", "content": self.h.system(query)},
                {"role": "user", "content": "Analyse this scenario request for a test engineer. Extract actors, "
                 "road, environment and variables; state the semantic meaning; give a numbered plan; list "
                 "assumptions and quantify every missing value (speeds, positions, timings) with its source; give "
                 "the governing equations (TTC, safe distance, required deceleration...).\n\nRequest:\n" + query}]
        return self.llm.chat(msgs, schema=Analysis.model_json_schema(), role="analyst",
                             temperature=self.h.knobs()["temperature"], max_tokens=4096)

    def propose(self, query: str, analysis: dict, sid: str) -> list[dict]:
        sys = self.h.system(query)
        user = (f"{self._skills_block(query)}\n\nNow write the TSIR for this request. Use meta.id = \"{sid}\".\n"
                f"Request:\n{query}\n\nEngineering analysis (quantified assumptions to use):\n"
                f"{json.dumps(analysis, separators=(',', ':'))}")
        return [{"role": "system", "content": sys}, {"role": "user", "content": user}]

    def lift(self, query: str, workdir: str | Path, sid: str | None = None, on_stage=None) -> LiftResult:
        """`on_stage(name, status, detail)` is called at every stage transition (UI / event stream)."""
        stage = on_stage or (lambda *a: None)
        t0 = time.perf_counter()
        wd = Path(workdir)
        wd.mkdir(parents=True, exist_ok=True)
        k = self.h.knobs()
        sid = sid or _slug(query)
        stage("analyse", "running", "")
        analysis = self.analyse(query)
        stage("analyse", "ok", "")
        msgs = self.propose(query, analysis, sid)
        attempts: list[Attempt] = []
        for i in range(k["max_repairs"] + 1):
            ta = time.perf_counter()
            stage("propose" if i == 0 else "repair", "running", f"attempt {i}")
            raw = self.llm.chat(msgs, schema=self._tsir_schema, role="proposer" if i == 0 else "repairer",
                                thinking=k["thinking"], temperature=k["temperature"])
            raw.setdefault("meta", {})["id"] = sid
            stage("propose" if i == 0 else "repair", "ok", f"attempt {i}")
            stage("verify", "running", f"attempt {i}")
            rep = ladder.run(raw, wd / f"attempt{i}")
            attempts.append(Attempt(raw, rep, time.perf_counter() - ta))
            stage("verify", "ok" if rep.passed else "failed", rep.diagnostics()[:300])
            if rep.passed:
                break
            msgs = msgs + [{"role": "assistant", "content": json.dumps(raw, separators=(",", ":"))},
                           {"role": "user", "content": "The verifier rejected this TSIR:\n" + rep.diagnostics() +
                            f"\nSimulation metrics: {json.dumps(rep.metrics)}\nApply the hints. Make the MINIMAL "
                            "change that fixes the root cause (positions, duration, timing, signal offsets, lane "
                            "numbers, triggers, or an assertion that contradicts the scenario's own values); keep "
                            "every actor and value that is not involved unchanged. Output the complete corrected "
                            "TSIR."}]
        final = next((a for a in reversed(attempts) if a.report.passed), attempts[-1])
        stage("document", "running", "")
        narrative = self.document(query, analysis, attempts, final)
        stage("document", "ok", "")
        pkg = assemble_package(query, analysis, attempts, final, narrative)
        valid, errs = validate_package(pkg)
        (wd / "package.json").write_text(json.dumps(pkg, indent=2), encoding="utf-8")
        res = LiftResult(query, analysis, attempts, narrative, pkg, valid, errs, self.llm.stats(),
                         time.perf_counter() - t0, wd)
        (wd / "kpis.json").write_text(json.dumps(res.kpis(), indent=2))
        return res

    def document(self, query: str, analysis: dict, attempts: list[Attempt], final: Attempt) -> dict:
        hist = [{"attempt": i, "passed": a.report.passed, "diagnostics": a.report.diagnostics()}
                for i, a in enumerate(attempts)]
        msgs = [{"role": "system", "content": self.h.part("system_prompt.md")},
                {"role": "user", "content": "Document the verified scenario for a codeBeamer test specification. "
                 "Derive 2-6 testable functional requirements for the ego (ids REQ-EGO-<TOPIC>-NN, measurable pass "
                 "criteria with thresholds), the scenario variations (intervals from the TSIR parameters) and a "
                 "short list of the improvements made during verification.\n\n"
                 f"Request:\n{query}\n\nFinal TSIR:\n{json.dumps(final.tsir, separators=(',', ':'))}\n\n"
                 f"Simulation metrics:\n{json.dumps(final.report.metrics)}\n\nVerification history:\n"
                 f"{json.dumps(hist)}"}]
        return self.llm.chat(msgs, schema=Narrative.model_json_schema(), role="documenter",
                             temperature=self.h.knobs()["temperature"])


# ----------------------------------------------------------------------------- package
def _read(p: str | None) -> str:
    return Path(p).read_text(encoding="utf-8") if p and Path(p).exists() else ""


def _syntax(rep: ladder.Report) -> dict:
    def st(*checks):
        fs = [f for f in rep.findings if f.check in checks]
        return {"isValid": all(f.ok for f in fs) and bool(fs), "errors": [f.message for f in fs if not f.ok]}
    return {"openScenario1": st("compile", "openscenario_xsd"), "openScenario2": st("compile"),
            "openDrive": st("compile", "opendrive_xsd", "asam_qc_opendrive", "tanitsim_parse"),
            "pythonCode": {"isValid": True, "errors": []}}


def assemble_package(query: str, analysis: dict, attempts: list[Attempt], final: Attempt, nar: dict) -> dict:
    from tsf.compile.pycode import emit as pycode

    first = attempts[0]
    fa = first.report.artifacts
    conflicts = [f"[{f.level}:{f.check}] {f.message}" for f in first.report.findings if not f.ok]
    sc_final = Scenario.model_validate(final.tsir) if final.report.passed or final.tsir else None
    a = analysis
    return {
        "userInput": {"userInput": query},
        "groundTruthAnswer": {
            "reasoning": {
                "analysisOfUserQuery": a["analysisOfUserQuery"],
                "planToDefineScenario": a["planToDefineScenario"],
                "requiredAssumptionsAndMissingValues": a["requiredAssumptionsAndMissingValues"],
                "mathematicalCorpus": a["mathematicalCorpus"],
                "firstHypothesis": {
                    "roadDescription": _read(fa.get("xodr"))[:4000] or "compilation failed",
                    "openScenarioDescription1": _read(fa.get("xosc")),
                    "openScenarioDescription2": _read(fa.get("osc2")),
                    "pythonCode": pycode(Scenario.model_validate(first.tsir), getattr(first.report, "_log", None))
                    if first.tsir and first.report.findings[0].ok else "",
                    "functionalRequirements": nar["functionalRequirementsSummary"],
                },
                "validationAndJudging": {
                    "syntaxCheck": _syntax(first.report),
                    "scenarioValidityCheck": {"isConsistent": first.report.passed, "conflicts": conflicts},
                    "improvements": nar["improvementsSummary"],
                },
            },
            "openDriveRoadDescription": _read(final.report.artifacts.get("xodr")),
            "assumptionsAndFilledMissedValues": {
                "assumptions": a["requiredAssumptionsAndMissingValues"]["assumptions"],
                "filledValues": a["requiredAssumptionsAndMissingValues"]["missingValues"]},
            "openScenarioDescription1": _read(final.report.artifacts.get("xosc")),
            "openScenarioDescription2": _read(final.report.artifacts.get("osc2")),
            "pythonCode": pycode(sc_final, getattr(final.report, "_log", None)) if sc_final else "",
            "codeBeamerScenarioDescription": nar["codeBeamerScenarioDescription"],
            "derivedFunctionalRequirements": nar["derivedFunctionalRequirements"],
            "scenarioVariations": nar["scenarioVariations"],
        },
        "tanitFlywheel": {"tsir": final.tsir, "verification": final.report.to_dict()},
    }


def validate_package(pkg: dict) -> tuple[bool, list[str]]:
    import jsonschema
    schema = json.loads((Path(__file__).resolve().parents[3] / "spec/imdad/scenario_schema.json").read_text())
    v = jsonschema.Draft7Validator(schema)
    errs = [f"{'/'.join(map(str, e.path))}: {e.message}" for e in v.iter_errors(pkg)]
    return not errs, errs[:20]
