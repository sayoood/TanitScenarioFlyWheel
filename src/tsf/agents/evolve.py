"""Harness² meta-loop for the Lifter, scored by the hard verifier ladder (Propose -> Probe -> Compose).

reference : run the probe set with the current harness (first TSIR attempt, analyses cached per query)
propose   : the harness editor reads the failing diagnostics and proposes one INVARIANT edit (a general rule
            for guardrails.md) and one DOMAIN lesson (EXPERIENCE entry with applicability keywords)
probe     : re-run the probe set with each candidate and with both, in throw-away copies of the harness
compose   : adopt the best candidate if the mean ladder score improves and no probe regresses pass -> fail;
            every decision is logged as a revision with its evidence; generations.jsonl feeds the UI
"""
from __future__ import annotations

import json
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from tsf.agents.harness import ROOT as HARNESS_ROOT, Harness, Lesson
from tsf.agents.lifter import Lifter, _slug
from tsf.agents.llm import LLM
from tsf.verify import ladder

PROBES = [
    "Generate data from a scenario where the ego vehicle changes the lane to the left to turn left in an "
    "intersection with traffic light at day in good weather conditions with vehicles surrounding",
    "The ego follows a lead car at 80 km/h on a two-lane rural road; the lead car brakes hard at 6 m/s2 and the "
    "ego must stop without collision.",
    "Unprotected left turn at a signalized four-way intersection: the ego waits for a gap in oncoming traffic "
    "and then turns left.",
    "A truck in the right lane of a three-lane highway cuts in close in front of the ego at 100 km/h; the ego "
    "decelerates moderately and keeps a safe gap.",
    "The ego approaches a red traffic light behind two queued cars, stops, and proceeds straight when the light "
    "turns green.",
    "A pedestrian crosses the road in front of the ego at an unsignalized T-junction while the ego approaches at "
    "40 km/h; the ego yields.",
]


class Edit(BaseModel):
    text: str = Field(description="The rule or lesson, imperative, at most 60 words, with numbers where useful")
    applies_to: list[str] = Field(description="Lower-case keywords of requests it applies to; empty = always")
    rationale: str = Field(description="Which observed failures it fixes and why")


class Proposal(BaseModel):
    invariant: Edit = Field(description="General authoring rule for guardrails.md (applies to every request)")
    domain: Edit = Field(description="Domain lesson for the experience library (scoped by keywords)")


def _harness_copy(edits: list[tuple[str, Edit]]) -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="tsf_h2_"))
    shutil.copytree(HARNESS_ROOT, tmp / "harness")
    root = tmp / "harness"
    h = Harness("lifter", root)
    for kind, e in edits:
        if kind == "INVARIANT":
            p = h.dir / "guardrails.md"
            p.write_text(p.read_text(encoding="utf-8").rstrip() + f"\n- {e.text}\n", encoding="utf-8")
        else:
            h.add_lesson(Lesson(f"les_{uuid.uuid4().hex[:8]}", "lifter", e.applies_to, e.text, "DOMAIN", "active"))
    return root


class Evolver:
    def __init__(self, llm: LLM | None = None, probes: list[str] = PROBES, workdir: str | Path = "runs/h2"):
        self.llm = llm or LLM()
        self.probes, self.wd = probes, Path(workdir)
        self.wd.mkdir(parents=True, exist_ok=True)
        self._analysis: dict[str, dict] = {}

    def evaluate(self, root: Path, tag: str) -> dict:
        """First-attempt ladder score per probe with the harness at `root`."""
        lif = Lifter(self.llm, Harness("lifter", root))
        rows = []
        for i, q in enumerate(self.probes):
            if q not in self._analysis:
                self._analysis[q] = lif.analyse(q)
            sid = _slug(q, 40)
            t0 = time.perf_counter()
            try:
                raw = self.llm.chat(lif.propose(q, self._analysis[q], sid), schema=lif._tsir_schema, role="probe",
                                    thinking=lif.h.knobs()["thinking"], temperature=lif.h.knobs()["temperature"])
                raw.setdefault("meta", {})["id"] = sid
                rep = ladder.run(raw, self.wd / tag / f"p{i}", qc=False)
                rows.append({"probe": i, "passed": rep.passed, "score": rep.score(), "diag": rep.diagnostics()[:700],
                             "s": round(time.perf_counter() - t0, 1)})
            except Exception as e:  # noqa: BLE001 - a crashed probe scores zero
                rows.append({"probe": i, "passed": False, "score": 0.0, "diag": f"{type(e).__name__}: {e}"[:300],
                             "s": round(time.perf_counter() - t0, 1)})
        return {"tag": tag, "score": round(sum(r["score"] for r in rows) / len(rows), 4),
                "valid_rate": round(sum(r["passed"] for r in rows) / len(rows), 3), "rows": rows}

    def propose(self, ref: dict) -> Proposal:
        h = Harness("lifter")
        fails = [f"probe {r['probe']} ({self.probes[r['probe']][:90]}...): {r['diag']}" for r in ref["rows"]
                 if not r["passed"]]
        msgs = [{"role": "system", "content": "You improve the harness (instructions) of an LLM agent that writes "
                 "TSIR traffic scenarios; you never see or change model weights. Edits must be general, short and "
                 "testable; never weaken the verifier or tell the agent to drop assertions."},
                {"role": "user", "content": "Current guardrails:\n" + h.part("guardrails.md") +
                 "\n\nVerifier failures of the current harness on the probe set:\n" + "\n".join(fails) +
                 "\n\nPropose one INVARIANT rule and one DOMAIN lesson that would have prevented the most failures."}]
        return Proposal.model_validate(self.llm.chat(msgs, schema=Proposal.model_json_schema(),
                                                     role="harness_editor", max_tokens=1500))

    def generation(self, gen: int) -> dict:
        ref = self.evaluate(HARNESS_ROOT, f"g{gen}_ref")
        if ref["valid_rate"] == 1.0:
            rec = {"gen": gen, "score": ref["score"], "valid_rate": ref["valid_rate"], "adopted": "none (all pass)"}
            self._log(rec, ref)
            return rec
        prop = self.propose(ref)
        cands = {"invariant": [("INVARIANT", prop.invariant)], "domain": [("DOMAIN", prop.domain)],
                 "both": [("INVARIANT", prop.invariant), ("DOMAIN", prop.domain)]}
        res = {k: self.evaluate(_harness_copy(v), f"g{gen}_{k}") for k, v in cands.items()}

        def regress(r):
            return any(a["passed"] and not b["passed"] for a, b in zip(ref["rows"], r["rows"]))

        best = max(res, key=lambda k: res[k]["score"])
        ok = res[best]["score"] > ref["score"] + 0.02 and not regress(res[best])
        h = Harness("lifter")
        evidence = {"ref": ref["score"], **{k: r["score"] for k, r in res.items()}, "valid_ref": ref["valid_rate"],
                    "valid_best": res[best]["valid_rate"], "n_probes": len(self.probes)}
        for kind, e in cands[best] if ok else []:
            if kind == "INVARIANT":
                p = h.dir / "guardrails.md"
                p.write_text(p.read_text(encoding="utf-8").rstrip() + f"\n- {e.text}\n", encoding="utf-8")
            else:
                h.add_lesson(Lesson(f"les_{uuid.uuid4().hex[:8]}", "lifter", e.applies_to, e.text, "DOMAIN", "active",
                                    {"delta_score": round(res[best]["score"] - ref["score"], 4), **evidence}))
        h.log_revision("COMPOSITE_MERGE" if best == "both" else f"{best.upper()}_EDITOR",
                       "guardrails.md+EXPERIENCE" if best == "both" else
                       ("guardrails.md" if best == "invariant" else "EXPERIENCE"),
                       json.dumps(prop.model_dump()), "adopted" if ok else "rejected", evidence)
        new = res[best] if ok else ref
        rec = {"gen": gen, "score": new["score"], "valid_rate": new["valid_rate"], "ref_score": ref["score"],
               "adopted": best if ok else "none"}
        self._log(rec, {"ref": ref, **res, "proposal": prop.model_dump()})
        return rec

    def _log(self, rec: dict, detail: dict):
        rec["t"] = time.time()
        with (HARNESS_ROOT / "generations.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        (self.wd / f"gen{rec['gen']}.json").write_text(json.dumps(detail, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    import sys
    ev = Evolver()
    start = 1 + sum(1 for _ in (HARNESS_ROOT / "generations.jsonl").open()) if (HARNESS_ROOT / "generations.jsonl").exists() else 1
    for g in range(start, start + int(sys.argv[1] if len(sys.argv) > 1 else 2)):
        print(ev.generation(g), flush=True)
