"""Harness² for the scenario augmenter: probe set, deterministic scoring and a Propose-Probe-Compose generation.

Same contract as ``tsf.agents.evolve.Evolver`` for the Lifter: ``evaluate(root, tag)`` returns
``{"tag", "score", "valid_rate", "rows": [{"probe", "passed", "score", "diag", "s"}]}`` for the harness at ``root``,
``generation(gen)`` proposes one INVARIANT rule (``guardrails.md``) and one DOMAIN lesson (experience library), probes
the candidates in throw-away copies of the harness and adopts the best one only if the mean score improves and no
probe regresses from pass to fail. The first editor proposal of every probe is scored (no repairs), because repairs
hide what the harness itself gets wrong.

Probe score (0..1) = 0.2 isolation + 0.3 verifier ladder score + 0.2 description names every changed attribute
+ 0.3 intent checks (expected values in the variant TSIR, see ``harness/augmenter/probes.jsonl``). A probe *passes*
when all four are perfect.

    python -m tsf.agents.augment_harness evaluate          # score the current harness
    python -m tsf.agents.augment_harness 1                  # run one generation (several minutes of LLM time)
"""
from __future__ import annotations

import fnmatch
import json
import shutil
import tempfile
import time
import uuid
from pathlib import Path

from tsf.agents import augment_patch as P
from tsf.agents import augment_plan as PL
from tsf.agents.evolve import Edit, Proposal
from tsf.agents.harness import Harness, Lesson
from tsf.agents.llm import LLM
from tsf.ir.scenario import Scenario

AGENT = "augmenter"
REPO = Path(__file__).resolve().parents[3]
W_ISO, W_VER, W_DESC, W_INTENT = 0.2, 0.3, 0.2, 0.3


def _root() -> Path:
    from tsf.agents.augment import harness_root
    return harness_root()


def load_probes(root: Path | None = None) -> list[dict]:
    p = (root or _root()) / AGENT / "probes.jsonl"
    if not p.exists():                                       # live state dirs only carry what was evolved
        p = Path(__file__).resolve().parents[3] / "harness" / AGENT / "probes.jsonl"
    rows = [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if "static_objects" not in Scenario.model_fields:        # probes that need the environment extension wait for it
        rows = [r for r in rows if r.get("requires") != "static_objects"]
    return rows


# ----------------------------------------------------------------------------- deterministic intent checks
def _get(doc, ptr: str):
    cur = doc
    for t in P.tokens(ptr):
        cur = cur[int(t)] if isinstance(cur, list) else cur[t]
    return cur


def intent_checks(expect: dict, parent: dict, variant: dict) -> list[tuple[str, bool]]:
    """(name, ok) for every expectation of a probe, evaluated on the variant TSIR."""
    out: list[tuple[str, bool]] = []
    env = variant["environment"]
    for k, want in (expect.get("env") or {}).items():
        if k == "band":
            out.append((f"env band {want}", PL.time_band(env["time_of_day"]) == want))
        else:
            ok = env.get(k) in (want if isinstance(want, list) else [want])
            out.append((f"env {k} in {want}", ok))
    for r in expect.get("road") or []:
        try:
            v = _get(variant, r["path"])
        except (KeyError, IndexError, TypeError):
            out.append((f"road {r['path']} exists", False))
            continue
        if "equals" in r:
            out.append((f"{r['path']} == {r['equals']}", v == r["equals"]))
        if "max" in r:
            out.append((f"{r['path']} <= {r['max']}", v <= r["max"]))
        if "min" in r:
            out.append((f"{r['path']} >= {r['min']}", v >= r["min"]))
    changes = P.diff(parent, variant)
    if expect.get("actors_touched"):
        ids = {a["id"] for a in variant["actors"]}
        touched = set()
        for c in changes:
            if c.path.startswith("/actions"):
                for x in (c.after, c.before):
                    if isinstance(x, dict) and "actor" in x:
                        touched.add(x["actor"])
                if not isinstance(c.after, dict) and not isinstance(c.before, dict):
                    i = c.path.split("/")[2] if len(c.path.split("/")) > 2 else None
                    if i is not None and i.isdigit() and int(i) < len(variant["actions"]):
                        touched.add(variant["actions"][int(i)]["actor"])
            elif c.path.startswith("/actors/"):
                i = c.path.split("/")[2]
                if i.isdigit() and int(i) < len(variant["actors"]):
                    touched.add(variant["actors"][int(i)]["id"])
        for pat in expect["actors_touched"]:
            out.append((f"behaviour of {pat} changed", any(fnmatch.fnmatch(t, pat) for t in touched & ids)))
    n_old = len(parent["actors"])
    new = variant["actors"][n_old:]
    if "new_actors_min" in expect:
        out.append((f"at least {expect['new_actors_min']} new actor(s)", len(new) >= expect["new_actors_min"]))
    for cat in expect.get("new_actor_categories") or []:
        out.append((f"new {cat}", any(a["category"] == cat for a in new)))
    objs = variant.get("static_objects") or []
    if expect.get("static_kinds_any"):
        out.append((f"static object of kind {expect['static_kinds_any']}", any(o["kind"] in expect["static_kinds_any"] for o in objs)))
    for s in expect.get("signs") or []:
        out.append((f"sign {s}", any(all(o.get(k) == v for k, v in s.items()) for o in objs)))
    return out


def score_variant(res: dict, probe: dict, parent: dict, variant: dict | None) -> tuple[float, bool, list[str]]:
    iso = 1.0 if (res.get("isolation") or {}).get("ok") else 0.0
    ver = res.get("verifier") or {}
    ver_s = float(ver.get("score", 0.0)) if res.get("status") == "ok" else float(ver.get("score", 0.0)) * 0.5
    ok_v = bool(res.get("status") == "ok" and ver.get("passed"))
    desc = 1.0 if (res.get("description_check") or {}).get("ok") else 0.0
    checks = intent_checks(probe.get("expect") or {}, parent, variant) if variant else []
    intent = (sum(ok for _, ok in checks) / len(checks)) if checks else (1.0 if variant else 0.0)
    score = W_ISO * iso + W_VER * ver_s + W_DESC * desc + W_INTENT * intent
    diag = []
    if not iso:
        diag.append("isolation FAILED: " + "; ".join((res.get("attempts") or [{}])[0].get("issues", [])[:2])[:240])
    if not ok_v:
        bad = [f"{f.get('level', '')} {f.get('check', '')}: {f.get('message', '')}" for f in ver.get("findings", [])
               if not f.get("ok", True)][:2]
        diag.append("verifier: " + (res.get("error") or "; ".join(bad) or "not passed")[:300])
    if desc < 1:
        diag.append("description misses: " + ", ".join((res.get("description_check") or {}).get("missing", [])))
    diag += [f"intent not met: {n}" for n, ok in checks if not ok]
    return score, bool(iso and ok_v and desc and intent == 1.0), diag


class AugmentEvaluator:
    def __init__(self, llm: LLM | None = None, probes: list[dict] | None = None, workdir: str | Path = "runs/h2_augment"):
        self.llm = llm or LLM()
        self.probes = probes if probes is not None else load_probes()
        self.wd = Path(workdir)
        self.wd.mkdir(parents=True, exist_ok=True)

    def evaluate(self, root: Path, tag: str) -> dict:
        """First-attempt score per probe with the augmenter harness at ``root`` (no repairs, no render, no package)."""
        from tsf.agents.augment import augment, load_parent
        h = Harness(AGENT, root)
        rows = []
        for i, pr in enumerate(self.probes):
            t0 = time.perf_counter()
            try:
                parent = load_parent(REPO / pr["parent"])
                res = augment(REPO / pr["parent"], self.wd / tag / f"p{i}", mode="manual", requests=pr["requests"],
                              separate=bool(pr.get("separate")), seed=int(pr.get("seed", 0)), render=False,
                              criticality=False, package=False, max_repairs=0, workers=1, llm=lambda: self.llm,
                              harness=h, parent_qc=False)
                scores, passed, diag = [], [], []
                for r in res:
                    variant = None
                    for k in ("tsir", "candidate"):
                        q = (r.get("paths") or {}).get(k)
                        if q and Path(q).exists():
                            variant = json.loads(Path(q).read_text(encoding="utf-8"))
                            break
                    s, ok, d = score_variant(r, pr, parent, variant)
                    scores.append(s)
                    passed.append(ok)
                    diag += d
                rows.append({"probe": i, "id": pr["id"], "passed": all(passed), "score": round(sum(scores) / len(scores), 4),
                             "diag": " | ".join(diag)[:700], "s": round(time.perf_counter() - t0, 1)})
            except Exception as e:  # noqa: BLE001 - a crashed probe scores zero
                rows.append({"probe": i, "id": pr["id"], "passed": False, "score": 0.0, "diag": f"{type(e).__name__}: {e}"[:300],
                             "s": round(time.perf_counter() - t0, 1)})
        return {"tag": tag, "score": round(sum(r["score"] for r in rows) / len(rows), 4),
                "valid_rate": round(sum(r["passed"] for r in rows) / len(rows), 3), "rows": rows}


# ----------------------------------------------------------------------------- Propose - Probe - Compose
def _harness_copy(root0: Path, edits: list[tuple[str, Edit]]) -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="tsf_h2_aug_"))
    shutil.copytree(root0, tmp / "harness")
    root = tmp / "harness"
    h = Harness(AGENT, root)
    for kind, e in edits:
        if kind == "INVARIANT":
            p = h.dir / "guardrails.md"
            p.write_text(p.read_text(encoding="utf-8").rstrip() + f"\n- {e.text}\n", encoding="utf-8")
        else:
            h.add_lesson(Lesson(f"les_{uuid.uuid4().hex[:8]}", AGENT, e.applies_to, e.text, "DOMAIN", "active"))
    return root


class AugmentEvolver:
    def __init__(self, llm: LLM | None = None, probes: list[dict] | None = None, workdir: str | Path = "runs/h2_augment"):
        self.llm = llm or LLM()
        self.ev = AugmentEvaluator(self.llm, probes, workdir)
        self.wd = self.ev.wd
        self.root = _root()
        self.log_path = self.root / AGENT / "generations.jsonl"

    def propose(self, ref: dict) -> Proposal:
        h = Harness(AGENT, self.root)
        fails = [f"probe {r['probe']} ({self.ev.probes[r['probe']]['id']}): {r['diag']}" for r in ref["rows"] if not r["passed"]]
        msgs = [{"role": "system", "content": "You improve the harness (instructions) of an LLM agent that derives scenario "
                 "variants from a verified TSIR by restricted JSON patches; you never see or change model weights. Edits must "
                 "be general, short and testable; never weaken the isolation check or the verifier."},
                {"role": "user", "content": "Current guardrails:\n" + h.part("guardrails.md") +
                 "\n\nFailures of the current harness on the probe set (first editor proposal):\n" + "\n".join(fails) +
                 "\n\nPropose one INVARIANT rule and one DOMAIN lesson that would have prevented the most failures."}]
        return Proposal.model_validate(self.llm.chat(msgs, schema=Proposal.model_json_schema(), role="harness_editor",
                                                      max_tokens=1500))

    def generation(self, gen: int) -> dict:
        ref = self.ev.evaluate(self.root, f"g{gen}_ref")
        if ref["valid_rate"] == 1.0:
            rec = {"gen": gen, "agent": AGENT, "score": ref["score"], "valid_rate": ref["valid_rate"], "adopted": "none (all pass)"}
            self._log(rec, ref)
            return rec
        prop = self.propose(ref)
        cands = {"invariant": [("INVARIANT", prop.invariant)], "domain": [("DOMAIN", prop.domain)],
                 "both": [("INVARIANT", prop.invariant), ("DOMAIN", prop.domain)]}
        res = {k: self.ev.evaluate(_harness_copy(self.root, v), f"g{gen}_{k}") for k, v in cands.items()}

        def regress(r):
            return any(a["passed"] and not b["passed"] for a, b in zip(ref["rows"], r["rows"]))

        best = max(res, key=lambda k: res[k]["score"])
        ok = res[best]["score"] > ref["score"] + 0.02 and not regress(res[best])
        h = Harness(AGENT, self.root)
        evidence = {"ref": ref["score"], **{k: r["score"] for k, r in res.items()}, "valid_ref": ref["valid_rate"],
                    "valid_best": res[best]["valid_rate"], "n_probes": len(self.ev.probes)}
        for kind, e in cands[best] if ok else []:
            if kind == "INVARIANT":
                p = h.dir / "guardrails.md"
                p.write_text(p.read_text(encoding="utf-8").rstrip() + f"\n- {e.text}\n", encoding="utf-8")
            else:
                h.add_lesson(Lesson(f"les_{uuid.uuid4().hex[:8]}", AGENT, e.applies_to, e.text, "DOMAIN", "active",
                                    {"delta_score": round(res[best]["score"] - ref["score"], 4), **evidence}))
        h.log_revision("COMPOSITE_MERGE" if best == "both" else f"{best.upper()}_EDITOR",
                       "guardrails.md+EXPERIENCE" if best == "both" else ("guardrails.md" if best == "invariant" else "EXPERIENCE"),
                       json.dumps(prop.model_dump()), "adopted" if ok else "rejected", evidence)
        new = res[best] if ok else ref
        rec = {"gen": gen, "agent": AGENT, "score": new["score"], "valid_rate": new["valid_rate"], "ref_score": ref["score"],
               "adopted": best if ok else "none"}
        self._log(rec, {"ref": ref, **res, "proposal": prop.model_dump()})
        return rec

    def _log(self, rec: dict, detail: dict):
        rec["t"] = time.time()
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        (self.wd / f"gen{rec['gen']}.json").write_text(json.dumps(detail, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "evaluate":
        r = AugmentEvaluator().evaluate(_root(), "eval")
        print(json.dumps({k: r[k] for k in ("score", "valid_rate")}), flush=True)
        for row in r["rows"]:
            print(row["id"], row["passed"], row["score"], row["diag"][:300], flush=True)
    else:
        ev = AugmentEvolver()
        start = 1 + (sum(1 for _ in ev.log_path.open()) if ev.log_path.exists() else 0)
        for g in range(start, start + int(sys.argv[1] if len(sys.argv) > 1 else 1)):
            print(ev.generation(g), flush=True)
