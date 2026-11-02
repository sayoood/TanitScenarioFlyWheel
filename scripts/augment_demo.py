"""Augmentation demo on the UC1 left-turn intersection: auto (seeds 0 and 1), manual combined and manual separate.

    python scripts/augment_demo.py run [--out runs/augment_demo] [--no-package] [--render] [--only NAME ...]
    python scripts/augment_demo.py summary [--out runs/augment_demo]

``run`` starts the CLI configurations concurrently (the LLM server batches them), waits, then writes ``summary.json``:
per variant first-pass validity, repairs, isolation, description check, verification, package validity, the lever the
editor chose, wall time per stage and LLM tokens. Raw outputs per configuration live in ``<out>/<name>/``.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

PARENT = "examples/uc1_left_turn_intersection.json"
MANUAL = {"weather": "night with heavy rain",
          "behavior": "the oncoming car runs the red light and the ego has to brake",
          "environment": "road works: cones close the right lane, a 30 km/h sign and a road-works sign before the junction"}
CONFIGS = {
    "auto_seed0": ["--mode", "auto", "--seed", "0"],
    "auto_seed1": ["--mode", "auto", "--seed", "1"],
    "manual_combined": ["--mode", "manual", "--seed", "0", *sum((["--" + k, v] for k, v in MANUAL.items()), [])],
    "manual_separate": ["--mode", "manual", "--seed", "0", "--separate", *sum((["--" + k, v] for k, v in MANUAL.items()), [])],
}
STAGES = ["plan_s", "edit_s", "verify_s", "criticality_s", "describe_s", "final_verify_s", "package_s", "render_s", "judge_s",
          "build_s", "total_s"]


def run(out: Path, package: bool, render: bool, only: list[str] | None):
    out.mkdir(parents=True, exist_ok=True)
    procs = {}
    for name, args in CONFIGS.items():
        if only and name not in only:
            continue
        cmd = [sys.executable, "-m", "tsf.agents.augment", PARENT, str(out / name), *args]
        if not package:
            cmd.append("--no-package")
        if not render:
            cmd.append("--no-render")
        log = (out / f"{name}.log").open("w")
        procs[name] = (subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), time.time())
        print("started", name, flush=True)
    for name, (p, t0) in procs.items():
        rc = p.wait()
        print(f"{name}: exit {rc} after {time.time() - t0:.0f} s", flush=True)
    summary(out)


def summary(out: Path):
    runs = {}
    for name, args in CONFIGS.items():
        f = out / name / "summary.json"
        if not f.exists():
            runs[name] = {"args": args, "error": "no summary.json (run failed or still running)"}
            continue
        s = json.loads(f.read_text())
        vs = []
        for v in s["variants"]:
            r = json.loads((out / name / v["id"] / "result.json").read_text())
            vs.append({
                "id": v["id"], "dimensions": v["dimensions"], "lever": v.get("lever"), "title": r.get("title"),
                "status": v["status"], "first_pass_valid": v["first_pass_valid"], "repairs": v["repairs"],
                "isolation_ok": v["isolation_ok"], "description_ok": v["description_ok"], "verified": v["verified"],
                "package": r.get("package"), "criticality": r.get("criticality"), "limitations": r.get("limitations"),
                "n_changes": len(r.get("diff") or []), "changed_paths": [c["path"] for c in r.get("diff") or []],
                "seconds": {k: v["timings"][k] for k in STAGES if k in v["timings"]},
                "tokens": v["tokens"], "error": r.get("error")})
        runs[name] = {"args": args, "wall_s": s["wall_s"], "parent_verified": s["parent_verified"], "variants": vs}
    allv = [v for r in runs.values() for v in r.get("variants", [])]
    n = max(len(allv), 1)
    tot = {"variants": len(allv), "verified": sum(bool(v["verified"]) for v in allv),
           "first_pass_valid": sum(bool(v["first_pass_valid"]) for v in allv),
           "isolation_ok": sum(bool(v["isolation_ok"]) for v in allv),
           "description_ok": sum(bool(v["description_ok"]) for v in allv),
           "mean_repairs": round(sum(v["repairs"] for v in allv) / n, 2),
           "llm_calls": sum(v["tokens"].get("calls", 0) for v in allv),
           "prompt_tokens": sum(v["tokens"].get("prompt_tokens", 0) for v in allv),
           "completion_tokens": sum(v["tokens"].get("completion_tokens", 0) for v in allv)}
    doc = {"parent": PARENT, "manual_requests": MANUAL, "totals": tot, "runs": runs, "t": time.time()}
    (out / "summary.json").write_text(json.dumps(doc, indent=1))
    print(json.dumps(tot), flush=True)
    for name, r in runs.items():
        for v in r.get("variants", []):
            print(f"{name:16s} {'+'.join(v['dimensions']):28s} {v['status']:7s} first={v['first_pass_valid']!s:5} rep={v['repairs']} "
                  f"iso={v['isolation_ok']} desc={v['description_ok']} ver={v['verified']} lever={v['lever']} "
                  f"build={v['seconds'].get('build_s')} tok={v['tokens'].get('prompt_tokens')}/{v['tokens'].get('completion_tokens')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["run", "summary"])
    ap.add_argument("--out", default="runs/augment_demo")
    ap.add_argument("--no-package", action="store_true")
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    if a.cmd == "run":
        run(Path(a.out), not a.no_package, a.render, a.only)
    else:
        summary(Path(a.out))
