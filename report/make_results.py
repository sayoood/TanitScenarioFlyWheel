"""Aggregate flywheel KPIs on thor3 into report/results.tex (aggregate numbers only, no dataset content).

Usage (on thor3): python report/make_results.py [--runs runs/api] [--out report/results.tex]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median

HOME = Path.home()


def _runs(root: Path) -> list[dict]:
    out = []
    for p in sorted(root.glob("*/run.json")):
        try:
            out.append(json.loads(p.read_text()))
        except json.JSONDecodeError:
            continue
    return out


def _j(p: Path) -> dict | list | None:
    return json.loads(p.read_text()) if p.exists() else None


def _f(v, d=2, unit=""):
    if v is None:
        return "--"
    if isinstance(v, bool):
        return r"\checkmark" if v else r"$\times$"
    if isinstance(v, (int, float)):
        return f"{v:.{d}f}{unit}"
    return str(v).replace("_", r"\_")


def _stage(r: dict, name: str):
    return next((s.get("seconds") for s in r.get("stages", []) if s["name"] == name), None)


def table(caption: str, label: str, cols: str, head: list[str], rows: list[list], note: str = "") -> str:
    body = "\n".join(" & ".join(_f(c) if not isinstance(c, str) else c.replace("_", r"\_") for c in r) + r" \\"
                     for r in rows) or r"\multicolumn{%d}{c}{no runs yet} \\" % len(head)
    return (r"\begin{table}[h]\centering\small\caption{%s}\label{%s}" % (caption, label) +
            r"\begin{tabular}{@{}%s@{}}\toprule " % cols + " & ".join(head) + r" \\\midrule" + "\n" + body +
            r"\bottomrule\end{tabular}" + (r"\par\smallskip{\footnotesize %s}" % note if note else "") + r"\end{table}")


def uc1(runs: list[dict]) -> str:
    q = [r for r in runs if r["uc"] == "uc1" and r["input"].get("query") and r["status"] in ("ok", "failed")]
    rows = []
    for r in q:
        k = r.get("kpis") or {}
        rows.append([r["id"][4:15], k.get("first_pass_score"), k.get("final_valid"), k.get("repairs"),
                     _stage(r, "analyse"), _stage(r, "propose"), k.get("time_to_valid_s"), k.get("llm_tok_per_s"),
                     k.get("package_valid"), k.get("epdms")])
    t1 = table("UC1 live runs of the specification query (Qwen3.8-27B NVFP4 on Thor). Times in s.", "tab:uc1",
               "lrcrrrrrcr", ["run", "1st score", "valid", "rep.", "analyse", "propose", "to valid", "tok/s", "pkg",
                              "EPDMS"], rows,
               "Earlier runs exposed the decoding failure modes described above; later runs include the fixes.")
    ex = [r for r in runs if r["uc"] == "uc1" and r["input"].get("example") and r["status"] == "ok"]
    rows = []
    for r in ex:
        k, m = r.get("kpis") or {}, r.get("metrics") or {}
        a, cl = m.get("augment") or {}, m.get("closed_loop") or {}
        rows.append([r["input"]["example"].replace("_", " ")[:28], k.get("time_to_valid_s"), k.get("sim_rtf"),
                     k.get("render_s"), a.get("sims"), a.get("kept"), a.get("coverage"), cl.get("policy"),
                     cl.get("EPDMS")])
    t2 = table("Pipeline on the verified example scenarios (verify, simulate, render, augment, closed loop).",
               "tab:examples", "lrrrrrrlr", ["scenario", "verify s", "sim RTF", "render s", "sims", "kept", "cov.",
                                             "policy", "EPDMS"], rows)
    return t1 + "\n" + t2


def uc3(runs: list[dict]) -> str:
    q = [r for r in runs if r["uc"] == "uc3" and r["status"] in ("ok", "failed")]
    rows = [[r["input"]["incident_id"], (r.get("kpis") or {}).get("first_pass_score"),
             (r.get("kpis") or {}).get("final_valid"), (r.get("kpis") or {}).get("repairs"),
             (r.get("kpis") or {}).get("twin_valid"), (r.get("kpis") or {}).get("package_valid"), r.get("seconds")]
            for r in q]
    return table("UC3 counterfactual positive twins of real NEGATIVE incidents (IMDAD v2).", "tab:uc3", "lrcrccr",
                 ["incident", "1st score", "TSIR valid", "rep.", "twin v2 valid", "pkg", "total s"], rows)


def uc45(runs: list[dict], batch: dict | None) -> str:
    q = [r for r in runs if r["uc"] in ("uc4", "uc5") and r["status"] in ("ok", "failed")]
    rows = []
    for r in q:
        k = r.get("kpis") or {}
        rows.append([r["uc"].upper(), r["input"]["snippet"].split(":")[1][:8], k.get("lift_s"), k.get("lift_ego_ade_s"),
                     k.get("lift_npc_ade_s"), k.get("neg_peak_decel"), k.get("pos_peak_decel"), k.get("neg_peak_jerk"),
                     k.get("pos_peak_jerk"), k.get("package_valid"), k.get("epdms")])
    t = table("UC4/UC5 on PhysicalAI-AV snippets: lifting time and fidelity (ADE of road coordinate $s$ vs. the "
              "log, m); UC5 negative vs. positive twin peak deceleration (m/s$^2$) and jerk (m/s$^3$).", "tab:uc45",
              "llrrrrrrrcr", ["UC", "clip", "lift s", "ego ADE", "NPC ADE", "dec$^-$", "dec$^+$", "jerk$^-$",
                              "jerk$^+$", "pkg", "EPDMS"], rows)
    if batch:
        t += "\n" + table("Dataset-level lifting over all PhysicalAI-AV clips with obstacle labels (3 chunks).",
                          "tab:batch", "rrrrrrrr", ["clips", "lifted", "1st-pass", "consistent", "ego ADE",
                                                     "ego FDE", "NPC ADE", "p90 NPC"],
                          [[batch["clips"], batch["lifted"], batch["first_pass_rate"], batch["consistent_rate"],
                            batch["median_ego_ade_s"], batch["median_ego_fde_s"], batch["median_npc_ade_s"],
                            batch.get("p90_npc_ade_s")]], "Medians over clips; 15 s windows; seconds per clip "
                          f"(median) {batch.get('median_seconds')}.")
    return t


def kpis(runs: list[dict], h2: list[dict]) -> str:
    ok = [r for r in runs if r["status"] == "ok"]
    tok = [k["llm_tok_per_s"] for r in runs if (k := r.get("kpis") or {}).get("llm_tok_per_s")]
    rtf = [k["sim_rtf"] for r in ok if (k := r.get("kpis") or {}).get("sim_rtf")]
    rows = [["LLM decode throughput (tok/s, median)", median(tok) if tok else None],
            ["TanitSim real-time factor (median)", median(rtf) if rtf else None],
            ["runs completed / total", f"{len(ok)} / {len(runs)}"],
            ["Harness$^2$ generations", len(h2)]]
    if h2:
        rows += [["Harness$^2$ score first $\\to$ last", f"{h2[0]['score']:.3f} $\\to$ {h2[-1]['score']:.3f}"],
                 ["Harness$^2$ valid rate first $\\to$ last", f"{h2[0]['valid_rate']:.2f} $\\to$ {h2[-1]['valid_rate']:.2f}"]]
    return table("Speed and self-improvement KPIs.", "tab:kpi", "lr", ["KPI", "value"], rows)


def free(path: Path, caption: str, label: str) -> str:
    d = _j(path)
    if not d:
        return r"\noindent\todo{%s: results pending}" % caption
    rows = [[k.replace("_", " "), v] for k, v in d.items() if not isinstance(v, (dict, list))]
    return table(caption, label, "lr", ["metric", "value"], rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(Path(__file__).resolve().parents[1] / "runs" / "api"))
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "results.tex"))
    a = ap.parse_args()
    runs = _runs(Path(a.runs))
    batch = _j(HOME / "tsf/dev/uc45/batch/summary.json")
    h2p = HOME / "tsf/state/harness/generations.jsonl"
    h2 = [json.loads(ln) for ln in h2p.read_text().splitlines() if ln.strip()] if h2p.exists() else []
    tex = "\n".join([
        r"\newcommand{\ResultsUCone}{%s}" % uc1(runs),
        r"\newcommand{\ResultsUCthree}{%s}" % uc3(runs),
        r"\newcommand{\ResultsUCfourfive}{%s}" % uc45(runs, batch),
        r"\newcommand{\ResultsKPI}{%s}" % kpis(runs, h2),
        r"\newcommand{\ResultsCosmos}{%s}" % free(HOME / "tsf/dev/cosmos_gen/results.json", "Cosmos~3 generation on Thor",
                                                    "tab:cosmos"),
        r"\newcommand{\ResultsUCtwo}{%s}" % free(HOME / "tsf/dev/vipe/results.json", "UC2 ViPE on CC-BY dash-cam clips",
                                                   "tab:uc2"),
    ])
    Path(a.out).write_text(tex)
    print(f"wrote {a.out}: {len(runs)} runs, batch={'yes' if batch else 'no'}, h2 gens={len(h2)}")


if __name__ == "__main__":
    main()
