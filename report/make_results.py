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


def uc45(runs: list[dict], batch: dict | None, batch_v1: dict | None = None) -> str:
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
        vers = ([("before fixes", batch_v1)] if batch_v1 else []) + [("after fixes" if batch_v1 else "current", batch)]
        t += "\n" + table("Dataset-level lifting over all PhysicalAI-AV clips with obstacle labels (3 chunks). "
                          "``after fixes'': lane-consistent speeds, free-road speed control at the commanded rate, "
                          "lateral offsets of queued oncoming vehicles.",
                          "tab:batch", "lrrrrrrrr", ["version", "clips", "lifted", "1st-pass", "consistent", "ego ADE",
                                                      "ego FDE", "NPC ADE", "p90 NPC"],
                          [[v, b["clips"], b["lifted"], b["first_pass_rate"], b["consistent_rate"],
                            b["median_ego_ade_s"], b["median_ego_fde_s"], b["median_npc_ade_s"],
                            b.get("p90_npc_ade_s")] for v, b in vers],
                          "Medians over clips (m); 15 s windows; seconds per clip (median) "
                          f"{batch.get('median_seconds')}.")
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


COND = {"A": "A one-shot", "B": "B one-shot + harness", "C": "C agentic", "D": "D agentic + repair",
        "E0": "E$_0$ agentic, H$^2$", "E": "E agentic + repair, H$^2$"}


def ablation(s: dict | None, fig: Path) -> str:
    if not s:
        return r"\noindent\todo{ablation pending}"
    rows = []
    for c, name in COND.items():
        a, p, h, w = (s.get(f"{c}/{k}") for k in ("all", "probe", "held_out", "vs_A"))
        if not a:
            continue
        ci = a["correct_ci"]
        rows.append([name, f"{a['valid_rate']:.2f}", f"{a['correct_rate']:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]",
                     f"{p['correct_rate']:.2f}" if p else "--", f"{h['correct_rate']:.2f}" if h else "--",
                     f"{a['mean_intent']:.2f}", f"{a['mean_llm_calls']:.1f}",
                     str(a["mean_completion_tokens"] or "--"), f"{a['median_s']:.0f}",
                     f"{w['wins']}/{w['losses']}" if w else "--"])
    _ablation_fig(s, fig)
    n = s.get("A/all", {}).get("n", 0)
    return table(f"Ablation over 12 queries $\\times$ 2 repetitions (n={n} per condition). correct = L3-valid and all "
                 "intent checks pass (95\\,\\% Wilson CI); intent = mean fraction of intent checks passed; tokens = "
                 "completion tokens per scenario (C/E$_0$: first attempt incl.\\ analysis, counted in D/E).",
                 "tab:ablation", "lrrrrrrrrr", ["condition", "valid", "correct [CI]", "probe", "held-out", "intent",
                                                "calls", "tokens", "s", "W/L vs A"], rows)


def _ablation_fig(s: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, muted, grid = "#0b0b0b", "#898781", "#e1e0d9"
    conds = [c for c in COND if f"{c}/all" in s]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.2, 3.0), gridspec_kw={"width_ratios": [2.2, 1]})
    w = 0.38
    for j, (split, col, lab) in enumerate([("probe", "#2a78d6", "probe (seen by Harness$^2$)"),
                                           ("held_out", "#eb6834", "held-out")]):
        xs, ys, lo, hi = [], [], [], []
        for i, c in enumerate(conds):
            d = s.get(f"{c}/{split}")
            if not d:
                continue
            xs.append(i + (j - 0.5) * (w + 0.02))
            ys.append(d["correct_rate"])
            lo.append(d["correct_rate"] - d["correct_ci"][0])
            hi.append(d["correct_ci"][1] - d["correct_rate"])
        ax.bar(xs, ys, w, color=col, label=lab, zorder=2)
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt="none", ecolor=ink, elinewidth=1, capsize=2.5, zorder=3)
    ax.set_xticks(range(len(conds)), [c.replace("E0", "E$_0$") for c in conds])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("correct rate", color=ink)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    tok = [s[f"{c}/all"].get("mean_completion_tokens") or 0 for c in conds]
    bx.bar(range(len(conds)), tok, 0.6, color=muted, zorder=2)
    bx.set_xticks(range(len(conds)), [c.replace("E0", "E$_0$") for c in conds])
    bx.set_ylabel("completion tokens / scenario", color=ink)
    for a_ in (ax, bx):
        a_.grid(axis="y", color=grid, linewidth=0.6, zorder=0)
        for sp in ("top", "right"):
            a_.spines[sp].set_visible(False)
        a_.spines["left"].set_color(muted)
        a_.spines["bottom"].set_color(muted)
        a_.tick_params(colors=ink, labelsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


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
    batch_v1 = _j(HOME / "tsf/dev/uc45/batch/summary_v1_prefix.json")
    h2p = HOME / "tsf/state/harness/generations.jsonl"
    h2 = [json.loads(ln) for ln in h2p.read_text().splitlines() if ln.strip()] if h2p.exists() else []
    tex = "\n".join([
        r"\newcommand{\ResultsUCone}{%s}" % uc1(runs),
        r"\newcommand{\ResultsUCthree}{%s}" % uc3(runs),
        r"\newcommand{\ResultsUCfourfive}{%s}" % uc45(runs, batch, batch_v1),
        r"\newcommand{\ResultsKPI}{%s}" % kpis(runs, h2),
        r"\newcommand{\ResultsCosmos}{%s}" % free(HOME / "tsf/dev/cosmos_gen/results.json", "Cosmos~3 generation on Thor",
                                                    "tab:cosmos"),
        r"\newcommand{\ResultsUCtwo}{%s}" % free(HOME / "tsf/dev/vipe/results.json", "UC2 ViPE on CC-BY dash-cam clips",
                                                   "tab:uc2"),
        r"\newcommand{\ResultsAblation}{%s}" % ablation(_j(HOME / "tsf/dev/ablation/summary.json"),
                                                        Path(a.out).parent / "figs" / "ablation.png"),
    ])
    Path(a.out).write_text(tex)
    print(f"wrote {a.out}: {len(runs)} runs, batch={'yes' if batch else 'no'}, h2 gens={len(h2)}")


if __name__ == "__main__":
    main()
