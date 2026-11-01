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
            out.append({**json.loads(p.read_text()), "_dir": str(p.parent)})
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
    q = [r for r in _latest(runs, ("uc1",)) if r["input"].get("query")]
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
    q = _latest(runs, ("uc3",))
    rows = [[r["input"]["incident_id"], (r.get("kpis") or {}).get("first_pass_score"),
             (r.get("kpis") or {}).get("final_valid"), (r.get("kpis") or {}).get("repairs"),
             (r.get("kpis") or {}).get("twin_valid"), (r.get("kpis") or {}).get("package_valid"), r.get("seconds")]
            for r in q]
    return table("UC3 counterfactual positive twins of real NEGATIVE incidents (IMDAD v2).", "tab:uc3", "lrcrccr",
                 ["incident", "1st score", "TSIR valid", "rep.", "twin v2 valid", "pkg", "total s"], rows)


def _latest(runs: list[dict], ucs: tuple[str, ...]) -> list[dict]:
    """Newest finished run per (use case, input); run dirs sort by creation time, so later ones win."""
    out: dict[tuple, dict] = {}
    for r in runs:
        if r["uc"] in ucs and r["status"] in ("ok", "failed"):
            out[(r["uc"], json.dumps(r["input"], sort_keys=True))] = r
    return list(out.values())


def uc2(runs: list[dict]) -> str:
    rows = []
    for r in _latest(runs, ("uc2",)):
        k, e = r.get("kpis") or {}, (r.get("uc2") or {}).get("ego") or {}
        st = sorted((Path(r["_dir"]) / "uc2" / "vipe").glob("w*/stats.json"))
        h = [(json.loads(p.read_text()).get("camera_height") or {}).get("median_m") for p in st]
        h = [x for x in h if x]
        rows.append([r["input"]["video"].split(":")[1], e.get("duration_s"), sum(h) / len(h) if h else None,
                     e.get("distance_m"), max(e.get("speed_kmh_per_s") or [0]), k.get("uc2_speed_mae_kmh"),
                     k.get("final_valid"), k.get("repairs"), k.get("package_valid"), k.get("epdms")])
    return table("UC2 on three CC-BY dash-cam clips without ego-motion data: ViPE ego track (15 fps, 480p, 20 s "
                 "windows), validity of the lifted scenario and how closely the simulated ego follows the measured "
                 "speed.", "tab:uc2", "lrrrrrcccr",
                 ["clip", "s", "cam.~h m", "dist. m", "peak km/h", "speed MAE km/h", "valid", "rep.", "pkg", "EPDMS"],
                 rows, r"Clips by YouTube user a62dave under the Creative Commons Attribution licence: "
                       r"youtu.be/amuldGGgNc4, youtu.be/dToIIqqvSt0, youtu.be/qdAI\_quNcqw. Video, frames and every "
                       r"derivative stay on the Thor; only these aggregate numbers are published.")


def uc45(runs: list[dict], batches: list[tuple[str, dict]]) -> str:
    q = _latest(runs, ("uc4", "uc5"))
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
    if batches:
        batch = batches[-1][1]
        t += "\n" + table("Dataset-level lifting over all PhysicalAI-AV clips with obstacle labels. v1: first lifter; "
                          "v3: lane-consistent speeds, free-road speed control at the commanded rate, track stitching, "
                          "lateral offsets of queued oncoming vehicles; v4: the 6-lane window is trimmed on the side "
                          "with fewer vehicles, and vehicles outside it are dropped instead of clamped into the edge lane.",
                          "tab:batch", "lrrrrrrrr", ["version", "clips", "lifted", "1st-pass", "consistent", "ego ADE",
                                                      "ego FDE", "NPC ADE", "p90 NPC"],
                          [[v, b["clips"], b["lifted"], b["first_pass_rate"], b["consistent_rate"],
                            b["median_ego_ade_s"], b["median_ego_fde_s"], b["median_npc_ade_s"],
                            b.get("p90_npc_ade_s")] for v, b in batches],
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


COND = {"A": "A one-shot (schema)", "G0": "A$'$ one-shot (replicate)", "G": "G one-shot + repair",
        "B": "B one-shot + harness", "C": "C agentic", "D": "D agentic + repair", "F0": "F$_0$ agentic + schema",
        "F": "F agentic + schema + repair", "E0": "E$_0$ agentic, H$^2$", "E": "E agentic + repair, H$^2$"}


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
    ax.set_xticks(range(len(conds)), [c.replace("E0", "E$_0$").replace("F0", "F$_0$").replace("G0", "A$'$") for c in conds])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("correct rate", color=ink)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    tok = [s[f"{c}/all"].get("mean_completion_tokens") or 0 for c in conds]
    bx.bar(range(len(conds)), tok, 0.6, color=muted, zorder=2)
    bx.set_xticks(range(len(conds)), [c.replace("E0", "E$_0$").replace("F0", "F$_0$").replace("G0", "A$'$") for c in conds])
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


def tone(root: Path) -> str:
    rows = []
    for p in sorted(root.glob("*/t1_summary.json")):
        s = json.loads(p.read_text())
        sc, tm = s["score"], s.get("timing") or [{}]
        cos = [x["cosmos_s"] for x in tm if "cosmos_s" in x]
        rows.append([s["scenario"].replace("_", " ")[:26], s["seconds"], s["chunks"], sc["EPDMS"], sc["NC"], sc["DAC"],
                     sc["TLC"], sc["TTC"], sc["EP"], sc["C"], round(sum(cos) / len(cos), 1) if cos else None,
                     round(s["wall_s"] / 60, 1)])
    if not rows:
        return r"\noindent\todo{T1 closed loop pending}"
    return table("T1 neural closed loop: REFe SFT-4 driving on Cosmos3-Nano-Sim-Auto frames (5 cameras, 480p).",
                 "tab:tone", "lrrrrrrrrrrr", ["scenario", "s", "chunks", "EPDMS", "NC", "DAC", "TLC", "TTC", "EP", "C",
                                              "Cosmos s/chunk", "wall min"], rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(Path(__file__).resolve().parents[1] / "runs" / "api"))
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "results.tex"))
    a = ap.parse_args()
    runs = _runs(Path(a.runs))
    bdir = HOME / "tsf/dev/uc45/batch"
    batches = [(v, b) for v, b in (("v1", _j(bdir / "summary_v1_prefix.json")), ("v3", _j(bdir / "summary_v3.json")),
                                   ("v4", _j(bdir / "summary.json"))) if b]
    h2p = HOME / "tsf/state/harness/generations.jsonl"
    h2 = [json.loads(ln) for ln in h2p.read_text().splitlines() if ln.strip()] if h2p.exists() else []
    tex = "\n".join([
        r"\newcommand{\ResultsUCone}{%s}" % uc1(runs),
        r"\newcommand{\ResultsUCthree}{%s}" % uc3(runs),
        r"\newcommand{\ResultsUCfourfive}{%s}" % uc45(runs, batches),
        r"\newcommand{\ResultsKPI}{%s}" % kpis(runs, h2),
        r"\newcommand{\ResultsCosmos}{%s}" % free(HOME / "tsf/dev/cosmos_gen/results.json", "Cosmos~3 generation on Thor",
                                                    "tab:cosmos"),
        r"\newcommand{\ResultsUCtwo}{%s}" % uc2(runs),
        r"\newcommand{\ResultsAblation}{%s}" % ablation(_j(HOME / "tsf/dev/ablation/summary.json"),
                                                        Path(a.out).parent / "figs" / "ablation.png"),
        r"\newcommand{\ResultsRefe}{%s}" % free(HOME / "tsf/dev/refe/results.json", "REFe SFT-4 on Thor: latency and "
                                                "open-loop sanity on a PhysicalAI-AV clip (ADE/FDE in m)", "tab:refe"),
        r"\newcommand{\ResultsTone}{%s}" % tone(Path(a.runs).parent / "t1"),
    ])
    Path(a.out).write_text(tex)
    print(f"wrote {a.out}: {len(runs)} runs, batches={[v for v, _ in batches]}, h2 gens={len(h2)}")


if __name__ == "__main__":
    main()
