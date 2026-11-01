"""Ablation: iterative Qwen3.8 agentic lifting + Harness² vs. one-shot, non-improving baselines.

Same model, same grammar-constrained decoding, same verifier (L0-L3, ASAM QC off for all) in every condition:
  A  one_shot        schema-prompted one call: system prompt + full TSIR JSON schema + request (no harness,
                     skills, analysis or repair)
  B  one_shot_h0     seed-harness system prompt + relevance-ranked skills, one call
  C  agentic_h0      analyse -> propose with the seed harness (first attempt of D)
  D  agentic_rep_h0  C + verifier-guided minimal-edit repair (<= max_repairs)
  E0 agentic_h2      analyse -> propose with the Harness²-evolved harness (first attempt of E)
  E  agentic_rep_h2  E0 + repair
Validity alone rewards trivial scenarios, so every query also carries deterministic intent checks on the TSIR and
on the simulated trajectory (road type, lanes, actors, environment, ego manoeuvres actually executed, no
collision). correct = L3 valid and all intent checks pass. Probe queries were seen by the evolver; held-out
queries were not.

Usage (thor3): python -m tsf.agents.ablation [--conds A,B,D,E] [--reps 2] [--workers 3]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tsf.agents.evolve import PROBES
from tsf.agents.harness import Harness
from tsf.agents.lifter import Lifter, _slug
from tsf.agents.llm import LLM
from tsf.verify import ladder

SEED = Path(__file__).resolve().parents[3] / "harness"
EVOLVED = Path(os.environ.get("TSF_HARNESS", Path.home() / "tsf/state/harness"))
OUT = Path.home() / "tsf/dev/ablation"


# ----------------------------------------------------------------------------- intent checks
def _ego(t):
    return next((a for a in t.get("actors", []) if a.get("role") == "ego"), {})


def _acts(t, ego: bool):
    e = _ego(t).get("id")
    return [a for a in t.get("actions", []) if (a.get("actor") == e) == ego]


def _hour(t):
    try:
        return int(str((t.get("environment") or {}).get("time_of_day", "12:00"))[:2])
    except ValueError:
        return 12


def _lanes(t):
    r = t.get("road") or {}
    if r.get("type") == "linear":
        return (r.get("forward") or {}).get("driving_lanes", 2)
    return max([(a.get("inbound") or {}).get("driving_lanes", 2) for a in r.get("approaches", [])] or [0])


def road(kind):
    return f"road {kind}", lambda t, m: (t.get("road") or {}).get("type") == kind


def signalized(v=True):
    return f"signalized={v}", lambda t, m: bool((t.get("road") or {}).get("signalized", True)) == v


def lanes_min(n):
    return f"lanes>={n}", lambda t, m: _lanes(t) >= n


def daytime(kind):
    rng = {"day": lambda h: 7 <= h <= 17, "night": lambda h: h >= 20 or h <= 5, "dusk": lambda h: 17 <= h <= 21}
    return f"time {kind}", lambda t, m: rng[kind](_hour(t))


def weather(*w):
    return f"weather {'/'.join(w)}", lambda t, m: (t.get("environment") or {}).get("weather", "clear") in w


def npc(cats=None, n=1):
    def f(t, m):
        return sum(1 for a in t.get("actors", []) if a.get("role") != "ego"
                   and (cats is None or a.get("category") in cats)) >= n
    return f"npc {'/'.join(cats) if cats else 'any'} x{n}", f


def npc_action(*types):
    return f"npc action {'/'.join(types)}", lambda t, m: any(a.get("type") in types for a in _acts(t, False))


def ego_action(*types):
    return f"ego action {'/'.join(types)}", lambda t, m: any(a.get("type") in types for a in _acts(t, True))


def ego_speed(kph, tol=0.3):
    return f"ego v0~{kph}km/h", lambda t, m: abs(_ego(t).get("initial_speed", 0) * 3.6 - kph) <= tol * kph


def static_npc(cats):
    return f"static {'/'.join(cats)}", lambda t, m: any(
        a.get("role") != "ego" and a.get("category") in cats and a.get("initial_speed", 1) < 0.5
        for a in t.get("actors", []))


def sim_lc(direction):
    return f"sim ego lane change {direction}", lambda t, m: m is not None and (
        m.get("ego_lane_changes_left", 0) >= 1 if direction == "left" else m.get("ego_lane_changes_left", 0) <= -1)


def sim_lc_both():
    def f(t, m):
        dirs = [a.get("direction") for a in _acts(t, True) if a.get("type") == "lane_change"]
        return m is not None and "left" in dirs and "right" in dirs
    return "ego overtakes (left+right)", f


def sim_turn(direction):
    return f"sim ego turn {direction}", lambda t, m: m is not None and direction in (m.get("ego_turns") or [])


def sim_no_collision():
    return "sim no collision", lambda t, m: m is not None and m.get("collision") is False


def sim_stops():
    return "sim ego stops", lambda t, m: m is not None and m.get("_ego_min_v", 99) < 0.5


QUERIES = [  # (query, split, checks)
    (PROBES[0], "probe", [road("junction"), signalized(), daytime("day"), weather("clear", "cloudy"), npc(n=2),
                          sim_lc("left"), sim_turn("left"), sim_no_collision()]),
    (PROBES[1], "probe", [road("linear"), npc(n=1), ego_speed(80), npc_action("brake", "speed_change"),
                          sim_stops(), sim_no_collision()]),
    (PROBES[2], "probe", [road("junction"), signalized(), npc(n=1), sim_turn("left"), sim_no_collision()]),
    (PROBES[3], "probe", [road("linear"), lanes_min(3), npc(["truck"]), npc_action("cut_in", "lane_change"),
                          ego_speed(100), sim_no_collision()]),
    (PROBES[4], "probe", [road("junction"), signalized(), npc(n=2), sim_stops(), sim_turn("straight"),
                          sim_no_collision()]),
    (PROBES[5], "probe", [road("junction"), signalized(False), npc(["pedestrian"]), npc_action("cross"),
                          ego_speed(40), sim_no_collision()]),
    ("On a two-lane urban road at night in the rain, the ego overtakes a cyclist riding ahead in its lane: it "
     "changes to the left lane, passes the cyclist and changes back to the right lane.", "held_out",
     [road("linear"), daytime("night"), weather("rain", "heavy_rain"), npc(["bicycle"]), sim_lc_both(),
      sim_no_collision()]),
    ("The ego drives at 120 km/h on a three-lane motorway; a car in the left lane slows down and changes into the "
     "ego lane 25 m ahead; the ego brakes without collision.", "held_out",
     [road("linear"), lanes_min(3), ego_speed(120), npc_action("cut_in", "lane_change"), sim_no_collision()]),
    ("At dusk the ego turns right at a signalized intersection while a pedestrian crosses the road it turns into; "
     "the ego waits for the pedestrian and then completes the turn.", "held_out",
     [road("junction"), signalized(), daytime("dusk"), npc(["pedestrian"]), npc_action("cross"),
      sim_turn("right"), sim_no_collision()]),
    ("Stop-and-go traffic on a two-lane highway: the lead vehicle alternates between 30 km/h and 5 km/h and the "
     "ego follows it with a 2 s time gap.", "held_out",
     [road("linear"), npc(n=1), ego_action("follow"), npc_action("speed_change"), sim_no_collision()]),
    ("At an unsignalized four-way intersection the ego goes straight while a car approaching from the right "
     "enters the intersection first; the ego yields to it.", "held_out",
     [road("junction"), signalized(False), npc(n=1), sim_turn("straight"), sim_no_collision()]),
    ("In dense fog a broken-down truck stands still in the right lane of a two-lane road; the ego, in the right "
     "lane at 70 km/h, changes to the left lane to pass it.", "held_out",
     [road("linear"), weather("fog"), static_npc(["truck"]), ego_speed(70), sim_lc("left"), sim_no_collision()]),
]


def _metrics(rep) -> dict | None:
    m = None
    if getattr(rep, "metrics", None):
        m = dict(rep.metrics)
        if hasattr(rep, "_log"):
            e = rep._log.roles.index("ego")
            m["_ego_min_v"] = float(rep._log.states[:, e, 3].min())
    return m


def score_intent(checks, tsir: dict, m: dict | None) -> tuple[float, list[str]]:
    failed = []
    for name, fn in checks:
        try:
            ok = bool(fn(tsir, m))
        except Exception:  # noqa: BLE001 - a malformed TSIR fails the check
            ok = False
        if not ok:
            failed.append(name)
    return round(1 - len(failed) / len(checks), 3), failed


# ----------------------------------------------------------------------------- conditions
BARE = ("You are an expert in automated-driving test scenario design. Write the requested scenario as a TSIR JSON "
        "document (the JSON schema is enforced by the decoder).")


def _row(cond, qi, rep_i, att, checks, t_s, st):
    m = _metrics(att.report)
    cov, failed = score_intent(checks, att.tsir or {}, m)
    return {"cond": cond, "q": qi, "split": QUERIES[qi][1], "rep": rep_i, "valid": att.report.passed,
            "score": att.report.score(), "intent": cov, "intent_failed": failed,
            "correct": bool(att.report.passed and cov == 1.0), "s": round(t_s, 1),
            "llm_calls": st["calls"], "completion_tokens": st["completion_tokens"],
            "prompt_tokens": st["prompt_tokens"], "first_failure": (att.report.diagnostics() or "")[:240]}


def run_job(cond: str, qi: int, rep_i: int) -> list[dict]:
    q, _, checks = QUERIES[qi]
    llm = LLM()
    root = EVOLVED if cond == "E" else SEED
    lif = Lifter(llm, Harness("lifter", root))
    k = lif.h.knobs()
    sid = _slug(q, 40)
    wd = OUT / "work" / f"{cond}_q{qi}_r{rep_i}"
    t0 = time.perf_counter()
    if cond in ("A", "B"):
        if cond == "A":
            msgs = [{"role": "system", "content": BARE + "\n\nTSIR JSON schema:\n" +
                     json.dumps(lif._tsir_schema, separators=(",", ":"))},
                    {"role": "user", "content": f"Use meta.id = \"{sid}\".\nRequest:\n{q}"}]
        else:
            msgs = [{"role": "system", "content": lif.h.system(q)},
                    {"role": "user", "content": f"{lif._skills_block(q)}\n\nNow write the TSIR for this request. Use "
                                                f"meta.id = \"{sid}\".\nRequest:\n{q}"}]
        att = lif.iterate(msgs, sid, wd, max_repairs=0, qc=False, role=f"oneshot_{cond}")[0]
        return [_row(cond, qi, rep_i, att, checks, time.perf_counter() - t0, llm.stats())]
    analysis = lif.analyse(q)
    t_an = time.perf_counter() - t0
    st_an = llm.stats()
    atts = lif.iterate(lif.propose(q, analysis, sid), sid, wd, k["max_repairs"], qc=False)
    first = {"calls": st_an["calls"] + 1, "completion_tokens": None, "prompt_tokens": None}
    final = next((a for a in reversed(atts) if a.report.passed), atts[-1])
    c0 = "C" if cond == "D" else "E0"
    return [_row(c0, qi, rep_i, atts[0], checks, t_an + atts[0].seconds, {**llm.stats(), **first}),
            {**_row(cond, qi, rep_i, final, checks, time.perf_counter() - t0, llm.stats()), "repairs": len(atts) - 1}]


# ----------------------------------------------------------------------------- summary
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return round(max(0.0, c - h), 3), round(min(1.0, c + h), 3)


def summarize(rows: list[dict]) -> dict:
    out = {}
    for cond in ("A", "B", "C", "D", "E0", "E"):
        for split in ("all", "probe", "held_out"):
            r = [x for x in rows if x["cond"] == cond and (split == "all" or x["split"] == split)]
            if not r:
                continue
            n, nv, nc = len(r), sum(x["valid"] for x in r), sum(x["correct"] for x in r)
            tok = [x["completion_tokens"] for x in r if x.get("completion_tokens") is not None]
            out[f"{cond}/{split}"] = {
                "n": n, "valid_rate": round(nv / n, 3), "valid_ci": wilson(nv, n),
                "correct_rate": round(nc / n, 3), "correct_ci": wilson(nc, n),
                "mean_score": round(sum(x["score"] for x in r) / n, 3),
                "mean_intent": round(sum(x["intent"] for x in r) / n, 3),
                "median_s": sorted(x["s"] for x in r)[n // 2],
                "mean_llm_calls": round(sum(x["llm_calls"] for x in r) / n, 2),
                "mean_completion_tokens": round(sum(tok) / len(tok)) if tok else None,
                "mean_repairs": round(sum(x.get("repairs", 0) for x in r) / n, 2)}
    # paired wins/losses vs. the one-shot baseline (same query and repetition)
    base = {(x["q"], x["rep"]): x["correct"] for x in rows if x["cond"] == "A"}
    for cond in ("B", "C", "D", "E0", "E"):
        pr = [(base[(x["q"], x["rep"])], x["correct"]) for x in rows if x["cond"] == cond and (x["q"], x["rep"]) in base]
        if pr:
            out[f"{cond}/vs_A"] = {"wins": sum(1 for a, b in pr if b and not a), "losses": sum(1 for a, b in pr if a and not b),
                                   "pairs": len(pr)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conds", default="A,B,D,E")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--queries", default="")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    res_p = OUT / "rows.jsonl"
    rows = [json.loads(ln) for ln in res_p.read_text().splitlines() if ln.strip()] if res_p.exists() else []
    done = {(x["cond"], x["q"], x["rep"]) for x in rows}
    qs = [int(x) for x in a.queries.split(",")] if a.queries else range(len(QUERIES))
    jobs = [(c, qi, r) for r in range(a.reps) for qi in qs for c in a.conds.split(",") if (c, qi, r) not in done]
    lock = threading.Lock()

    def work(job):
        try:
            new = run_job(*job)
        except Exception as e:  # noqa: BLE001 - a crashed job counts as invalid / incorrect
            new = [{"cond": job[0], "q": job[1], "split": QUERIES[job[1]][1], "rep": job[2], "valid": False,
                    "score": 0.0, "intent": 0.0, "intent_failed": ["crash"], "correct": False, "s": 0.0,
                    "llm_calls": 0, "completion_tokens": None, "prompt_tokens": None,
                    "first_failure": f"{type(e).__name__}: {e}"[:240]}]
        with lock:
            with res_p.open("a") as f:
                for x in new:
                    f.write(json.dumps(x) + "\n")
            rows.extend(new)
            print(job, [(x["cond"], x["valid"], x["intent"], x["s"]) for x in new], flush=True)

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(work, jobs))
    summ = summarize(rows)
    (OUT / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
