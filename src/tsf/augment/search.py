"""Smart augmentation: coverage- and criticality-guided search over TSIR logical parameters.

Behaviour/kinematic parameters (TSIR `parameters`, JSON pointers) are explored with TanitSim as the
cheap inner loop:
  1. space-filling seed (scrambled Sobol)               -> coverage of the logical scenario space
  2. criticality refinement (cross-entropy method)        -> concentrate samples near the safety
     boundary (target band of min-TTC / required decel, no trivial collisions)
  3. novelty filter on a coverage grid                   -> keep diverse variants (LAMBDA-style cells)
Appearance parameters (weather, time of day, lighting, object colour) do not change the simulation;
they are crossed in at render time (Cosmos prompt + WSM), so every behavioural variant fans out.
Every kept variant is re-verified (L0-L3) and labelled SYNTHETIC_AUGMENTATION with its parent id.
"""
from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import qmc

from tsf.ir.scenario import Scenario
from tsf.sim.engine import TanitSim
from tsf.sim.odr import OdrMap

APPEARANCE = {"weather", "time_of_day", "road_surface", "cloud_cover", "precipitation", "fog_visibility_m"}


def _set(doc: dict, pointer: str, value):
    parts = [p.replace("~1", "/").replace("~0", "~") for p in pointer.lstrip("/").split("/")]
    cur = doc
    for p in parts[:-1]:
        cur = cur[int(p)] if isinstance(cur, list) else cur[p]
    last = parts[-1]
    if isinstance(cur, list):
        cur[int(last)] = value
    else:
        cur[last] = value


@dataclass
class Variant:
    values: dict
    tsir: dict
    metrics: dict
    criticality: float
    valid: bool
    cell: tuple = ()
    reason: str = ""


@dataclass
class SearchResult:
    parent_id: str
    params: list[str]
    variants: list[Variant] = field(default_factory=list)
    kept: list[Variant] = field(default_factory=list)
    coverage: float = 0.0
    sims: int = 0

    def summary(self) -> dict:
        crit = [v for v in self.kept if 0.0 < v.criticality < 1.0]
        return {"parent": self.parent_id, "sims": self.sims, "kept": len(self.kept),
                "coverage": round(self.coverage, 3), "near_miss": len(crit),
                "invalid": sum(not v.valid for v in self.variants)}


def criticality(metrics: dict, ttc_band=(0.8, 3.0)) -> float:
    """1 = at the boundary (near miss), 0 = benign; collisions are 'beyond boundary' (returned as -1)."""
    if metrics["collision"]:
        return -1.0
    ttc = metrics["min_ttc"]
    c_ttc = 0.0 if ttc is None else float(np.clip((ttc_band[1] - ttc) / (ttc_band[1] - ttc_band[0]), 0, 1))
    c_dec = float(np.clip((metrics["ego_max_decel"] - 2.0) / 4.0, 0, 1))
    return max(c_ttc, c_dec)


class Augmenter:
    def __init__(self, sc: Scenario, m: OdrMap, grid: int = 8, seed: int = 0):
        self.sc, self.m, self.grid = sc, m, grid
        self.base = json.loads(sc.model_dump_json())
        self.params = [p for p in sc.parameters if p.name not in APPEARANCE and p.distribution == "uniform"
                       and p.path.split("/")[-1] not in APPEARANCE]
        self.rng = np.random.default_rng(seed)

    def _eval(self, u: np.ndarray, keep_assertions: set[str]) -> Variant:
        doc = copy.deepcopy(self.base)
        vals = {}
        for p, ui in zip(self.params, u):
            v = float(p.low + ui * (p.high - p.low))
            _set(doc, p.path, round(v, 3))
            vals[p.name] = round(v, 3)
        doc["meta"]["parent_id"] = self.sc.meta.id
        doc["meta"]["id"] = f"{self.sc.meta.id}_aug_{abs(hash(tuple(vals.values()))) % 10**8:08d}"
        doc["meta"]["tags"] = list(dict.fromkeys(doc["meta"]["tags"] + ["SYNTHETIC_AUGMENTATION"]))
        try:
            sc = Scenario.model_validate(doc)
            log = TanitSim(sc, self.m).run()
        except Exception as e:  # invalid combination of parameters
            return Variant(vals, doc, {}, 0.0, False, reason=f"{type(e).__name__}: {e}")
        from tsf.verify.ladder import l3_assertions
        intent = [f for f in l3_assertions(sc, log) if f.check.split(":")[0] in keep_assertions]
        ok = all(f.ok for f in intent)
        cell = tuple(int(min(self.grid - 1, ui * self.grid)) for ui in u)
        return Variant(vals, doc, log.metrics, criticality(log.metrics), ok, cell,
                       "" if ok else "; ".join(f.message for f in intent if not f.ok))

    def run(self, n_seed: int = 32, n_iter: int = 4, n_per_iter: int = 16, elite: float = 0.25) -> SearchResult:
        d = len(self.params)
        res = SearchResult(self.sc.meta.id, [p.name for p in self.params])
        if d == 0:
            return res
        # intent assertions that define "the same scenario" (safety thresholds are allowed to move)
        keep = {"ego_turn", "ego_lane_change", "red_light_respected", "ego_reaches_s"}
        U = qmc.Sobol(d, scramble=True, seed=int(self.rng.integers(1 << 31))).random(n_seed)
        evals = [self._eval(u, keep) for u in U]
        mu, sigma = np.full(d, 0.5), np.full(d, 0.3)
        for _ in range(n_iter):  # cross-entropy method toward the criticality boundary
            scored = sorted((v for v in evals if v.valid and v.criticality >= 0), key=lambda v: -v.criticality)
            top = scored[:max(2, int(len(scored) * elite))]
            if top:
                X = np.array([[(v.values[p.name] - p.low) / (p.high - p.low) for p in self.params] for v in top])
                mu, sigma = X.mean(0), np.maximum(X.std(0), 0.05)
            U = np.clip(self.rng.normal(mu, sigma, (n_per_iter, d)), 0, 1)
            evals += [self._eval(u, keep) for u in U]
        res.variants, res.sims = evals, len(evals)
        cells = {}
        for v in sorted((v for v in evals if v.valid and v.criticality >= 0), key=lambda v: -v.criticality):
            cells.setdefault(v.cell, v)       # novelty: best variant per coverage cell
        res.kept = list(cells.values())
        res.coverage = len({v.cell for v in evals if v.valid}) / float(self.grid ** d)
        return res


def plot_coverage(res: SearchResult, aug: Augmenter, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    p = aug.params
    if len(p) < 2:
        return None
    fig, ax = plt.subplots(figsize=(7.5, 6), dpi=140)
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#11151b")
    for v in res.variants:
        x, y = v.values[p[0].name], v.values[p[1].name]
        if not v.valid:
            ax.scatter(x, y, marker="x", c="#6b7280", s=22)
        elif v.criticality < 0:
            ax.scatter(x, y, marker="X", c="#ff4d5e", s=40)
        else:
            ax.scatter(x, y, c=[[0.18 + 0.82 * v.criticality, 0.9 - 0.6 * v.criticality, 0.84 - 0.7 * v.criticality]],
                       s=26 + 50 * v.criticality, edgecolors="none")
    for v in res.kept:
        ax.scatter(v.values[p[0].name], v.values[p[1].name], facecolors="none", edgecolors="#2ee6d6", s=110, lw=1.2)
    ax.set_xlabel(p[0].name, color="#e8edf3")
    ax.set_ylabel(p[1].name, color="#e8edf3")
    ax.tick_params(colors="#8b96a5")
    for s in ax.spines.values():
        s.set_color("#2a3240")
    ax.set_title(f"Criticality-guided augmentation · {res.sims} sims · {len(res.kept)} kept · coverage "
                 f"{res.coverage:.0%}", color="#e8edf3", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_png
