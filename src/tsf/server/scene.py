"""SimLog -> `Scene` JSON for the interactive viewer (docs/API.md)."""
from __future__ import annotations

import numpy as np

from tsf.sim.engine import SimLog, TanitSim
from tsf.sim.render2d import VEH_COLORS


def _pts(a: np.ndarray, step: int = 2) -> list[list[float]]:
    a = np.asarray(a)
    idx = list(range(0, len(a), step))
    if idx[-1] != len(a) - 1:
        idx.append(len(a) - 1)
    return np.round(a[idx, :2], 2).tolist()


def scene(sim: TanitSim, log: SimLog, hz: float = 10.0) -> dict:
    m = sim.m
    dt = float(log.t[1] - log.t[0]) if len(log.t) > 1 else 0.1
    k = max(1, int(round(1.0 / (dt * hz))))
    ticks = list(range(0, len(log.t), k))
    lanes, stop_lines = [], []
    for key, ln in m.lanes.items():
        lanes.append({"id": "/".join(map(str, key)) if isinstance(key, tuple) else str(key),
                      "junction": ln.junction != "-1", "left": _pts(ln.left_border), "right": _pts(ln.right_border),
                      "center": _pts(ln.xy)})
        if ln.junction == "-1" and ln.successors and m.lanes[ln.successors[0]].junction != "-1":
            stop_lines.append(np.round([ln.left_border[-1], ln.right_border[-1]], 2).tolist())
    signals = []
    for arm, states in log.signals.items():
        for sg in m.signals():
            if sg.road == str(arm):
                signals.append({"arm": int(arm), "x": round(sg.x, 2), "y": round(sg.y, 2),
                                "states": [states[i] for i in ticks]})
    colors = {a.id: a.color for a in sim.sc.actors}
    actors = [{"id": aid, "role": "ego" if role == "ego" else "npc", "category": cat,
               "lwh": [round(float(v), 2) for v in lwh],
               "color": VEH_COLORS.get(colors.get(aid, "silver"), colors.get(aid, "#b9c0ca"))}
              for aid, role, cat, lwh in zip(log.ids, log.roles, log.categories, log.lwh)]
    S = log.states[ticks]
    tracks = {aid: np.round(S[:, j, :4], 3).tolist() for j, aid in enumerate(log.ids)}
    return {"bounds": [round(v, 2) for v in m.bounds()], "lanes": lanes, "stop_lines": stop_lines,
            "signals": signals, "actors": actors, "t": np.round(log.t[ticks], 3).tolist(), "tracks": tracks,
            "events": log.events}
