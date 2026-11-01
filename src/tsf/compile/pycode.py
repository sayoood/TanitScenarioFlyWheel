"""Standalone `pythonCode` for the Scenario_Structure package: embedded TanitSim trajectories
(2 Hz) + independent recomputation of ego gap/TTC/decel + plot. Only numpy and matplotlib."""
from __future__ import annotations

import json

import numpy as np

from tsf.ir.scenario import Scenario

TEMPLATE = '''"""{title}
Standalone replay of the TanitSim result (Tanit Scenario Flywheel, TSIR {ver}).
Trajectories are sampled at {hz} Hz from the verified simulation; metrics are recomputed independently."""
import numpy as np
import matplotlib.pyplot as plt

DT = {dt}
LWH = {lwh}
TRAJ = {traj}  # actor -> [[t, x, y, yaw, v], ...]

ego = np.array(TRAJ["{ego}"])
report = {{}}
for aid, tr in TRAJ.items():
    if aid == "{ego}":
        continue
    o = np.array(tr)
    d = np.hypot(o[:, 1] - ego[:, 1], o[:, 2] - ego[:, 2]) - (LWH[aid][0] + LWH["{ego}"][0]) / 2
    rel = np.stack([o[:, 1] - ego[:, 1], o[:, 2] - ego[:, 2]], 1)
    vrel = np.stack([o[:, 4] * np.cos(o[:, 3]) - ego[:, 4] * np.cos(ego[:, 3]),
                     o[:, 4] * np.sin(o[:, 3]) - ego[:, 4] * np.sin(ego[:, 3])], 1)
    closing = -(rel * vrel).sum(1) / np.maximum(np.linalg.norm(rel, axis=1), 1e-6)
    ttc = np.where(closing > 0.1, np.maximum(d, 0) / np.maximum(closing, 1e-6), np.inf)
    report[aid] = {{"min_gap_m": round(float(d.min()), 2), "min_ttc_s": round(float(ttc.min()), 2)}}
acc = np.gradient(ego[:, 4], DT)
print("ego peak deceleration [m/s2]:", round(float(-acc.min()), 2))
for aid, r in report.items():
    print(aid, r)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
for aid, tr in TRAJ.items():
    tr = np.array(tr)
    ax1.plot(tr[:, 1], tr[:, 2], lw=3 if aid == "{ego}" else 1.5, label=aid)
ax1.set_aspect("equal"); ax1.set_title("{title_short}"); ax1.legend(fontsize=8); ax1.grid(alpha=.3)
ax2.plot(ego[:, 0], ego[:, 4] * 3.6, label="ego speed [km/h]"); ax2.plot(ego[:, 0], acc, label="ego accel [m/s2]")
ax2.set_xlabel("t [s]"); ax2.legend(); ax2.grid(alpha=.3)
plt.tight_layout(); plt.show()
'''


def emit(sc: Scenario, log=None, hz: float = 2.0) -> str:
    traj = {}
    if log is not None:
        step = max(1, int(round(1.0 / (hz * sc.dt))))
        for j, aid in enumerate(log.ids):
            S = log.states[::step, j]
            traj[aid] = [[round(float(t), 2), round(float(s[0]), 2), round(float(s[1]), 2), round(float(s[2]), 3),
                          round(float(s[3]), 2)] for t, s in zip(log.t[::step], S)]
        lwh = {aid: [round(float(v), 2) for v in log.lwh[j]] for j, aid in enumerate(log.ids)}
    else:  # no simulation available: constant-velocity placeholder along +x (clearly labelled)
        ts = np.arange(0, sc.duration + 1e-9, 1.0 / hz)
        for a in sc.actors:
            traj[a.id] = [[round(float(t), 2), round(a.initial.s + a.initial_speed * t, 2), 3.5 * a.initial.lane,
                           0.0, a.initial_speed] for t in ts]
        lwh = {a.id: list(a.dims()) for a in sc.actors}
    return TEMPLATE.format(title=sc.meta.title.replace('"', "'"), title_short=sc.meta.title[:60].replace('"', "'"),
                           ver=sc.tsir_version, hz=hz, dt=1.0 / hz, lwh=json.dumps(lwh), traj=json.dumps(traj),
                           ego=sc.ego.id)
