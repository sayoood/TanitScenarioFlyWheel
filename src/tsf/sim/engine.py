"""TanitSim engine: deterministic traffic simulation on an OpenDRIVE lane graph driven by TSIR.

Ego can be scripted (TSIR actions, open loop) or driven by an external policy (closed loop,
kinematic bicycle + pure pursuit on the policy trajectory). Output is a `SimLog` with
per-tick states, signal states, fired events and safety metrics, consumed by the verifier,
the 2D/3D renderers and the RDS-HQ/WSM exporter.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from tsf.compile.mapping import odr_position
from tsf.ir.scenario import Junction, Scenario
from tsf.sim.odr import Lane, LaneKey, OdrMap

TURN_THRESH = 0.6  # rad heading change classifying a junction lane as left/right


# ----------------------------------------------------------------------------- helpers
def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def lane_turn(ln: Lane) -> str:
    d = wrap(float(ln.hdg[-1] - ln.hdg[0]))
    return "left" if d > TURN_THRESH else "right" if d < -TURN_THRESH else "straight"


def tsir_lane_key(sc: Scenario, m: OdrMap, pos) -> tuple[LaneKey, float]:
    """TSIR LanePosition -> (OpenDRIVE lane key, progress along that lane in driving direction)."""
    rid, lid, s_road = odr_position(sc, pos)
    road = m.roads[rid]
    if not -1e-6 <= s_road <= road.length + 1e-6:
        raise ValueError(f"position s={pos.s} lies outside road {pos.road} ({pos.direction}), whose length is "
                         f"{road.length:.0f} m: place the actor within 0..{road.length:.0f} m or lengthen the road/approach")
    si = max(i for i, sec in enumerate(road.sections) if sec.s <= s_road + 1e-6)
    key = (road.id, si, lid)
    if key not in m.lanes:
        raise ValueError(f"lane {key} does not exist for {pos}")
    ln = m.lanes[key]
    xs, ys = (ln.s_road, ln.cum) if ln.s_road[0] < ln.s_road[-1] else (ln.s_road[::-1], ln.cum[::-1])
    return key, float(np.interp(s_road, xs, ys))


class Path:
    """Concatenated lanes an actor follows; progress p is arc length along the concatenation."""

    def __init__(self, m: OdrMap, lanes: list[LaneKey]):
        self.keys = lanes
        xy, hd, first = [], [], []
        n = 0
        for i, k in enumerate(lanes):
            a, h = m.lanes[k].xy, m.lanes[k].hdg
            if i and np.hypot(*(a[0] - xy[-1][-1])) < 0.05:   # shared joint point
                a, h = a[1:], h[1:]
                first.append(n - 1)
            else:
                first.append(n)
            xy.append(a)
            hd.append(h)
            n += len(a)
        self.xy = np.concatenate(xy)
        self.hdg = np.unwrap(np.concatenate(hd))
        d = np.linalg.norm(np.diff(self.xy, axis=0), axis=1)
        self.cum = np.concatenate([[0.0], np.cumsum(d)])
        self.starts = [float(self.cum[i]) for i in first]
        self.length = float(self.cum[-1])

    def pose(self, p: float, d: float = 0.0):
        p = min(max(p, 0.0), self.length)
        x = float(np.interp(p, self.cum, self.xy[:, 0]))
        y = float(np.interp(p, self.cum, self.xy[:, 1]))
        h = float(np.interp(p, self.cum, self.hdg))
        return x - math.sin(h) * d, y + math.cos(h) * d, h

    def lane_index(self, p: float) -> int:
        return max(0, int(np.searchsorted(self.starts, p, side="right")) - 1)

    def project(self, x: float, y: float, p_from: float, horizon: float):
        """Project a world point onto [p_from, p_from+horizon] -> (p, signed lateral offset)."""
        i0 = int(np.searchsorted(self.cum, p_from))
        i1 = int(np.searchsorted(self.cum, p_from + horizon)) + 1
        if i1 - i0 < 2:
            return None
        seg = self.xy[i0:i1]
        dd = np.hypot(seg[:, 0] - x, seg[:, 1] - y)
        j = int(dd.argmin())
        h = self.hdg[i0 + j]
        lat = -math.sin(h) * (x - seg[j, 0]) + math.cos(h) * (y - seg[j, 1])
        lon = math.cos(h) * (x - seg[j, 0]) + math.sin(h) * (y - seg[j, 1])
        return float(self.cum[i0 + j] + lon), float(lat)


def build_route(m: OdrMap, start: LaneKey, turns: list[str], min_len: float) -> tuple[list[LaneKey], list[str]]:
    """Follow successors from `start`; at junctions take the next requested turn if available."""
    keys, total, remaining = [start], m.lanes[start].length, list(turns)
    cur = start
    while total < min_len:
        succ = m.lanes[cur].successors
        if not succ:
            break
        jn = [k for k in succ if m.lanes[k].junction != "-1"]
        if jn:
            want = remaining[0] if remaining else "straight"
            pick = next((k for k in jn if lane_turn(m.lanes[k]) == want), None)
            if pick is not None and remaining:
                remaining.pop(0)
            if pick is None:
                pick = next((k for k in jn if lane_turn(m.lanes[k]) == "straight"), jn[0])
        else:
            pick = succ[0]
        keys.append(pick)
        total += m.lanes[pick].length
        cur = pick
        if len(keys) > 64:
            break
    return keys, remaining


# ----------------------------------------------------------------------------- signals
class SignalPlan:
    """Fixed-time controller over junction arms (two opposing groups, optional protected left)."""

    def __init__(self, sc: Scenario):
        self.enabled = isinstance(sc.road, Junction) and sc.road.signalized
        if not self.enabled:
            return
        j: Junction = sc.road
        p = j.signal_plan
        h0 = j.approaches[0].heading_deg
        grp = [0 if abs(math.remainder(a.heading_deg - h0, 180.0)) < 45 else 1 for a in j.approaches]
        left_g = max(6.0, 0.4 * p.green_s)
        phases = []
        for g in (0, 1):
            arms = {i for i, x in enumerate(grp) if x == g}
            if p.protected_left:
                phases.append((arms, {"left"}, left_g))
                phases.append((arms, {"straight", "right"}, p.green_s))
            else:
                phases.append((arms, {"left", "straight", "right"}, p.green_s))
        self.phases, self.yellow, self.red = phases, p.yellow_s, p.all_red_s
        self.cycle = sum(g + self.yellow + self.red for _, _, g in phases)
        self.offset = p.initial_phase_offset_s
        self.protected = p.protected_left

    def state(self, arm: int, turn: str, t: float) -> str:
        if not self.enabled:
            return "green"
        tc = (t + self.offset) % self.cycle
        for arms, turns, g in self.phases:
            if tc < g + self.yellow + self.red:
                if arm in arms and turn in turns:
                    return "green" if tc < g else "yellow" if tc < g + self.yellow else "red"
                return "red"
            tc -= g + self.yellow + self.red
        return "red"

    def arm_state(self, arm: int, t: float) -> str:
        """Head state shown to an arm (green if any movement of the arm is green)."""
        s = [self.state(arm, tr, t) for tr in ("left", "straight", "right")]
        return "green" if "green" in s else "yellow" if "yellow" in s else "red"


# ----------------------------------------------------------------------------- actors
@dataclass
class Agent:
    id: str
    role: str
    category: str
    lwh: tuple[float, float, float]
    path: Path | None
    p: float
    v: float
    turns: list[str]
    d: float = 0.0
    v_des: float = 13.9
    a_max: float = 2.0
    b_comf: float = 3.0
    time_gap: float = 1.5
    headway: float = 1.5              # desired time headway to any leader (driver parameter)
    follow_target: str | None = None
    lc: dict | None = None            # active lane change {d0, t0, T}
    d_off: float = 0.0                # persistent in-lane offset (lane_offset / initial t_offset)
    lo: dict | None = None            # active lane offset transition {d0, d1, t0, T}
    braking: float | None = None      # emergency brake decel
    stop_line: bool = False           # stop_at_line active
    a: float = 0.0
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    free: dict | None = None          # free motion (pedestrian crossing) {vx, vy}
    external: bool = False            # closed-loop ego
    steer: float = 0.0
    done: bool = False


@dataclass
class SimLog:
    t: np.ndarray
    ids: list[str]
    states: np.ndarray                # (T, N, 6): x, y, yaw, v, a, d
    lanes: list[list[LaneKey | None]]
    lwh: np.ndarray                   # (N, 3)
    categories: list[str]
    roles: list[str]
    signals: dict[int, list[str]]     # arm -> state per tick
    events: list[dict] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)

    def idx(self, actor: str) -> int:
        return self.ids.index(actor)


def obb_corners(x, y, yaw, l, w):
    c, s = math.cos(yaw), math.sin(yaw)
    dx = np.array([l, l, -l, -l]) / 2
    dy = np.array([w, -w, -w, w]) / 2
    return np.stack([x + c * dx - s * dy, y + s * dx + c * dy], 1)


def obb_overlap(A, B) -> bool:
    for P in (A, B):
        for i in range(2):
            e = P[(i + 1) % 4] - P[i]
            n = np.array([-e[1], e[0]])
            pa, pb = A @ n, B @ n
            if pa.max() < pb.min() or pb.max() < pa.min():
                return False
    return True


# ----------------------------------------------------------------------------- simulator
class TanitSim:
    def __init__(self, sc: Scenario, m: OdrMap, ego_policy=None):
        self.sc, self.m, self.t = sc, m, 0.0
        self.dt = sc.dt
        self.signals = SignalPlan(sc)
        self.ego_policy = ego_policy
        horizon = sc.duration * 40 + 200
        self.agents: dict[str, Agent] = {}
        for a in sc.actors:
            key, prog = tsir_lane_key(sc, m, a.initial)
            l, w, h = a.dims()
            if a.category == "pedestrian":
                ln = m.lanes[key]
                x, y, hd = ln.pose(prog, a.initial.t_offset)
                ag = Agent(a.id, a.role, a.category, (l, w, h), None, 0.0, 0.0, [], x=x, y=y, yaw=hd + math.pi / 2,
                           v_des=1.4)
            else:
                keys, rem = build_route(m, key, list(a.route), horizon)
                ag = Agent(a.id, a.role, a.category, (l, w, h), Path(m, keys), prog, a.initial_speed, rem,
                           d=a.initial.t_offset, d_off=a.initial.t_offset, v_des=max(a.initial_speed, 0.1))
                ag.turns_all = list(a.route)
                ag.headway = a.headway or 1.5
                if a.role == "ego" and ego_policy is not None:
                    ag.external = True          # integrated kinematically from here on: start at its scenario pose
                    ag.x, ag.y, ag.yaw = ag.path.pose(ag.p, ag.d)
            self.agents[a.id] = ag
        self.actions = {act.id: act for act in sc.actions}
        self.started: dict[str, float] = {}
        self._lc_wait: dict[str, float] = {}           # safe_gap lane changes waiting for a gap
        self.ended: dict[str, float] = {}
        self.events: list[dict] = []
        self._junction_conflicts = self._conflicts()
        for ag in self.agents.values():
            self._update_pose(ag)

    # ---------------------------------------------------------------- junction conflicts
    def _conflicts(self) -> dict[LaneKey, set[LaneKey]]:
        from shapely.geometry import LineString
        jl = {k: ln for k, ln in self.m.lanes.items() if ln.junction != "-1"}
        geo = {k: LineString(ln.xy) for k, ln in jl.items()}
        out = {k: set() for k in jl}
        ks = list(jl)
        for i, a in enumerate(ks):
            for b in ks[i + 1:]:
                if jl[a].predecessors and jl[b].predecessors and \
                        set(jl[a].predecessors) & set(jl[b].predecessors):
                    continue  # same entry lane: not a crossing conflict
                if geo[a].distance(geo[b]) < 1.2:
                    out[a].add(b)
                    out[b].add(a)
        return out

    # ---------------------------------------------------------------- triggers
    def _fires(self, trig) -> bool:
        k = trig.kind
        if k == "sim_time":
            return self.t >= trig.t - 1e-9
        if k == "after_action":
            ref = (self.started if trig.on == "start" else self.ended).get(trig.action_id)
            return ref is not None and self.t >= ref + trig.delay - 1e-9
        a = self.agents.get(trig.actor)
        if k == "distance_to_actor":
            b = self.agents[trig.other]
            if trig.mode == "euclidean":
                dist = math.hypot(a.x - b.x, a.y - b.y)
            else:
                lon = math.cos(b.yaw) * (a.x - b.x) + math.sin(b.yaw) * (a.y - b.y)
                if (lon > 0) != (trig.mode == "ahead"):
                    return False
                dist = abs(lon)
            if trig.freespace:
                dist -= (a.lwh[0] + b.lwh[0]) / 2
            return dist < trig.distance if trig.rule == "less" else dist > trig.distance
        if k == "ttc":
            return self._ttc(a, self.agents[trig.other]) < trig.ttc
        if k == "distance_to_stop_line":
            dsl = self._dist_to_stop_line(a)
            return dsl is not None and dsl <= trig.distance
        return False

    def _start_action(self, act):
        ag = self.agents[act.actor]
        self.started[act.id] = self.t
        self.events.append({"t": round(self.t, 3), "type": "action_start", "id": act.id, "actor": act.actor})
        ty = act.type
        if ty == "speed_change":
            ag.v_des, ag.a_max = act.target_speed, act.accel
            ag.follow_target = None
        elif ty == "follow":
            ag.follow_target, ag.time_gap = act.target, act.time_gap
        elif ty == "lane_change":
            self._begin_lane_change(ag, act.direction, act.lanes, act.duration, act.id)
        elif ty == "cut_in":
            tgt = self.agents[act.target]
            lat = tgt.path.project(ag.x, ag.y, max(0, tgt.p - 80), 160) if tgt.path else None
            direction = "left" if lat and lat[1] < 0 else "right"
            self._begin_lane_change(ag, direction, 1, act.duration, act.id)
        elif ty == "stop_at_line":
            ag.stop_line = True
            ag.b_comf = act.decel
        elif ty == "brake":
            ag.braking = act.decel
        elif ty == "lane_offset":
            ag.lo = {"d0": ag.d, "d1": act.offset, "t0": self.t, "T": act.duration, "id": act.id}
        elif ty == "cross":
            sgn = 1 if act.side == "right_to_left" else -1
            ag.free = {"vx": -math.sin(ag.yaw - math.pi / 2) * act.speed * sgn * -1,
                       "vy": math.cos(ag.yaw - math.pi / 2) * act.speed * sgn * -1}
            ag.v = act.speed

    def _end_action(self, act_id: str, actor: str):
        if act_id not in self.ended:
            self.ended[act_id] = self.t
            self.events.append({"t": round(self.t, 3), "type": "action_end", "id": act_id, "actor": actor})

    # ---------------------------------------------------------------- lane changes
    def _defer(self, act) -> bool:
        """`safe_gap` lane changes (replayed drivers) wait until the target lane is safe: no vehicle within
        2 m + 0.3 s ahead, and the follower there keeps 2 m + 0.5 s and needs <= 3 m/s2 to adapt (MOBIL b_safe).
        After 5 s of waiting the lane change is dropped. A replayed lane change is otherwise timed to the logged
        ego; once the ego deviates from the log (UC5 twins, closed loop) it could cut into the ego."""
        if act.type != "lane_change" or not getattr(act, "safe_gap", False):
            return False
        ag = self.agents[act.actor]
        if ag.path is None or ag.done or self._lc_safe(ag, act.direction, act.lanes):
            self._lc_wait.pop(act.id, None)
            return False
        t0 = self._lc_wait.setdefault(act.id, self.t)
        if self.t - t0 < 5.0:
            return True
        self.started[act.id] = self.t
        self.events.append({"t": round(self.t, 3), "type": "lane_change_skipped_unsafe", "actor": ag.id, "id": act.id})
        self._end_action(act.id, ag.id)
        return True

    def _lc_safe(self, ag: Agent, direction: str, n: int) -> bool:
        key = ag.path.keys[ag.path.lane_index(ag.p)]
        tgt = key
        for _ in range(n):
            nb = self.m.lanes[tgt].left if direction == "left" else self.m.lanes[tgt].right
            if nb is None:
                break
            tgt = nb
        if tgt == key:
            return True                                        # impossible: _begin_lane_change reports it
        ln = self.m.lanes[tgt]
        p0 = float(ln.cum[int(np.hypot(ln.xy[:, 0] - ag.x, ln.xy[:, 1] - ag.y).argmin())])
        for o in self.agents.values():
            if o is ag or o.done or o.path is None or math.hypot(o.x - ag.x, o.y - ag.y) > 80:
                continue
            j = int(np.hypot(ln.xy[:, 0] - o.x, ln.xy[:, 1] - o.y).argmin())
            if math.hypot(ln.xy[j, 0] - o.x, ln.xy[j, 1] - o.y) > 1.6 + o.lwh[1] / 2:
                continue                                       # not in the target lane
            gap = float(ln.cum[j]) - p0
            free = abs(gap) - (ag.lwh[0] + o.lwh[0]) / 2
            if gap >= 0 and free < 2.0 + 0.3 * ag.v:
                return False
            if gap < 0:
                closing = max(0.0, o.v - ag.v)
                if free < 2.0 + 0.5 * o.v or closing ** 2 / (2 * max(free - 2.0, 0.1)) > 3.0:
                    return False
        return True

    def _begin_lane_change(self, ag: Agent, direction: str, n: int, T: float, act_id: str):
        key = ag.path.keys[ag.path.lane_index(ag.p)]
        tgt = key
        for _ in range(n):
            nb = self.m.lanes[tgt].left if direction == "left" else self.m.lanes[tgt].right
            if nb is None:
                break
            tgt = nb
        if tgt == key:
            self.events.append({"t": round(self.t, 3), "type": "lane_change_impossible", "actor": ag.id})
            self._end_action(act_id, ag.id)
            return
        x, y = ag.x, ag.y
        ln = self.m.lanes[tgt]
        dd = np.hypot(ln.xy[:, 0] - x, ln.xy[:, 1] - y)
        prog = float(ln.cum[int(dd.argmin())])
        keys, rem = build_route(self.m, tgt, getattr(ag, "turns_all", ag.turns), self.sc.duration * 40 + 200)
        newp = Path(self.m, keys)
        proj = newp.project(x, y, max(0.0, prog - 5), 10.0)
        ag.path, ag.turns = newp, rem
        ag.p, ag.d = (proj if proj else (prog, 0.0))
        ag.lc = {"d0": ag.d, "t0": self.t, "T": T, "id": act_id}

    # ---------------------------------------------------------------- geometry queries
    def _dist_to_stop_line(self, ag: Agent) -> float | None:
        if ag.path is None:
            return None
        i = ag.path.lane_index(ag.p)
        for j in range(i, min(i + 3, len(ag.path.keys))):
            if self.m.lanes[ag.path.keys[j]].junction != "-1":
                return ag.path.starts[j] - ag.p - ag.lwh[0] / 2 if j > i else None
        return None

    def _next_junction_lane(self, ag: Agent) -> tuple[int, LaneKey] | None:
        i = ag.path.lane_index(ag.p)
        for j in range(i, min(i + 3, len(ag.path.keys))):
            k = ag.path.keys[j]
            if self.m.lanes[k].junction != "-1":
                return (j, k) if j > i else None
        return None

    def _arm_of(self, ag: Agent) -> int | None:
        k = ag.path.keys[ag.path.lane_index(ag.p)]
        return int(k[0]) if isinstance(self.sc.road, Junction) and k[0].isdigit() and int(k[0]) < 100 else None

    def _ttc(self, a: Agent, b: Agent, horizon: float = 6.0, step: float = 0.1) -> float:
        va = np.array([math.cos(a.yaw), math.sin(a.yaw)]) * a.v
        vb = np.array([math.cos(b.yaw), math.sin(b.yaw)]) * b.v
        rel_p = np.array([b.x - a.x, b.y - a.y])
        reach = (a.lwh[0] + b.lwh[0]) / 2 + 0.5
        dist0 = float(np.hypot(*rel_p))
        if dist0 - (np.hypot(*(vb - va)) * horizon) > reach + max(a.lwh[0], b.lwh[0]):
            return math.inf
        for tau in np.arange(step, horizon + 1e-9, step):
            A = obb_corners(a.x + va[0] * tau, a.y + va[1] * tau, a.yaw, a.lwh[0], a.lwh[1])
            B = obb_corners(b.x + vb[0] * tau, b.y + vb[1] * tau, b.yaw, b.lwh[0], b.lwh[1])
            if obb_overlap(A, B):
                return float(tau)
        return math.inf

    # ---------------------------------------------------------------- longitudinal model
    def _leader_gap(self, ag: Agent) -> tuple[float, float, str | None]:
        """Gap (m) and speed of the closest obstacle ahead in the actor's corridor."""
        best, bv, bid = math.inf, 0.0, None
        horizon = max(40.0, ag.v * 6 + 20)
        for o in self.agents.values():
            if o is ag or o.done:
                continue
            if math.hypot(o.x - ag.x, o.y - ag.y) > horizon + 20:
                continue
            pr = ag.path.project(o.x, o.y, ag.p, horizon)
            if pr is None:
                continue
            po, lat = pr
            if po <= ag.p:
                continue
            if abs(lat - ag.d) > (ag.lwh[1] + o.lwh[1]) / 2 + 0.25:
                continue
            gap = po - ag.p - (ag.lwh[0] + o.lwh[0]) / 2
            if gap < best:
                ov = o.v * math.cos(wrap(o.yaw - ag.yaw))
                best, bv, bid = gap, max(ov, 0.0), o.id
        return best, bv, bid

    def _must_stop(self, ag: Agent) -> float | None:
        """Distance to a virtual stop (red light, stop_at_line, yield) or None."""
        nj = self._next_junction_lane(ag)
        if nj is None:
            return None
        j, jkey = nj
        dist = ag.path.starts[j] - ag.p - ag.lwh[0] / 2  # front bumper to stop line
        if dist > 120:
            return None
        turn = lane_turn(self.m.lanes[jkey])
        arm = self._arm_of(ag)
        if self.signals.enabled and arm is not None:
            st = self.signals.state(arm, turn, self.t)
            if st == "red" or (st == "yellow" and dist > ag.v ** 2 / (2 * 3.0) + 2):
                return dist
            if st == "green" and turn == "left" and not self.signals.protected and self._yield(ag, jkey, dist):
                return dist
            if st == "green" and ag.stop_line:
                ag.stop_line = False
        elif ag.stop_line:
            return dist
        elif turn == "left" and self._yield(ag, jkey, dist):
            return dist
        return None

    def _yield(self, ag: Agent, jkey: LaneKey, dist: float) -> bool:
        if dist > 25:
            return False
        conf = self._junction_conflicts.get(jkey, set())
        for o in self.agents.values():
            if o is ag or o.path is None:
                continue
            oi = o.path.lane_index(o.p)
            upcoming = o.path.keys[oi:oi + 3]
            hit = [k for k in upcoming if k in conf]
            if not hit:
                continue
            if lane_turn(self.m.lanes[hit[0]]) == "left":
                continue
            j = o.path.keys.index(hit[0])
            o_dist = o.path.starts[j] - o.p
            o_end = o_dist + self.m.lanes[hit[0]].length
            if o_end < 0:
                continue
            eta = o_dist / max(o.v, 0.5)
            if eta < 4.5:
                return True
        return False

    def _idm(self, ag: Agent, gap: float, v_lead: float, T: float, a_lead: float | None = None) -> float:
        """ACC model (Kesting, Treiber & Helbing 2010): IDM blended with the constant-acceleration
        heuristic (CAH), which avoids IDM's over-braking on cut-ins with a faster/opening leader."""
        s0, coolness = 2.0, 0.99
        v0 = max(ag.v_des, 0.1)
        free = 1 - (ag.v / v0) ** 4 if ag.v <= v0 else -(1 - (v0 / max(ag.v, 1e-3)) ** 4) * 2
        if math.isinf(gap):
            # free road: saturating P-controller (tau 0.6 s) instead of IDM's (v/v0)^4 creep, so a speed_change
            # reaches its target at the commanded rate (OpenSCENARIO linear SpeedAction semantics)
            return ag.a_max * min(1.0, max(-2.0, (v0 - ag.v) / (ag.a_max * 0.6)))
        s = max(gap, 0.1)
        s_star = s0 + max(0.0, ag.v * T + ag.v * (ag.v - v_lead) / (2 * math.sqrt(ag.a_max * ag.b_comf)))
        a_idm = ag.a_max * (free - (s_star / s) ** 2)
        if a_lead is None:
            return a_idm
        al = min(a_lead, ag.a_max)
        if v_lead * (ag.v - v_lead) <= -2 * s * al:
            a_cah = ag.v ** 2 * al / max(v_lead ** 2 - 2 * s * al, 1e-6)
        else:
            a_cah = al - (max(ag.v - v_lead, 0.0) ** 2) / (2 * s)
        if a_idm >= a_cah:
            return a_idm
        b = ag.b_comf
        return (1 - coolness) * a_idm + coolness * (a_cah + b * math.tanh((a_idm - a_cah) / b))

    # ---------------------------------------------------------------- step
    def _update_pose(self, ag: Agent):
        if ag.path is None or ag.external:
            return
        x, y, h = ag.path.pose(ag.p, ag.d)
        if ag.lc:
            el = (self.t - ag.lc["t0"]) / ag.lc["T"]
            dd = -ag.lc["d0"] * (30 * el**2 - 60 * el**3 + 30 * el**4) / ag.lc["T"] if 0 <= el <= 1 else 0.0
            h += math.atan2(dd, max(ag.v, 0.5))
        ag.x, ag.y, ag.yaw = x, y, h

    def step(self):
        dt, t = self.dt, self.t
        for act in self.actions.values():
            if act.id not in self.started and self._fires(act.start) and not self._defer(act):
                self._start_action(act)
        for ag in self.agents.values():
            if ag.done:
                continue
            if ag.free is not None:
                ag.x += ag.free["vx"] * dt
                ag.y += ag.free["vy"] * dt
                continue
            if ag.path is None:
                continue
            if ag.external:
                continue
            gap, vl, lid = self._leader_gap(ag)
            T = ag.time_gap if ag.follow_target and lid == ag.follow_target else ag.headway
            acc = self._idm(ag, gap, vl, T, self.agents[lid].a if lid else None)
            sd = self._must_stop(ag)
            if sd is not None:  # virtual standing obstacle 1 m past the line; IDM s0=2 m -> stop ~1 m before it
                acc = min(acc, self._idm(ag, max(sd + 1.0, 0.05), 0.0, 1.0))
            if ag.braking is not None:
                acc = -ag.braking
            acc = max(min(acc, ag.a_max), -9.0)
            # jerk limits: braking may build up fast (30 m/s3), everything else is smooth (6 m/s3)
            j_up = 30.0 if acc < ag.a and acc < -1.0 else 6.0
            acc = min(max(acc, ag.a - j_up * dt), ag.a + 6.0 * dt)
            ag.a = acc
            ag.v = max(0.0, ag.v + acc * dt)
            ag.p += ag.v * dt
            if ag.lc:
                el = (t + dt - ag.lc["t0"]) / ag.lc["T"]
                if el >= 1:
                    ag.d = ag.d_off
                    self._end_action(ag.lc["id"], ag.id)
                    ag.lc = None
                else:
                    ag.d = ag.d_off + (ag.lc["d0"] - ag.d_off) * (1 - (10 * el**3 - 15 * el**4 + 6 * el**5))
            elif ag.lo:
                el = (t + dt - ag.lo["t0"]) / ag.lo["T"]
                if el >= 1:
                    ag.d = ag.d_off = ag.lo["d1"]
                    self._end_action(ag.lo["id"], ag.id)
                    ag.lo = None
                else:
                    ag.d = ag.lo["d0"] + (ag.lo["d1"] - ag.lo["d0"]) * (10 * el**3 - 15 * el**4 + 6 * el**5)
            if ag.p >= ag.path.length - 0.5:
                ag.done = True
                self.events.append({"t": round(t, 3), "type": "route_end", "actor": ag.id})
            for aid in [a.id for a in self.actions.values() if a.actor == ag.id and a.id in self.started]:
                act = self.actions[aid]
                if act.type == "speed_change" and abs(ag.v - act.target_speed) < 0.2:
                    self._end_action(aid, ag.id)
                if act.type == "brake" and ag.v < 0.05:
                    self._end_action(aid, ag.id)
        ego = next(a for a in self.agents.values() if a.role == "ego")
        if ego.external:
            self._step_external(ego)
        self.t = t + dt
        for ag in self.agents.values():
            self._update_pose(ag)

    # ---------------------------------------------------------------- closed-loop ego
    def _step_external(self, ego: Agent):
        """Kinematic bicycle tracking the policy trajectory [(x_fwd, y_left) @ 0.5 s ...] in ego frame."""
        cmd = self.ego_policy(self, ego)
        traj, dt_traj = cmd["traj"], cmd.get("dt", 0.5)
        L = 2.9
        look = max(4.0, ego.v * 0.9)
        pts = np.asarray(traj, dtype=float)
        dist = np.hypot(pts[:, 0], pts[:, 1])
        k = int(np.searchsorted(dist, look))
        tx, ty = pts[min(k, len(pts) - 1)]
        alpha = math.atan2(ty, max(tx, 0.1))
        ego.steer = float(np.clip(math.atan2(2 * L * math.sin(alpha), max(math.hypot(tx, ty), 0.1)), -0.6, 0.6))
        v_target = float(np.hypot(*(pts[1] - pts[0])) / dt_traj) if len(pts) > 1 else float(dist[0] / dt_traj)
        ego.a = float(np.clip((v_target - ego.v) / 0.8, -8.0, 3.0))
        dt = self.dt
        ego.v = max(0.0, ego.v + ego.a * dt)
        ego.yaw += ego.v / L * math.tan(ego.steer) * dt
        ego.x += ego.v * math.cos(ego.yaw) * dt
        ego.y += ego.v * math.sin(ego.yaw) * dt
        pr = ego.path.project(ego.x, ego.y, max(0.0, ego.p - 5), 30)
        if pr:
            ego.p, ego.d = pr

    # ---------------------------------------------------------------- run
    def run(self) -> SimLog:
        ids = list(self.agents)
        n = int(round(self.sc.duration / self.dt)) + 1
        S = np.zeros((n, len(ids), 6))
        lanes, sig = [], {}
        arms = range(len(self.sc.road.approaches)) if isinstance(self.sc.road, Junction) else []
        for arm in arms:
            sig[arm] = []
        ts = np.arange(n) * self.dt
        for i in range(n):
            row = []
            for j, aid in enumerate(ids):
                ag = self.agents[aid]
                S[i, j] = (ag.x, ag.y, ag.yaw, ag.v, ag.a, ag.d)
                row.append(ag.path.keys[ag.path.lane_index(ag.p)] if ag.path else None)
            lanes.append(row)
            for arm in arms:
                sig[arm].append(self.signals.arm_state(arm, self.t))
            if i < n - 1:
                self.step()
        log = SimLog(ts, ids, S, lanes, np.array([self.agents[a].lwh for a in ids]),
                     [self.agents[a].category for a in ids], [self.agents[a].role for a in ids], sig, self.events)
        log.metrics = metrics(self, log)
        return log


# ----------------------------------------------------------------------------- metrics
def metrics(sim: TanitSim, log: SimLog) -> dict:
    e = log.roles.index("ego")
    S, n = log.states, len(log.t)
    collisions, min_ttc, min_ttc_actor = [], math.inf, None
    stride = max(1, int(round(0.1 / sim.dt)))
    for i in range(0, n, stride):
        ex, ey, eyaw, ev = S[i, e, :4]
        A = obb_corners(ex, ey, eyaw, log.lwh[e, 0], log.lwh[e, 1])
        for j in range(len(log.ids)):
            if j == e:
                continue
            B = obb_corners(S[i, j, 0], S[i, j, 1], S[i, j, 2], log.lwh[j, 0], log.lwh[j, 1])
            if obb_overlap(A, B):
                collisions.append({"t": float(log.t[i]), "with": log.ids[j]})
            a = Agent("e", "", "", tuple(log.lwh[e]), None, 0, ev, [], x=ex, y=ey, yaw=eyaw)
            b = Agent("o", "", "", tuple(log.lwh[j]), None, 0, S[i, j, 3], [], x=S[i, j, 0], y=S[i, j, 1],
                      yaw=S[i, j, 2])
            ttc = sim._ttc(a, b, horizon=5.0, step=0.1)
            if ttc < min_ttc:
                min_ttc, min_ttc_actor = ttc, log.ids[j]
    acc = S[:, e, 4]
    jerk = np.diff(acc) / sim.dt if n > 1 else np.zeros(1)
    lane_seq = [ln[e] for ln in log.lanes]
    lane_changes, turns = 0, []
    prev = lane_seq[0]
    for k in lane_seq[1:]:
        if k is None or prev is None:
            prev = k
            continue
        if k != prev:
            if k[0] == prev[0] and k[1] == prev[1] and k[2] != prev[2]:
                lane_changes += 1 if abs(k[2]) < abs(prev[2]) else -1  # + = toward centre (left in RHT)
            if sim.m.lanes[k].junction != "-1" and sim.m.lanes[prev].junction == "-1":
                turns.append(lane_turn(sim.m.lanes[k]))
        prev = k
    red_violation = _red_violations(sim, log, e)
    return {
        "collision": bool(collisions), "collisions": collisions[:5],
        "min_ttc": None if math.isinf(min_ttc) else round(min_ttc, 2), "min_ttc_actor": min_ttc_actor,
        "ego_max_decel": round(float(-acc.min()), 2), "ego_max_accel": round(float(acc.max()), 2),
        "ego_max_abs_jerk": round(float(np.abs(jerk).max()), 2),
        "ego_mean_speed": round(float(S[:, e, 3].mean()), 2),
        "ego_distance": round(float(np.sum(np.hypot(np.diff(S[:, e, 0]), np.diff(S[:, e, 1])))), 1),
        "ego_lane_changes_left": lane_changes, "ego_turns": turns,
        "red_light_violations": red_violation,
    }


def _red_violations(sim: TanitSim, log: SimLog, e: int) -> list[float]:
    if not sim.signals.enabled:
        return []
    out = []
    seq = [ln[e] for ln in log.lanes]
    for i in range(1, len(seq)):
        a, b = seq[i - 1], seq[i]
        if a and b and a != b and sim.m.lanes[b].junction != "-1" and sim.m.lanes[a].junction == "-1":
            arm = int(a[0]) if a[0].isdigit() and int(a[0]) < 100 else None
            if arm is not None and sim.signals.state(arm, lane_turn(sim.m.lanes[b]), float(log.t[i])) == "red":
                out.append(float(log.t[i]))
    return out
