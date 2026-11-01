/**
 * Procedural Scene generator for the mock layer.
 *
 * Builds a 4-arm signalized intersection (right-hand traffic, 2 inbound + 2 outbound lanes per arm,
 * arm headings 0/90/180/270 as in examples/uc1_left_turn_intersection.json) and runs a tiny
 * microsimulation: the ego starts in the right lane of arm 2, changes one lane to the left, then
 * takes the protected-green left turn. NPCs follow IDM and respect the signals.
 *
 * Self-contained on purpose (type-only imports), so it can also be run with plain Node for testing.
 */
import type { Scene } from "../types";

type V2 = [number, number];

export const LANE_W = 3.5;
export const BOX = 8.5; // half-size of the junction box
export const STOP = 9.8; // stop line distance from the centre
export const ARM_HEADINGS = [0, 90, 180, 270];
export const ARM_LEN = [180, 180, 200, 180];
const STEP = 0.5; // route resolution (m)
const r3 = (v: number) => Math.round(v * 1000) / 1000;
const clean = (v: number) => (Math.abs(v) < 1e-9 ? 0 : v);

export type Move = "left" | "straight" | "right";

const armU = (a: number): V2 => {
  const th = (ARM_HEADINGS[a] * Math.PI) / 180;
  return [clean(Math.cos(th)), clean(Math.sin(th))];
};
const rightIn = (a: number): V2 => {
  const u = armU(a);
  return [clean(-u[1]), clean(u[0])];
};
const add = (p: V2, q: V2, k = 1): V2 => [p[0] + q[0] * k, p[1] + q[1] * k];

const inLaneCentre = (a: number, k: number, dist: number): V2 =>
  add(add([0, 0], armU(a), dist), rightIn(a), LANE_W * (k - 0.5));
const outLaneCentre = (a: number, k: number, dist: number): V2 =>
  add(add([0, 0], armU(a), dist), rightIn(a), -LANE_W * (k - 0.5));

const turnTarget = (a: number, m: Move): number => (m === "left" ? (a + 3) % 4 : m === "right" ? (a + 1) % 4 : (a + 2) % 4);

function offsetBoundaries(c: V2[], half: number): { left: V2[]; right: V2[] } {
  const left: V2[] = [];
  const right: V2[] = [];
  for (let i = 0; i < c.length; i++) {
    const p0 = c[Math.max(0, i - 1)];
    const p1 = c[Math.min(c.length - 1, i + 1)];
    let tx = p1[0] - p0[0];
    let ty = p1[1] - p0[1];
    const n = Math.hypot(tx, ty) || 1;
    tx /= n;
    ty /= n;
    left.push([c[i][0] - ty * half, c[i][1] + tx * half]);
    right.push([c[i][0] + ty * half, c[i][1] - tx * half]);
  }
  return { left, right };
}

function sampleLine(p0: V2, p1: V2, n: number): V2[] {
  const out: V2[] = [];
  for (let i = 0; i <= n; i++) out.push([p0[0] + ((p1[0] - p0[0]) * i) / n, p0[1] + ((p1[1] - p0[1]) * i) / n]);
  return out;
}

interface Connector {
  pts: V2[];
  outArm: number;
  outLane: number;
  ctrl: [V2, V2, V2, V2];
  entryHeading: number;
}

function connector(a: number, k: number, m: Move): Connector {
  const b = turnTarget(a, m);
  const outLane = k; // inner->inner, outer->outer
  const E = inLaneCentre(a, k, BOX);
  const X = outLaneCentre(b, outLane, BOX);
  const d: V2 = [-armU(a)[0], -armU(a)[1]];
  const ub = armU(b);
  const W: V2 = [X[0] - E[0], X[1] - E[1]];
  const det = d[0] * ub[1] - d[1] * ub[0];
  let P1: V2;
  let P2: V2;
  if (Math.abs(det) < 1e-6) {
    const c = Math.hypot(W[0], W[1]) / 3;
    P1 = add(E, d, c);
    P2 = add(X, ub, -c);
  } else {
    const s = (W[0] * ub[1] - W[1] * ub[0]) / det;
    const t = (d[0] * W[1] - d[1] * W[0]) / det;
    P1 = add(E, d, 0.55 * s);
    P2 = add(X, ub, -0.55 * t);
  }
  const N = m === "straight" ? 8 : 28;
  const pts: V2[] = [];
  for (let i = 0; i <= N; i++) {
    const u = i / N;
    const w = 1 - u;
    pts.push([
      w * w * w * E[0] + 3 * w * w * u * P1[0] + 3 * w * u * u * P2[0] + u * u * u * X[0],
      w * w * w * E[1] + 3 * w * w * u * P1[1] + 3 * w * u * u * P2[1] + u * u * u * X[1],
    ]);
  }
  return { pts, outArm: b, outLane, ctrl: [E, P1, P2, X], entryHeading: Math.atan2(d[1], d[0]) };
}

export interface ConnectorInfo {
  id: string;
  arm: number;
  lane: number;
  move: Move;
  outArm: number;
  outLane: number;
  ctrl: [V2, V2, V2, V2];
  entryHeading: number;
  length: number;
}

const CONNECTOR_MOVES: [number, Move][] = [
  [1, "left"],
  [1, "straight"],
  [2, "straight"],
  [2, "right"],
];

export function getConnectors(): ConnectorInfo[] {
  const out: ConnectorInfo[] = [];
  for (let a = 0; a < 4; a++) {
    for (const [k, m] of CONNECTOR_MOVES) {
      const c = connector(a, k, m);
      let length = 0;
      for (let i = 1; i < c.pts.length; i++) length += Math.hypot(c.pts[i][0] - c.pts[i - 1][0], c.pts[i][1] - c.pts[i - 1][1]);
      out.push({ id: `j${a}_${k}_${m}`, arm: a, lane: k, move: m, outArm: c.outArm, outLane: c.outLane, ctrl: c.ctrl, entryHeading: c.entryHeading, length });
    }
  }
  return out;
}

// ----------------------------------------------------------------------------- map
export function buildMap(): Pick<Scene, "bounds" | "lanes" | "stop_lines" | "signals"> {
  const lanes: Scene["lanes"] = [];
  const pushLane = (id: string, junction: boolean, centre: V2[]) => {
    const { left, right } = offsetBoundaries(centre, LANE_W / 2);
    lanes.push({ id, junction, left: left.map(rp), right: right.map(rp), center: centre.map(rp) });
  };
  for (let a = 0; a < 4; a++) {
    for (let k = 1; k <= 2; k++) {
      pushLane(`arm${a}_in${k}`, false, sampleLine(inLaneCentre(a, k, ARM_LEN[a]), inLaneCentre(a, k, BOX), 10));
      pushLane(`arm${a}_out${k}`, false, sampleLine(outLaneCentre(a, k, BOX), outLaneCentre(a, k, ARM_LEN[a]), 10));
    }
    for (const [k, m] of CONNECTOR_MOVES) pushLane(`j${a}_${k}_${m}`, true, connector(a, k, m).pts);
  }
  const stop_lines: Scene["stop_lines"] = [];
  const signals: Scene["signals"] = [];
  for (let a = 0; a < 4; a++) {
    const u = armU(a);
    const r = rightIn(a);
    stop_lines.push([rp(add(add([0, 0], u, STOP), r, 0.1)), rp(add(add([0, 0], u, STOP), r, 2 * LANE_W - 0.1))]);
    const sp = add(add([0, 0], u, STOP + 1.5), r, 2 * LANE_W + 1.4);
    signals.push({ arm: a, x: r3(sp[0]), y: r3(sp[1]), states: [] });
  }
  return { bounds: [-ARM_LEN[2], -ARM_LEN[3], ARM_LEN[0], ARM_LEN[1]], lanes, stop_lines, signals };
}
const rp = (p: V2): number[] => [r3(p[0]), r3(p[1])];

// ----------------------------------------------------------------------------- routes
interface Route {
  pts: V2[];
  s: number[];
  len: number;
  connStart: number;
  connEnd: number;
  stopS: number;
  move: Move;
}

function densify(poly: V2[], step: number): V2[] {
  const out: V2[] = [poly[0]];
  for (let i = 1; i < poly.length; i++) {
    const a = poly[i - 1];
    const b = poly[i];
    const L = Math.hypot(b[0] - a[0], b[1] - a[1]);
    const n = Math.max(1, Math.round(L / step));
    for (let j = 1; j <= n; j++) out.push([a[0] + ((b[0] - a[0]) * j) / n, a[1] + ((b[1] - a[1]) * j) / n]);
  }
  return out;
}

function buildRoute(arm: number, lane: number, m: Move): Route {
  const c = connector(arm, lane, m);
  const approach = densify([inLaneCentre(arm, lane, ARM_LEN[arm]), inLaneCentre(arm, lane, BOX)], STEP);
  const conn = densify(c.pts, STEP);
  const exit = densify([outLaneCentre(c.outArm, c.outLane, BOX), outLaneCentre(c.outArm, c.outLane, ARM_LEN[c.outArm])], STEP);
  const pts = [...approach, ...conn.slice(1), ...exit.slice(1)];
  const s = [0];
  for (let i = 1; i < pts.length; i++) s.push(s[i - 1] + Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]));
  const connStart = s[approach.length - 1];
  const connEnd = s[approach.length + conn.length - 2];
  return { pts, s, len: s[s.length - 1], connStart, connEnd, stopS: ARM_LEN[arm] - STOP, move: m };
}

function routeAt(r: Route, sv: number): { p: V2; n: V2; yaw: number } {
  const x = Math.min(Math.max(sv, 0), r.len - 1e-6);
  let lo = 0;
  let hi = r.s.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (r.s[mid] <= x) lo = mid;
    else hi = mid;
  }
  const f = (x - r.s[lo]) / (r.s[hi] - r.s[lo] || 1);
  const a = r.pts[lo];
  const b = r.pts[hi];
  const tx = b[0] - a[0];
  const ty = b[1] - a[1];
  const L = Math.hypot(tx, ty) || 1;
  return { p: [a[0] + tx * f, a[1] + ty * f], n: [-ty / L, tx / L], yaw: Math.atan2(ty, tx) };
}

// ----------------------------------------------------------------------------- actors
interface Spec {
  id: string;
  role: "ego" | "npc";
  category: string;
  color: string;
  arm: number;
  lane: number; // lane the route is built from
  s0: number; // distance from the junction centre at t=0
  v0: number;
  move: Move;
  leader?: string;
}

const LWH: Record<string, [number, number, number]> = {
  car: [4.6, 1.9, 1.5],
  suv: [4.9, 2.0, 1.8],
  van: [5.4, 2.0, 2.1],
  truck: [8.5, 2.5, 3.3],
  bus: [12, 2.55, 3.2],
};

const SPECS: Spec[] = [
  { id: "ego", role: "ego", category: "car", color: "white", arm: 2, lane: 1, s0: 140, v0: 12.5, move: "left" },
  { id: "lead_right", role: "npc", category: "van", color: "blue", arm: 2, lane: 2, s0: 115, v0: 12, move: "straight" },
  { id: "follow_left", role: "npc", category: "suv", color: "black", arm: 2, lane: 1, s0: 175, v0: 11, move: "left", leader: "ego" },
  { id: "oncoming_1", role: "npc", category: "car", color: "red", arm: 0, lane: 2, s0: 120, v0: 13, move: "straight" },
  { id: "oncoming_2", role: "npc", category: "truck", color: "white", arm: 0, lane: 2, s0: 165, v0: 11.5, move: "straight", leader: "oncoming_1" },
  { id: "cross_north", role: "npc", category: "car", color: "grey", arm: 1, lane: 1, s0: 60, v0: 8, move: "straight" },
  { id: "cross_south", role: "npc", category: "bus", color: "yellow", arm: 3, lane: 2, s0: 70, v0: 8, move: "right" },
];

export type SigState = "green" | "yellow" | "red";

export interface SceneOptions {
  seed?: number;
  egoSpeed?: number;
  lcStart?: number;
  duration?: number;
  /** Time at which arm 2 turns yellow (green runs out). */
  greenUntil?: number;
}

export interface SceneMetrics {
  min_ttc: number;
  min_ttc_actor: string;
  collision: boolean;
  max_decel: number;
  max_jerk: number;
  max_speed: number;
  turns: { left: number; right: number };
  lane_changes: number;
  red_light_violations: number;
  ego_distance_m: number;
  ego_mean_speed: number;
  duration_s: number;
}

function mulberry32(a: number) {
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const ease = (x: number) => 0.5 - 0.5 * Math.cos(Math.PI * Math.min(1, Math.max(0, x)));

interface Rect {
  x: number;
  y: number;
  yaw: number;
  l: number;
  w: number;
}
function rectsOverlap(a: Rect, b: Rect): boolean {
  const axes = (r: Rect): V2[] => [
    [Math.cos(r.yaw), Math.sin(r.yaw)],
    [-Math.sin(r.yaw), Math.cos(r.yaw)],
  ];
  const corners = (r: Rect): V2[] => {
    const c = Math.cos(r.yaw);
    const s = Math.sin(r.yaw);
    return [
      [r.x + (c * r.l) / 2 - (s * r.w) / 2, r.y + (s * r.l) / 2 + (c * r.w) / 2],
      [r.x + (c * r.l) / 2 + (s * r.w) / 2, r.y + (s * r.l) / 2 - (c * r.w) / 2],
      [r.x - (c * r.l) / 2 + (s * r.w) / 2, r.y - (s * r.l) / 2 - (c * r.w) / 2],
      [r.x - (c * r.l) / 2 - (s * r.w) / 2, r.y - (s * r.l) / 2 + (c * r.w) / 2],
    ];
  };
  const ca = corners(a);
  const cb = corners(b);
  for (const ax of [...axes(a), ...axes(b)]) {
    const pa = ca.map((p) => p[0] * ax[0] + p[1] * ax[1]);
    const pb = cb.map((p) => p[0] * ax[0] + p[1] * ax[1]);
    if (Math.max(...pa) < Math.min(...pb) || Math.max(...pb) < Math.min(...pa)) return false;
  }
  return true;
}

export function generateScene(opts: SceneOptions = {}): { scene: Scene; metrics: SceneMetrics } {
  const rnd = mulberry32(opts.seed ?? 7);
  const egoSpeed = opts.egoSpeed ?? 12.5;
  const lcStart = opts.lcStart ?? 1.5;
  const duration = opts.duration ?? 22;
  const greenUntil = opts.greenUntil ?? 17;
  const LC_DUR = 4;
  const SAMPLE_DT = 0.1;
  const SIM_DT = 0.05;

  const map = buildMap();
  const sigState = (arm: number, t: number): SigState => {
    if (arm === 2) return t < greenUntil ? "green" : t < greenUntil + 3 ? "yellow" : "red";
    return "red";
  };

  interface Sim {
    spec: Spec;
    route: Route;
    len: number;
    s: number;
    v: number;
    a: number;
    track: number[][];
  }
  const sims: Sim[] = SPECS.map((spec) => {
    const route = buildRoute(spec.arm, spec.lane, spec.move);
    const v0 = spec.id === "ego" ? egoSpeed : spec.v0 * (1 + (rnd() - 0.5) * 0.02);
    return { spec, route, len: LWH[spec.category][0], s: ARM_LEN[spec.arm] - spec.s0, v: v0, a: 0, track: [] };
  });
  const byId = new Map(sims.map((s) => [s.spec.id, s]));

  const nSamples = Math.round(duration / SAMPLE_DT) + 1;
  const ts: number[] = [];
  const stepsPerSample = Math.round(SAMPLE_DT / SIM_DT);
  const egoSlowAt: { t?: number } = {};
  const prevYaw = new Map<string, number>();

  const latOf = (sim: Sim, t: number): number =>
    sim.spec.id === "ego" ? -LANE_W * (1 - ease((t - lcStart) / LC_DUR)) : 0;

  const record = (t: number) => {
    for (const sim of sims) {
      const ra = routeAt(sim.route, sim.s);
      const lat = latOf(sim, t);
      const px = ra.p[0] + ra.n[0] * lat;
      const py = ra.p[1] + ra.n[1] * lat;
      // yaw from route tangent plus the lateral-offset slope (so the ego visibly angles into the new lane)
      const dl = (latOf(sim, t + 0.05) - latOf(sim, t - 0.05)) / 0.1;
      const slope = sim.v > 0.5 ? Math.atan2(dl, Math.max(sim.v, 0.5)) : 0;
      let yaw = ra.yaw + slope;
      if (sim.v < 0.3 && prevYaw.has(sim.spec.id)) yaw = prevYaw.get(sim.spec.id) as number;
      prevYaw.set(sim.spec.id, yaw);
      sim.track.push([r3(px), r3(py), r3(yaw), r3(sim.v)]);
    }
    ts.push(r3(t));
  };

  let t = 0;
  record(0);
  for (let k = 1; k < nSamples; k++) {
    for (let q = 0; q < stepsPerSample; q++) {
      for (const sim of sims) {
        const { spec } = sim;
        const distToStop = sim.route.stopS - (sim.s + sim.len / 2); // from front bumper
        const cap = spec.move === "left" ? 5.8 : spec.move === "right" ? 4.2 : 99;
        let vt = spec.v0;
        if (spec.id === "ego") {
          vt = egoSpeed;
          if (distToStop < 45 && sim.s < sim.route.connEnd) {
            vt = 7;
            if (egoSlowAt.t === undefined) egoSlowAt.t = t;
          }
          if (sim.s >= sim.route.connEnd) vt = 9.5;
        }
        if (sim.s > sim.route.connStart - 14 && sim.s < sim.route.connEnd + 1) vt = Math.min(vt, cap);
        let a: number;
        if (spec.id === "ego") {
          const cmd = Math.max(-1.5, Math.min(1.5, (vt - sim.v) * 1.2));
          a = sim.a + Math.max(-2 * SIM_DT, Math.min(2 * SIM_DT, cmd - sim.a)); // jerk limit 2 m/s^3
        } else {
          let gap = 1e9;
          let vl = 0;
          if (spec.leader) {
            const ld = byId.get(spec.leader);
            if (ld) {
              const g = ld.s - ld.len / 2 - (sim.s + sim.len / 2);
              if (g > -5) {
                gap = g;
                vl = ld.v;
              }
            }
          }
          const st = sigState(spec.arm, t);
          const mustStop = st === "red" || (st === "yellow" && distToStop > (sim.v * sim.v) / 6);
          if (mustStop && distToStop > -0.5 && distToStop < gap) {
            gap = Math.max(0.1, distToStop + 1.3);
            vl = 0;
          }
          const amax = 2.0;
          const b = 2.5;
          const free = amax * (1 - Math.pow(sim.v / Math.max(vt, 0.1), 4));
          const sStar = 2.5 + Math.max(0, sim.v * 1.4 + (sim.v * (sim.v - vl)) / (2 * Math.sqrt(amax * b)));
          a = gap < 1e8 ? free - amax * Math.pow(sStar / Math.max(gap, 0.1), 2) : free;
          a = Math.max(-6, Math.min(amax, a));
        }
        const nv = Math.max(0, sim.v + a * SIM_DT);
        sim.s = Math.min(sim.route.len - 1, sim.s + ((sim.v + nv) / 2) * SIM_DT);
        sim.v = nv;
        sim.a = a;
      }
      t += SIM_DT;
    }
    record(k * SAMPLE_DT);
  }

  // ------------------------------------------------------------------ signals
  const signals = map.signals.map((sg) => ({ ...sg, states: ts.map((tt) => sigState(sg.arm, tt)) }));

  // ------------------------------------------------------------------ events
  const events: Scene["events"] = [];
  const ego = byId.get("ego") as Sim;
  const egoTrack = ego.track;
  events.push({ t: r3(lcStart), type: "lane_change_start", id: "ego_lc", actor: "ego" });
  events.push({ t: r3(lcStart + LC_DUR), type: "lane_change_end", id: "ego_lc", actor: "ego" });
  if (egoSlowAt.t !== undefined) events.push({ t: r3(egoSlowAt.t), type: "speed_change", id: "ego_slow", actor: "ego" });
  events.push({ t: r3(greenUntil), type: "signal_yellow", id: "arm2" });
  events.push({ t: r3(greenUntil + 3), type: "signal_red", id: "arm2" });

  // junction / stop line crossings
  const sSeries = (sim: Sim): number[] => {
    // recover s per sample from the track by nearest route point search is expensive; re-simulate cheaply instead
    const out: number[] = [];
    let idx = 0;
    for (const row of sim.track) {
      let best = idx;
      let bd = Infinity;
      for (let i = idx; i < Math.min(sim.route.pts.length, idx + 160); i++) {
        const d = (sim.route.pts[i][0] - row[0]) ** 2 + (sim.route.pts[i][1] - row[1]) ** 2;
        if (d < bd) {
          bd = d;
          best = i;
        }
      }
      idx = best;
      out.push(sim.route.s[best]);
    }
    return out;
  };
  let redViolations = 0;
  for (const sim of sims) {
    const ss = sSeries(sim);
    let crossed = false;
    let stopped = false;
    let inConn = false;
    for (let i = 1; i < ss.length; i++) {
      const front = ss[i] + sim.len / 2;
      if (!crossed && front >= sim.route.stopS && ss[i - 1] + sim.len / 2 < sim.route.stopS) {
        crossed = true;
        events.push({ t: ts[i], type: "stop_line_cross", actor: sim.spec.id });
        if (sigState(sim.spec.arm, ts[i]) === "red") redViolations++;
      }
      if (!stopped && sim.track[i][3] < 0.15 && sim.track[i - 1][3] >= 0.15 && sim.route.stopS - front < 6) {
        stopped = true;
        events.push({ t: ts[i], type: "stopped_at_line", actor: sim.spec.id });
      }
      if (sim.spec.id === "ego") {
        const inside = ss[i] >= sim.route.connStart && ss[i] <= sim.route.connEnd;
        if (inside && !inConn) events.push({ t: ts[i], type: "turn_start", id: "ego_turn", actor: "ego" });
        if (!inside && inConn) events.push({ t: ts[i], type: "turn_end", id: "ego_turn", actor: "ego" });
        inConn = inside;
      }
    }
  }

  // ------------------------------------------------------------------ metrics
  let minTtc = 99;
  let minTtcT = 0;
  let minTtcActor = "";
  let collision = false;
  const egoL = LWH.car;
  for (let i = 0; i < ts.length; i++) {
    const e = egoTrack[i];
    const ev: V2 = [Math.cos(e[2]) * e[3], Math.sin(e[2]) * e[3]];
    for (const sim of sims) {
      if (sim.spec.id === "ego") continue;
      const o = sim.track[i];
      const dx = o[0] - e[0];
      const dy = o[1] - e[1];
      if (dx * dx + dy * dy > 90 * 90) continue;
      if (rectsOverlap({ x: e[0], y: e[1], yaw: e[2], l: egoL[0], w: egoL[1] }, { x: o[0], y: o[1], yaw: o[2], l: sim.len, w: LWH[sim.spec.category][1] })) collision = true;
      const ov: V2 = [Math.cos(o[2]) * o[3], Math.sin(o[2]) * o[3]];
      const w: V2 = [ov[0] - ev[0], ov[1] - ev[1]];
      const R = (egoL[1] + LWH[sim.spec.category][1]) / 2 + 0.9;
      const A = w[0] * w[0] + w[1] * w[1];
      const B = 2 * (dx * w[0] + dy * w[1]);
      const C = dx * dx + dy * dy - R * R;
      if (C <= 0 || A < 1e-6 || B >= 0) continue;
      const disc = B * B - 4 * A * C;
      if (disc < 0) continue;
      const tau = (-B - Math.sqrt(disc)) / (2 * A);
      if (tau >= 0 && tau < minTtc) {
        minTtc = tau;
        minTtcT = ts[i];
        minTtcActor = sim.spec.id;
      }
    }
  }
  if (minTtc < 99) events.push({ t: r3(minTtcT), type: "min_ttc", actor: minTtcActor });
  let maxDecel = 0;
  let maxJerk = 0;
  let maxSpeed = 0;
  let dist = 0;
  let prevA = 0;
  for (let i = 1; i < egoTrack.length; i++) {
    const a = (egoTrack[i][3] - egoTrack[i - 1][3]) / SAMPLE_DT;
    maxDecel = Math.max(maxDecel, -a);
    if (i > 1) maxJerk = Math.max(maxJerk, Math.abs((a - prevA) / SAMPLE_DT));
    prevA = a;
    maxSpeed = Math.max(maxSpeed, egoTrack[i][3]);
    dist += Math.hypot(egoTrack[i][0] - egoTrack[i - 1][0], egoTrack[i][1] - egoTrack[i - 1][1]);
  }
  events.sort((a, b) => a.t - b.t);

  const scene: Scene = {
    bounds: map.bounds,
    lanes: map.lanes,
    stop_lines: map.stop_lines,
    signals,
    actors: SPECS.map((s) => ({ id: s.id, role: s.role, category: s.category, lwh: LWH[s.category], color: s.color })),
    t: ts,
    tracks: Object.fromEntries(sims.map((s) => [s.spec.id, s.track])),
    events,
  };
  return {
    scene,
    metrics: {
      min_ttc: r3(minTtc),
      min_ttc_actor: minTtcActor,
      collision,
      max_decel: r3(maxDecel),
      max_jerk: r3(maxJerk),
      max_speed: r3(maxSpeed),
      turns: { left: 1, right: 0 },
      lane_changes: 1,
      red_light_violations: redViolations,
      ego_distance_m: r3(dist),
      ego_mean_speed: r3(dist / duration),
      duration_s: duration,
    },
  };
}
