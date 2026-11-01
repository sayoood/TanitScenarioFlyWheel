/** Run fixtures for the mock layer: verifier attempts, stages, artifacts and RunDetail assembly. */
import { CAMERA_NAMES, LEVELS } from "../types";
import type {
  Artifact,
  Attempt,
  Finding,
  RunDetail,
  RunOptions,
  RunStatus,
  Stage,
  StageName,
  UseCaseId,
  VerifierLevel,
} from "../types";
import { generateScene, type SceneMetrics } from "./scene";

export function mulberry32(a: number) {
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function hashStr(s: string): number {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

// ----------------------------------------------------------------------------- verifier ladder fixtures
const CHECKS: Record<VerifierLevel, string[]> = {
  L0: ["tsir_schema", "referential_integrity"],
  L1: ["opendrive_xsd", "openscenario_xsd", "asam_qc_opendrive"],
  L2: ["xodr_readback", "junction_movements", "signal_plan_match"],
  L3: ["ego_lane_change", "ego_turn", "no_collision", "min_ttc_above", "red_light_respected", "max_decel_below"],
  L4: ["intent_fidelity"],
  L5: ["kinematic_limits", "jerk_limit", "ttc_distribution"],
};

interface Failure {
  level: VerifierLevel;
  check: string;
  message: string;
}

const FAILURES: Failure[] = [
  { level: "L2", check: "junction_movements", message: "1 movement(s) not realised: arm2 lane 1 -left-> arm1 lane 1 (arm 1 outbound declares driving_lanes=1)" },
  { level: "L3", check: "ego_lane_change", message: "ego ended in lane 2, expected a net lane change of 1 (action ego_lc starts at t=1.5 s but lead_right occupies the gap)" },
  { level: "L1", check: "opendrive_xsd", message: "line 41: Element 'lane': Missing child element(s). Expected is one of ( link, border, width )." },
  { level: "L3", check: "min_ttc_above", message: "min TTC 0.84 s < 1.5 s (ego vs oncoming_1 at t=11.2 s)" },
  { level: "L0", check: "tsir_schema", message: "/actions/3/start/kind: Input should be 'sim_time', 'after_action', 'distance_to_actor', 'ttc' or 'distance_to_stop_line'" },
  { level: "L4", check: "intent_fidelity", message: "judge score 0.58 < 0.70: query asks for day and good weather, TSIR environment has weather='cloudy', time_of_day='17:45'" },
  { level: "L3", check: "red_light_respected", message: "cross_north enters the junction at t=9.4 s while arm 1 shows red" },
];

const PASS_MESSAGES: Record<string, string> = {
  tsir_schema: "valid against TSIR v0.1",
  referential_integrity: "7 actors, 6 actions, 6 assertions resolved",
  opendrive_xsd: "OpenDRIVE 1.8 XSD ok",
  openscenario_xsd: "OpenSCENARIO 1.3 XSD ok",
  asam_qc_opendrive: "0 errors, 2 warnings",
  xodr_readback: "TanitSim parsed 4 roads + 16 connecting roads",
  junction_movements: "16/16 movements realised",
  signal_plan_match: "phase plan equals TSIR (25 / 3 / 2 s)",
  ego_lane_change: "net lane change +1 at t=5.5 s",
  ego_turn: "left turn completed at t=14.8 s",
  no_collision: "no OBB overlap in 22.0 s",
  min_ttc_above: "min TTC 2.55 s >= 1.5 s",
  red_light_respected: "0 violations",
  max_decel_below: "max decel 1.5 m/s^2 < 4.0",
  intent_fidelity: "judge score 0.91 >= 0.70",
  kinematic_limits: "speed, accel within limits",
  jerk_limit: "max jerk 2.1 m/s^3",
  ttc_distribution: "within naturalistic prior (KS 0.12)",
};

function weight(l: VerifierLevel): number {
  return { L0: 1, L1: 2, L2: 3, L3: 4 }[l as "L0" | "L1" | "L2" | "L3"] ?? 1;
}

export function makeAttempt(index: number, failure: Failure | null, rnd: () => number, metrics?: SceneMetrics): Attempt {
  const findings: Finding[] = [];
  for (const level of LEVELS) {
    for (const check of CHECKS[level]) {
      if (failure && failure.level === level && failure.check === check) {
        findings.push({ level, check, ok: false, message: failure.message });
        break;
      }
      let message = PASS_MESSAGES[check] ?? "ok";
      if (check === "min_ttc_above" && metrics) message = `min TTC ${metrics.min_ttc.toFixed(2)} s >= 1.5 s`;
      if (check === "max_decel_below" && metrics) message = `max decel ${metrics.max_decel.toFixed(1)} m/s^2 < 4.0`;
      if (check === "intent_fidelity") message = `judge score ${(0.86 + rnd() * 0.08).toFixed(2)} >= 0.70`;
      findings.push({ level, check, ok: true, message });
    }
    if (failure && failure.level === level) break; // first failing level stops the ladder
  }
  const tot = findings.reduce((a, f) => a + weight(f.level), 0) || 1;
  const ok = findings.filter((f) => f.ok).reduce((a, f) => a + weight(f.level), 0);
  // score is computed as the fraction of passed weighted checks over the checks that were actually run
  const score = failure ? Math.round((ok / (tot + 2)) * 1000) / 1000 : 1;
  const seconds = failure ? 3 + rnd() * 4 : 9 + rnd() * 5;
  return { index, passed: !failure, score, findings, seconds: Math.round(seconds * 10) / 10 };
}

// ----------------------------------------------------------------------------- stage plan
export interface PlannedStage {
  name: StageName;
  /** realistic duration shown in the UI (seconds) */
  seconds: number;
  /** wall-clock duration of the mock animation (ms) */
  wallMs: number;
  detail?: string;
  skip?: boolean;
}

export function planStages(uc: UseCaseId, options: RunOptions, nAttempts: number, outcome: RunStatus, rnd: () => number): PlannedStage[] {
  const r = (a: number, b: number) => Math.round((a + rnd() * (b - a)) * 10) / 10;
  const render = options.render;
  const frames = render?.frames ?? 121;
  const views = render?.views ?? 7;
  const perFrame = 3.6 + rnd() * 1.4;
  const plan: PlannedStage[] = [];
  const analyseDetail: Record<UseCaseId, string> = {
    uc1: "intent parsed, 5 lessons retrieved",
    uc2: "ViPE poses + depth, SAM3 tracks lifted, map fit",
    uc3: "IMDAD record loaded, incident reproduced in sim",
    uc4: "snippet replayed to TSIR (ego GT + tracked agents)",
    uc5: "erroneous trajectory embedded in negative TSIR",
  };
  plan.push({ name: "analyse", seconds: uc === "uc2" ? r(180, 320) : r(1.2, 3), wallMs: 1500, detail: analyseDetail[uc] });
  plan.push({ name: "propose", seconds: r(10, 20), wallMs: 3400, detail: uc === "uc3" || uc === "uc5" ? "Lifter + Counterfactual agent (positive twin)" : "Lifter agent, xgrammar-constrained TSIR" });
  const failed = outcome === "failed";
  plan.push({ name: "verify", seconds: r(4, 9) * Math.max(1, nAttempts), wallMs: 2400, detail: `${nAttempts} attempt${nAttempts === 1 ? "" : "s"}, ladder L0 to L5`, skip: false });
  plan.push({ name: "repair", seconds: nAttempts > 1 ? r(8, 14) * (nAttempts - 1) : 0, wallMs: 2800, detail: nAttempts > 1 ? `${nAttempts - 1} repair round${nAttempts > 2 ? "s" : ""} (Propose-Probe-Compose)` : "not needed", skip: nAttempts <= 1 });
  if (failed) return finishFailed(plan);
  plan.push({ name: "document", seconds: r(0.6, 1.4), wallMs: 1100, detail: "xodr + xosc 1.3 + osc2 + package" });
  plan.push({ name: "simulate", seconds: r(2.5, 5), wallMs: 2400, detail: "TanitSim 2D, 22.0 s at 104x real time" });
  const doRender = !!render;
  plan.push({ name: "wsm", seconds: doRender ? r(30, 60) : 0, wallMs: 3800, detail: doRender ? `RDS-HQ export, ${views} views${render?.lidar ? " + LiDAR range maps" : ""}` : "render disabled", skip: !doRender });
  plan.push({ name: "cosmos_transfer", seconds: doRender ? Math.round(frames * views * perFrame) : 0, wallMs: 7000, detail: doRender ? `Transfer-Auto, ${views} cams x ${frames} frames @ ${render?.res ?? "480p"}, ${perFrame.toFixed(1)} s/frame` : "render disabled", skip: !doRender });
  const aug = options.augment;
  plan.push({ name: "augment", seconds: aug && aug.n > 0 ? r(50, 140) : 0, wallMs: 4200, detail: aug && aug.n > 0 ? `${aug.n} variants: ${aug.strategies.join(", ")}` : "no augmentation requested", skip: !(aug && aug.n > 0) });
  const cl = options.closed_loop;
  plan.push({ name: "closed_loop", seconds: cl ? r(120, 260) : 0, wallMs: 5600, detail: cl ? `policy ${cl.policy}, ${cl.steps} steps` : "closed loop not requested", skip: !cl });
  return plan;
}

function finishFailed(plan: PlannedStage[]): PlannedStage[] {
  // after a failed verify/repair all remaining stages are skipped
  for (const n of ["document", "simulate", "wsm", "cosmos_transfer", "augment", "closed_loop"] as StageName[]) {
    plan.push({ name: n, seconds: 0, wallMs: 0, skip: true, detail: "blocked by failed verification" });
  }
  return plan;
}

// ----------------------------------------------------------------------------- RunDetail assembly
export interface BuildArgs {
  id: string;
  uc: UseCaseId;
  title: string;
  input: Record<string, unknown>;
  options: RunOptions;
  created: string;
  outcome: RunStatus; // "ok" | "failed"
  nAttempts: number;
  seed: number;
  failures?: number[]; // indices into FAILURES
}

export interface RunRecordBase {
  detail: RunDetail;
  plan: PlannedStage[];
  sceneMetrics?: SceneMetrics;
}

const KIND_SIZES = { xodr: 14_800, xosc: 21_300, osc2: 3_900, package: 9_400 } as const;

export function buildRun(a: BuildArgs): RunRecordBase {
  const rnd = mulberry32(a.seed);
  const failed = a.outcome === "failed";
  // scene is only meaningful for runs that got past verification
  const egoSpeed = 10.5 + rnd() * 3.5;
  const lcStart = 1 + rnd() * 2;
  const sceneRes = failed ? undefined : generateScene({ seed: a.seed, egoSpeed, lcStart });
  const m = sceneRes?.metrics;

  const attempts: Attempt[] = [];
  const fi = a.failures ?? [];
  for (let i = 0; i < a.nAttempts; i++) {
    const isLast = i === a.nAttempts - 1;
    const f = !isLast || failed ? FAILURES[fi[i] ?? (a.seed + i * 3) % FAILURES.length] : null;
    attempts.push(makeAttempt(i, f, rnd, m));
  }

  const plan = planStages(a.uc, a.options, a.nAttempts, a.outcome, rnd);
  const t0 = Date.parse(a.created);
  let cursor = t0;
  const stages: Stage[] = plan.map((p) => {
    const st: Stage = { name: p.name, status: p.skip ? "skipped" : "ok", detail: p.detail };
    if (!p.skip) {
      st.started = new Date(cursor).toISOString();
      st.seconds = p.seconds;
      cursor += p.seconds * 1000;
    }
    return st;
  });
  if (failed) {
    // the ladder ran to completion on every attempt; the pipeline stops when the repair budget is exhausted
    const rp = stages.find((s) => s.name === "repair");
    const last = attempts[attempts.length - 1]?.findings.find((f) => !f.ok);
    if (rp) {
      rp.status = "failed";
      rp.detail = `repair budget exhausted after ${a.nAttempts} attempts${last ? `, last failure ${last.level}:${last.check}` : ""}`;
    }
  }
  const total = stages.reduce((s, x) => s + (x.seconds ?? 0), 0);

  const id = a.id;
  const artifacts: Artifact[] = [];
  if (!failed) {
    artifacts.push({ name: `${id}.xodr`, kind: "xodr", size: KIND_SIZES.xodr });
    artifacts.push({ name: `${id}.xosc`, kind: "xosc", size: KIND_SIZES.xosc });
    artifacts.push({ name: `${id}.osc`, kind: "osc2", size: KIND_SIZES.osc2 });
    artifacts.push({ name: `${id}.package.json`, kind: "package", size: KIND_SIZES.package });
    artifacts.push({ name: "tsir.json", kind: "json", size: 6_200 });
    artifacts.push({ name: "bev_keyframe.png", kind: "image", size: 184_000 });
  } else {
    artifacts.push({ name: "tsir_attempt_last.json", kind: "json", size: 5_900 });
  }
  let cameras: RunDetail["cameras"];
  const views = a.options.render?.views ?? 0;
  if (!failed && views > 0) {
    cameras = CAMERA_NAMES.slice(0, views).map((n) => ({ name: n, artifact: `cam_${n}.mp4` }));
    for (const c of cameras) artifacts.push({ name: c.artifact, kind: "video", size: 3_400_000 + Math.round(rnd() * 900_000) });
    if (a.options.render?.lidar) artifacts.push({ name: "lidar_range.json", kind: "json", size: 2_100_000 });
  }

  const metrics: Record<string, any> = {};
  if (m) {
    metrics.min_ttc = m.min_ttc;
    metrics.min_ttc_actor = m.min_ttc_actor;
    metrics.collision = m.collision;
    metrics.max_decel = m.max_decel;
    metrics.max_jerk = m.max_jerk;
    metrics.turns = m.turns;
    metrics.lane_changes = m.lane_changes;
    metrics.red_light_violations = m.red_light_violations;
    metrics.ego_distance_m = m.ego_distance_m;
    metrics.ego_mean_speed = m.ego_mean_speed;
    metrics.sim = { duration_s: m.duration_s, real_time_factor: Math.round(96 + rnd() * 22), steps: Math.round(m.duration_s / 0.05) };
    if (views > 0) {
      metrics.cosmos = { s_per_frame: Math.round((3.6 + rnd() * 1.4) * 100) / 100, frames: a.options.render?.frames ?? 121, views, resolution: a.options.render?.res ?? "480p" };
    }
    if (a.options.closed_loop) {
      const pol = a.options.closed_loop.policy;
      metrics.closed_loop = {
        policy: pol,
        steps: a.options.closed_loop.steps,
        epdms: pol === "refe_sft4" ? 0.84 : pol === "const_vel" ? 0.61 : 0.67,
        collision_rate: pol === "const_vel" ? 0.12 : 0,
        policy_latency_ms: pol === "refe_sft4" ? 38 : 1,
      };
    }
  }

  const finalScore = attempts[attempts.length - 1]?.score ?? 0;
  const firstPass = attempts[0]?.passed ?? false;
  const detail: RunDetail = {
    id,
    uc: a.uc,
    title: a.title,
    status: a.outcome,
    created: a.created,
    seconds: Math.round(total * 10) / 10,
    score: finalScore,
    kpis: {
      first_pass: firstPass,
      attempts: a.nAttempts,
      time_to_valid_s: Math.round(attempts.reduce((s, x) => s + x.seconds, 0) + (stages[0]?.seconds ?? 0) + (stages[1]?.seconds ?? 0)),
      ...(m ? { min_ttc: m.min_ttc, collision: m.collision } : {}),
    },
    input: a.input,
    stages,
    attempts,
    metrics,
    artifacts,
    cameras,
    llm: { calls: 2 + a.nAttempts * 2, tok_per_s: Math.round((33 + rnd() * 3.5) * 10) / 10 },
  };
  return { detail, plan, sceneMetrics: m };
}

export { FAILURES };
