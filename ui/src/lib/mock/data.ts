/** Harness, KPI and config fixtures for the mock layer. */
import type { Generation, HarnessState, Kpis, Lesson, PipelineConfig, Revision } from "../types";
import { mulberry32 } from "./runs";

const DAY = 86400;
const nowS = () => Math.floor(Date.now() / 1000);

export function makeHarness(): HarnessState {
  const n = nowS();
  const L = (
    id: string,
    kind: Lesson["kind"],
    status: Lesson["status"],
    scope: string,
    applies_to: string[],
    lesson: string,
    delta: number,
    nRuns: number,
    ageDays: number,
  ): Lesson => ({
    id,
    scope,
    applies_to,
    lesson,
    kind,
    status,
    evidence: { delta_score: delta, n: nRuns, runs: ["run_7f3a", "run_5c10", "run_21de"].slice(0, Math.min(3, nRuns)) },
    created: n - ageDays * DAY,
  });
  const lessons: Lesson[] = [
    L("exp_a01", "INVARIANT", "active", "lifter", ["junction", "turn", "left", "right"], "Every inbound lane that an actor routes through needs a matching outbound lane on the target arm; declare driving_lanes on both sides of each approach.", 0.12, 18, 12),
    L("exp_a02", "INVARIANT", "active", "*", ["lane", "lane_change"], "Count lane indices from the centre line outward (1 = innermost). Never count from the kerb.", 0.09, 14, 12),
    L("exp_a03", "INVARIANT", "active", "lifter", [], "Declare speeds in m/s. Convert km/h limits before writing initial_speed or target_speed.", 0.06, 22, 11),
    L("exp_a04", "INVARIANT", "active", "lifter", ["spawn", "initial"], "Never spawn two actors with overlapping bounding boxes at t=0; keep at least 6 m between centres on one lane.", 0.08, 11, 9),
    L("exp_a05", "INVARIANT", "candidate", "lifter", ["assertions"], "Assertions and parameters may only reference action and actor ids that exist in the same TSIR.", 0.05, 6, 3),
    L("exp_a06", "INVARIANT", "active", "counterfactual", ["twin", "incident"], "A positive twin edits ego behaviour only: same ODD, same actors, same map. Report the minimal edit distance.", 0.1, 9, 8),
    L("exp_a07", "INVARIANT", "candidate", "*", ["follow"], "Prefer keep_time_gap over a fixed distance for follow actions so the gap scales with speed.", 0.04, 5, 2),
    L("exp_d01", "DOMAIN", "active", "lifter", ["traffic_light", "signal", "left", "protected"], "For protected left turns set signal_plan.protected_left=true and choose the phase offset so the ego reaches the stop line inside the green window.", 0.07, 13, 10),
    L("exp_d02", "DOMAIN", "active", "lifter", ["lane_change", "intersection"], "Finish a lane change at least 60 m before the stop line at 12 m/s; otherwise the oracle sees an aborted manoeuvre.", 0.04, 8, 7),
    L("exp_d03", "DOMAIN", "active", "lifter", ["cut_in", "highway"], "Cut-in scenarios are informative when TTC at cut-in start is 1.5 to 3 s; larger gaps produce uneventful data.", 0.05, 10, 6),
    L("exp_d04", "DOMAIN", "candidate", "augmenter", ["night", "dusk", "rain"], "At dusk and night set headlights on and mention them in the Cosmos prompt; raindrops need wet road_surface.", 0.03, 5, 2),
    L("exp_d05", "DOMAIN", "candidate", "lifter", ["pedestrian", "occlusion"], "For occluded pedestrian crossings place the occluder at least 3 m before the crossing point.", 0.02, 4, 1),
    L("exp_d06", "DOMAIN", "retired", "lifter", ["fog"], "In fog reduce speed_limit_kph by 20 percent to satisfy L5 plausibility.", -0.01, 9, 14),
    L("exp_d07", "DOMAIN", "retired", "lifter", ["truck", "bus"], "Use category 'truck' for every vehicle longer than 7 m.", 0.0, 7, 13),
  ];
  const rev = (
    i: number,
    agent: string,
    editor: string,
    target: string,
    proposal: string,
    verdict: Revision["verdict"],
    ev: Record<string, unknown>,
    ageDays: number,
  ): Revision => ({ revision_id: `rev_${(0x3a7c1d0 + i * 0x9e3b).toString(16).slice(0, 10)}`, agent, editor_type: editor, target, proposal, verdict, evidence: ev, t: n - ageDays * DAY });
  const revisions: Revision[] = [
    rev(14, "lifter", "invariant", "guardrails.md", "Add rule: outbound lane count must cover every routed turn.", "adopt", { reference: 0.71, invariant: 0.8, domain: 0.73, composed: 0.8, n: 24 }, 0.4),
    rev(13, "lifter", "domain", "tool_guidance.md", "Document protected-left phase offset recipe with a worked example.", "merge", { reference: 0.78, invariant: 0.79, domain: 0.82, composed: 0.85, n: 24 }, 1.3),
    rev(12, "counterfactual", "invariant", "system_prompt.md", "Restrict the twin to ego-only edits and print edit distance.", "adopt", { reference: 0.64, invariant: 0.77, domain: 0.66, composed: 0.77, n: 18 }, 2.1),
    rev(11, "lifter", "domain", "skills/uc1_cut_in.json", "Add verified cut-in exemplar (TTC 2.1 s) as few-shot skill.", "keep", { reference: 0.8, invariant: 0.79, domain: 0.8, composed: 0.8, n: 24 }, 3.4),
    rev(10, "augmenter", "domain", "tool_guidance.md", "Teach weather sweep to keep road_surface consistent with rain.", "adopt", { reference: 0.69, invariant: 0.7, domain: 0.76, composed: 0.76, n: 16 }, 4.2),
    rev(9, "lifter", "invariant", "system_prompt.md", "Force knobs.thinking=true for junction topologies.", "keep", { reference: 0.76, invariant: 0.74, domain: 0.77, composed: 0.77, n: 20 }, 5.5),
    rev(8, "lifter", "invariant", "guardrails.md", "Convert all speeds to m/s before emitting TSIR.", "adopt", { reference: 0.66, invariant: 0.72, domain: 0.67, composed: 0.72, n: 20 }, 7.1),
    rev(7, "lifter", "domain", "skills/uc1_left_turn_intersection.json", "Swap in repaired left-turn exemplar (run_7f3a, attempt 1).", "merge", { reference: 0.6, invariant: 0.63, domain: 0.69, composed: 0.71, n: 16 }, 9.0),
    rev(6, "judge", "domain", "system_prompt.md", "Tighten L4 rubric: weather and time of day are hard constraints.", "reject", { reference: 0.71, invariant: 0.7, domain: 0.64, composed: 0.71, n: 12 }, 10.6),
    rev(5, "lifter", "invariant", "tool_guidance.md", "Lane indices count from the centre line.", "adopt", { reference: 0.55, invariant: 0.64, domain: 0.56, composed: 0.64, n: 16 }, 12.2),
  ];
  const rnd = mulberry32(42);
  const generations: Generation[] = [];
  let s = 0.55;
  let v = 0.5;
  for (let g = 0; g < 15; g++) {
    s = Math.min(0.94, s + 0.018 + rnd() * 0.02);
    v = Math.min(0.96, v + 0.02 + rnd() * 0.025);
    generations.push({ gen: g, score: Math.round(s * 1000) / 1000, valid_rate: Math.round(v * 1000) / 1000 });
  }
  return {
    agents: ["lifter", "counterfactual", "augmenter", "judge"],
    knobs: {
      lifter: { thinking: false, temperature: 0.3, max_repairs: 3, n_skills: 1, n_lessons: 6 },
      counterfactual: { thinking: true, temperature: 0.2, max_repairs: 2, n_skills: 1, n_lessons: 4 },
      augmenter: { thinking: false, temperature: 0.6, max_repairs: 2, n_skills: 2, n_lessons: 4 },
      judge: { thinking: true, temperature: 0.0, max_repairs: 0, n_skills: 0, n_lessons: 2 },
    },
    lessons,
    revisions,
    generations,
  };
}

// ----------------------------------------------------------------------------- KPIs
export function makeKpis(): Kpis {
  const rnd = mulberry32(2026);
  const now = Date.now();
  const points = 14 * 4; // 14 days, every 6 h
  const stepMs = 6 * 3600 * 1000;
  const make = (from: number, to: number, noise: number, clampLo: number, clampHi: number, curve = 1): [string, number][] =>
    Array.from({ length: points }, (_, i) => {
      const u = i / (points - 1);
      const base = from + (to - from) * Math.pow(u, curve);
      const val = Math.min(clampHi, Math.max(clampLo, base + (rnd() - 0.5) * 2 * noise));
      return [new Date(now - (points - 1 - i) * stepMs).toISOString(), Math.round(val * 1000) / 1000] as [string, number];
    });
  const series = {
    first_pass_valid_rate: make(0.41, 0.74, 0.05, 0, 1, 0.8),
    time_to_valid_s: make(96, 41, 7, 10, 200, 0.9),
    tok_s: make(31.5, 34.8, 1.6, 20, 45),
    cosmos_s_per_frame: make(6.1, 4.1, 0.35, 1, 12, 0.7),
    sim_rtf: make(78, 106, 6, 20, 200, 0.85),
  };
  const last = (k: keyof typeof series) => series[k][series[k].length - 1][1];
  return {
    series,
    latest: {
      first_pass_valid_rate: last("first_pass_valid_rate"),
      time_to_valid_s: last("time_to_valid_s"),
      tok_s: last("tok_s"),
      cosmos_s_per_frame: last("cosmos_s_per_frame"),
      sim_rtf: last("sim_rtf"),
      valid_after_3_repairs: 0.96,
      l4_intent_score: 0.88,
    },
  };
}

// ----------------------------------------------------------------------------- config
export function makeConfig(): PipelineConfig {
  return {
    rig: {
      cameras: [
        { name: "front_wide", fov_deg: 110, yaw_deg: 0 },
        { name: "front_tele", fov_deg: 35, yaw_deg: 0, tele: true },
        { name: "cross_left", fov_deg: 110, yaw_deg: 65 },
        { name: "cross_right", fov_deg: 110, yaw_deg: -65 },
        { name: "rear_left", fov_deg: 110, yaw_deg: 145 },
        { name: "rear_right", fov_deg: 110, yaw_deg: -145 },
        { name: "rear_tele", fov_deg: 35, yaw_deg: 180, tele: true },
      ],
      lidar: { beams: 128, hz: 10 },
    },
    render: { views: 7, res: "480p", frames: 121, lidar: true },
    augmentation: {
      default_n: 8,
      strategies: [
        { id: "weather", label: "Weather" },
        { id: "time_of_day", label: "Time of day" },
        { id: "behavior", label: "Behaviour" },
        { id: "object", label: "Objects" },
        { id: "map", label: "Map" },
      ],
    },
    closed_loop: {
      default_steps: 200,
      policies: [
        { id: "refe_sft4", label: "REFe SFT-4" },
        { id: "const_vel", label: "Constant velocity" },
        { id: "accel_decel", label: "Accelerate / decelerate" },
      ],
    },
  };
}
