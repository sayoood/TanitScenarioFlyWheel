/**
 * Typed contract for the Flywheel API v0.1 (docs/API.md).
 * The first block mirrors the document 1:1. Everything below "UI-side extensions" is either
 * unspecified in API.md (Lesson, Revision, PipelineConfig, event payloads) or UI-only; those
 * shapes follow src/tsf/agents/harness.py and are parsed defensively.
 */

// ----------------------------------------------------------------------------- health
/** A NEGATIVE real-world incident available for UC3 (titles only; records stay on the Thor). */
export type IncidentRef = { incident_id: string; title: string; has_cf_positive: boolean };

export interface ThorHealth {
  host: string;
  mem_total_gb: number;
  mem_used_gb: number;
  gpu_util: number; // 0..1
  power_w: number;
  disk_free_gb: number;
  /** Optional extensions (not in v0.1): used for gauge scaling when present. */
  disk_total_gb?: number;
  power_max_w?: number;
}

export interface Health {
  thor: ThorHealth;
  services: {
    llm: { up: boolean; model?: string; tok_s?: number };
    cosmos_sim: { up: boolean; model?: string };
    cosmos_transfer: { up: boolean; model?: string };
    scheduler: { queue: number; running?: string | null };
  };
}

// ----------------------------------------------------------------------------- runs
export type UseCaseId = "uc1" | "uc2" | "uc3" | "uc4" | "uc5";

export type StageName =
  | "analyse"
  | "propose"
  | "verify"
  | "repair"
  | "document"
  | "simulate"
  | "wsm"
  | "cosmos_transfer"
  | "augment"
  | "closed_loop";

export type StageStatus = "pending" | "running" | "ok" | "failed" | "skipped";

export interface Stage {
  name: StageName;
  status: StageStatus;
  started?: string;
  seconds?: number;
  detail?: string;
}

export type RunStatus = "queued" | "running" | "ok" | "failed";

export interface RunSummary {
  id: string;
  uc: string;
  title: string;
  status: RunStatus;
  created: string;
  seconds?: number;
  score?: number;
  kpis?: Record<string, number | boolean>;
}

export type VerifierLevel = "L0" | "L1" | "L2" | "L3" | "L4" | "L5";

export interface Finding {
  level: VerifierLevel;
  check: string;
  ok: boolean;
  message: string;
}

export interface Attempt {
  index: number;
  passed: boolean;
  score: number;
  findings: Finding[];
  seconds: number;
}

export type ArtifactKind = "xodr" | "xosc" | "osc2" | "package" | "video" | "image" | "json";

export interface Artifact {
  name: string;
  kind: ArtifactKind;
  size: number;
}

export interface RunDetail extends RunSummary {
  input: any;
  stages: Stage[];
  attempts: Attempt[];
  metrics: Record<string, any>;
  artifacts: Artifact[];
  cameras?: { name: string; artifact: string }[];
  package?: any;
  llm?: { calls: number; tok_per_s: number };
}

export interface Scene {
  /** [xmin, ymin, xmax, ymax] in metres (x east, y north, z up). */
  bounds: [number, number, number, number];
  lanes: { id: string; junction: boolean; left: number[][]; right: number[][]; center: number[][] }[];
  stop_lines: number[][][];
  signals: { arm: number; x: number; y: number; states: string[] }[];
  actors: { id: string; role: "ego" | "npc"; category: string; lwh: [number, number, number]; color: string }[];
  t: number[];
  /** actor id -> [[x, y, yaw, v], ...] sampled at `t`. */
  tracks: Record<string, number[][]>;
  events: { t: number; type: string; id?: string; actor?: string }[];
}

// ----------------------------------------------------------------------------- run creation
export interface AugmentOptions {
  n: number;
  strategies: string[];
}
export interface RenderOptions {
  views: number;
  res: string;
  frames: number;
  lidar: boolean;
}
export interface ClosedLoopOptions {
  policy: "refe_sft4" | "const_vel" | "accel_decel" | (string & {});
  steps: number;
}
export interface RunOptions {
  augment?: AugmentOptions;
  render?: RenderOptions;
  closed_loop?: ClosedLoopOptions;
}

export type RunInput =
  | { query: string }
  | { video: string }
  | { incident_id: string }
  | { snippet: string };

export interface CreateRunRequest {
  uc: UseCaseId;
  input: RunInput;
  options: RunOptions;
}

// ----------------------------------------------------------------------------- harness
export type LessonKind = "INVARIANT" | "DOMAIN";
export type LessonStatus = "candidate" | "active" | "retired";

export interface Lesson {
  id: string;
  scope: string;
  applies_to: string[];
  lesson: string;
  kind: LessonKind;
  status: LessonStatus;
  evidence: { delta_score?: number; n?: number; runs?: string[]; [k: string]: unknown };
  created: number; // epoch seconds
}

export type RevisionVerdict = "adopt" | "merge" | "keep" | "reject" | (string & {});

export interface Revision {
  revision_id: string;
  agent: string;
  editor_type: string; // "invariant" | "domain" | "compose" ...
  target: string;
  proposal: string;
  verdict: RevisionVerdict;
  /** e.g. {reference, invariant, domain, composed, n} (all optional). */
  evidence: Record<string, any>;
  t: number; // epoch seconds
}

export interface Generation {
  gen: number;
  score: number;
  valid_rate: number;
}

export type AgentKnobs = Record<string, number | boolean | string>;

export interface HarnessState {
  agents: string[];
  knobs: Record<string, AgentKnobs>;
  lessons: Lesson[];
  revisions: Revision[];
  generations: Generation[];
}

// ----------------------------------------------------------------------------- kpis
export type KpiSeriesName =
  | "first_pass_valid_rate"
  | "time_to_valid_s"
  | "tok_s"
  | "cosmos_s_per_frame"
  | "sim_rtf";

export interface Kpis {
  series: Record<string, [string, number][]>;
  latest: Record<string, number>;
}

// ----------------------------------------------------------------------------- config
export interface PipelineConfig {
  rig?: {
    cameras?: { name: string; fov_deg?: number; yaw_deg?: number; tele?: boolean }[];
    lidar?: { beams?: number; hz?: number };
  };
  render?: Partial<RenderOptions>;
  augmentation?: {
    default_n?: number;
    strategies?: { id: string; label?: string; description?: string }[];
  };
  closed_loop?: {
    default_steps?: number;
    policies?: { id: string; label?: string; description?: string }[];
  };
  [k: string]: unknown;
}

// ----------------------------------------------------------------------------- events
export interface FlywheelEvent {
  type: "stage" | "run" | "log" | "health";
  run_id?: string;
  [k: string]: unknown;
}

// ----------------------------------------------------------------------------- UI-side constants
export const STAGE_ORDER: StageName[] = [
  "analyse",
  "propose",
  "verify",
  "repair",
  "document",
  "simulate",
  "wsm",
  "cosmos_transfer",
  "augment",
  "closed_loop",
];

export const LEVELS: VerifierLevel[] = ["L0", "L1", "L2", "L3", "L4", "L5"];

export const CAMERA_NAMES = [
  "front_wide",
  "front_tele",
  "cross_left",
  "cross_right",
  "rear_left",
  "rear_right",
  "rear_tele",
] as const;
export type CameraName = (typeof CAMERA_NAMES)[number];
