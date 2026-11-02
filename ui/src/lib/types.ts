/**
 * Typed contract for the Flywheel API v0.1 (docs/API.md).
 * The first block mirrors the document 1:1. Everything below "UI-side extensions" is either
 * unspecified in API.md (Lesson, Revision, PipelineConfig, event payloads) or UI-only; those
 * shapes follow src/tsf/agents/harness.py and are parsed defensively.
 */

// ----------------------------------------------------------------------------- health
/** A NEGATIVE real-world incident available for UC3 (titles only; records stay on the Thor). */
export type IncidentRef = { incident_id: string; title: string; has_cf_positive: boolean };

/** Evidence video of a positive result (tsf.evidence). `public: false` = derived from a dataset or incident; served from the Thor only. */
export interface EvidenceItem {
  kind: "uc5" | "uc3" | "uc2" | "closed_loop" | "ablation";
  file: string;
  public: boolean;
  run?: string;
  source?: string;
  query?: string;
  q?: number;
  rep?: number;
  split?: string;
}

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
  | "brief"
  | "replay_negative"
  | "twin"
  | "lift"
  | "vipe"
  | "scene"
  | "sensors"
  | "parent"
  | "plan"
  | "edit"
  | "criticality"
  | "package"
  | "render"
  | "judge"
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

// ----------------------------------------------------------------------------- scenario augmentation
export type AugmentDimension = "weather" | "behavior" | "environment";
export type AugmentMode = "auto" | "manual";

export interface AugmentRequest {
  mode: AugmentMode;
  requests?: Partial<Record<AugmentDimension, string>>;
  seed: number;
  separate?: boolean;
  render?: boolean;
  criticality?: boolean;
  package?: boolean;
}

export interface AugmentChange {
  path: string;
  op: "add" | "replace" | "remove";
  before?: unknown;
  after?: unknown;
}

export interface AugmentAttempt {
  index: number;
  lever?: string;
  stage: string;
  passed: boolean;
  issues: string[];
  seconds?: number;
  limitations?: string | null;
  tokens?: { prompt?: number; completion?: number };
}

export interface AugmentJudge {
  ok?: boolean | null;
  skipped?: string;
  camera?: {
    ok?: boolean | null;
    skipped?: string;
    weather_agrees?: boolean;
    time_band_score?: number;
    objects_visible_ratio?: number | null;
    expected?: { weather: string; time_band: string; objects: string[] };
    answer?: { weather: string; time_of_day: string; road_wet_or_snowy: boolean; note: string };
  };
  lidar?: { ok?: boolean | null; skipped?: string; error?: string; warnings?: string[]; numbers?: Record<string, number | null> };
  rerender?: unknown;
  parent?: { camera?: AugmentJudge["camera"]; lidar?: AugmentJudge["lidar"] };
}

export interface AugmentVariant {
  index: number;
  id: string;
  parent_id: string;
  mode: AugmentMode;
  seed: number;
  dimensions: AugmentDimension[];
  requests?: Partial<Record<AugmentDimension, string>> | null;
  status: "ok" | "failed";
  error?: string;
  title?: string;
  description?: string;
  lever?: string;
  rationale?: string;
  limitations?: string | null;
  first_pass_valid: boolean;
  repairs: number;
  isolation?: { ok: boolean; violations: { path: string; op: string; why: string }[]; by_dim: Record<string, number>; untouched: string[] };
  description_check?: { ok: boolean; attempts: number; first_pass: boolean; missing: string[]; attributes: string[] };
  criticality?: { band: [number, number]; searched: boolean; landed: boolean; initial_min_ttc?: number | null; min_ttc?: number | null; reason?: string; sims?: number; params?: string[]; retuned?: { path: string; before: number; after: number }[] } | null;
  effect?: { ego_dev_m: number; actors_changed: string[]; new_actors: string[]; min_ttc: [number | null, number | null]; ego_max_decel: [number, number]; ego_relevant: boolean } | null;
  plan?: { weather?: { weather: string; band: string; environment: Record<string, unknown> }; levers?: Record<string, { id: string; text: string }[]> };
  verifier?: { passed: boolean; score: number; metrics?: Record<string, unknown>; qc_run?: boolean; findings: Finding[] };
  package?: { scenario_structure_valid: boolean; imdad_record_valid: boolean; imdad_record_errors: string[]; scenario_structure_errors: string[] };
  patch?: { op: string; path: string; value?: unknown }[];
  diff?: AugmentChange[];
  housekeeping?: string[];
  attempts: AugmentAttempt[];
  timings: Record<string, number>;
  tokens?: { calls: number; prompt_tokens: number; completion_tokens: number; tok_per_s: number };
  tokens_judge?: { calls: number; prompt_tokens: number; completion_tokens: number };
  render?: { skipped?: string; error?: string; seed?: number; t0?: number; render_s?: number; geometry_reused?: boolean };
  judge?: AugmentJudge;
  /** artifact names (relative to the augment run): tsir, xodr, xosc, osc2, package, imdad_record, mosaic, lidar_bev, sensors */
  media: Record<string, string>;
}

export interface AugmentInfo {
  mode: AugmentMode;
  seed: number;
  separate: boolean;
  requests?: Partial<Record<AugmentDimension, string>> | null;
  render: boolean;
  progress: Record<string, { stage: string; status: string; detail: string }>;
  variants: AugmentVariant[];
  parent_media?: { mosaic?: string; lidar_bev?: string };
  wall_s?: number;
}

export interface AugmentationRef extends RunSummary {
  mode?: AugmentMode;
  seed?: number;
  n_variants: number;
  dimensions: AugmentDimension[];
}

export interface RunDetail extends RunSummary {
  error?: string;
  augment?: AugmentInfo;
  augmentations?: AugmentationRef[];
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
    policies?: { id: string; label?: string; description?: string }[] | Record<string, { tier?: string; desc?: string }>;
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
