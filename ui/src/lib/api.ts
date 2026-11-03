import type {
  AugmentRequest,
  CreateRunRequest,
  FlywheelEvent,
  AgentKnobs,
  Health,
  HarnessState,
  Kpis,
  Lesson,
  LessonStatus,
  PipelineConfig,
  RunDetail,
  RunSummary,
  Scene, IncidentRef, EvidenceItem } from "./types";

export type ApiKind = "live" | "mock";

/** One interface, two implementations (LiveApi talks to the Thor, MockApi runs in the browser). */
export interface FlywheelApi {
  readonly kind: ApiKind;
  readonly base: string;
  health(): Promise<Health>;
  listRuns(limit?: number): Promise<RunSummary[]>;
  createRun(body: CreateRunRequest): Promise<RunSummary>;
  /** Child job: weather / behaviour / environment variants of a finished run's verified TSIR. */
  augmentRun(runId: string, body: AugmentRequest): Promise<RunSummary>;
  getRun(id: string): Promise<RunDetail>;
  getScene(id: string): Promise<Scene>;
  /** Absolute URL of an artifact (usable in <video src> or <a download>). */
  artifactUrl(runId: string, name: string): string;
  artifactText(runId: string, name: string): Promise<string>;
  /** URL of a file inside the run directory (sensor videos, WSM controls); `download` only for synthetic runs. */
  runFileUrl(runId: string, path: string, download?: boolean): string;
  /** Generated LiDAR sweep: float32 [x, y, z, intensity] per point, sensor frame (x forward, y left, z up). */
  lidarSweep(runId: string, index: number, every?: number): Promise<Float32Array>;
  /** Dashcam2Scenario source clip (streamed). */
  sourceVideoUrl(runId: string): string;
  /** Zip of the run's sensor data (synthetic runs only). */
  sensorsZipUrl(runId: string): string;
  getHarness(): Promise<HarnessState>;
  putKnobs(agent: string, knobs: AgentKnobs): Promise<void>;
  setLessonStatus(id: string, status: Exclude<LessonStatus, "candidate">): Promise<Lesson | void>;
  getKpis(): Promise<Kpis>;
  getConfig(): Promise<PipelineConfig>;
  listIncidents(): Promise<IncidentRef[]>;
  listEvidence(): Promise<EvidenceItem[]>;
  /** Absolute URL of an evidence video. */
  evidenceUrl(file: string): string;
  putConfig(cfg: PipelineConfig): Promise<PipelineConfig | void>;
  /** Subscribe to live events (WS /api/events). Returns an unsubscribe function. */
  subscribe(handler: (e: FlywheelEvent) => void, onState?: (connected: boolean) => void): () => void;
}

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status?: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}
