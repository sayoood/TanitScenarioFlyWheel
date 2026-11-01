/**
 * In-browser mock of the Flywheel API. Same interface as LiveApi, with a tiny scheduler that
 * advances queued runs through the pipeline stages and emits the same kind of live events.
 */
import { ApiError, type FlywheelApi } from "../api";
import { readRaw, writeJSON } from "../storage";
import { UC1_EXAMPLES } from "../meta";
import type {
  AgentKnobs,
  CreateRunRequest,
  FlywheelEvent,
  Health,
  HarnessState,
  Kpis,
  Lesson,
  LessonStatus,
  PipelineConfig,
  RunDetail,
  RunOptions,
  RunStatus,
  RunSummary,
  Scene,
  Stage,
  StageName,
  UseCaseId, IncidentRef } from "../types";
import { makeOsc2, makePackage, makeTsir, makeXodr, makeXosc } from "./code";
import { makeConfig, makeHarness, makeKpis } from "./data";
import { buildRun, hashStr, mulberry32, type PlannedStage } from "./runs";
import { generateScene } from "./scene";

type Handler = (e: FlywheelEvent) => void;

interface Rec {
  id: string;
  final: RunDetail;
  plan: PlannedStage[];
  state: RunStatus;
  /** index into final.stages of the stage that is running (when state === "running") */
  cursor: number;
  stageT0: number;
  wallMs: number[];
  seed: number;
}

const FULL_OPTIONS: RunOptions = {
  augment: { n: 8, strategies: ["weather", "time_of_day", "behavior"] },
  render: { views: 7, res: "480p", frames: 121, lidar: true },
  closed_loop: { policy: "refe_sft4", steps: 200 },
};

interface SeedRun {
  id: string;
  uc: UseCaseId;
  title: string;
  input: Record<string, unknown>;
  options: RunOptions;
  ageMin: number;
  outcome: RunStatus;
  attempts: number;
  failures?: number[];
  state?: "running" | "queued";
}

const SEEDS: SeedRun[] = [
  { id: "run_c4e8", uc: "uc4", title: "Augment snippet clip_0a41c7e2: weather and behaviour variants", input: { snippet: "physicalai_av:clip_0a41c7e2" }, options: { augment: { n: 12, strategies: ["weather", "behavior"] }, render: { views: 7, res: "480p", frames: 121, lidar: true } }, ageMin: 6, outcome: "ok", attempts: 1, state: "running" },
  { id: "run_d93b", uc: "uc1", title: "Truck cut-in on a three-lane highway in fog", input: { query: UC1_EXAMPLES[2] }, options: { render: { views: 4, res: "480p", frames: 121, lidar: false } }, ageMin: 3, outcome: "ok", attempts: 2, failures: [3], state: "queued" },
  { id: "run_e07a", uc: "uc3", title: "Positive twin of incident inc_000231 (red-light conflict)", input: { incident_id: "imdad:inc_000231" }, options: { closed_loop: { policy: "const_vel", steps: 200 } }, ageMin: 1, outcome: "ok", attempts: 1, state: "queued" },
  { id: "run_7f3a", uc: "uc1", title: "Lane change left, then protected-green left turn at a signalized 4-way intersection", input: { query: UC1_EXAMPLES[0] }, options: FULL_OPTIONS, ageMin: 38, outcome: "ok", attempts: 2, failures: [0] },
  { id: "run_5c10", uc: "uc1", title: "Pedestrian steps out from behind a parked van at dusk in light rain", input: { query: UC1_EXAMPLES[1] }, options: { render: { views: 7, res: "720p", frames: 121, lidar: true }, augment: { n: 6, strategies: ["weather", "time_of_day"] } }, ageMin: 190, outcome: "ok", attempts: 3, failures: [4, 6] },
  { id: "run_21de", uc: "uc3", title: "Positive twin of incident inc_000187 (late braking at junction)", input: { incident_id: "imdad:inc_000187" }, options: { render: { views: 7, res: "480p", frames: 121, lidar: true }, closed_loop: { policy: "refe_sft4", steps: 200 } }, ageMin: 260, outcome: "ok", attempts: 1 },
  { id: "run_88b2", uc: "uc2", title: "Dashcam 2026-10-22 urban_0417: reconstruct and replay", input: { video: "/data/dashcam/audi_a6/2026-10-22_urban_0417.mp4" }, options: { render: { views: 7, res: "480p", frames: 121, lidar: true } }, ageMin: 410, outcome: "failed", attempts: 3, failures: [1, 3, 1] },
  { id: "run_9a77", uc: "uc5", title: "Corrective data for negative snippet clip_3d92aa05", input: { snippet: "physicalai_av:clip_3d92aa05" }, options: { augment: { n: 8, strategies: ["behavior", "object"] }, closed_loop: { policy: "refe_sft4", steps: 200 } }, ageMin: 640, outcome: "ok", attempts: 2, failures: [5] },
  { id: "run_3e19", uc: "uc4", title: "Augment snippet clip_77be9f10: time-of-day sweep", input: { snippet: "physicalai_av:clip_77be9f10" }, options: { augment: { n: 8, strategies: ["time_of_day"] }, render: { views: 7, res: "480p", frames: 121, lidar: true } }, ageMin: 900, outcome: "ok", attempts: 1 },
  { id: "run_b602", uc: "uc1", title: "Cyclist overtaking on a two-lane road, light fog", input: { query: "Ego follows a cyclist on a two-lane road in light fog and overtakes when the oncoming lane is free" }, options: { render: { views: 3, res: "480p", frames: 121, lidar: false } }, ageMin: 1300, outcome: "ok", attempts: 2, failures: [1] },
  { id: "run_4d85", uc: "uc1", title: "Roundabout entry with yielding traffic at night", input: { query: "Ego enters a two-lane roundabout at night and yields to a bus already circulating" }, options: {}, ageMin: 1750, outcome: "failed", attempts: 3, failures: [0, 4, 0] },
  { id: "run_1fa0", uc: "uc3", title: "Positive twin of incident inc_000302 (pedestrian conflict)", input: { incident_id: "imdad:inc_000302" }, options: { closed_loop: { policy: "accel_decel", steps: 200 } }, ageMin: 2300, outcome: "ok", attempts: 1 },
  { id: "run_a6c3", uc: "uc1", title: "Unprotected left turn across oncoming traffic, day, dry", input: { query: "Ego turns left at a signalized intersection with an unprotected green and oncoming vehicles" }, options: { render: { views: 7, res: "480p", frames: 121, lidar: true } }, ageMin: 2900, outcome: "ok", attempts: 2, failures: [3] },
];

const iso = (ms: number) => new Date(ms).toISOString();

export class MockApi implements FlywheelApi {
  readonly kind = "mock" as const;
  readonly base = "mock://tanit";

  private recs = new Map<string, Rec>();
  private order: string[] = [];
  private listeners = new Set<Handler>();
  private backlog: FlywheelEvent[] = [];
  private harness: HarnessState;
  private config: PipelineConfig;
  private kpis: Kpis;
  private scenes = new Map<string, Scene>();
  private completed = 0;
  private started = Date.now();

  constructor() {
    this.harness = makeHarness();
    // persisted human-gate decisions and knobs survive reloads
    const st = readRaw<Record<string, LessonStatus>>("tsf.mock.lessons", {});
    for (const l of this.harness.lessons) if (st[l.id]) l.status = st[l.id];
    const kn = readRaw<Record<string, AgentKnobs>>("tsf.mock.knobs", {});
    for (const [a, k] of Object.entries(kn)) this.harness.knobs[a] = { ...(this.harness.knobs[a] ?? {}), ...k };
    this.config = readRaw<PipelineConfig>("tsf.mock.config", makeConfig());
    this.kpis = makeKpis();

    const now = Date.now();
    for (const s of SEEDS) {
      const created = iso(now - s.ageMin * 60_000);
      const built = buildRun({ id: s.id, uc: s.uc, title: s.title, input: s.input, options: s.options, created, outcome: s.outcome, nAttempts: s.attempts, seed: hashStr(s.id), failures: s.failures });
      const rec: Rec = {
        id: s.id,
        final: built.detail,
        plan: built.plan,
        state: s.state ?? built.detail.status,
        cursor: 0,
        stageT0: now,
        wallMs: built.plan.map((p) => p.wallMs),
        seed: hashStr(s.id),
      };
      if (s.state === "running") {
        const idx = built.detail.stages.findIndex((x) => x.name === "cosmos_transfer");
        rec.cursor = idx;
        rec.wallMs[idx] = 70_000;
        rec.stageT0 = now - 6_000;
      }
      this.recs.set(s.id, rec);
      this.order.push(s.id);
    }
    this.order.sort((a, b) => Date.parse(this.recs.get(b)!.final.created) - Date.parse(this.recs.get(a)!.final.created));
    // recent history so the live log is not empty on first load
    const t0 = Date.now();
    this.log("run_7f3a", "run ok (score 1.00)", "info", iso(t0 - 36 * 60_000));
    this.log("run_7f3a", "closed_loop done in 133.0 s", "info", iso(t0 - 37 * 60_000));
    this.log(undefined, "lesson exp_d05 proposed as candidate (delta +0.02, n=4)", "info", iso(t0 - 20 * 60_000));
    this.log("run_c4e8", "run started (uc4)", "info", iso(t0 - 6 * 60_000));
    this.log("run_c4e8", "wsm done in 41.0 s", "info", iso(t0 - 60_000));
    this.log("run_c4e8", "cosmos_transfer started: Transfer-Auto, 7 cams", "info", iso(t0 - 6_000));
    setInterval(() => this.tick(), 500);
  }

  // ------------------------------------------------------------------ scheduler
  private emit(e: FlywheelEvent) {
    for (const h of this.listeners) h(e);
  }
  private log(run_id: string | undefined, message: string, level: "info" | "warn" = "info", ts = new Date().toISOString()) {
    const e: FlywheelEvent = { type: "log", run_id, level, message, ts };
    this.backlog = [...this.backlog.slice(-29), e];
    this.emit(e);
  }
  private get running(): Rec | undefined {
    for (const r of this.recs.values()) if (r.state === "running") return r;
    return undefined;
  }

  private tick() {
    const now = Date.now();
    const cur = this.running;
    if (!cur) {
      const next = this.order
        .map((id) => this.recs.get(id) as Rec)
        .filter((r) => r.state === "queued")
        .sort((a, b) => Date.parse(a.final.created) - Date.parse(b.final.created))[0];
      if (next) this.startRun(next, now);
      return;
    }
    if (now - cur.stageT0 < cur.wallMs[cur.cursor]) return;
    // finish the current stage
    const done = cur.final.stages[cur.cursor];
    this.emit({ type: "stage", run_id: cur.id, name: done.name, status: done.status === "failed" ? "failed" : "ok", seconds: done.seconds, detail: done.detail });
    this.log(cur.id, `${done.name} ${done.status === "failed" ? "failed" : "done"} in ${done.seconds?.toFixed(1) ?? "0"} s`, done.status === "failed" ? "warn" : "info");
    let i = cur.cursor + 1;
    while (i < cur.final.stages.length && cur.final.stages[i].status === "skipped") i++;
    if (i >= cur.final.stages.length || done.status === "failed") {
      cur.state = cur.final.status === "failed" ? "failed" : "ok";
      this.completed++;
      this.emit({ type: "run", run_id: cur.id, status: cur.state });
      this.log(cur.id, `run ${cur.state}${cur.final.score !== undefined ? ` (score ${cur.final.score.toFixed(2)})` : ""}`);
      return;
    }
    cur.cursor = i;
    cur.stageT0 = now;
    const st = cur.final.stages[i];
    this.emit({ type: "stage", run_id: cur.id, name: st.name, status: "running", detail: st.detail });
    this.log(cur.id, `${st.name} started: ${st.detail ?? ""}`);
  }

  private startRun(r: Rec, now: number) {
    r.state = "running";
    r.cursor = r.final.stages.findIndex((s) => s.status !== "skipped");
    r.stageT0 = now;
    this.emit({ type: "run", run_id: r.id, status: "running" });
    const st = r.final.stages[r.cursor];
    this.emit({ type: "stage", run_id: r.id, name: st.name, status: "running", detail: st.detail });
    this.log(r.id, `run started (${r.final.uc})`);
  }

  // ------------------------------------------------------------------ snapshots
  private snapshot(r: Rec): RunDetail {
    const f = r.final;
    if (r.state === "ok" || r.state === "failed") return f;
    const queued = r.state === "queued";
    // never leak the final outcome of a run that has not got there yet
    const blocked = (s: Stage) => f.status === "failed" && (s.detail ?? "").startsWith("blocked");
    const future = (s: Stage): Stage => ({ name: s.name, status: s.status === "skipped" && !blocked(s) ? "skipped" : "pending", detail: s.status === "skipped" && !blocked(s) ? s.detail : undefined });
    const stages: Stage[] = f.stages.map((s, i) => {
      if (queued) return future(s);
      if (i < r.cursor) return s;
      if (i === r.cursor) return { name: s.name, status: "running", started: iso(r.stageT0), detail: s.name === "repair" && f.status === "failed" ? undefined : s.detail };
      return future(s);
    });
    const idx = (n: StageName) => f.stages.findIndex((s) => s.name === n);
    const past = (n: StageName) => !queued && r.cursor > idx(n);
    const attempts = past("verify") ? (past("repair") || f.attempts.length <= 1 ? f.attempts : f.attempts.slice(0, 1)) : [];
    const docDone = past("document");
    const simDone = past("simulate");
    const trDone = past("cosmos_transfer") || (past("wsm") && !f.cameras);
    return {
      ...f,
      status: queued ? "queued" : "running",
      seconds: undefined,
      score: attempts.length ? attempts[attempts.length - 1].score : undefined,
      kpis: undefined,
      llm: undefined,
      stages,
      attempts,
      metrics: simDone ? f.metrics : {},
      artifacts: f.artifacts.filter((a) => (a.kind === "video" || a.name.includes("lidar") ? past("cosmos_transfer") : a.kind === "image" || a.kind === "json" ? simDone || a.name.startsWith("tsir") : docDone)),
      cameras: trDone ? f.cameras : undefined,
    };
  }

  // ------------------------------------------------------------------ API
  async health(): Promise<Health> {
    const run = this.running;
    const stage = run ? run.final.stages[run.cursor]?.name : undefined;
    const t = (Date.now() - this.started) / 1000;
    const wob = (a: number, f: number) => Math.sin(t / f + a);
    const llmBusy = stage === "analyse" || stage === "propose" || stage === "verify" || stage === "repair" || stage === "augment";
    const transfer = stage === "cosmos_transfer";
    const sim = stage === "closed_loop";
    const gpu = transfer ? 0.93 + wob(1, 2.3) * 0.03 : sim ? 0.86 + wob(2, 2.1) * 0.05 : stage === "wsm" ? 0.38 + wob(3, 3) * 0.06 : llmBusy ? 0.6 + wob(4, 1.7) * 0.08 : 0.06 + Math.abs(wob(5, 4)) * 0.03;
    const mem = 27.8 + wob(0, 11) * 0.3 + (transfer ? 38.5 : 0) + (sim ? 34.2 : 0) + (stage === "wsm" ? 6 : 0);
    const power = 29 + gpu * 92 + wob(6, 1.9) * 2.5;
    const queue = [...this.recs.values()].filter((r) => r.state === "queued").length;
    return {
      thor: {
        host: "thor3",
        mem_total_gb: 122,
        mem_used_gb: Math.round(mem * 10) / 10,
        gpu_util: Math.round(Math.max(0, Math.min(1, gpu)) * 1000) / 1000,
        power_w: Math.round(power),
        disk_free_gb: Math.round(590 - this.completed * 0.7),
        disk_total_gb: 1000,
        power_max_w: 130,
      },
      services: {
        llm: { up: true, model: "qwen3.8-27b", tok_s: llmBusy ? Math.round((34.8 + wob(7, 1.3) * 1.4) * 10) / 10 : 0 },
        cosmos_sim: { up: sim, model: "cosmos3-nano-sim-auto" },
        cosmos_transfer: { up: transfer, model: "cosmos3-nano-transfer-auto" },
        scheduler: { queue, running: run?.id ?? null },
      },
    };
  }

  async listRuns(limit = 50): Promise<RunSummary[]> {
    return this.order.slice(0, limit).map((id) => {
      const r = this.recs.get(id) as Rec;
      const d = this.snapshot(r);
      const s: RunSummary = { id: d.id, uc: d.uc, title: d.title, status: d.status, created: d.created, seconds: d.seconds, score: d.score, kpis: d.kpis };
      return s;
    });
  }

  async createRun(body: CreateRunRequest): Promise<RunSummary> {
    const uc = body.uc;
    const input = body.input as Record<string, unknown>;
    const val = String(input.query ?? input.video ?? input.incident_id ?? input.snippet ?? "").trim();
    if (!val) throw new ApiError("422 input is empty", 422);
    const id = `run_${Math.floor(Math.random() * 0xffff).toString(16).padStart(4, "0")}`;
    const seed = hashStr(id + val);
    const rnd = mulberry32(seed);
    const title = this.titleFor(uc, val);
    const nAttempts = rnd() < 0.5 ? 1 : 2;
    const outcome: RunStatus = /fail/i.test(val) || rnd() < 0.03 ? "failed" : "ok";
    const built = buildRun({ id, uc, title, input, options: body.options ?? {}, created: iso(Date.now()), outcome, nAttempts: outcome === "failed" ? 3 : nAttempts, seed });
    const rec: Rec = { id, final: built.detail, plan: built.plan, state: "queued", cursor: 0, stageT0: Date.now(), wallMs: built.plan.map((p) => p.wallMs), seed };
    this.recs.set(id, rec);
    this.order.unshift(id);
    this.emit({ type: "run", run_id: id, status: "queued" });
    this.log(id, `run queued (${uc})`);
    const d = this.snapshot(rec);
    return { id: d.id, uc: d.uc, title: d.title, status: d.status, created: d.created };
  }

  private titleFor(uc: UseCaseId, val: string): string {
    if (uc === "uc1") return val.length > 88 ? val.slice(0, 85).trimEnd() + "..." : val;
    if (uc === "uc2") return `Dashcam ${val.split("/").pop() ?? val}: reconstruct and replay`;
    if (uc === "uc3") return `Positive twin of incident ${val.replace("imdad:", "")}`;
    if (uc === "uc4") return `Augment snippet ${val.split(":").pop()}`;
    return `Corrective data for negative snippet ${val.split(":").pop()}`;
  }

  async getRun(id: string): Promise<RunDetail> {
    const r = this.recs.get(id);
    if (!r) throw new ApiError(`404 run ${id} not found`, 404);
    const d = this.snapshot(r);
    const docDone = d.artifacts.some((a) => a.kind === "package");
    if (docDone && !d.package) {
      const q = String((d.input as Record<string, unknown>)?.query ?? d.title);
      return { ...d, package: makePackage(id, q, makeXodr(id), makeXosc(id, q), makeOsc2(id)) };
    }
    return d;
  }

  async getScene(id: string): Promise<Scene> {
    const r = this.recs.get(id);
    if (!r) throw new ApiError(`404 run ${id} not found`, 404);
    if (r.final.status === "failed") throw new ApiError("409 no scene: run failed before simulation", 409);
    let sc = this.scenes.get(id);
    if (!sc) {
      const rnd = mulberry32(r.seed);
      const featured = id === "run_7f3a";
      sc = generateScene(featured ? { seed: r.seed } : { seed: r.seed, egoSpeed: 10.5 + rnd() * 3.5, lcStart: 1 + rnd() * 2 }).scene;
      this.scenes.set(id, sc);
    }
    return sc;
  }

  artifactUrl(runId: string, name: string): string {
    return `mock://${runId}/${name}`;
  }

  async artifactText(runId: string, name: string): Promise<string> {
    const r = this.recs.get(runId);
    if (!r) throw new ApiError(`404 run ${runId} not found`, 404);
    const art = r.final.artifacts.find((a) => a.name === name);
    if (!art) throw new ApiError(`404 artifact ${name} not found`, 404);
    const q = String((r.final.input as Record<string, unknown>)?.query ?? r.final.title);
    switch (art.kind) {
      case "xodr":
        return makeXodr(runId);
      case "xosc":
        return makeXosc(runId, q);
      case "osc2":
        return makeOsc2(runId);
      case "package":
        return JSON.stringify(makePackage(runId, q, makeXodr(runId), makeXosc(runId, q), makeOsc2(runId)), null, 2);
      case "json":
        return JSON.stringify(makeTsir(q), null, 2);
      default:
        throw new ApiError("415 binary artifact (mock)", 415);
    }
  }

  async getHarness(): Promise<HarnessState> {
    return structuredClone(this.harness);
  }

  async putKnobs(agent: string, knobs: AgentKnobs): Promise<void> {
    this.harness.knobs[agent] = { ...knobs };
    writeJSON("tsf.mock.knobs", this.harness.knobs);
  }

  async setLessonStatus(id: string, status: Exclude<LessonStatus, "candidate">): Promise<Lesson> {
    const l = this.harness.lessons.find((x) => x.id === id);
    if (!l) throw new ApiError(`404 lesson ${id} not found`, 404);
    l.status = status;
    const st = readRaw<Record<string, LessonStatus>>("tsf.mock.lessons", {});
    st[id] = status;
    writeJSON("tsf.mock.lessons", st);
    this.log(undefined, `lesson ${id} ${status === "active" ? "activated" : "retired"} by operator`);
    return structuredClone(l);
  }

  async getKpis(): Promise<Kpis> {
    return structuredClone(this.kpis);
  }

  async listIncidents(): Promise<IncidentRef[]> {
    return [
      { incident_id: "imdad:inc_000231", title: "Red-light conflict at a signalized junction (collision, NEGATIVE)", has_cf_positive: false },
      { incident_id: "imdad:inc_000187", title: "Late braking before a junction, rear-end contact", has_cf_positive: true },
      { incident_id: "imdad:inc_000302", title: "Pedestrian conflict at an unsignalized crossing", has_cf_positive: false },
    ];
  }

  async getConfig(): Promise<PipelineConfig> {
    return structuredClone(this.config);
  }

  async putConfig(cfg: PipelineConfig): Promise<PipelineConfig> {
    this.config = structuredClone(cfg);
    writeJSON("tsf.mock.config", this.config);
    return structuredClone(this.config);
  }

  subscribe(handler: Handler, onState?: (connected: boolean) => void): () => void {
    this.listeners.add(handler);
    for (const e of this.backlog) handler(e); // replay recent log lines to a new subscriber
    onState?.(true);
    return () => {
      this.listeners.delete(handler);
    };
  }
}

let singleton: MockApi | null = null;
export function getMockApi(): MockApi {
  if (!singleton) singleton = new MockApi();
  return singleton;
}
