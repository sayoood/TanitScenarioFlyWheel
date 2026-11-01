import { ApiError, type FlywheelApi } from "./api";
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
  RunSummary,
  Scene, IncidentRef, EvidenceItem } from "./types";

export const normalizeBase = (b: string): string => b.trim().replace(/\/+$/, "");

/** Typed client implemented exactly against docs/API.md. */
export class LiveApi implements FlywheelApi {
  readonly kind = "live" as const;
  readonly base: string;

  constructor(base: string) {
    this.base = normalizeBase(base);
  }

  private url(path: string): string {
    return `${this.base}${path}`;
  }

  private async req<T>(path: string, init?: RequestInit & { timeoutMs?: number }): Promise<T> {
    const ctl = new AbortController();
    const timer = setTimeout(() => ctl.abort(), init?.timeoutMs ?? 15000);
    try {
      const res = await fetch(this.url(path), {
        ...init,
        signal: ctl.signal,
        headers: { Accept: "application/json", ...(init?.body ? { "Content-Type": "application/json" } : {}), ...init?.headers },
      });
      if (!res.ok) {
        let msg = res.statusText;
        try {
          const j = await res.json();
          msg = typeof j?.detail === "string" ? j.detail : JSON.stringify(j?.detail ?? j);
        } catch {
          /* ignore */
        }
        throw new ApiError(`${res.status} ${msg}`, res.status);
      }
      if (res.status === 204) return undefined as T;
      const ct = res.headers.get("content-type") ?? "";
      return (ct.includes("json") ? await res.json() : undefined) as T;
    } catch (e) {
      if (e instanceof ApiError) throw e;
      throw new ApiError(e instanceof Error ? e.message : "Network error");
    } finally {
      clearTimeout(timer);
    }
  }

  health = () => this.req<Health>("/api/health", { timeoutMs: 4000 });
  listRuns = (limit = 50) => this.req<RunSummary[]>(`/api/runs?limit=${limit}`);
  createRun = (body: CreateRunRequest) => this.req<RunSummary>("/api/runs", { method: "POST", body: JSON.stringify(body) });
  getRun = (id: string) => this.req<RunDetail>(`/api/runs/${encodeURIComponent(id)}`);
  getScene = (id: string) => this.req<Scene>(`/api/runs/${encodeURIComponent(id)}/scene`, { timeoutMs: 30000 });
  artifactUrl = (runId: string, name: string) =>
    this.url(`/api/runs/${encodeURIComponent(runId)}/artifacts/${encodeURIComponent(name)}`);
  async artifactText(runId: string, name: string): Promise<string> {
    const res = await fetch(this.artifactUrl(runId, name));
    if (!res.ok) throw new ApiError(`${res.status} ${res.statusText}`, res.status);
    return res.text();
  }
  getHarness = () => this.req<HarnessState>("/api/harness");
  putKnobs = (agent: string, knobs: AgentKnobs) =>
    this.req<void>("/api/harness/knobs", { method: "PUT", body: JSON.stringify({ agent, knobs }) });
  setLessonStatus = (id: string, status: Exclude<LessonStatus, "candidate">) =>
    this.req<Lesson | void>(`/api/harness/lessons/${encodeURIComponent(id)}`, { method: "POST", body: JSON.stringify({ status }) });
  getKpis = () => this.req<Kpis>("/api/kpis");
  getConfig = () => this.req<PipelineConfig>("/api/config");
  listIncidents = () => this.req<IncidentRef[]>("/api/incidents");
  listEvidence = () => this.req<EvidenceItem[]>("/api/evidence");
  evidenceUrl = (file: string) => this.url(`/api/evidence/files/${file.split("/").map(encodeURIComponent).join("/")}`);
  putConfig = (cfg: PipelineConfig) => this.req<PipelineConfig | void>("/api/config", { method: "PUT", body: JSON.stringify(cfg) });

  subscribe(handler: (e: FlywheelEvent) => void, onState?: (connected: boolean) => void): () => void {
    let ws: WebSocket | null = null;
    let closed = false;
    let retry = 0;
    let timer: ReturnType<typeof setTimeout> | undefined;

    const wsUrl = (): string => {
      if (/^https?:/i.test(this.base)) return this.base.replace(/^http/i, "ws") + "/api/events";
      const proto = location.protocol === "https:" ? "wss:" : "ws:";
      return `${proto}//${location.host}${this.base}/api/events`;
    };

    const connect = () => {
      if (closed) return;
      try {
        ws = new WebSocket(wsUrl());
      } catch {
        schedule();
        return;
      }
      ws.onopen = () => {
        retry = 0;
        onState?.(true);
      };
      ws.onmessage = (m) => {
        try {
          const e = JSON.parse(String(m.data)) as FlywheelEvent;
          if (e && typeof e.type === "string") handler(e);
        } catch {
          /* ignore malformed frames */
        }
      };
      ws.onclose = () => {
        onState?.(false);
        schedule();
      };
      ws.onerror = () => ws?.close();
    };
    const schedule = () => {
      if (closed) return;
      retry = Math.min(retry + 1, 6);
      timer = setTimeout(connect, 800 * 2 ** retry);
    };
    connect();
    return () => {
      closed = true;
      if (timer) clearTimeout(timer);
      ws?.close();
    };
  }
}

/** Probe a base URL: resolves with latency if /api/health answers with the expected shape. */
export async function probe(base: string, timeoutMs = 2500): Promise<{ ok: true; ms: number } | { ok: false; error: string }> {
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), timeoutMs);
  const t0 = performance.now();
  try {
    const res = await fetch(`${normalizeBase(base)}/api/health`, { signal: ctl.signal, headers: { Accept: "application/json" } });
    if (!res.ok) return { ok: false, error: `HTTP ${res.status}` };
    const j = await res.json();
    if (!j || typeof j !== "object" || !("thor" in j) || !("services" in j)) return { ok: false, error: "Unexpected response shape" };
    return { ok: true, ms: Math.round(performance.now() - t0) };
  } catch (e) {
    return { ok: false, error: e instanceof DOMException && e.name === "AbortError" ? "Timed out" : "Unreachable (offline or CORS)" };
  } finally {
    clearTimeout(timer);
  }
}
