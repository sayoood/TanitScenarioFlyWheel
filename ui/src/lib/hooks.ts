import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useMemo } from "react";
import { useApi } from "./provider";
import type { AgentKnobs, CreateRunRequest, LessonStatus, PipelineConfig, StageName } from "./types";

/** Query keys always carry the API identity so mock and live data never mix in the cache. */
function useKey() {
  const { api } = useApi();
  return (...k: (string | number | undefined)[]) => [api.kind, api.base, ...k] as const;
}

export function useHealth() {
  const { api } = useApi();
  const key = useKey();
  return useQuery({ queryKey: key("health"), queryFn: () => api.health(), refetchInterval: 2500, placeholderData: (p) => p });
}

export function useRuns(limit = 50) {
  const { api } = useApi();
  const key = useKey();
  return useQuery({ queryKey: key("runs", limit), queryFn: () => api.listRuns(limit), refetchInterval: 4000, placeholderData: (p) => p });
}

export function useRun(id: string | undefined) {
  const { api } = useApi();
  const key = useKey();
  return useQuery({
    queryKey: key("run", id),
    queryFn: () => api.getRun(id as string),
    enabled: !!id,
    refetchInterval: (q) => (q.state.data && (q.state.data.status === "running" || q.state.data.status === "queued") ? 1500 : false),
  });
}

export function useScene(id: string | undefined, enabled: boolean) {
  const { api } = useApi();
  const key = useKey();
  return useQuery({ queryKey: key("scene", id), queryFn: () => api.getScene(id as string), enabled: !!id && enabled, staleTime: 60_000, retry: 0 });
}

export function useArtifactText(runId: string | undefined, name: string | undefined) {
  const { api } = useApi();
  const key = useKey();
  return useQuery({
    queryKey: key("artifact", runId, name),
    queryFn: () => api.artifactText(runId as string, name as string),
    enabled: !!runId && !!name,
    staleTime: 5 * 60_000,
    retry: 0,
  });
}

export function useHarness() {
  const { api } = useApi();
  const key = useKey();
  return useQuery({ queryKey: key("harness"), queryFn: () => api.getHarness(), refetchInterval: 20_000 });
}

export function useKpis() {
  const { api } = useApi();
  const key = useKey();
  return useQuery({ queryKey: key("kpis"), queryFn: () => api.getKpis(), refetchInterval: 60_000 });
}

export function useIncidents() {
  const { api } = useApi();
  const key = useKey();
  return useQuery({ queryKey: key("incidents"), queryFn: () => api.listIncidents(), staleTime: 300_000 });
}

export function useEvidence() {
  const { api } = useApi();
  const key = useKey();
  return useQuery({ queryKey: key("evidence"), queryFn: () => api.listEvidence(), refetchInterval: 60_000 });
}

export function useConfig() {
  const { api } = useApi();
  const key = useKey();
  return useQuery({ queryKey: key("config"), queryFn: () => api.getConfig(), staleTime: 30_000 });
}

export function useCreateRun() {
  const { api } = useApi();
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: CreateRunRequest) => api.createRun(body),
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: [api.kind, api.base, "runs"] });
      void qc.invalidateQueries({ queryKey: [api.kind, api.base, "health"] });
    },
  });
}

export function useSetLesson() {
  const { api } = useApi();
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (v: { id: string; status: Exclude<LessonStatus, "candidate"> }) => api.setLessonStatus(v.id, v.status),
    onSuccess: () => void qc.invalidateQueries({ queryKey: [api.kind, api.base, "harness"] }),
  });
}

export function usePutKnobs() {
  const { api } = useApi();
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (v: { agent: string; knobs: AgentKnobs }) => api.putKnobs(v.agent, v.knobs),
    onSuccess: () => void qc.invalidateQueries({ queryKey: [api.kind, api.base, "harness"] }),
  });
}

export function usePutConfig() {
  const { api } = useApi();
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (cfg: PipelineConfig) => api.putConfig(cfg),
    onSuccess: () => void qc.invalidateQueries({ queryKey: [api.kind, api.base, "config"] }),
  });
}

/** Which pipeline stage is currently running, for the flywheel hero. */
export function useActivity(): { runId?: string; uc?: string; stage?: StageName; active: boolean; queue: number } {
  const health = useHealth();
  const runId = health.data?.services.scheduler.running ?? undefined;
  const run = useRun(runId ?? undefined);
  return useMemo(() => {
    const stage = run.data?.stages.find((s) => s.status === "running")?.name;
    return { runId, uc: run.data?.uc, stage, active: !!runId, queue: health.data?.services.scheduler.queue ?? 0 };
  }, [runId, run.data, health.data]);
}
