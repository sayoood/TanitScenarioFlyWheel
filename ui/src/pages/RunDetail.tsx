import { ArrowLeft, Box, Camera, Code2, Copy, FileJson, ListChecks, Workflow } from "lucide-react";
import { useEffect, useMemo, type ReactNode } from "react";
import { Link, useParams } from "react-router";
import { StatusChip, UcBadge } from "../components/common";
import { CameraGrid } from "../components/run/CameraGrid";
import { CodeViewer } from "../components/run/CodeViewer";
import { LadderViz } from "../components/run/LadderViz";
import { MetricsPanel } from "../components/run/MetricsPanel";
import { Playback, PlaybackContext } from "../components/run/playback";
import { SceneViewer } from "../components/run/SceneViewer";
import { StageTimeline } from "../components/run/StageTimeline";
import { CodeBlock } from "../components/run/CodeBlock";
import { Chip, EmptyState, Panel, Skeleton } from "../components/ui/primitives";
import { bytes, duration, shortDate } from "../lib/format";
import { ucMeta } from "../lib/meta";
import { useRun, useScene } from "../lib/hooks";
import type { RunDetail } from "../lib/types";

function SceneSection({ run }: { run: RunDetail }) {
  const simulated = run.stages.find((s) => s.name === "simulate")?.status === "ok";
  const scene = useScene(run.id, simulated);
  const duration_ = scene.data?.t[scene.data.t.length - 1] ?? 0;
  const pb = useMemo(() => new Playback(duration_ || 1), []); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    pb.setDuration(duration_ || 1);
  }, [pb, duration_]);
  useEffect(() => () => pb.dispose(), [pb]);

  let sceneBody: ReactNode;
  if (!simulated) {
    sceneBody = (
      <EmptyState icon={<Box size={26} />} title={run.status === "failed" ? "No scene: the run failed before simulation" : "Scene not available yet"}>
        {run.status === "failed" ? "Open the verifier ladder above to see why the repair budget ran out." : "The interactive scene appears once TanitSim has finished the simulate stage."}
      </EmptyState>
    );
  } else if (scene.isLoading) {
    sceneBody = <Skeleton className="h-[520px]" />;
  } else if (scene.isError || !scene.data) {
    sceneBody = (
      <EmptyState icon={<Box size={26} />} title="Scene could not be loaded">
        {(scene.error as Error | null)?.message ?? "Unknown error"}
      </EmptyState>
    );
  } else {
    sceneBody = <SceneViewer scene={scene.data} />;
  }

  return (
    <PlaybackContext.Provider value={pb}>
      <Panel id="scene" eyebrow="Interactive" title="Scene viewer" actions={scene.data && <Chip>{scene.data.actors.length} actors · {scene.data.t.length} samples</Chip>} className="scroll-mt-24">
        {sceneBody}
      </Panel>
      <Panel id="cameras" eyebrow="Cosmos 3 Transfer" title="Multi-camera video" className="mt-5 scroll-mt-24">
        <CameraGrid run={run} scene={scene.data} />
      </Panel>
    </PlaybackContext.Provider>
  );
}

export default function RunDetailPage() {
  const { id } = useParams();
  const run = useRun(id);

  if (run.isLoading) {
    return (
      <div className="space-y-4">
        <Skeleton className="h-10 w-96" />
        <Skeleton className="h-48" />
        <Skeleton className="h-96" />
      </div>
    );
  }
  if (run.isError || !run.data) {
    return (
      <div>
        <Link to="/runs" className="btn btn-sm mb-4">
          <ArrowLeft size={14} /> Runs
        </Link>
        <EmptyState title={`Run ${id} could not be loaded`}>{(run.error as Error | null)?.message}</EmptyState>
      </div>
    );
  }
  const r = run.data;
  const uc = ucMeta(r.uc);
  const running = r.status === "running" || r.status === "queued";

  return (
    <div>
      <div className="mb-6">
        <Link to="/runs" className="mb-3 inline-flex items-center gap-1.5 text-[12.5px] text-fg-2 hover:text-fg">
          <ArrowLeft size={14} /> All runs
        </Link>
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div className="min-w-0">
            <div className="mb-2 flex flex-wrap items-center gap-2">
              <UcBadge uc={r.uc} />
              <StatusChip status={r.status} />
              <span className="text-[12.5px] text-fg-2">{uc.name}</span>
            </div>
            <h1 className="max-w-4xl text-[26px] leading-tight font-semibold text-fg">{r.title}</h1>
            <div className="mono mt-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-[12px] text-fg-2">
              <button className="inline-flex items-center gap-1.5 hover:text-fg" onClick={() => void navigator.clipboard?.writeText(r.id)} title="Copy run id">
                {r.id} <Copy size={12} />
              </button>
              <span>{shortDate(r.created)}</span>
              {r.seconds !== undefined && <span>{duration(r.seconds)}</span>}
              {r.score != null && <span>score {r.score.toFixed(2)}</span>}
            </div>
          </div>
          <nav aria-label="Sections" className="flex flex-wrap gap-1.5">
            {[
              ["#pipeline", "Pipeline", Workflow],
              ["#verifier", "Verifier", ListChecks],
              ["#scene", "Scene", Box],
              ["#cameras", "Cameras", Camera],
              ["#code", "Code", Code2],
            ].map(([href, label, Icon]) => {
              const I = Icon as typeof Workflow;
              return (
                <a key={String(href)} href={String(href)} className="btn btn-sm">
                  <I size={14} /> {String(label)}
                </a>
              );
            })}
          </nav>
        </div>
      </div>

      <div className="space-y-5">
        <Panel id="pipeline" eyebrow="Pipeline" title="Stage timeline" className="scroll-mt-24" actions={running && <Chip tone="info"><span className="live-dot !h-1.5 !w-1.5" /> live</Chip>}>
          <StageTimeline run={r} />
        </Panel>

        <div className="grid grid-cols-12 gap-5">
          <Panel id="verifier" eyebrow="Reward signal" title="Verifier ladder L0 to L5" className="col-span-12 scroll-mt-24 xl:col-span-7">
            <LadderViz attempts={r.attempts} running={running} />
          </Panel>
          <div className="col-span-12 space-y-5 xl:col-span-5">
            <Panel eyebrow="TanitSim" title="Metrics">
              <MetricsPanel run={r} />
            </Panel>
            <Panel eyebrow="Request" title="Input and artifacts">
              <CodeBlock code={JSON.stringify(r.input, null, 2)} lang="json" maxHeight={150} label="Run input" />
              <ul className="m-0 mt-3 list-none space-y-1 p-0">
                {r.artifacts.length === 0 && <li className="text-[12.5px] text-fg-2">No artifacts yet.</li>}
                {r.artifacts.map((a) => (
                  <li key={a.name} className="flex items-center justify-between gap-3 text-[12.5px]">
                    <span className="flex min-w-0 items-center gap-2 text-fg-1">
                      <FileJson size={13} className="shrink-0 text-fg-3" />
                      <span className="mono truncate">{a.name}</span>
                    </span>
                    <span className="flex shrink-0 items-center gap-2 text-fg-2">
                      <Chip>{a.kind}</Chip>
                      {bytes(a.size)}
                    </span>
                  </li>
                ))}
              </ul>
            </Panel>
          </div>
        </div>

        <SceneSection run={r} />

        <Panel id="code" eyebrow="Compilers" title="Generated artifacts" className="scroll-mt-24">
          <CodeViewer run={r} />
        </Panel>
      </div>
    </div>
  );
}
