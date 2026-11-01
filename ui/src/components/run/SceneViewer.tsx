import { Box, Map as MapIcon, Orbit, Video } from "lucide-react";
import { Suspense, lazy, useMemo, useState } from "react";
import clsx from "clsx";
import type { Scene } from "../../lib/types";
import { ErrorBoundary } from "../ui/ErrorBoundary";
import { Segmented, Skeleton } from "../ui/primitives";
import { MS_TO_KMH, eventStyle, nearestSignal, samplePose, signalColor, signalState, usePlayback, usePlaybackState } from "./playback";
import { SceneBEV } from "./SceneBEV";
import { Scrubber } from "./Scrubber";
import type { CamMode } from "./Scene3D";

const Scene3D = lazy(() => import("./Scene3D"));

type Layout = "bev" | "3d" | "split";

function Readout({ scene }: { scene: Scene }) {
  const pb = usePlayback();
  const s = usePlaybackState(pb);
  const ego = scene.actors.find((a) => a.role === "ego") ?? scene.actors[0];
  const p = samplePose(scene.t, ego && scene.tracks[ego.id], s.t);
  const prev = samplePose(scene.t, ego && scene.tracks[ego.id], Math.max(0, s.t - 0.2));
  const accel = s.t > 0.2 ? (p.v - prev.v) / 0.2 : 0;
  const si = nearestSignal(scene, [[samplePose(scene.t, ego && scene.tracks[ego.id], 0).x, samplePose(scene.t, ego && scene.tracks[ego.id], 0).y]]);
  const st = si >= 0 ? signalState(scene.signals[si].states, scene.t, s.t) : "off";
  const recent = [...scene.events].filter((e) => e.t <= s.t).slice(-1)[0];
  const rs = recent ? eventStyle(recent.type) : null;
  const items: [string, string, string?][] = [
    ["Ego speed", `${(p.v * MS_TO_KMH).toFixed(1)} km/h`],
    ["Accel", `${accel >= 0 ? "+" : ""}${accel.toFixed(2)} m/s²`],
    ["Signal", st.toUpperCase(), signalColor(st)],
  ];
  return (
    <div className="mb-3 flex flex-wrap items-center gap-x-6 gap-y-2 rounded-xl border border-line bg-bg/50 px-4 py-2.5" aria-live="off">
      {items.map(([k, v, c]) => (
        <div key={k} className="flex items-baseline gap-2">
          <span className="text-[11px] font-semibold tracking-wider text-fg-2 uppercase">{k}</span>
          <span className="mono text-[13px] font-semibold tabular-nums" style={{ color: c ?? "var(--color-fg)" }}>
            {v}
          </span>
        </div>
      ))}
      {rs && recent && (
        <div className="ml-auto flex items-center gap-2 text-[12px] text-fg-1">
          <span className="inline-block h-2 w-2 rounded-sm" style={{ background: rs.color }} />
          Last event: {rs.label}
          {recent.actor ? ` (${recent.actor})` : ""} <span className="mono text-fg-3">{recent.t.toFixed(1)} s</span>
        </div>
      )}
    </div>
  );
}

export function SceneViewer({ scene }: { scene: Scene }) {
  const [layout, setLayout] = useState<Layout>("split");
  const [cam, setCam] = useState<CamMode>("orbit");
  const show3d = layout !== "bev";
  const showBev = layout !== "3d";
  const height = layout === "split" ? "h-[420px]" : "h-[520px]";
  const fallback = useMemo(
    () => (e: Error) => (
      <div className={clsx("grid place-items-center rounded-xl border border-line bg-bg text-center", height)}>
        <div className="max-w-xs px-4">
          <Box className="mx-auto text-fg-3" />
          <div className="mt-2 text-[13px] font-medium text-fg-1">3D view unavailable</div>
          <div className="mt-1 text-[12px] text-fg-2">{e.message}. The BEV view still works.</div>
        </div>
      </div>
    ),
    [height],
  );

  return (
    <div>
      <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
        <Segmented
          label="Viewer layout"
          value={layout}
          onChange={setLayout}
          options={[
            { value: "bev", label: (<><MapIcon size={14} /> 2D BEV</>) },
            { value: "3d", label: (<><Box size={14} /> 3D</>) },
            { value: "split", label: (<><Orbit size={14} /> Split</>) },
          ]}
        />
        {show3d && (
          <Segmented
            size="sm"
            label="3D camera"
            value={cam}
            onChange={setCam}
            options={[
              { value: "orbit", label: "Orbit", title: "Drag to orbit, scroll to zoom, right-drag to pan" },
              { value: "chase", label: (<><Video size={13} /> Chase</>), title: "Follow the ego from behind" },
              { value: "top", label: "Top", title: "Top-down overview" },
            ]}
          />
        )}
      </div>
      <Readout scene={scene} />
      <div className={clsx("grid gap-3", layout === "split" ? "grid-cols-1 xl:grid-cols-2" : "grid-cols-1")}>
        {showBev && <SceneBEV scene={scene} className={height} />}
        {show3d && (
          <div className={clsx("relative overflow-hidden rounded-xl border border-line", height)}>
            <ErrorBoundary fallback={fallback}>
              <Suspense fallback={<Skeleton className="h-full w-full !rounded-none" />}>
                <Scene3D scene={scene} mode={cam} />
              </Suspense>
            </ErrorBoundary>
            <span className="pointer-events-none absolute top-2.5 left-3 rounded-md bg-bg/70 px-2 py-0.5 text-[11px] font-semibold tracking-wider text-fg-1 backdrop-blur">3D · {cam.toUpperCase()}</span>
          </div>
        )}
      </div>
      <div className="mt-3">
        <Scrubber scene={scene} />
      </div>
    </div>
  );
}
