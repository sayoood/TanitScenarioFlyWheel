import { Pause, Play, Repeat, SkipBack, SkipForward } from "lucide-react";
import { useCallback, useMemo, useRef, type KeyboardEvent, type PointerEvent as ReactPointerEvent } from "react";
import clsx from "clsx";
import type { Scene } from "../../lib/types";
import { Segmented } from "../ui/primitives";
import { eventStyle, nearestSignal, samplePose, signalColor, usePlayback, usePlaybackState } from "./playback";

const SPEEDS = [0.25, 0.5, 1, 2, 4];

/** Shared timeline: play/pause, speed, draggable playhead, signal phase lane and event markers lane. */
export function Scrubber({ scene }: { scene: Scene }) {
  const pb = usePlayback();
  const s = usePlaybackState(pb);
  const track = useRef<HTMLDivElement>(null);
  const dragging = useRef(false);
  const dur = s.duration || 1;

  // signal lane for the arm the ego approaches on (nearest signal to the ego's first pose)
  const phases = useMemo(() => {
    const ego = scene.actors.find((a) => a.role === "ego") ?? scene.actors[0];
    if (!ego || scene.signals.length === 0) return [];
    const p0 = samplePose(scene.t, scene.tracks[ego.id], 0);
    const si = nearestSignal(scene, [[p0.x, p0.y]]);
    const states = scene.signals[si]?.states ?? [];
    const out: { from: number; to: number; color: string; state: string }[] = [];
    states.forEach((st, i) => {
      const t0 = scene.t[i] ?? 0;
      const t1 = scene.t[i + 1] ?? dur;
      const last = out[out.length - 1];
      if (last && last.state === st) last.to = t1;
      else out.push({ from: t0, to: t1, color: signalColor(st), state: st });
    });
    return out;
  }, [scene, dur]);

  const seekFromEvent = useCallback(
    (clientX: number) => {
      const el = track.current;
      if (!el) return;
      const r = el.getBoundingClientRect();
      pb.seek(((clientX - r.left) / r.width) * dur);
    },
    [pb, dur],
  );
  const onDown = (e: ReactPointerEvent) => {
    dragging.current = true;
    (e.currentTarget as Element).setPointerCapture(e.pointerId);
    seekFromEvent(e.clientX);
  };
  const onMove = (e: ReactPointerEvent) => {
    if (dragging.current) seekFromEvent(e.clientX);
  };
  const onUp = () => {
    dragging.current = false;
  };
  const onKey = (e: KeyboardEvent) => {
    if (e.key === "ArrowRight") pb.seek(pb.t + (e.shiftKey ? 1 : 0.1));
    else if (e.key === "ArrowLeft") pb.seek(pb.t - (e.shiftKey ? 1 : 0.1));
    else if (e.key === " ") pb.toggle();
    else if (e.key === "Home") pb.seek(0);
    else if (e.key === "End") pb.seek(dur);
    else return;
    e.preventDefault();
  };

  const pct = (t: number) => `${(t / dur) * 100}%`;
  const ticks = useMemo(() => {
    const step = dur > 60 ? 10 : dur > 30 ? 5 : 2;
    return Array.from({ length: Math.floor(dur / step) + 1 }, (_, i) => i * step);
  }, [dur]);

  return (
    <div className="rounded-xl border border-line bg-bg/60 p-3">
      <div className="flex flex-wrap items-center gap-3">
        <div className="flex items-center gap-1.5">
          <button className="btn btn-icon btn-sm" onClick={() => pb.seek(0)} title="Back to start" aria-label="Back to start">
            <SkipBack size={15} />
          </button>
          <button className="btn btn-primary btn-icon !h-9 !w-9 !rounded-full" onClick={() => pb.toggle()} aria-label={s.playing ? "Pause" : "Play"} title="Play / pause (space)">
            {s.playing ? <Pause size={16} fill="currentColor" /> : <Play size={16} fill="currentColor" className="ml-0.5" />}
          </button>
          <button className="btn btn-icon btn-sm" onClick={() => pb.seek(dur)} title="Jump to end" aria-label="Jump to end">
            <SkipForward size={15} />
          </button>
          <button
            className={clsx("btn btn-icon btn-sm", pb.loop && "!border-accent-2/60 !text-accent-2")}
            onClick={() => {
              pb.loop = !pb.loop;
              pb.seek(pb.t);
            }}
            aria-pressed={pb.loop}
            title="Loop"
            aria-label="Loop"
          >
            <Repeat size={15} />
          </button>
        </div>
        <div className="mono text-[13px] text-fg tabular-nums">
          {s.t.toFixed(2)} <span className="text-fg-3">/ {dur.toFixed(1)} s</span>
        </div>
        <div className="ml-auto">
          <Segmented size="sm" label="Playback speed" value={String(s.speed)} onChange={(v) => pb.setSpeed(Number(v))} options={SPEEDS.map((v) => ({ value: String(v), label: `${v}x` }))} />
        </div>
      </div>

      <div
        ref={track}
        role="slider"
        tabIndex={0}
        aria-label="Timeline"
        aria-valuemin={0}
        aria-valuemax={Number(dur.toFixed(1))}
        aria-valuenow={Number(s.t.toFixed(2))}
        aria-valuetext={`${s.t.toFixed(1)} seconds`}
        onPointerDown={onDown}
        onPointerMove={onMove}
        onPointerUp={onUp}
        onKeyDown={onKey}
        className="relative mt-3 cursor-pointer touch-none select-none rounded-lg outline-offset-4"
      >
        {/* signal phase lane */}
        <div className="relative h-2.5 overflow-hidden rounded-full bg-bg-3" title="Signal phase of the ego approach">
          {phases.map((p, i) => (
            <div key={i} className="absolute top-0 h-full" style={{ left: pct(p.from), width: `${((p.to - p.from) / dur) * 100}%`, background: p.color, opacity: 0.85 }} />
          ))}
        </div>
        {/* events lane */}
        <div className="relative mt-1.5 h-[22px] rounded-md border border-line/70 bg-bg-1/60">
          {scene.events.map((e, i) => {
            const st = eventStyle(e.type);
            return (
              <button
                key={i}
                title={`${st.label}${e.actor ? ` (${e.actor})` : e.id ? ` (${e.id})` : ""} at ${e.t.toFixed(1)} s`}
                aria-label={`${st.label} at ${e.t.toFixed(1)} seconds`}
                onPointerDown={(ev) => ev.stopPropagation()}
                onClick={() => pb.seek(e.t)}
                className="absolute top-1/2 -translate-x-1/2 -translate-y-1/2 p-1"
                style={{ left: pct(e.t) }}
              >
                <span
                  className="block"
                  style={{
                    width: st.shape === "bar" ? 3 : 9,
                    height: st.shape === "bar" ? 14 : 9,
                    background: st.color,
                    borderRadius: st.shape === "dot" ? 999 : st.shape === "bar" ? 2 : 2,
                    transform: st.shape === "diamond" ? "rotate(45deg)" : undefined,
                    boxShadow: `0 0 8px ${st.color}88`,
                  }}
                />
              </button>
            );
          })}
        </div>
        {/* axis */}
        <div className="relative mt-1 h-4 text-[10.5px] text-fg-3">
          {ticks.map((t) => (
            <span key={t} className="mono absolute -translate-x-1/2" style={{ left: pct(t) }}>
              {t}
            </span>
          ))}
        </div>
        {/* playhead */}
        <div className="pointer-events-none absolute top-[-4px] bottom-3 w-0.5 -translate-x-1/2 bg-white shadow-[0_0_10px_rgba(255,255,255,0.7)]" style={{ left: pct(s.t) }}>
          <div className="absolute -top-0.5 left-1/2 h-2.5 w-2.5 -translate-x-1/2 rounded-full border-2 border-bg bg-white" />
        </div>
      </div>
      <div className="mt-1 flex flex-wrap items-center gap-x-4 gap-y-1 text-[11.5px] text-fg-2">
        <span>
          <span className="mr-1 inline-block h-2 w-2 rounded-full" style={{ background: "#34d399" }} />
          green
          <span className="mx-1 inline-block h-2 w-2 rounded-full" style={{ background: "#fbbf24" }} />
          yellow
          <span className="mx-1 inline-block h-2 w-2 rounded-full" style={{ background: "#fb7185" }} />
          red (signal lane)
        </span>
        <span>Events: click a marker to jump. Arrow keys step 0.1 s, shift for 1 s.</span>
      </div>
    </div>
  );
}
