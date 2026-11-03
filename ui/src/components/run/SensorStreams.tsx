import { Download, Lock, Pause, Play } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useApi } from "../../lib/provider";
import type { RunDetail, SensorData } from "../../lib/types";
import { Chip, Segmented } from "../ui/primitives";

type Mode = "generated" | "control" | "both";

/** A group of <video> elements driven by one transport: play/pause together, one scrubber, drift corrected.
 *  ``refAt(offset)`` registers a video whose own clock runs ``offset`` seconds ahead of the group clock. */
export function useVideoSync() {
  const vids = useRef(new Map<HTMLVideoElement, number>());
  const [t, setT] = useState(0);
  const [dur, setDur] = useState(0);
  const [playing, setPlaying] = useState(false);

  const refAt = useCallback(
    (offset: number) => (v: HTMLVideoElement | null) => {
      if (v && !vids.current.has(v)) {
        vids.current.set(v, offset);
        if (offset) v.addEventListener("loadedmetadata", () => (v.currentTime = offset), { once: true });
      }
    },
    [],
  );
  const ref = useMemo(() => refAt(0), [refAt]);
  useEffect(() => {
    const id = setInterval(() => {
      for (const v of vids.current.keys()) if (!v.isConnected) vids.current.delete(v);
      const [m, mOff] = [...vids.current.entries()].find(([, o]) => o === 0) ?? [...vids.current.entries()][0] ?? [];
      if (!m) return;
      const clock = m.currentTime - (mOff ?? 0);
      setT(clock);
      if (m.duration && Number.isFinite(m.duration)) setDur(m.duration - (mOff ?? 0));
      if (!m.paused)
        for (const [v, off] of vids.current) if (v !== m && Math.abs(v.currentTime - (clock + off)) > 0.08) v.currentTime = clock + off;
    }, 100);
    return () => clearInterval(id);
  }, []);
  const toggle = () => {
    const play = !playing;
    for (const v of vids.current.keys()) {
      if (play) v.play().catch(() => undefined);
      else v.pause();
    }
    setPlaying(play);
  };
  const seek = (s: number) => {
    for (const [v, off] of vids.current) v.currentTime = s + off;
    setT(s);
  };
  return { ref, refAt, t, dur, playing, toggle, seek };
}

export function Transport({ sync, t0 = 0, label }: { sync: ReturnType<typeof useVideoSync>; t0?: number; label?: string }) {
  return (
    <div className="flex items-center gap-3 rounded-xl border border-line bg-bg/50 px-3 py-2">
      <button className="btn btn-sm !px-2.5" onClick={sync.toggle} aria-label={sync.playing ? "Pause all" : "Play all"}>
        {sync.playing ? <Pause size={14} /> : <Play size={14} />}
      </button>
      <input type="range" className="flex-1 accent-[var(--color-accent)]" aria-label="Scrub all videos" min={0} max={sync.dur || 1} step={1 / 30}
        value={Math.min(sync.t, sync.dur || 1)} onChange={(e) => sync.seek(Number(e.target.value))} />
      <span className="mono w-40 text-right text-[11.5px] text-fg-1">
        t = {(t0 + sync.t).toFixed(2)} s{label ? ` · ${label}` : ""}
      </span>
    </div>
  );
}

function Clip({ src, label, sync, download }: { src: string; label: string; sync: ReturnType<typeof useVideoSync>; download?: string }) {
  return (
    <figure className="relative m-0">
      <video ref={sync.ref} className="w-full rounded-lg border border-line bg-black" src={src} muted playsInline preload="auto" loop />
      <figcaption className="pointer-events-none absolute top-1.5 left-2 rounded bg-bg/75 px-1.5 py-0.5 text-[10.5px] font-semibold text-fg-1">{label}</figcaption>
      {download && (
        <a className="btn btn-sm absolute top-1.5 right-1.5 !h-6 !px-1.5" href={download} title={`Download ${label}`}>
          <Download size={12} />
        </a>
      )}
    </figure>
  );
}

/** Per-camera Cosmos output next to the WSM control it was conditioned on, one transport for all. */
export function SensorStreams({ run, data }: { run: RunDetail; data: SensorData }) {
  const { api } = useApi();
  const [mode, setMode] = useState<Mode>("both");
  const sync = useVideoSync();
  const file = (p: string, dl = false) => api.runFileUrl(run.id, p, dl);
  const seconds = data.frames / data.fps;
  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <Segmented label="Camera stream" size="sm" value={mode} onChange={setMode}
          options={[{ value: "generated", label: "Generated" }, { value: "control", label: "WSM control" }, { value: "both", label: "Side by side" }]} />
        <div className="flex items-center gap-2 text-[12px] text-fg-2">
          <span>{data.cameras.length} cameras · {data.frames} frames ({seconds.toFixed(1)} s from t = {data.t0_s.toFixed(1)} s)</span>
          {data.public ? (
            <a className="btn btn-sm" href={api.sensorsZipUrl(run.id)}><Download size={13} /> All sensor data (.zip)</a>
          ) : (
            <Chip title="Derived from a dataset, incident or video clip: streamed to this console, never downloaded"><Lock size={11} /> Thor only</Chip>
          )}
        </div>
      </div>
      <div className={mode === "both" ? "grid gap-3 lg:grid-cols-2 2xl:grid-cols-3" : "grid gap-3 md:grid-cols-2 xl:grid-cols-3"}>
        {data.cameras.map((c) => (
          <div key={c.rig} className="rounded-xl border border-line bg-bg/40 p-2">
            <div className="mb-1.5 flex items-center justify-between">
              <span className="mono text-[12px] font-semibold text-fg">{c.name}</span>
              <span className="text-[11px] text-fg-3">{c.rig}</span>
            </div>
            <div className={mode === "both" ? "grid grid-cols-1 gap-2" : ""}>
              {mode !== "control" && <Clip src={file(c.generated)} label="Cosmos generated" sync={sync} download={data.public ? file(c.generated, true) : undefined} />}
              {mode !== "generated" && c.control && <Clip src={file(c.control)} label="WSM control" sync={sync} download={data.public ? file(c.control, true) : undefined} />}
            </div>
          </div>
        ))}
      </div>
      <Transport sync={sync} t0={data.t0_s} label={`${data.fps} fps`} />
    </div>
  );
}
