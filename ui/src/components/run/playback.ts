import { createContext, useContext, useSyncExternalStore } from "react";
import type { Scene } from "../../lib/types";

/** Shared playback clock: the BEV canvas, the 3D view, the scrubber and the camera videos all follow it. */
export interface PlaybackState {
  t: number;
  playing: boolean;
  speed: number;
  duration: number;
}

export class Playback {
  private state: PlaybackState;
  private subs = new Set<() => void>();
  private raf = 0;
  private last = 0;
  loop = false;

  constructor(duration: number) {
    this.state = { t: 0, playing: false, speed: 1, duration };
  }

  subscribe = (cb: () => void): (() => void) => {
    this.subs.add(cb);
    return () => this.subs.delete(cb);
  };
  getSnapshot = (): PlaybackState => this.state;
  get t(): number {
    return this.state.t;
  }

  private set(patch: Partial<PlaybackState>) {
    this.state = { ...this.state, ...patch };
    this.subs.forEach((s) => s());
  }

  setDuration(d: number) {
    if (d !== this.state.duration) this.set({ duration: d, t: Math.min(this.state.t, d) });
  }
  seek(t: number) {
    this.set({ t: Math.min(Math.max(t, 0), this.state.duration) });
  }
  setSpeed(speed: number) {
    this.set({ speed });
  }
  play() {
    if (this.state.playing) return;
    if (this.state.t >= this.state.duration - 1e-3) this.state = { ...this.state, t: 0 };
    this.set({ playing: true });
    this.last = performance.now();
    const step = (now: number) => {
      if (!this.state.playing) return;
      const dt = ((now - this.last) / 1000) * this.state.speed;
      this.last = now;
      let t = this.state.t + dt;
      if (t >= this.state.duration) {
        if (this.loop) t = 0;
        else {
          this.set({ t: this.state.duration, playing: false });
          return;
        }
      }
      this.set({ t });
      this.raf = requestAnimationFrame(step);
    };
    this.raf = requestAnimationFrame(step);
  }
  pause() {
    cancelAnimationFrame(this.raf);
    if (this.state.playing) this.set({ playing: false });
  }
  toggle() {
    if (this.state.playing) this.pause();
    else this.play();
  }
  dispose() {
    cancelAnimationFrame(this.raf);
    this.subs.clear();
  }
}

export const PlaybackContext = createContext<Playback | null>(null);

export function usePlayback(): Playback {
  const pb = useContext(PlaybackContext);
  if (!pb) throw new Error("PlaybackContext missing");
  return pb;
}

export function usePlaybackState(pb: Playback): PlaybackState {
  return useSyncExternalStore(pb.subscribe, pb.getSnapshot);
}

// ----------------------------------------------------------------------------- sampling
export function findIndex(ts: number[], t: number): number {
  let lo = 0;
  let hi = ts.length - 1;
  if (t <= ts[0]) return 0;
  if (t >= ts[hi]) return hi;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (ts[mid] <= t) lo = mid;
    else hi = mid;
  }
  return lo;
}

export interface Pose {
  x: number;
  y: number;
  yaw: number;
  v: number;
  i: number;
}

const angLerp = (a: number, b: number, f: number) => {
  let d = b - a;
  while (d > Math.PI) d -= 2 * Math.PI;
  while (d < -Math.PI) d += 2 * Math.PI;
  return a + d * f;
};

export function samplePose(ts: number[], track: number[][] | undefined, t: number): Pose {
  if (!track || track.length === 0) return { x: 0, y: 0, yaw: 0, v: 0, i: 0 };
  const i = findIndex(ts, t);
  const j = Math.min(i + 1, track.length - 1);
  const a = track[i];
  const b = track[j];
  const span = ts[j] - ts[i];
  const f = span > 1e-9 ? Math.min(1, Math.max(0, (t - ts[i]) / span)) : 0;
  return { x: a[0] + (b[0] - a[0]) * f, y: a[1] + (b[1] - a[1]) * f, yaw: angLerp(a[2], b[2], f), v: a[3] + (b[3] - a[3]) * f, i };
}

export function signalState(states: string[], ts: number[], t: number): string {
  if (states.length === 0) return "off";
  return states[Math.min(findIndex(ts, t), states.length - 1)] ?? "off";
}

export function signalColor(state: string): string {
  const s = state.toLowerCase();
  if (s.startsWith("green")) return "#34d399";
  if (s.startsWith("yellow") || s.startsWith("amber")) return "#fbbf24";
  if (s.startsWith("red")) return "#fb7185";
  return "#4b5877";
}

const COLORS: Record<string, string> = {
  white: "#e8ecf4",
  black: "#1c2230",
  grey: "#8d97ab",
  gray: "#8d97ab",
  silver: "#b7bfd0",
  red: "#ef4b5f",
  blue: "#3b82f6",
  yellow: "#f5c04d",
  green: "#34d399",
  orange: "#fb923c",
  brown: "#8b5e3c",
  purple: "#a78bfa",
};

export function actorColor(name: string): string {
  if (!name) return "#8d97ab";
  if (name.startsWith("#") || name.startsWith("rgb")) return name;
  return COLORS[name.toLowerCase()] ?? "#8d97ab";
}

/** Nearest signal for a stop line (matches by distance, no assumption about ordering). */
export function nearestSignal(scene: Scene, line: number[][]): number {
  if (line.length === 0 || scene.signals.length === 0) return -1;
  const mx = line.reduce((s, p) => s + p[0], 0) / line.length;
  const my = line.reduce((s, p) => s + p[1], 0) / line.length;
  let best = 0;
  let bd = Infinity;
  scene.signals.forEach((s, k) => {
    const d = (s.x - mx) ** 2 + (s.y - my) ** 2;
    if (d < bd) {
      bd = d;
      best = k;
    }
  });
  return best;
}

export const MS_TO_KMH = 3.6;

export interface EventStyle {
  label: string;
  color: string;
  shape: "dot" | "diamond" | "bar";
}

const EVENT_STYLES: Record<string, EventStyle> = {
  lane_change_start: { label: "Lane change start", color: "#a78bfa", shape: "bar" },
  lane_change_end: { label: "Lane change end", color: "#a78bfa", shape: "bar" },
  speed_change: { label: "Speed change", color: "#60a5fa", shape: "dot" },
  turn_start: { label: "Turn start", color: "#22d3ee", shape: "bar" },
  turn_end: { label: "Turn end", color: "#22d3ee", shape: "bar" },
  stop_line_cross: { label: "Stop line crossed", color: "#cbd5e1", shape: "dot" },
  stopped_at_line: { label: "Stopped at line", color: "#fbbf24", shape: "dot" },
  signal_yellow: { label: "Signal yellow", color: "#fbbf24", shape: "diamond" },
  signal_red: { label: "Signal red", color: "#fb7185", shape: "diamond" },
  signal_green: { label: "Signal green", color: "#34d399", shape: "diamond" },
  min_ttc: { label: "Minimum TTC", color: "#fb7185", shape: "diamond" },
  collision: { label: "Collision", color: "#ef4444", shape: "diamond" },
};

export function eventStyle(type: string): EventStyle {
  return EVENT_STYLES[type] ?? { label: type.replace(/_/g, " "), color: "#94a3c0", shape: "dot" };
}
