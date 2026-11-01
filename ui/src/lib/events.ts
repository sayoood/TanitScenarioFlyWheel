import { useSyncExternalStore } from "react";
import type { FlywheelEvent } from "./types";

export interface LogLine {
  id: number;
  ts: string;
  level: string;
  run_id?: string;
  message: string;
}

/** Small external store: ring buffer of log lines plus a pulse counter for the hero animation. */
let lines: LogLine[] = [];
let seq = 0;
let pulses = 0;
const subs = new Set<() => void>();
const pulseSubs = new Set<(e: FlywheelEvent) => void>();

function notify() {
  subs.forEach((s) => s());
}

export function pushEvent(e: FlywheelEvent): void {
  pulseSubs.forEach((p) => p(e));
  pulses++;
  if (e.type !== "log") return;
  const key = `${String(e.ts)}|${e.run_id ?? ""}|${String(e.message)}`;
  if (lines.some((l) => `${l.ts}|${l.run_id ?? ""}|${l.message}` === key)) return;
  const line: LogLine = {
    id: ++seq,
    ts: typeof e.ts === "string" ? e.ts : new Date().toISOString(),
    level: typeof e.level === "string" ? e.level : "info",
    run_id: e.run_id,
    message: typeof e.message === "string" ? e.message : JSON.stringify(e),
  };
  lines = [...lines.slice(-199), line];
  notify();
}

export function useLogLines(): LogLine[] {
  return useSyncExternalStore(
    (cb) => {
      subs.add(cb);
      return () => subs.delete(cb);
    },
    () => lines,
  );
}

export function onPulse(fn: (e: FlywheelEvent) => void): () => void {
  pulseSubs.add(fn);
  return () => pulseSubs.delete(fn);
}

export const eventCount = () => pulses;
