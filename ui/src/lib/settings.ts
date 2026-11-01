import { useSyncExternalStore } from "react";
import { readRaw, writeJSON } from "./storage";
import type { AugmentOptions, ClosedLoopOptions, RenderOptions } from "./types";

export type ApiMode = "auto" | "mock" | "live";

export interface Settings {
  baseUrl: string;
  mode: ApiMode;
  render: RenderOptions;
  augment: AugmentOptions;
  closedLoop: ClosedLoopOptions;
  reduceMotion: boolean;
}

export const THOR_URL = "http://192.168.178.182:8080";
const KEY = "tsf.settings.v1";

export const DEFAULT_SETTINGS: Settings = {
  baseUrl: THOR_URL,
  mode: "auto",
  render: { views: 7, res: "480p", frames: 121, lidar: true },
  augment: { n: 8, strategies: ["weather", "time_of_day", "behavior"] },
  closedLoop: { policy: "refe_sft4", steps: 200 },
  reduceMotion: false,
};

function load(): Settings {
  const s = readRaw<Partial<Settings>>(KEY, {});
  return {
    ...DEFAULT_SETTINGS,
    ...s,
    render: { ...DEFAULT_SETTINGS.render, ...s.render },
    augment: { ...DEFAULT_SETTINGS.augment, ...s.augment },
    closedLoop: { ...DEFAULT_SETTINGS.closedLoop, ...s.closedLoop },
  };
}

let state: Settings = load();
const listeners = new Set<() => void>();

export function getSettings(): Settings {
  return state;
}

export function updateSettings(patch: Partial<Settings>): void {
  state = { ...state, ...patch };
  writeJSON(KEY, state);
  listeners.forEach((l) => l());
}

export function resetSettings(): void {
  state = { ...DEFAULT_SETTINGS };
  writeJSON(KEY, state);
  listeners.forEach((l) => l());
}

export function useSettings(): [Settings, typeof updateSettings] {
  const s = useSyncExternalStore(
    (cb) => {
      listeners.add(cb);
      return () => listeners.delete(cb);
    },
    () => state,
  );
  return [s, updateSettings];
}
