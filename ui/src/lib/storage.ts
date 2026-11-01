/** localStorage helpers that never throw (private windows, blocked storage, thumbnail capture). */
export function readJSON<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(key);
    return raw ? ({ ...(typeof fallback === "object" && fallback ? fallback : {}), ...JSON.parse(raw) } as T) : fallback;
  } catch {
    return fallback;
  }
}

export function readRaw<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(key);
    return raw ? (JSON.parse(raw) as T) : fallback;
  } catch {
    return fallback;
  }
}

export function writeJSON(key: string, value: unknown): void {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    /* storage unavailable: the app keeps working without persistence */
  }
}
