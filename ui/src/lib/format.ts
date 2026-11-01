export const fmt = (v: number | undefined | null, digits = 1): string =>
  v === undefined || v === null || Number.isNaN(v) ? "n/a" : v.toFixed(digits);

export const pct = (v: number | undefined | null, digits = 0): string =>
  v === undefined || v === null || Number.isNaN(v) ? "n/a" : `${(v * 100).toFixed(digits)}%`;

export function duration(sec: number | undefined | null): string {
  if (sec === undefined || sec === null || Number.isNaN(sec)) return "n/a";
  if (sec < 1) return `${Math.round(sec * 1000)} ms`;
  if (sec < 60) return `${sec.toFixed(sec < 10 ? 1 : 0)} s`;
  const m = Math.floor(sec / 60);
  const s = Math.round(sec % 60);
  if (m < 60) return `${m}m ${s.toString().padStart(2, "0")}s`;
  const h = Math.floor(m / 60);
  return `${h}h ${(m % 60).toString().padStart(2, "0")}m`;
}

export function bytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 ** 2) return `${(n / 1024).toFixed(1)} kB`;
  if (n < 1024 ** 3) return `${(n / 1024 ** 2).toFixed(1)} MB`;
  return `${(n / 1024 ** 3).toFixed(2)} GB`;
}

export function timeAgo(iso: string, now = Date.now()): string {
  const t = Date.parse(iso);
  if (Number.isNaN(t)) return iso;
  const s = Math.max(0, (now - t) / 1000);
  if (s < 45) return "just now";
  if (s < 3600) return `${Math.round(s / 60)} min ago`;
  if (s < 86400) return `${Math.round(s / 3600)} h ago`;
  return `${Math.round(s / 86400)} d ago`;
}

export const clamp = (v: number, lo: number, hi: number): number => Math.min(hi, Math.max(lo, v));
export const lerp = (a: number, b: number, t: number): number => a + (b - a) * t;

export function shortDate(iso: string): string {
  const d = new Date(iso);
  return Number.isNaN(d.getTime())
    ? iso
    : d.toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
}

export function kpiLabel(name: string): string {
  const map: Record<string, string> = {
    first_pass_valid_rate: "First-pass validity",
    time_to_valid_s: "Time to valid scenario",
    tok_s: "LLM throughput",
    cosmos_s_per_frame: "Cosmos time per frame",
    sim_rtf: "Sim real-time factor",
  };
  return map[name] ?? name.replace(/_/g, " ");
}
