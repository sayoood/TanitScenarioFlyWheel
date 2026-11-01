import { motion } from "framer-motion";
import { ArrowLeftRight, CarFront, Gauge, OctagonAlert, Timer, TrafficCone } from "lucide-react";
import type { LucideIcon } from "lucide-react";
import clsx from "clsx";
import type { RunDetail } from "../../lib/types";
import { Chip, EmptyState } from "../ui/primitives";

interface Tile {
  label: string;
  value: string;
  unit?: string;
  tone: "ok" | "bad" | "warn" | "neutral";
  hint: string;
  Icon: LucideIcon;
}

const toneColor = { ok: "var(--color-ok)", bad: "var(--color-bad)", warn: "var(--color-warn)", neutral: "var(--color-fg-1)" };

function num(v: unknown): number | undefined {
  return typeof v === "number" && Number.isFinite(v) ? v : undefined;
}

function flatten(prefix: string, v: unknown, out: [string, string][]) {
  if (v && typeof v === "object" && !Array.isArray(v)) {
    for (const [k, x] of Object.entries(v)) flatten(prefix ? `${prefix} / ${k}` : k, x, out);
  } else {
    out.push([prefix, typeof v === "number" ? (Number.isInteger(v) ? String(v) : v.toFixed(3)) : Array.isArray(v) ? v.join(", ") : String(v)]);
  }
}

export function MetricsPanel({ run }: { run: RunDetail }) {
  const m = run.metrics ?? {};
  if (Object.keys(m).length === 0) {
    return <EmptyState title="No metrics yet">Metrics appear after the simulate stage has finished.</EmptyState>;
  }
  const ttc = num(m.min_ttc);
  const decel = num(m.max_decel ?? m.ego_max_decel);
  const collision = typeof m.collision === "boolean" ? m.collision : undefined;
  const rawTurns = m.turns ?? m.ego_turns;
  const turns = (Array.isArray(rawTurns) ? (rawTurns.length ? rawTurns.join(", ") : "none") : rawTurns) as { left?: number; right?: number } | number | string | undefined;
  const turnText = typeof turns === "object" && turns ? `L${turns.left ?? 0} · R${turns.right ?? 0}` : turns !== undefined ? String(turns) : "n/a";
  const lc = num(m.lane_changes ?? m.ego_lane_changes_left);
  const red = num(m.red_light_violations);

  const tiles: Tile[] = [
    { label: "Min TTC", value: ttc != null ? ttc.toFixed(2) : "n/a", unit: "s", tone: ttc == null ? "neutral" : ttc >= 1.5 ? "ok" : "bad", hint: `Threshold 1.5 s${m.min_ttc_actor ? `, closest actor ${m.min_ttc_actor}` : ""}`, Icon: Timer },
    { label: "Collision", value: collision === undefined ? "n/a" : collision ? "Yes" : "None", tone: collision === undefined ? "neutral" : collision ? "bad" : "ok", hint: "OBB overlap anywhere in the run", Icon: OctagonAlert },
    { label: "Max decel", value: decel != null ? decel.toFixed(2) : "n/a", unit: "m/s²", tone: decel == null ? "neutral" : decel < 4 ? "ok" : "warn", hint: "Limit 4.0 m/s²", Icon: Gauge },
    { label: "Turns", value: turnText, tone: "neutral", hint: "Completed ego turns by direction", Icon: ArrowLeftRight },
    { label: "Lane changes", value: lc !== undefined ? String(lc) : "n/a", tone: "neutral", hint: "Net ego lane changes", Icon: CarFront },
    { label: "Red-light violations", value: red !== undefined ? String(red) : "n/a", tone: red === undefined ? "neutral" : red === 0 ? "ok" : "bad", hint: "Ego crossing a stop line on red", Icon: TrafficCone },
  ];

  const known = new Set(["min_ttc", "min_ttc_actor", "collision", "max_decel", "turns", "lane_changes", "red_light_violations"]);
  const extra: [string, string][] = [];
  for (const [k, v] of Object.entries(m)) if (!known.has(k)) flatten(k, v, extra);

  return (
    <div>
      <div className="grid grid-cols-2 gap-2.5 sm:grid-cols-3" role="list">
        {tiles.map((t, i) => (
          <motion.div
            role="listitem"
            key={t.label}
            title={t.hint}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.04 }}
            className="rounded-xl border bg-bg/50 p-3"
            style={{ borderColor: t.tone === "neutral" ? "var(--color-line)" : `color-mix(in srgb, ${toneColor[t.tone]} 35%, transparent)` }}
          >
            <div className="flex items-center gap-1.5 text-[11px] font-semibold tracking-wide text-fg-2 uppercase">
              <t.Icon size={13} style={{ color: toneColor[t.tone] }} /> {t.label}
            </div>
            <div className="mt-1.5 flex items-baseline gap-1">
              <span className="num text-[24px] leading-none font-semibold" style={{ color: t.tone === "neutral" ? "var(--color-fg)" : toneColor[t.tone] }}>
                {t.value}
              </span>
              {t.unit && <span className="text-[12px] text-fg-2">{t.unit}</span>}
            </div>
          </motion.div>
        ))}
      </div>
      {extra.length > 0 && (
        <dl className="m-0 mt-4 grid grid-cols-1 gap-x-6 sm:grid-cols-2">
          {extra.map(([k, v]) => (
            <div key={k} className={clsx("flex items-baseline justify-between gap-3 border-b border-line/60 py-1.5")}>
              <dt className="truncate text-[12px] text-fg-2">{k.replace(/_/g, " ")}</dt>
              <dd className="mono m-0 text-[12px] text-fg">{v}</dd>
            </div>
          ))}
        </dl>
      )}
      {m.min_ttc_actor && (
        <div className="mt-3">
          <Chip tone="neutral">closest approach: {String(m.min_ttc_actor)}</Chip>
        </div>
      )}
    </div>
  );
}
