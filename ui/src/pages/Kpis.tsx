import { ArrowDownRight, ArrowUpRight } from "lucide-react";
import { useMemo, useState } from "react";
import { Area, AreaChart, CartesianGrid, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { AXIS, ChartTooltip, GRID_STROKE } from "../components/charts/common";
import { Chip, EmptyState, PageHeader, Panel, Segmented, Skeleton } from "../components/ui/primitives";
import { kpiLabel } from "../lib/format";
import { useKpis } from "../lib/hooks";

interface Def {
  key: string;
  unit: string;
  color: string;
  good: "up" | "down";
  digits: number;
  scale?: number;
  note: string;
}

const DEFS: Def[] = [
  { key: "first_pass_valid_rate", unit: "%", color: "#34d399", good: "up", digits: 1, scale: 100, note: "Share of scenarios valid at L0 to L5 without a repair round." },
  { key: "time_to_valid_s", unit: "s", color: "#22d3ee", good: "down", digits: 0, note: "Wall time from query to the first verified scenario." },
  { key: "tok_s", unit: "tok/s", color: "#8b7cf6", good: "up", digits: 1, note: "Qwen3.8-27B NVFP4 decode throughput on vLLM." },
  { key: "cosmos_s_per_frame", unit: "s/frame", color: "#f5c04d", good: "down", digits: 2, note: "Cosmos 3 Transfer-Auto seconds per frame and view." },
  { key: "sim_rtf", unit: "x", color: "#60a5fa", good: "up", digits: 0, note: "TanitSim 2D: simulated seconds per wall second." },
];

type Range = "24h" | "3d" | "7d" | "all";
const RANGE_MS: Record<Range, number> = { "24h": 86400e3, "3d": 3 * 86400e3, "7d": 7 * 86400e3, all: Infinity };

function fmtTime(ms: number, range: Range): string {
  const d = new Date(ms);
  return range === "24h" || range === "3d" ? d.toLocaleString(undefined, { weekday: "short", hour: "2-digit", minute: "2-digit" }) : d.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

function KpiChart({ def, series, range }: { def: Def; series: [string, number][]; range: Range }) {
  const scale = def.scale ?? 1;
  const data = useMemo(() => {
    const now = series.length ? Date.parse(series[series.length - 1][0]) : Date.now();
    return series.filter((p) => now - Date.parse(p[0]) <= RANGE_MS[range]).map((p) => ({ t: Date.parse(p[0]), v: p[1] * scale }));
  }, [series, range, scale]);
  const first = data[0]?.v;
  const last = data[data.length - 1]?.v;
  const delta = first !== undefined && last !== undefined ? last - first : undefined;
  const improving = delta === undefined ? undefined : def.good === "up" ? delta >= 0 : delta <= 0;
  const vals = data.map((d) => d.v);
  const min = Math.min(...vals);
  const max = Math.max(...vals);
  const mean = vals.reduce((a, b) => a + b, 0) / (vals.length || 1);
  const pad = (max - min) * 0.2 || 1;
  const gid = `kg-${def.key}`;
  return (
    <Panel
      eyebrow={def.unit}
      title={kpiLabel(def.key)}
      actions={
        delta !== undefined && (
          <Chip tone={improving ? "ok" : "warn"}>
            {delta >= 0 ? <ArrowUpRight size={12} /> : <ArrowDownRight size={12} />}
            {delta >= 0 ? "+" : ""}
            {delta.toFixed(def.digits)} {def.unit} over range
          </Chip>
        )
      }
    >
      <div className="mb-2 flex flex-wrap items-end justify-between gap-3">
        <div className="flex items-baseline gap-1.5">
          <span className="num text-[34px] leading-none font-semibold text-fg">{last !== undefined ? last.toFixed(def.digits) : "n/a"}</span>
          <span className="text-[13px] text-fg-2">{def.unit}</span>
        </div>
        <dl className="m-0 flex gap-5 text-[11.5px] text-fg-2">
          {[
            ["min", min],
            ["mean", mean],
            ["max", max],
          ].map(([k, v]) => (
            <div key={String(k)}>
              <dt className="eyebrow !text-[10px]">{String(k)}</dt>
              <dd className="num m-0 text-[13px] text-fg-1">{Number.isFinite(v as number) ? (v as number).toFixed(def.digits) : "n/a"}</dd>
            </div>
          ))}
        </dl>
      </div>
      <div className="h-[220px]" role="img" aria-label={`${kpiLabel(def.key)} over time, latest ${last?.toFixed(def.digits) ?? "n/a"} ${def.unit}`}>
        {data.length < 2 ? (
          <EmptyState title="Not enough data in this range" />
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={data} margin={{ top: 8, right: 8, left: -12, bottom: 0 }}>
              <defs>
                <linearGradient id={gid} x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0" stopColor={def.color} stopOpacity={0.4} />
                  <stop offset="1" stopColor={def.color} stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid stroke={GRID_STROKE} strokeDasharray="3 6" vertical={false} />
              <XAxis dataKey="t" type="number" domain={["dataMin", "dataMax"]} scale="time" tickFormatter={(v: number) => fmtTime(v, range)} {...AXIS} minTickGap={42} />
              <YAxis domain={[min - pad, max + pad]} tickFormatter={(v: number) => v.toFixed(def.digits > 1 ? 1 : def.digits)} width={52} {...AXIS} />
              <Tooltip
                cursor={{ stroke: "#2b3858" }}
                content={<ChartTooltip title={(l) => (typeof l === "number" ? new Date(l).toLocaleString() : l)} format={(_, v) => `${v.toFixed(def.digits)} ${def.unit}`} />}
              />
              <ReferenceLine y={mean} stroke={def.color} strokeOpacity={0.35} strokeDasharray="4 5" />
              <Area type="monotone" dataKey="v" name={kpiLabel(def.key)} stroke={def.color} strokeWidth={2} fill={`url(#${gid})`} dot={false} activeDot={{ r: 4, stroke: "#060912", strokeWidth: 2 }} isAnimationActive={false} />
            </AreaChart>
          </ResponsiveContainer>
        )}
      </div>
      <p className="mt-2 mb-0 text-[12px] text-fg-2">{def.note}</p>
    </Panel>
  );
}

export default function Kpis() {
  const kpis = useKpis();
  const [range, setRange] = useState<Range>("7d");
  return (
    <div>
      <PageHeader
        eyebrow="Telemetry"
        title="KPIs"
        description="Time series from GET /api/kpis. Dashed lines mark the mean of the selected range."
        actions={
          <Segmented
            label="Time range"
            value={range}
            onChange={setRange}
            options={[
              { value: "24h", label: "24 h" },
              { value: "3d", label: "3 d" },
              { value: "7d", label: "7 d" },
              { value: "all", label: "All" },
            ]}
          />
        }
      />
      {!kpis.data ? (
        <div className="grid grid-cols-1 gap-5 xl:grid-cols-2">
          {[0, 1, 2, 3].map((i) => (
            <Skeleton key={i} className="h-[380px]" />
          ))}
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-5 xl:grid-cols-2">
          {DEFS.filter((d) => kpis.data.series[d.key]).map((d) => (
            <KpiChart key={d.key} def={d} series={kpis.data.series[d.key]} range={range} />
          ))}
          <Panel eyebrow="Latest" title="Snapshot">
            <dl className="m-0 divide-y divide-line/70">
              {Object.entries(kpis.data.latest).map(([k, v]) => (
                <div key={k} className="flex items-baseline justify-between py-2">
                  <dt className="text-[13px] text-fg-1">{kpiLabel(k)}</dt>
                  <dd className="num m-0 text-[15px] text-fg">{typeof v === "number" ? (v <= 1 && v >= 0 ? v.toFixed(3) : v.toFixed(2)) : String(v)}</dd>
                </div>
              ))}
            </dl>
          </Panel>
        </div>
      )}
    </div>
  );
}
