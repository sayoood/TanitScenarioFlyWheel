import { motion } from "framer-motion";
import type { ReactNode } from "react";
import { ArrowDownRight, ArrowUpRight, Cpu, MemoryStick, Server, Terminal } from "lucide-react";
import { Link } from "react-router";
import { FlywheelHero, STAGE_NODE } from "../components/flywheel/FlywheelHero";
import { StatusChip, UcBadge } from "../components/common";
import { RadialGauge, Sparkline } from "../components/ui/Gauge";
import { Chip, EmptyState, PageHeader, Panel, Skeleton } from "../components/ui/primitives";
import { fmt, pct, timeAgo, duration } from "../lib/format";
import { STAGE_LABEL } from "../lib/meta";
import { useLogLines } from "../lib/events";
import { useActivity, useHarness, useHealth, useKpis, useRuns } from "../lib/hooks";
import { useSettings } from "../lib/settings";
import type { Health, Kpis } from "../lib/types";

interface TileDef {
  key: string;
  title: string;
  hint: string;
  color: string;
  good: "up" | "down";
  format: (v: number) => string;
  unit: string;
  deltaFormat: (d: number) => string;
}

const TILES: TileDef[] = [
  { key: "first_pass_valid_rate", title: "First-pass validity", hint: "Scenarios that pass L0 to L5 without repair", color: "var(--color-ok)", good: "up", format: (v) => (v * 100).toFixed(0), unit: "%", deltaFormat: (d) => `${d >= 0 ? "+" : ""}${(d * 100).toFixed(1)} pp` },
  { key: "time_to_valid_s", title: "Time to valid", hint: "Query to first verified scenario", color: "var(--color-accent-2)", good: "down", format: (v) => v.toFixed(0), unit: "s", deltaFormat: (d) => `${d >= 0 ? "+" : ""}${d.toFixed(0)} s` },
  { key: "tok_s", title: "LLM throughput", hint: "Qwen3.8-27B NVFP4 on vLLM", color: "var(--color-accent)", good: "up", format: (v) => v.toFixed(1), unit: "tok/s", deltaFormat: (d) => `${d >= 0 ? "+" : ""}${d.toFixed(1)}` },
  { key: "cosmos_s_per_frame", title: "Cosmos time per frame", hint: "Transfer-Auto, seconds per frame and view", color: "var(--color-gold)", good: "down", format: (v) => v.toFixed(2), unit: "s/frame", deltaFormat: (d) => `${d >= 0 ? "+" : ""}${d.toFixed(2)} s` },
  { key: "sim_rtf", title: "Sim real-time factor", hint: "TanitSim 2D, simulated time over wall time", color: "var(--color-info)", good: "up", format: (v) => v.toFixed(0), unit: "x", deltaFormat: (d) => `${d >= 0 ? "+" : ""}${d.toFixed(0)}x` },
];

function KpiTile({ def, kpis }: { def: TileDef; kpis: Kpis | undefined }) {
  if (!kpis) return <Skeleton className="h-[148px]" />;
  const s = kpis.series[def.key] ?? [];
  const values = s.map((p) => p[1]);
  const latest = kpis.latest[def.key] ?? values[values.length - 1];
  const ref = values.length > 28 ? values[values.length - 29] : values[0];
  const delta = latest !== undefined && ref !== undefined ? latest - ref : undefined;
  const improving = delta === undefined ? undefined : def.good === "up" ? delta >= 0 : delta <= 0;
  return (
    <Panel className="overflow-hidden" bodyClassName="!p-4">
      <div className="eyebrow" title={def.hint}>
        {def.title}
      </div>
      <div className="mt-2 flex items-baseline justify-between gap-2">
        <div className="flex items-baseline gap-1.5">
          <span className="num text-[34px] leading-none font-semibold text-fg">{latest === undefined ? "n/a" : def.format(latest)}</span>
          <span className="text-[13px] text-fg-2">{def.unit}</span>
        </div>
        {delta !== undefined && (
          <Chip tone={improving ? "ok" : "warn"} className="shrink-0" title="Change over the last 7 days">
            {delta >= 0 ? <ArrowUpRight size={12} /> : <ArrowDownRight size={12} />}
            {def.deltaFormat(delta)}
          </Chip>
        )}
      </div>
      <div className="mt-3 -mb-1">
        <Sparkline data={values.slice(-48)} color={def.color} height={44} />
      </div>
      <div className="mt-1 text-[11.5px] text-fg-3">12-day trend · chip: 7-day change</div>
    </Panel>
  );
}

function ThorPanel({ health }: { health: Health | undefined }) {
  const t = health?.thor;
  return (
    <Panel
      className="h-full"
      eyebrow="Jetson AGX Thor"
      title={t ? `${t.host} resources` : "Resources"}
      actions={
        health && (
          <Chip tone="neutral" title="Unified memory is shared by the LLM and one Cosmos model at a time">
            <Server size={12} /> 122 GB unified
          </Chip>
        )
      }
    >
      {!t ? (
        <div className="grid grid-cols-4 gap-4">
          {[0, 1, 2, 3].map((i) => (
            <Skeleton key={i} className="h-32" />
          ))}
        </div>
      ) : (
        <div className="grid min-h-[210px] grid-cols-2 content-center items-center justify-items-center gap-x-4 gap-y-3 sm:grid-cols-4">
          <RadialGauge size={150} label="Unified memory" value={t.mem_used_gb} max={t.mem_total_gb} display={`${Math.round((t.mem_used_gb / t.mem_total_gb) * 100)}%`} sub={`${fmt(t.mem_used_gb, 1)} / ${t.mem_total_gb} GB`} color="var(--color-accent)" />
          <RadialGauge size={150} label="GPU" value={t.gpu_util} max={1} display={`${Math.round(t.gpu_util * 100)}%`} sub="utilisation" color="var(--color-accent-2)" warnAt={0.985} />
          <RadialGauge size={150} label="Power" value={t.power_w} max={t.power_max_w ?? 130} display={`${t.power_w}`} sub={`W of ${t.power_max_w ?? 130}`} color="var(--color-gold)" warnAt={0.92} />
          <RadialGauge size={150} label="Disk" value={(t.disk_total_gb ?? 1000) - t.disk_free_gb} max={t.disk_total_gb ?? 1000} display={`${t.disk_free_gb}`} sub={`GB free${t.disk_total_gb ? "" : " (of ~1 TB)"}`} color="var(--color-ok)" warnAt={0.9} />
        </div>
      )}
    </Panel>
  );
}

function ServiceRow({ icon, name, detail, up, standby, right }: { icon: ReactNode; name: string; detail: string; up: boolean; standby?: boolean; right?: ReactNode }) {
  return (
    <div className="flex items-center gap-3 rounded-xl border border-line bg-bg/40 px-3.5 py-3">
      <div className="grid h-9 w-9 shrink-0 place-items-center rounded-[10px] border border-line-2 bg-bg-2 text-fg-1">{icon}</div>
      <div className="min-w-0 flex-1">
        <div className="truncate text-[13.5px] font-semibold text-fg">{name}</div>
        <div className="truncate text-[12px] text-fg-2">{detail}</div>
      </div>
      {right}
      {up ? (
        <Chip tone="ok">
          <span className="live-dot !h-1.5 !w-1.5" /> UP
        </Chip>
      ) : standby ? (
        <Chip title="Loaded on demand by the GPU scheduler">STANDBY</Chip>
      ) : (
        <Chip tone="bad" dot>
          DOWN
        </Chip>
      )}
    </div>
  );
}

function ServicesPanel({ health }: { health: Health | undefined }) {
  const s = health?.services;
  return (
    <Panel className="h-full" eyebrow="Services" title="Model endpoints">
      {!s ? (
        <div className="space-y-2.5">
          {[0, 1, 2].map((i) => (
            <Skeleton key={i} className="h-[62px]" />
          ))}
        </div>
      ) : (
        <div className="space-y-2.5">
          <ServiceRow
            icon={<Cpu size={18} />}
            name="LLM · lifter agents"
            detail={`${s.llm.model ?? "unknown model"}${s.llm.up ? (s.llm.tok_s ? ` · ${s.llm.tok_s.toFixed(1)} tok/s` : " · idle") : ""}`}
            up={s.llm.up}
          />
          <ServiceRow icon={<Terminal size={18} />} name="Cosmos Sim" detail={s.cosmos_sim.model ?? "Cosmos3-Nano-Sim-Auto · closed loop"} up={s.cosmos_sim.up} standby />
          <ServiceRow icon={<MemoryStick size={18} />} name="Cosmos Transfer" detail={s.cosmos_transfer.model ?? "Cosmos3-Nano-Transfer-Auto · 7 cams + LiDAR"} up={s.cosmos_transfer.up} standby />
        </div>
      )}
    </Panel>
  );
}

function RecentRuns() {
  const runs = useRuns(8);
  return (
    <Panel
      eyebrow="Activity"
      title="Recent runs"
      actions={
        <Link to="/runs" className="btn btn-sm">
          All runs
        </Link>
      }
      flush
    >
      {!runs.data ? (
        <div className="space-y-2 px-4 pb-4">
          {[0, 1, 2, 3].map((i) => (
            <Skeleton key={i} className="h-10" />
          ))}
        </div>
      ) : runs.data.length === 0 ? (
        <div className="p-4">
          <EmptyState title="No runs yet">Start one in the Studio.</EmptyState>
        </div>
      ) : (
        <table className="tbl">
          <thead>
            <tr>
              <th>Use case</th>
              <th>Title</th>
              <th>Status</th>
              <th className="text-right">Score</th>
              <th className="text-right">Duration</th>
              <th className="text-right">Started</th>
            </tr>
          </thead>
          <tbody>
            {runs.data.slice(0, 7).map((r, i) => (
              <motion.tr key={r.id} initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.03 }}>
                <td>
                  <UcBadge uc={r.uc} />
                </td>
                <td className="max-w-[420px]">
                  <Link to={`/runs/${r.id}`} className="block truncate font-medium text-fg hover:text-accent-2">
                    {r.title}
                  </Link>
                  <span className="mono text-[11.5px] text-fg-3">{r.id}</span>
                </td>
                <td>
                  <StatusChip status={r.status} />
                </td>
                <td className="num text-right">{r.score != null ? r.score.toFixed(2) : "n/a"}</td>
                <td className="num text-right text-fg-1">{duration(r.seconds)}</td>
                <td className="text-right whitespace-nowrap text-fg-2">{timeAgo(r.created)}</td>
              </motion.tr>
            ))}
          </tbody>
        </table>
      )}
    </Panel>
  );
}

function LogPanel() {
  const lines = useLogLines();
  const shown = lines.slice(-9).reverse();
  return (
    <Panel eyebrow="Event stream" title="Live log" actions={<Chip tone="neutral">WS /api/events</Chip>}>
      {shown.length === 0 ? (
        <div className="text-[13px] text-fg-2">Waiting for events. Stage transitions and operator actions appear here.</div>
      ) : (
        <ul className="m-0 list-none space-y-1.5 p-0" aria-live="polite">
          {shown.map((l) => (
            <li key={l.id} className="mono flex gap-2 text-[11.5px] leading-relaxed">
              <span className="shrink-0 text-fg-3">{new Date(l.ts).toLocaleTimeString([], { hour12: false })}</span>
              {l.run_id && <span className="shrink-0 text-accent">{l.run_id}</span>}
              <span className={l.level === "warn" ? "text-warn" : "text-fg-1"}>{l.message}</span>
            </li>
          ))}
        </ul>
      )}
    </Panel>
  );
}

export default function Dashboard() {
  const health = useHealth();
  const kpis = useKpis();
  const harness = useHarness();
  const act = useActivity();
  const [settings] = useSettings();
  const gens = harness.data?.generations;
  const gen = gens?.[gens.length - 1];
  const node = act.stage ? STAGE_NODE[act.stage] : undefined;
  const latest = kpis.data?.latest;

  return (
    <div>
      <PageHeader
        eyebrow="Mission control"
        title={
          <>
            Scenario <span className="text-grad">Flywheel</span>
          </>
        }
        description="Five sources, one typed IR, hard verifiers as the reward signal. Everything below is live from the Thor, or mock data when it is unreachable."
        actions={
          <Link to="/studio" className="btn btn-primary">
            New run
          </Link>
        }
      />

      <div className="grid grid-cols-12 gap-5">
        <Panel accent className="col-span-12 overflow-hidden" bodyClassName="!px-3 !pb-3 !pt-2">
          <div className="flex items-center justify-between px-2 pt-1 pb-1">
            <div>
              <div className="eyebrow">The loop</div>
              <div className="text-[15px] font-semibold">Flywheel in motion</div>
            </div>
            <div className="flex items-center gap-2">
              {latest && <Chip tone="ok">validity after repairs {pct(latest.valid_after_3_repairs, 0)}</Chip>}
              {act.active ? (
                <Chip tone="info">
                  <span className="live-dot !h-1.5 !w-1.5" /> {act.stage ? STAGE_LABEL[act.stage] : "starting"}
                </Chip>
              ) : (
                <Chip>idle</Chip>
              )}
            </div>
          </div>
          <FlywheelHero
            activeNode={node}
            activeUc={act.uc}
            runLabel={act.runId ? `${act.runId}${act.stage ? ` · ${act.stage}` : ""}` : undefined}
            generation={gen}
            busy={act.active}
            queue={act.queue}
            reduceMotion={settings.reduceMotion}
          />
        </Panel>

        <div className="col-span-12 xl:col-span-7">
          <ThorPanel health={health.data} />
        </div>
        <div className="col-span-12 xl:col-span-5">
          <ServicesPanel health={health.data} />
        </div>

        <div className="col-span-12 grid grid-cols-2 gap-5 lg:grid-cols-3 xl:grid-cols-5" role="list" aria-label="Key performance indicators">
          {TILES.map((t) => (
            <div role="listitem" key={t.key}>
              <KpiTile def={t} kpis={kpis.data} />
            </div>
          ))}
        </div>

        <div className="col-span-12 xl:col-span-8">
          <RecentRuns />
        </div>
        <div className="col-span-12 xl:col-span-4">
          <LogPanel />
        </div>
      </div>
    </div>
  );
}
