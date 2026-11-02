import { motion } from "framer-motion";
import { Search } from "lucide-react";
import { useMemo, useState } from "react";
import { Link } from "react-router";
import clsx from "clsx";
import { StatusChip, UcBadge } from "../components/common";
import { EmptyState, PageHeader, Panel, Skeleton } from "../components/ui/primitives";
import { duration, timeAgo } from "../lib/format";
import { useRuns } from "../lib/hooks";
import { displayTitle, USE_CASES } from "../lib/meta";

export default function Runs() {
  const runs = useRuns(100);
  const [uc, setUc] = useState<string>("all");
  const [status, setStatus] = useState<string>("all");
  const [q, setQ] = useState("");

  const rows = useMemo(
    () =>
      (runs.data ?? []).filter((r) => (uc === "all" || r.uc === uc) && (status === "all" || r.status === status) && (!q || `${r.title} ${r.id}`.toLowerCase().includes(q.toLowerCase()))),
    [runs.data, uc, status, q],
  );

  const pill = (active: boolean) => clsx("h-8 rounded-full border px-3 text-[12.5px] font-medium transition-colors", active ? "border-accent/60 bg-accent/15 text-fg" : "border-line bg-bg-1/70 text-fg-1 hover:border-line-2");

  return (
    <div>
      <PageHeader eyebrow="History" title="Runs" description="Every flywheel execution on the Thor, newest first." />
      <Panel flush>
        <div className="flex flex-wrap items-center gap-2 px-4 pb-3">
          <div className="relative min-w-[220px] flex-1 sm:max-w-[320px]">
            <Search size={15} className="pointer-events-none absolute top-1/2 left-3 -translate-y-1/2 text-fg-3" />
            <input className="field !h-9 !py-0 !pl-9" placeholder="Search title or id" value={q} onChange={(e) => setQ(e.target.value)} aria-label="Search runs" />
          </div>
          <div className="flex flex-wrap gap-1.5" role="group" aria-label="Filter by use case">
            <button className={pill(uc === "all")} onClick={() => setUc("all")}>
              All use cases
            </button>
            {USE_CASES.map((u) => (
              <button key={u.id} className={pill(uc === u.id)} onClick={() => setUc(u.id)}>
                {u.short}
              </button>
            ))}
          </div>
          <div className="flex flex-wrap gap-1.5" role="group" aria-label="Filter by status">
            {["all", "running", "queued", "ok", "failed"].map((s) => (
              <button key={s} className={pill(status === s)} onClick={() => setStatus(s)}>
                {s === "all" ? "Any status" : s === "ok" ? "passed" : s}
              </button>
            ))}
          </div>
        </div>
        {!runs.data ? (
          <div className="space-y-2 px-4 pb-4">
            {[0, 1, 2, 3, 4].map((i) => (
              <Skeleton key={i} className="h-11" />
            ))}
          </div>
        ) : rows.length === 0 ? (
          <div className="p-4">
            <EmptyState title="No runs match">Adjust the filters or start a run in the Studio.</EmptyState>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="tbl">
              <thead>
                <tr>
                  <th>Use case</th>
                  <th>Run</th>
                  <th>Status</th>
                  <th className="text-right">Attempts</th>
                  <th className="text-right">Score</th>
                  <th className="text-right">Duration</th>
                  <th className="text-right">Started</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r, i) => (
                  <motion.tr key={r.id} initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: Math.min(i, 12) * 0.02 }}>
                    <td>
                      <UcBadge uc={r.uc} />
                    </td>
                    <td className="max-w-[560px]">
                      <Link to={`/runs/${r.id}`} className="block truncate font-medium text-fg hover:text-accent-2">
                        {displayTitle(r.title)}
                      </Link>
                      <span className="mono text-[11.5px] text-fg-3">{r.id}</span>
                    </td>
                    <td>
                      <StatusChip status={r.status} />
                    </td>
                    <td className="num text-right text-fg-1">{typeof r.kpis?.attempts === "number" ? r.kpis.attempts : "n/a"}</td>
                    <td className="num text-right">{r.score != null ? r.score.toFixed(2) : "n/a"}</td>
                    <td className="num text-right text-fg-1">{duration(r.seconds)}</td>
                    <td className="text-right whitespace-nowrap text-fg-2">{timeAgo(r.created)}</td>
                  </motion.tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Panel>
    </div>
  );
}
