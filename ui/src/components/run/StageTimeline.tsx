import { motion } from "framer-motion";
import { useEffect, useState } from "react";
import clsx from "clsx";
import { duration } from "../../lib/format";
import { STAGE_HINT, STAGE_LABEL } from "../../lib/meta";
import type { RunDetail, Stage } from "../../lib/types";
import { StatusIcon } from "../common";

const TONE: Record<string, string> = {
  ok: "var(--color-ok)",
  failed: "var(--color-bad)",
  running: "var(--color-accent-2)",
  pending: "var(--color-line-2)",
  skipped: "var(--color-line)",
};

function useElapsed(started: string | undefined, active: boolean): number | undefined {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!active) return;
    const id = setInterval(() => setNow(Date.now()), 500);
    return () => clearInterval(id);
  }, [active]);
  if (!active || !started) return undefined;
  return Math.max(0, (now - Date.parse(started)) / 1000);
}

function StageCard({ s, total, index }: { s: Stage; total: number; index: number }) {
  const running = s.status === "running";
  const elapsed = useElapsed(s.started, running);
  const secs = s.seconds ?? elapsed;
  const share = total > 0 && s.seconds ? Math.max(0.03, s.seconds / total) : 0;
  return (
    <motion.li
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.035 }}
      className={clsx(
        "relative overflow-hidden rounded-xl border p-3",
        running ? "border-accent-2/50 bg-accent-2/[0.06]" : s.status === "failed" ? "border-bad/45 bg-bad/[0.06]" : "border-line bg-bg/50",
        s.status === "skipped" && "opacity-55",
      )}
      aria-label={`${STAGE_LABEL[s.name]}: ${s.status}`}
    >
      {running && <span className="absolute inset-x-0 top-0 h-0.5 animate-shimmer bg-[linear-gradient(90deg,transparent,var(--color-accent-2),transparent)] bg-[length:200%_100%]" />}
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <StatusIcon status={s.status} size={15} />
          <span className="text-[13px] font-semibold text-fg">{STAGE_LABEL[s.name]}</span>
        </div>
        <span className="num text-[13px] text-fg-1" title={s.seconds === undefined && !running ? "not run" : undefined}>
          {s.status === "skipped" ? "skipped" : secs !== undefined ? duration(secs) : "n/a"}
        </span>
      </div>
      <p className="mt-1.5 mb-0 line-clamp-2 min-h-[32px] text-[11.5px] leading-snug text-fg-2" title={s.detail ?? STAGE_HINT[s.name]}>
        {s.detail ?? STAGE_HINT[s.name]}
      </p>
      <div className="mt-2 h-1 overflow-hidden rounded-full bg-bg-3">
        <div className="h-full rounded-full" style={{ width: `${(running ? 0.35 : share) * 100}%`, background: TONE[s.status], opacity: s.status === "skipped" ? 0.3 : 1 }} />
      </div>
    </motion.li>
  );
}

export function StageTimeline({ run }: { run: RunDetail }) {
  const total = run.stages.reduce((a, s) => a + (s.seconds ?? 0), 0);
  // the strip uses sqrt widths so a 1 s stage stays visible next to a 1 h render
  const weights = run.stages.map((s) => (s.status === "skipped" ? 0.4 : Math.max(1.2, Math.sqrt(s.seconds ?? (s.status === "running" ? 30 : 4)))));
  const sum = weights.reduce((a, b) => a + b, 0) || 1;
  return (
    <div>
      <div className="mb-4 flex h-3 w-full gap-[3px] overflow-hidden rounded-full" role="img" aria-label="Stage durations (square-root scaled)">
        {run.stages.map((s, i) => (
          <motion.div
            key={s.name}
            title={`${STAGE_LABEL[s.name]}: ${s.status}${s.seconds !== undefined ? `, ${duration(s.seconds)}` : ""}`}
            initial={{ scaleX: 0 }}
            animate={{ scaleX: 1 }}
            transition={{ delay: i * 0.04, duration: 0.4 }}
            style={{ width: `${(weights[i] / sum) * 100}%`, background: TONE[s.status], transformOrigin: "left", opacity: s.status === "pending" || s.status === "skipped" ? 0.45 : 1 }}
            className={clsx("h-full rounded-[3px]", s.status === "running" && "animate-blink")}
          />
        ))}
      </div>
      <ol className="m-0 grid list-none grid-cols-2 gap-2.5 p-0 md:grid-cols-3 xl:grid-cols-5">
        {run.stages.map((s, i) => (
          <StageCard key={s.name} s={s} total={total} index={i} />
        ))}
      </ol>
      <div className="mt-3 flex flex-wrap items-center gap-4 text-[12px] text-fg-2">
        <span>
          Total <span className="num text-fg">{run.seconds !== undefined || total > 0 ? duration(run.seconds ?? total) : "n/a"}</span>
        </span>
        {run.llm && (
          <span>
            LLM <span className="num text-fg">{run.llm.calls}</span> calls at <span className="num text-fg">{run.llm.tok_per_s.toFixed(1)}</span> tok/s
          </span>
        )}
      </div>
    </div>
  );
}
