import { AnimatePresence, motion } from "framer-motion";
import { Check, ChevronDown, CornerDownRight, RefreshCcw, X } from "lucide-react";
import { Fragment, useState } from "react";
import clsx from "clsx";
import { LEVEL_INFO } from "../../lib/meta";
import { LEVELS, type Attempt, type Finding, type VerifierLevel } from "../../lib/types";
import { Chip, Skeleton } from "../ui/primitives";

type LevelState = "pass" | "fail" | "idle";

function levelSummary(a: Attempt, lv: VerifierLevel): { state: LevelState; ok: number; total: number; findings: Finding[] } {
  const f = a.findings.filter((x) => x.level === lv);
  if (f.length === 0) return { state: "idle", ok: 0, total: 0, findings: f };
  const bad = f.some((x) => !x.ok);
  return { state: bad ? "fail" : "pass", ok: f.filter((x) => x.ok).length, total: f.length, findings: f };
}

function Node({ lv, s, active, onClick }: { lv: VerifierLevel; s: ReturnType<typeof levelSummary>; active: boolean; onClick: () => void }) {
  const color = s.state === "pass" ? "var(--color-ok)" : s.state === "fail" ? "var(--color-bad)" : "var(--color-line-2)";
  return (
    <button
      onClick={onClick}
      aria-pressed={active}
      title={`${lv} ${LEVEL_INFO[lv].title}: ${LEVEL_INFO[lv].tool}`}
      className={clsx(
        "group relative flex min-w-0 flex-1 flex-col items-center gap-1.5 rounded-xl border px-2 py-2.5 transition-colors",
        s.state === "idle" ? "border-dashed border-line bg-transparent" : "bg-bg/60",
        active ? "border-line-2 bg-bg-2" : s.state === "idle" ? "" : "border-line hover:border-line-2",
      )}
    >
      <span
        className="grid h-9 w-9 place-items-center rounded-full border-2 text-[12px] font-bold"
        style={{ borderColor: color, color: s.state === "idle" ? "var(--color-fg-3)" : color, background: s.state === "idle" ? "transparent" : `color-mix(in srgb, ${color} 12%, transparent)`, boxShadow: s.state === "fail" ? `0 0 18px -2px ${color}` : undefined }}
      >
        {s.state === "pass" ? <Check size={17} strokeWidth={3} /> : s.state === "fail" ? <X size={17} strokeWidth={3} /> : lv}
      </span>
      <span className="text-[12px] font-semibold text-fg">{lv}</span>
      <span className="truncate text-[11px] text-fg-2">{LEVEL_INFO[lv].title}</span>
      <span className="mono text-[10.5px] text-fg-3">{s.state === "idle" ? "not reached" : `${s.ok}/${s.total}`}</span>
    </button>
  );
}

function FindingRow({ f }: { f: Finding }) {
  return (
    <li className="flex items-start gap-2.5 py-1.5">
      <span className={clsx("mt-0.5 grid h-4 w-4 shrink-0 place-items-center rounded-full", f.ok ? "bg-ok/15 text-ok" : "bg-bad/15 text-bad")}>{f.ok ? <Check size={11} strokeWidth={3} /> : <X size={11} strokeWidth={3} />}</span>
      <span className="mono w-[34px] shrink-0 text-[11.5px] text-fg-3">{f.level}</span>
      <span className="mono w-[170px] shrink-0 truncate text-[11.5px] text-fg-1" title={f.check}>
        {f.check}
      </span>
      <span className={clsx("min-w-0 text-[12.5px] leading-snug", f.ok ? "text-fg-2" : "text-fg")}>{f.message || (f.ok ? "ok" : "failed")}</span>
    </li>
  );
}

function AttemptCard({ a, index }: { a: Attempt; index: number }) {
  const [open, setOpen] = useState(!a.passed);
  const [level, setLevel] = useState<VerifierLevel | null>(null);
  const failing = a.findings.find((f) => !f.ok);
  const shown = level ? a.findings.filter((f) => f.level === level) : a.findings;
  return (
    <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: index * 0.08 }} className={clsx("rounded-2xl border p-4", a.passed ? "border-ok/35 bg-ok/[0.035]" : "border-bad/35 bg-bad/[0.035]")}>
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2.5">
          <span className="font-display text-[15px] font-semibold text-fg">Attempt {a.index}</span>
          <Chip tone={a.passed ? "ok" : "bad"} dot>
            {a.passed ? "passed" : `failed at ${failing?.level ?? "?"}`}
          </Chip>
        </div>
        <div className="flex items-center gap-4 text-[12px] text-fg-2">
          <span>
            score <span className="num text-[14px] text-fg">{a.score.toFixed(2)}</span>
          </span>
          <span>
            <span className="num text-fg-1">{a.seconds.toFixed(1)}</span> s
          </span>
        </div>
      </div>

      <div className="flex items-stretch gap-1.5">
        {LEVELS.map((lv, i) => {
          const s = levelSummary(a, lv);
          const next = i < LEVELS.length - 1;
          return (
            <Fragment key={lv}>
              <Node lv={lv} s={s} active={level === lv} onClick={() => (setOpen(true), setLevel(level === lv ? null : lv))} />
              {next && (
                <div className="flex w-3 items-center sm:w-5" aria-hidden>
                  <div className="h-0.5 w-full rounded" style={{ background: s.state === "pass" ? "var(--color-ok)" : "var(--color-line-2)", opacity: s.state === "pass" ? 0.7 : 0.6 }} />
                </div>
              )}
            </Fragment>
          );
        })}
      </div>

      {failing && (
        <div className="mt-3 flex items-start gap-2 rounded-xl border border-bad/40 bg-bad/10 p-3" role="alert">
          <X size={15} className="mt-0.5 shrink-0 text-bad" />
          <div className="min-w-0 text-[12.5px] leading-snug text-fg">
            <span className="mono mr-1.5 text-bad">
              {failing.level}:{failing.check}
            </span>
            {failing.message}
          </div>
        </div>
      )}
      {a.passed && <div className="mt-3 text-[12.5px] text-ok">All {a.findings.length} checks passed.</div>}

      <button onClick={() => setOpen((o) => !o)} className="mt-3 flex items-center gap-1.5 text-[12px] font-medium text-fg-1 hover:text-fg" aria-expanded={open}>
        <ChevronDown size={14} className={clsx("transition-transform", !open && "-rotate-90")} />
        {open ? "Hide findings" : `Show ${a.findings.length} findings`}
        {level && <span className="text-fg-2">(filtered: {level})</span>}
      </button>
      <AnimatePresence initial={false}>
        {open && (
          <motion.ul initial={{ height: 0, opacity: 0 }} animate={{ height: "auto", opacity: 1 }} exit={{ height: 0, opacity: 0 }} transition={{ duration: 0.2 }} className="m-0 mt-1 list-none overflow-hidden p-0">
            {shown.map((f, i) => (
              <FindingRow key={`${f.level}-${f.check}-${i}`} f={f} />
            ))}
          </motion.ul>
        )}
      </AnimatePresence>
    </motion.div>
  );
}

function RepairLink({ from }: { from: Attempt }) {
  const f = from.findings.find((x) => !x.ok);
  return (
    <div className="flex items-stretch gap-3 pl-8" aria-label="Repair loop">
      <div className="relative w-5 shrink-0">
        <div className="absolute inset-y-0 left-1/2 w-px bg-[repeating-linear-gradient(180deg,var(--color-accent)_0_4px,transparent_4px_8px)]" />
        <div className="absolute top-1/2 left-1/2 grid h-6 w-6 -translate-x-1/2 -translate-y-1/2 place-items-center rounded-full border border-accent/60 bg-bg-2 text-accent">
          <RefreshCcw size={12} />
        </div>
      </div>
      <div className="my-2 min-w-0 flex-1 rounded-xl border border-dashed border-accent/40 bg-accent/[0.05] px-3 py-2">
        <div className="flex items-center gap-1.5 text-[11px] font-bold tracking-wider text-accent">
          <CornerDownRight size={12} /> REPAIR LOOP · DIAGNOSTICS FED BACK TO THE LIFTER
        </div>
        {f && (
          <div className="mono mt-1 text-[11.5px] leading-snug break-words text-fg-1">
            [{f.level}:{f.check}] {f.message}
          </div>
        )}
      </div>
    </div>
  );
}

export function LadderViz({ attempts, running }: { attempts: Attempt[]; running: boolean }) {
  if (attempts.length === 0) {
    return running ? (
      <div className="space-y-3" aria-busy>
        <Skeleton className="h-36" />
        <p className="m-0 text-[12.5px] text-fg-2">Waiting for the first verifier attempt.</p>
      </div>
    ) : (
      <p className="m-0 text-[13px] text-fg-2">No verifier attempts recorded for this run.</p>
    );
  }
  const passed = attempts.findIndex((a) => a.passed);
  return (
    <div>
      <div className="mb-4 flex flex-wrap items-center gap-2 text-[12.5px] text-fg-1">
        {passed === 0 && <Chip tone="ok">first-pass valid</Chip>}
        {passed > 0 && <Chip tone="warn">valid after {passed} repair{passed > 1 ? "s" : ""}</Chip>}
        {passed < 0 && !running && <Chip tone="bad">no valid scenario within the repair budget</Chip>}
        <span className="text-fg-2">The first failing level stops the ladder and returns structured diagnostics to the agent.</span>
      </div>
      <div className="space-y-0">
        {attempts.map((a, i) => (
          <Fragment key={a.index}>
            <AttemptCard a={a} index={i} />
            {i < attempts.length - 1 && !a.passed && <RepairLink from={a} />}
          </Fragment>
        ))}
      </div>
    </div>
  );
}
