import { AnimatePresence, motion } from "framer-motion";
import { Check, FlaskConical, GitMerge, Lock, RotateCcw, ShieldAlert, Sparkles, X } from "lucide-react";
import { useEffect, useMemo, useState, type ReactNode } from "react";
import { CartesianGrid, ComposedChart, Legend, Line, ResponsiveContainer, Tooltip, XAxis, YAxis, Area } from "recharts";
import clsx from "clsx";
import { AXIS, ChartTooltip, GRID_STROKE } from "../components/charts/common";
import { Chip, PageHeader, Panel, Segmented, Skeleton, Slider, Toggle } from "../components/ui/primitives";
import { timeAgo } from "../lib/format";
import { useHarness, usePutKnobs, useSetLesson } from "../lib/hooks";
import type { AgentKnobs, HarnessState, Lesson, LessonKind, LessonStatus, Revision } from "../lib/types";

function Stat({ label, value, sub, tone }: { label: string; value: string; sub: string; tone?: "ok" | "warn" | "neutral" }) {
  return (
    <Panel bodyClassName="!p-4">
      <div className="eyebrow">{label}</div>
      <div className={clsx("num mt-1.5 text-[32px] leading-none font-semibold", tone === "ok" ? "text-ok" : tone === "warn" ? "text-warn" : "text-fg")}>{value}</div>
      <div className="mt-1.5 text-[12px] text-fg-2">{sub}</div>
    </Panel>
  );
}

function GenerationChart({ data }: { data: HarnessState["generations"] }) {
  const first = data[0];
  const last = data[data.length - 1];
  return (
    <Panel
      eyebrow="Self-improvement"
      title="Generation score"
      actions={first && last && <Chip tone="ok">+{(last.score - first.score).toFixed(2)} score since gen {first.gen}</Chip>}
    >
      <div className="h-[290px]" role="img" aria-label={`Verifier score and valid rate per generation, from ${first?.score.toFixed(2)} to ${last?.score.toFixed(2)}`}>
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={data} margin={{ top: 10, right: 12, left: -14, bottom: 0 }}>
            <defs>
              <linearGradient id="gen-score" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0" stopColor="#8b7cf6" stopOpacity={0.4} />
                <stop offset="1" stopColor="#8b7cf6" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid stroke={GRID_STROKE} strokeDasharray="3 6" vertical={false} />
            <XAxis dataKey="gen" {...AXIS} label={{ value: "generation", position: "insideBottomRight", offset: -2, fill: "#7684a2", fontSize: 11 }} />
            <YAxis domain={[0.4, 1]} tickFormatter={(v: number) => v.toFixed(1)} width={48} {...AXIS} />
            <Tooltip content={<ChartTooltip title={(l) => `Generation ${l}`} format={(_, v) => v.toFixed(3)} />} cursor={{ stroke: "#2b3858" }} />
            <Legend iconType="circle" wrapperStyle={{ fontSize: 12, color: "#bac5da", paddingTop: 6 }} />
            <Area type="monotone" dataKey="score" name="Verifier score" stroke="none" fill="url(#gen-score)" legendType="none" tooltipType="none" isAnimationActive={false} />
            <Line type="monotone" dataKey="score" name="Verifier score" stroke="#8b7cf6" strokeWidth={2.5} dot={{ r: 3, fill: "#8b7cf6", stroke: "#060912", strokeWidth: 2 }} activeDot={{ r: 5 }} isAnimationActive={false} />
            <Line type="monotone" dataKey="valid_rate" name="Valid rate" stroke="#22d3ee" strokeWidth={2.5} dot={{ r: 3, fill: "#22d3ee", stroke: "#060912", strokeWidth: 2 }} activeDot={{ r: 5 }} isAnimationActive={false} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </Panel>
  );
}

function KnobsPanel({ state }: { state: HarnessState }) {
  const put = usePutKnobs();
  const [agent, setAgent] = useState(state.agents[0] ?? "lifter");
  const stored = state.knobs[agent];
  const [draft, setDraft] = useState<AgentKnobs>(stored ?? {});
  useEffect(() => setDraft(state.knobs[agent] ?? {}), [agent, state.knobs]);
  const dirty = JSON.stringify(draft) !== JSON.stringify(stored ?? {});
  const num = (k: string, d = 0) => (typeof draft[k] === "number" ? (draft[k] as number) : d);
  const set = (k: string, v: number | boolean) => setDraft((x) => ({ ...x, [k]: v }));
  return (
    <Panel eyebrow="PUT /api/harness/knobs" title="Agent knobs" actions={dirty && <Chip tone="warn">unsaved</Chip>}>
      <div className="mb-4 flex flex-wrap gap-1.5" role="tablist" aria-label="Agent">
        <Segmented size="sm" label="Agent" value={agent} onChange={setAgent} options={state.agents.map((a) => ({ value: a, label: a }))} />
      </div>
      <div className="space-y-4">
        <div className="flex items-center justify-between gap-4">
          <div>
            <div className="text-[13px] font-medium text-fg">Thinking mode</div>
            <div className="text-[12px] text-fg-2">Reasoning tokens before the TSIR. Slower, helps hard topologies.</div>
          </div>
          <Toggle label="Thinking" checked={draft.thinking === true} onChange={(v) => set("thinking", v)} />
        </div>
        {(
          [
            ["temperature", "Temperature", 0, 1, 0.05, "Sampling temperature for proposals."],
            ["max_repairs", "Max repairs", 0, 6, 1, "Repair rounds before a run is failed."],
            ["n_skills", "Few-shot skills", 0, 5, 1, "Verified TSIR exemplars in the prompt."],
            ["n_lessons", "Lessons in prompt", 0, 12, 1, "Active experience entries retrieved per run."],
          ] as const
        ).map(([k, label, min, max, step, hint]) => (
          <div key={k}>
            <div className="mb-1.5 flex items-baseline justify-between">
              <span className="text-[13px] font-medium text-fg">{label}</span>
              <span className="num text-[13px] text-fg">{num(k).toFixed(step < 1 ? 2 : 0)}</span>
            </div>
            <Slider label={label} value={num(k)} min={min} max={max} step={step} onChange={(v) => set(k, v)} />
            <div className="mt-1 text-[11.5px] text-fg-2">{hint}</div>
          </div>
        ))}
      </div>
      <div className="mt-5 flex items-center gap-2">
        <button className="btn btn-primary btn-sm" disabled={!dirty || put.isPending} onClick={() => put.mutate({ agent, knobs: draft })}>
          {put.isPending ? "Saving..." : "Save knobs"}
        </button>
        <button className="btn btn-sm" disabled={!dirty} onClick={() => setDraft(stored ?? {})}>
          <RotateCcw size={13} /> Reset
        </button>
        {put.isSuccess && !dirty && (
          <span className="flex items-center gap-1 text-[12px] text-ok">
            <Check size={13} /> saved
          </span>
        )}
        {put.isError && <span className="text-[12px] text-bad">{(put.error as Error).message}</span>}
      </div>
    </Panel>
  );
}

function StatusChip({ s }: { s: LessonStatus }) {
  return s === "active" ? (
    <Chip tone="ok" dot>
      active
    </Chip>
  ) : s === "retired" ? (
    <Chip dot>retired</Chip>
  ) : (
    <Chip tone="warn" dot>
      candidate
    </Chip>
  );
}

function Evidence({ l }: { l: Lesson }) {
  const d = typeof l.evidence.delta_score === "number" ? l.evidence.delta_score : undefined;
  const n = typeof l.evidence.n === "number" ? l.evidence.n : undefined;
  if (d === undefined) return <span className="text-fg-3">no evidence</span>;
  const frac = Math.min(1, Math.abs(d) / 0.14);
  const pos = d >= 0;
  return (
    <div className="min-w-[132px]" title={`Verifier score delta against matched runs without this lesson${n ? `, n = ${n}` : ""}`}>
      <div className="flex items-baseline justify-between">
        <span className={clsx("num text-[14px] font-semibold", d > 0.001 ? "text-ok" : d < -0.001 ? "text-bad" : "text-fg-1")}>
          {d >= 0 ? "+" : ""}
          {d.toFixed(2)}
        </span>
        <span className="text-[11px] text-fg-3">n = {n ?? "?"}</span>
      </div>
      <div className="mt-1 flex h-1.5 overflow-hidden rounded-full bg-bg-3">
        <div className="h-full w-1/2 flex justify-end">{!pos && <div className="h-full rounded-l-full bg-bad" style={{ width: `${frac * 100}%` }} />}</div>
        <div className="h-full w-1/2">{pos && <div className="h-full rounded-r-full bg-ok" style={{ width: `${frac * 100}%` }} />}</div>
      </div>
    </div>
  );
}

function LessonActions({ l }: { l: Lesson }) {
  const set = useSetLesson();
  const [confirm, setConfirm] = useState<"active" | "retired" | null>(null);
  const busy = set.isPending;
  const apply = (status: "active" | "retired") => {
    set.mutate({ id: l.id, status }, { onSettled: () => setConfirm(null) });
  };
  // invariants alter validators, so they need an explicit second confirmation (human gate)
  const start = (status: "active" | "retired") => (l.kind === "INVARIANT" ? setConfirm(status) : apply(status));
  if (confirm) {
    return (
      <div className="flex items-center gap-1.5" role="group" aria-label="Confirm invariant change">
        <span className="flex items-center gap-1 text-[11.5px] text-warn">
          <ShieldAlert size={13} /> Invariant: {confirm === "active" ? "activate" : "retire"}?
        </span>
        <button className={clsx("btn btn-sm", confirm === "active" ? "btn-ok" : "btn-bad")} disabled={busy} onClick={() => apply(confirm)}>
          Confirm
        </button>
        <button className="btn btn-sm" onClick={() => setConfirm(null)}>
          Cancel
        </button>
      </div>
    );
  }
  return (
    <div className="flex items-center gap-1.5">
      {l.status !== "active" && (
        <button className="btn btn-sm btn-ok" disabled={busy} onClick={() => start("active")} aria-label={`Activate ${l.id}`}>
          <Check size={13} /> Activate
        </button>
      )}
      {l.status !== "retired" && (
        <button className="btn btn-sm btn-bad" disabled={busy} onClick={() => start("retired")} aria-label={`Retire ${l.id}`}>
          <X size={13} /> Retire
        </button>
      )}
    </div>
  );
}

function Library({ lessons }: { lessons: Lesson[] }) {
  const [status, setStatus] = useState<"all" | LessonStatus>("all");
  const [kind, setKind] = useState<"all" | LessonKind>("all");
  const rows = useMemo(
    () => lessons.filter((l) => (status === "all" || l.status === status) && (kind === "all" || l.kind === kind)).sort((a, b) => (a.status === "candidate" ? -1 : 0) - (b.status === "candidate" ? -1 : 0) || (b.evidence.delta_score ?? 0) - (a.evidence.delta_score ?? 0)),
    [lessons, status, kind],
  );
  const counts = { candidate: lessons.filter((l) => l.status === "candidate").length };
  return (
    <Panel
      eyebrow="Human gate"
      title="Experience library"
      actions={
        <div className="flex flex-wrap items-center gap-2">
          <Segmented
            size="sm"
            label="Status filter"
            value={status}
            onChange={setStatus}
            options={[
              { value: "all", label: "All" },
              { value: "candidate", label: `Candidates (${counts.candidate})` },
              { value: "active", label: "Active" },
              { value: "retired", label: "Retired" },
            ]}
          />
          <Segmented
            size="sm"
            label="Kind filter"
            value={kind}
            onChange={setKind}
            options={[
              { value: "all", label: "Any kind" },
              { value: "INVARIANT", label: "Invariant" },
              { value: "DOMAIN", label: "Domain" },
            ]}
          />
        </div>
      }
      flush
    >
      <div className="mx-4 mb-3 flex items-start gap-2 rounded-xl border border-line bg-bg/50 px-3.5 py-2.5 text-[12.5px] text-fg-1">
        <Lock size={14} className="mt-0.5 shrink-0 text-accent" />
        Entries are kept only when verifier deltas support them. Validators and invariants change only with your approval: activating or retiring an invariant asks for a second confirmation.
      </div>
      <div className="overflow-x-auto">
        <table className="tbl">
          <thead>
            <tr>
              <th>Lesson</th>
              <th>Kind</th>
              <th>Scope</th>
              <th>Status</th>
              <th>Evidence Δscore</th>
              <th className="text-right">Decision</th>
            </tr>
          </thead>
          <tbody>
            <AnimatePresence initial={false}>
              {rows.map((l) => (
                <motion.tr key={l.id} layout initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.18 }}>
                  <td className="max-w-[560px]">
                    <div className="text-[13px] leading-snug text-fg">{l.lesson}</div>
                    <div className="mt-1.5 flex flex-wrap items-center gap-1.5">
                      <span className="mono text-[11px] text-fg-3">{l.id}</span>
                      {l.applies_to.slice(0, 4).map((a) => (
                        <span key={a} className="rounded-md border border-line px-1.5 py-px text-[10.5px] text-fg-2">
                          {a}
                        </span>
                      ))}
                    </div>
                  </td>
                  <td>
                    <Chip tone={l.kind === "INVARIANT" ? "accent" : "info"}>{l.kind}</Chip>
                  </td>
                  <td className="mono text-[12px] text-fg-1">{l.scope}</td>
                  <td>
                    <StatusChip s={l.status} />
                    <div className="mt-1 text-[11px] text-fg-3">{timeAgo(new Date(l.created * 1000).toISOString())}</div>
                  </td>
                  <td>
                    <Evidence l={l} />
                  </td>
                  <td className="text-right">
                    <div className="flex justify-end">
                      <LessonActions l={l} />
                    </div>
                  </td>
                </motion.tr>
              ))}
            </AnimatePresence>
          </tbody>
        </table>
        {rows.length === 0 && <div className="px-4 pb-6 text-center text-[13px] text-fg-2">No entries match the filters.</div>}
      </div>
    </Panel>
  );
}

function ScoreBars({ ev }: { ev: Revision["evidence"] }) {
  const items = [
    ["reference", "Reference", "#7684a2"],
    ["invariant", "Invariant edit", "#8b7cf6"],
    ["domain", "Domain edit", "#22d3ee"],
    ["composed", "Composed", "#34d399"],
  ] as const;
  const vals = items.map(([k]) => (typeof ev[k] === "number" ? (ev[k] as number) : undefined));
  const best = Math.max(...vals.filter((v): v is number => v !== undefined));
  return (
    <div className="space-y-1.5">
      {items.map(([k, label, color], i) => {
        const v = vals[i];
        if (v === undefined) return null;
        return (
          <div key={k} className="flex items-center gap-2 text-[11.5px]">
            <span className="w-[92px] shrink-0 text-fg-2">{label}</span>
            <div className="h-2 flex-1 overflow-hidden rounded-full bg-bg-3">
              <motion.div initial={{ width: 0 }} animate={{ width: `${v * 100}%` }} transition={{ duration: 0.6 }} className="h-full rounded-full" style={{ background: color, boxShadow: v === best ? `0 0 10px ${color}` : undefined }} />
            </div>
            <span className={clsx("num w-9 text-right", v === best ? "text-fg" : "text-fg-2")}>{v.toFixed(2)}</span>
          </div>
        );
      })}
    </div>
  );
}

const VERDICT: Record<string, { tone: "ok" | "info" | "neutral" | "bad"; label: string }> = {
  adopt: { tone: "ok", label: "Adopted" },
  merge: { tone: "info", label: "Merged" },
  keep: { tone: "neutral", label: "Kept reference" },
  reject: { tone: "bad", label: "Rejected" },
};

function Revisions({ revisions }: { revisions: Revision[] }) {
  const list = [...revisions].sort((a, b) => b.t - a.t);
  return (
    <Panel eyebrow="Task level" title="Revisions: Propose, Probe, Compose">
      <ol className="relative m-0 list-none space-y-5 p-0 pl-6">
        <div className="absolute top-2 bottom-2 left-[7px] w-px bg-line-2" aria-hidden />
        {list.map((r, i) => {
          const v = VERDICT[r.verdict] ?? { tone: "neutral" as const, label: r.verdict };
          const n = typeof r.evidence.n === "number" ? r.evidence.n : undefined;
          return (
            <motion.li key={r.revision_id} initial={{ opacity: 0, x: -8 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: Math.min(i, 6) * 0.05 }} className="relative">
              <span
                className="absolute top-3 -left-6 grid h-[15px] w-[15px] place-items-center rounded-full border-2 border-bg"
                style={{ background: v.tone === "ok" ? "var(--color-ok)" : v.tone === "info" ? "var(--color-info)" : v.tone === "bad" ? "var(--color-bad)" : "var(--color-fg-3)" }}
              />
              <div className="rounded-2xl border border-line bg-bg/50 p-4">
                <div className="mb-3 flex flex-wrap items-center gap-2">
                  <span className="mono text-[12px] text-fg-1">{r.revision_id}</span>
                  <Chip>{r.agent}</Chip>
                  <Chip tone={r.editor_type === "invariant" ? "accent" : "info"}>{r.editor_type}</Chip>
                  <span className="mono text-[11.5px] text-fg-2">{r.target}</span>
                  <span className="ml-auto text-[12px] text-fg-2">{timeAgo(new Date(r.t * 1000).toISOString())}</span>
                </div>
                <div className="grid grid-cols-1 gap-5 lg:grid-cols-[1fr_320px]">
                  <div className="space-y-3">
                    <Phase n={1} name="Propose" icon={<Sparkles size={13} />}>
                      {r.proposal}
                    </Phase>
                    <Phase n={2} name="Probe" icon={<FlaskConical size={13} />}>
                      Matched runs{n ? ` (n = ${n})` : ""} scored by the verifier ladder against the reference harness.
                    </Phase>
                    <Phase n={3} name="Compose" icon={<GitMerge size={13} />}>
                      <Chip tone={v.tone}>{v.label}</Chip>
                    </Phase>
                  </div>
                  <ScoreBars ev={r.evidence} />
                </div>
              </div>
            </motion.li>
          );
        })}
      </ol>
    </Panel>
  );
}

function Phase({ n, name, icon, children }: { n: number; name: string; icon: ReactNode; children: ReactNode }) {
  return (
    <div className="flex items-start gap-3">
      <span className="grid h-6 w-6 shrink-0 place-items-center rounded-full border border-line-2 bg-bg-2 text-[11px] font-bold text-fg-1">{n}</span>
      <div className="min-w-0">
        <div className="flex items-center gap-1.5 text-[11px] font-bold tracking-wider text-fg-2 uppercase">
          {icon} {name}
        </div>
        <div className="mt-0.5 text-[13px] leading-snug text-fg">{children}</div>
      </div>
    </div>
  );
}

export default function Harness() {
  const h = useHarness();
  if (!h.data) {
    return (
      <div className="space-y-5">
        <Skeleton className="h-10 w-72" />
        <div className="grid grid-cols-4 gap-5">
          {[0, 1, 2, 3].map((i) => (
            <Skeleton key={i} className="h-24" />
          ))}
        </div>
        <Skeleton className="h-80" />
      </div>
    );
  }
  const d = h.data;
  const active = d.lessons.filter((l) => l.status === "active").length;
  const cand = d.lessons.filter((l) => l.status === "candidate").length;
  const retired = d.lessons.filter((l) => l.status === "retired").length;
  const last = d.generations[d.generations.length - 1];
  const prev = d.generations[d.generations.length - 2];
  return (
    <div>
      <PageHeader eyebrow="Harness²" title="Self-improvement" description="Prompts, guardrails and skills evolve under hard-verifier rewards. You decide which lessons enter the library." />
      <div className="grid grid-cols-12 gap-5">
        <div className="col-span-12 grid grid-cols-2 gap-5 lg:grid-cols-4">
          <Stat label="Active lessons" value={String(active)} sub="injected into prompts" tone="ok" />
          <Stat label="Awaiting your decision" value={String(cand)} sub="candidates with evidence" tone={cand ? "warn" : "neutral"} />
          <Stat label="Retired" value={String(retired)} sub="no longer applied" />
          <Stat label="Generation" value={last ? `${last.gen}` : "n/a"} sub={last ? `score ${last.score.toFixed(2)}${prev ? ` (${last.score - prev.score >= 0 ? "+" : ""}${(last.score - prev.score).toFixed(2)})` : ""}` : ""} />
        </div>
        <div className="col-span-12 xl:col-span-7">
          <GenerationChart data={d.generations} />
        </div>
        <div className="col-span-12 xl:col-span-5">
          <KnobsPanel state={d} />
        </div>
        <div className="col-span-12">
          <Library lessons={d.lessons} />
        </div>
        <div className="col-span-12">
          <Revisions revisions={d.revisions} />
        </div>
      </div>
    </div>
  );
}
