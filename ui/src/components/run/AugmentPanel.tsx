import { Dices, Sparkles } from "lucide-react";
import { useEffect, useState } from "react";
import { Link } from "react-router";
import { StatusChip } from "../common";
import { DIM_META, AugmentResults } from "./AugmentResults";
import { Chip, Field, Panel, Segmented, Toggle } from "../ui/primitives";
import { timeAgo } from "../../lib/format";
import { useAugmentRun, useRun } from "../../lib/hooks";
import type { AugmentDimension, AugmentMode, AugmentRequest, RunDetail } from "../../lib/types";

const FIELDS: { dim: AugmentDimension; label: string; placeholder: string }[] = [
  { dim: "weather", label: "Weather & time of day", placeholder: "e.g. night with heavy rain" },
  { dim: "behavior", label: "Behaviour", placeholder: "e.g. the oncoming car brakes late and the ego has to react" },
  { dim: "environment", label: "Environment", placeholder: "e.g. road works: cones close the right lane, a 30 km/h sign before the junction" },
];

/** Start an augmentation job for this run and browse the jobs it already has. */
export function AugmentPanel({ run }: { run: RunDetail }) {
  const start = useAugmentRun(run.id);
  const [mode, setMode] = useState<AugmentMode>("auto");
  const [texts, setTexts] = useState<Record<AugmentDimension, string>>({ weather: "", behavior: "", environment: "" });
  const [seed, setSeed] = useState(0);
  const [render, setRender] = useState(false);
  const [separate, setSeparate] = useState(false);
  const [criticality, setCriticality] = useState(true);
  const kids = run.augmentations ?? [];
  const [sel, setSel] = useState<string | undefined>(kids[0]?.id);
  useEffect(() => {
    if (!sel && kids[0]) setSel(kids[0].id);
  }, [sel, kids]);

  const given = FIELDS.filter((f) => texts[f.dim].trim());
  const ready = !!run.artifacts.some((a) => a.name === "tsir.json") && (mode === "auto" || given.length > 0);
  const submit = () => {
    const body: AugmentRequest = { mode, seed, render, criticality, separate: mode === "manual" && separate };
    if (mode === "manual") body.requests = Object.fromEntries(given.map((f) => [f.dim, texts[f.dim].trim()]));
    start.mutate(body, { onSuccess: (r) => setSel(r.id) });
  };

  return (
    <Panel id="augment" eyebrow="Scenario flywheel" title="Augment this scenario" className="scroll-mt-24"
      actions={<span className="hidden sm:inline-flex"><Chip tone="accent"><Sparkles size={11} /> weather · behaviour · environment</Chip></span>}>
      <div className="grid grid-cols-1 gap-5 xl:grid-cols-12">
        <div className="space-y-4 xl:col-span-5">
          <Segmented<AugmentMode> label="Augmentation mode" value={mode} onChange={setMode}
            options={[{ value: "auto", label: "Auto", title: "three variants, one per dimension" }, { value: "manual", label: "Manual", title: "your free-text variation" }]} />
          {mode === "auto" ? (
            <p className="m-0 text-[12.5px] leading-relaxed text-fg-1">
              Exactly three variants of this verified scenario: <b>weather &amp; time of day</b> drawn at random from the seed (never the parent's), and a
              <b> behaviour</b> and an <b>environment</b> change that the LLM chooses as meaningful for this scenario. Every variant is verified, described and packaged.
            </p>
          ) : (
            <div className="space-y-3">
              {FIELDS.map((f) => {
                const m = DIM_META[f.dim];
                return (
                  <Field key={f.dim} label={<span className="inline-flex items-center gap-1.5"><m.Icon size={13} /> {f.label}</span>}>
                    <textarea className="field" rows={2} value={texts[f.dim]} onChange={(e) => setTexts((t) => ({ ...t, [f.dim]: e.target.value }))} placeholder={f.placeholder} aria-label={f.label} />
                  </Field>
                );
              })}
              <div className="flex items-center justify-between gap-3 rounded-xl border border-line bg-bg/50 px-3 py-2.5">
                <div>
                  <div className="text-[12.5px] font-semibold text-fg">One variant per dimension</div>
                  <div className="text-[11.5px] text-fg-2">Off: all given texts are applied together in a single variant.</div>
                </div>
                <Toggle checked={separate} onChange={setSeparate} label="Separate variants" />
              </div>
            </div>
          )}
          <div className="grid grid-cols-2 gap-3">
            <Field label="Seed" hint="Same seed, same weather draw and same render seed for parent and variants.">
              <div className="flex gap-1.5">
                <input className="field mono" type="number" min={0} value={seed} onChange={(e) => setSeed(Math.max(0, Math.trunc(Number(e.target.value) || 0)))} aria-label="Seed" />
                <button className="btn btn-icon" onClick={() => setSeed(Math.floor(Math.random() * 1000))} title="Random seed" aria-label="Random seed"><Dices size={15} /></button>
              </div>
            </Field>
            <div className="space-y-2.5 pt-6">
              <label className="flex items-center justify-between gap-3 text-[12.5px] text-fg-1">
                Render camera + LiDAR
                <Toggle checked={render} onChange={setRender} label="Render sensors" />
              </label>
              <label className="flex items-center justify-between gap-3 text-[12.5px] text-fg-1" title="Land the minimum TTC of behaviour variants in 1.0-2.5 s">
                Criticality targeting
                <Toggle checked={criticality} onChange={setCriticality} label="Criticality targeting" />
              </label>
            </div>
          </div>
          <button className="btn btn-primary h-10 w-full" disabled={!ready || start.isPending} onClick={submit}>
            {start.isPending ? "Starting..." : mode === "auto" ? "Create 3 variants" : separate ? `Create ${Math.max(given.length, 1)} variants` : "Create variant"}
          </button>
          {start.isError && <p className="m-0 text-[12.5px] text-bad">{(start.error as Error).message}</p>}
          {!ready && mode === "manual" && <p className="m-0 text-[12px] text-fg-2">Describe at least one dimension.</p>}
          {!run.artifacts.some((a) => a.name === "tsir.json") && <p className="m-0 text-[12px] text-fg-2">This run has no verified TSIR yet.</p>}
          {render && <p className="m-0 text-[12px] text-fg-2">Rendering waits for the single Cosmos slot and 35 GB of free memory; the parent is rendered once for comparison.</p>}
        </div>

        <div className="min-w-0 xl:col-span-7">
          {kids.length === 0 ? (
            <div className="rounded-xl border border-dashed border-line-2 p-6 text-center text-[13px] text-fg-2">No augmentation jobs for this run yet.</div>
          ) : (
            <>
              <div className="mb-3 flex flex-wrap gap-1.5" role="tablist" aria-label="Augmentation jobs">
                {kids.map((k) => (
                  <button key={k.id} role="tab" aria-selected={sel === k.id} onClick={() => setSel(k.id)}
                    className={`btn btn-sm ${sel === k.id ? "!border-accent/60 !bg-accent/15" : ""}`}>
                    <span className="mono">{k.id.slice(-9)}</span>
                    <span className="hidden text-fg-2 sm:inline">{k.mode ?? "auto"} · seed {k.seed ?? 0}</span>
                    <StatusChip status={k.status} />
                  </button>
                ))}
              </div>
              {sel && <JobResults id={sel} created={kids.find((k) => k.id === sel)?.created} />}
            </>
          )}
        </div>
      </div>
    </Panel>
  );
}

function JobResults({ id, created }: { id: string; created?: string }) {
  const job = useRun(id);
  if (job.isLoading || !job.data) return <p className="text-[13px] text-fg-2">Loading job {id}...</p>;
  const j = job.data;
  const running = j.status === "running" || j.status === "queued";
  return (
    <div>
      <div className="mb-3 flex flex-wrap items-center gap-2 text-[12px] text-fg-2">
        <StatusChip status={j.status} />
        {created && <span>{timeAgo(created)}</span>}
        <Link className="underline decoration-line-2 hover:text-fg" to={`/runs/${j.id}`}>open job {j.id}</Link>
        {j.error && <span className="text-bad">{j.error.slice(0, 200)}</span>}
      </div>
      {j.augment ? <AugmentResults runId={j.id} info={j.augment} running={running} /> : <p className="text-[13px] text-fg-2">{running ? "Queued..." : "No result."}</p>}
    </div>
  );
}
