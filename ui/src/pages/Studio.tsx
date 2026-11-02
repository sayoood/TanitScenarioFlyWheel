import { AnimatePresence, motion } from "framer-motion";
import { Check, Film, Lightbulb, Rocket, Shuffle, Route as RouteIcon } from "lucide-react";
import { useMemo, useState, type ReactNode } from "react";
import { useNavigate } from "react-router";
import clsx from "clsx";
import { CodeBlock } from "../components/run/CodeBlock";
import { Chip, Field, PageHeader, Panel, Segmented, Slider, Toggle } from "../components/ui/primitives";
import { duration } from "../lib/format";
import { useConfig, useCreateRun, useIncidents, useKpis } from "../lib/hooks";
import { AUGMENT_STRATEGIES, POLICIES, MAX_RENDER_VIEWS, RENDER_CAMERAS, RESOLUTIONS, UC1_EXAMPLES, UC_EXAMPLE_INPUTS, USE_CASES } from "../lib/meta";
import { useSettings } from "../lib/settings";
import type { CreateRunRequest, RunInput, RunOptions, UseCaseId } from "../lib/types";

const INCIDENTS: Record<string, string> = {
  "imdad:inc_000231": "Red-light conflict at a signalized junction (collision, NEGATIVE)",
  "imdad:inc_000187": "Late braking before a junction, rear-end contact",
  "imdad:inc_000302": "Pedestrian conflict at an unsignalized crossing",
};

const INPUT_META: Record<Exclude<UseCaseId, "uc1">, { label: string; placeholder: string; hint: string; key: "video" | "incident_id" | "snippet" }> = {
  uc2: { label: "Video path or URL", placeholder: "youtube:<id> or a video path on the Thor", hint: "Single dash-cam video without ego data. ViPE estimates intrinsics, metric depth and ego motion.", key: "video" },
  uc3: { label: "IMDAD incident id", placeholder: "imdad:inc_000231", hint: "An IMDAD v2 NEGATIVE record. The Counterfactual agent writes a positive twin (ego edits only).", key: "incident_id" },
  uc4: { label: "Sensor snippet id", placeholder: "pai_av:<clip uuid>[@t0]", hint: "PhysicalAI-AV clip on the Thor. GT egomotion and auto-labelled tracks are lifted into a verified TSIR (fidelity vs. log is reported).", key: "snippet" },
  uc5: { label: "Negative snippet id", placeholder: "pai_av:<clip uuid>[@t0]", hint: "A critical snippet (the NEGATIVE). The twin editor rewrites only the ego plan; the verified POSITIVE plus variations become corrective training data.", key: "snippet" },
};

function Section({ title, hint, enabled, onToggle, children, icon }: { title: string; hint: string; enabled: boolean; onToggle: (v: boolean) => void; children: ReactNode; icon: ReactNode }) {
  return (
    <Panel
      eyebrow={hint}
      title={
        <span className="inline-flex items-center gap-2">
          {icon} {title}
        </span>
      }
      actions={<Toggle checked={enabled} onChange={onToggle} label={`Enable ${title}`} />}
    >
      <AnimatePresence initial={false}>
        {enabled ? (
          <motion.div key="on" initial={{ height: 0, opacity: 0 }} animate={{ height: "auto", opacity: 1 }} exit={{ height: 0, opacity: 0 }} transition={{ duration: 0.22 }} className="overflow-hidden">
            <div className="space-y-5 pt-1">{children}</div>
          </motion.div>
        ) : (
          <motion.p key="off" initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="m-0 text-[13px] text-fg-2">
            Disabled: this stage is skipped.
          </motion.p>
        )}
      </AnimatePresence>
    </Panel>
  );
}

export default function Studio() {
  const [settings] = useSettings();
  const navigate = useNavigate();
  const create = useCreateRun();
  const kpis = useKpis();
  const config = useConfig();

  const [uc, setUc] = useState<UseCaseId>("uc1");
  const [query, setQuery] = useState(UC1_EXAMPLES[0]);
  const incidents = useIncidents();
  const incidentTitles = useMemo(
    () => (incidents.data?.length ? Object.fromEntries(incidents.data.map((i) => [i.incident_id, i.title])) : null),
    [incidents.data],
  );
  const incidentsByID: Record<string, string> = incidentTitles ?? INCIDENTS;
  const [other, setOther] = useState<Record<string, string>>({ uc2: "", uc3: "", uc4: "", uc5: "" });

  const [augOn, setAugOn] = useState(true);
  const [aug, setAug] = useState(settings.augment);
  const [renOn, setRenOn] = useState(true);
  const [ren, setRen] = useState({ ...settings.render, views: Math.min(settings.render.views, MAX_RENDER_VIEWS) });
  const [clOn, setClOn] = useState(false);
  const [cl, setCl] = useState(settings.closedLoop);

  const strategies = config.data?.augmentation?.strategies?.map((s) => ({ ...AUGMENT_STRATEGIES.find((a) => a.id === s.id), id: s.id, label: s.label ?? s.id, description: s.description ?? AUGMENT_STRATEGIES.find((a) => a.id === s.id)?.description ?? "" })) ?? AUGMENT_STRATEGIES;
  // the Thor sends {id: {tier, desc}}; older configs sent [{id, label, description}]
  const cfgPolicies = config.data?.closed_loop?.policies;
  const policies = cfgPolicies
    ? (Array.isArray(cfgPolicies) ? cfgPolicies : Object.entries(cfgPolicies).map(([id, p]) => ({ id, description: p.desc }))).map((p) => {
        const known = POLICIES.find((x) => x.id === p.id);
        return { id: p.id, label: ("label" in p && p.label) || known?.label || p.id, description: p.description ?? known?.description ?? "" };
      })
    : POLICIES;

  const input: RunInput | null = useMemo(() => {
    if (uc === "uc1") return query.trim() ? { query: query.trim() } : null;
    const v = (other[uc] ?? "").trim();
    if (!v) return null;
    const k = INPUT_META[uc].key;
    return { [k]: v } as RunInput;
  }, [uc, query, other]);

  const options: RunOptions = useMemo(() => {
    const o: RunOptions = {};
    if (augOn && aug.strategies.length > 0 && aug.n > 0) o.augment = { n: aug.n, strategies: aug.strategies };
    if (renOn) o.render = { ...ren };
    if (clOn) o.closed_loop = { ...cl };
    return o;
  }, [augOn, aug, renOn, ren, clOn, cl]);

  const body: CreateRunRequest | null = input ? { uc, input, options } : null;
  const sPerFrame = kpis.data?.latest.cosmos_s_per_frame ?? 4.1;
  const renderEstimate = renOn ? ren.frames * ren.views * sPerFrame : 0;
  const meta = USE_CASES.find((u) => u.id === uc) ?? USE_CASES[0];

  const submit = () => {
    if (!body) return;
    create.mutate(body, { onSuccess: (r) => navigate(`/runs/${r.id}`) });
  };

  return (
    <div>
      <PageHeader
        eyebrow="Studio"
        title="Start a new run"
        description="Pick a use case, describe the input and choose how far down the pipeline the run should go. Cheap stages gate the expensive GPU stages."
      />

      <div className="mb-5 grid grid-cols-1 gap-2.5 sm:grid-cols-2 xl:grid-cols-5" role="tablist" aria-label="Use case">
        {USE_CASES.map((u) => {
          const active = u.id === uc;
          return (
            <button
              key={u.id}
              role="tab"
              aria-selected={active}
              onClick={() => setUc(u.id)}
              className={clsx("relative overflow-hidden rounded-[14px] border p-3.5 text-left transition-colors", active ? "border-transparent bg-bg-2" : "border-line bg-bg-1/70 hover:border-line-2")}
              style={active ? { boxShadow: `0 0 0 1px ${u.color}, 0 10px 30px -14px ${u.color}` } : undefined}
            >
              <div className="flex items-center gap-2">
                <span className="text-[11px] font-bold tracking-widest" style={{ color: u.color }}>
                  {u.short}
                </span>
                {active && <Check size={13} style={{ color: u.color }} />}
              </div>
              <div className="mt-1 text-[14px] font-semibold text-fg">{u.name}</div>
              <div className="mt-0.5 line-clamp-2 text-[12px] text-fg-2">{u.blurb}</div>
            </button>
          );
        })}
      </div>

      <div className="grid grid-cols-12 gap-5">
        <div className="col-span-12 space-y-5 xl:col-span-8">
          <Panel eyebrow={`${meta.short} input`} title={meta.name} accent>
            <AnimatePresence mode="wait" initial={false}>
              <motion.div key={uc} initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -6 }} transition={{ duration: 0.16 }}>
                {uc === "uc1" ? (
                  <div className="space-y-4">
                    <Field label="Scenario description" right={<span className="mono text-[11.5px] text-fg-3">{query.length} chars</span>} hint="Natural language is lifted into TSIR by the Lifter agent (constrained decoding), then verified L0 to L5.">
                      <textarea className="field" rows={5} value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Describe the driving scenario to generate..." aria-label="Scenario description" />
                    </Field>
                    <div>
                      <div className="mb-2 flex items-center gap-1.5 text-[12.5px] font-medium text-fg-1">
                        <Lightbulb size={14} className="text-gold" /> Example queries
                      </div>
                      <div className="grid grid-cols-1 gap-2 lg:grid-cols-3">
                        {UC1_EXAMPLES.map((q, i) => (
                          <button key={i} onClick={() => setQuery(q)} className={clsx("rounded-xl border p-3 text-left text-[12.5px] leading-relaxed transition-colors", q === query ? "border-uc1/60 bg-uc1/10 text-fg" : "border-line bg-bg/50 text-fg-1 hover:border-line-2")}>
                            <span className="line-clamp-4">{q}</span>
                          </button>
                        ))}
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="space-y-4">
                    <Field label={INPUT_META[uc].label} hint={INPUT_META[uc].hint}>
                      <input className="field mono" value={other[uc] ?? ""} onChange={(e) => setOther((o) => ({ ...o, [uc]: e.target.value }))} placeholder={INPUT_META[uc].placeholder} aria-label={INPUT_META[uc].label} spellCheck={false} />
                    </Field>
                    <div>
                      <div className="mb-2 text-[12.5px] font-medium text-fg-1">{uc === "uc3" ? "Recent incidents" : "Examples"}</div>
                      <div className="flex flex-wrap gap-2">
                        {(uc === "uc3" && incidentTitles ? Object.keys(incidentTitles) : UC_EXAMPLE_INPUTS[uc]).map((ex) => (
                          <button key={ex} className="btn btn-sm mono !font-medium" title={incidentsByID[ex]} onClick={() => setOther((o) => ({ ...o, [uc]: ex }))}>
                            {ex}
                          </button>
                        ))}
                      </div>
                      {uc === "uc3" && other.uc3 && incidentsByID[other.uc3] && <p className="mt-2 text-[12.5px] text-fg-2">{incidentsByID[other.uc3]}</p>}
                    </div>
                  </div>
                )}
              </motion.div>
            </AnimatePresence>
          </Panel>

          <Section title="Augmentation" hint="Semantic variants, re-verified" enabled={augOn} onToggle={setAugOn} icon={<Shuffle size={16} className="text-accent" />}>
            <div>
              <div className="mb-2 text-[12.5px] font-medium text-fg-1">Strategies</div>
              <div className="grid grid-cols-1 gap-2 sm:grid-cols-2 xl:grid-cols-3">
                {strategies.map((s) => {
                  const on = aug.strategies.includes(s.id);
                  return (
                    <button
                      key={s.id}
                      role="checkbox"
                      aria-checked={on}
                      onClick={() => setAug((a) => ({ ...a, strategies: on ? a.strategies.filter((x) => x !== s.id) : [...a.strategies, s.id] }))}
                      className={clsx("flex items-start gap-2.5 rounded-xl border p-3 text-left transition-colors", on ? "border-accent/60 bg-accent/10" : "border-line bg-bg/50 hover:border-line-2")}
                    >
                      <span className={clsx("mt-0.5 grid h-4 w-4 shrink-0 place-items-center rounded border", on ? "border-transparent bg-accent text-bg" : "border-line-2")}>{on && <Check size={11} strokeWidth={3} />}</span>
                      <span>
                        <span className="block text-[13px] font-semibold text-fg">{s.label}</span>
                        <span className="block text-[11.5px] leading-snug text-fg-2">{s.description}</span>
                      </span>
                    </button>
                  );
                })}
              </div>
            </div>
            <Field label={<>Variants: <span className="num text-fg">{aug.n}</span></>} hint="Every variant is re-verified (L0 to L5) and labelled SYNTHETIC_AUGMENTATION.">
              <Slider label="Number of variants" value={aug.n} min={1} max={32} onChange={(n) => setAug((a) => ({ ...a, n }))} />
            </Field>
          </Section>

          <Section title="Rendering" hint="Cosmos3 Sim-Auto: cameras + LiDAR" enabled={renOn} onToggle={setRenOn} icon={<Film size={16} className="text-accent-2" />}>
            <div className="grid grid-cols-1 gap-5 md:grid-cols-2">
              <Field label={<>Camera views: <span className="num text-fg">{ren.views}</span></>}>
                <Slider label="Camera views" value={ren.views} min={1} max={MAX_RENDER_VIEWS} onChange={(views) => setRen((r) => ({ ...r, views }))} />
                <div className="mt-2.5 flex flex-wrap gap-1.5">
                  {RENDER_CAMERAS.map((c, i) => (
                    <Chip key={c} tone={i < ren.views ? "info" : "neutral"} className={clsx(i >= ren.views && "opacity-50")}>
                      {c}
                    </Chip>
                  ))}
                </div>
              </Field>
              <div className="space-y-5">
                <Field label="Resolution">
                  <Segmented label="Resolution" value={ren.res} onChange={(res) => setRen((r) => ({ ...r, res }))} options={RESOLUTIONS.map((r) => ({ value: r, label: r }))} />
                </Field>
                <Field label={<>Frames: <span className="num text-fg">{ren.frames}</span></>}>
                  <div className="flex items-center gap-3">
                    <Slider label="Frames" value={ren.frames} min={25} max={241} step={4} onChange={(frames) => setRen((r) => ({ ...r, frames }))} />
                  </div>
                  <div className="mt-2 flex gap-1.5">
                    {[61, 93, 121, 241].map((f) => (
                      <button key={f} className="btn btn-sm !h-6 !px-2 mono" onClick={() => setRen((r) => ({ ...r, frames: f }))}>
                        {f}
                      </button>
                    ))}
                  </div>
                </Field>
              </div>
            </div>
            <div className="flex items-center justify-between gap-4 rounded-xl border border-line bg-bg/50 px-3.5 py-3">
              <div>
                <div className="text-[13px] font-semibold text-fg">128-beam LiDAR range maps</div>
                <div className="text-[12px] text-fg-2">Generated jointly with the cameras (Sim-Auto joint mode) and checked against the control range maps.</div>
              </div>
              <Toggle checked={ren.lidar} onChange={(lidar) => setRen((r) => ({ ...r, lidar }))} label="LiDAR" />
            </div>
          </Section>

          <Section title="Closed loop" hint="Policy under test in TanitSim + Cosmos Sim" enabled={clOn} onToggle={setClOn} icon={<RouteIcon size={16} className="text-gold" />}>
            <div>
              <div className="mb-2 text-[12.5px] font-medium text-fg-1">Policy</div>
              <div className="grid grid-cols-1 gap-2 md:grid-cols-3" role="radiogroup" aria-label="Policy">
                {policies.map((p) => {
                  const on = cl.policy === p.id;
                  return (
                    <button key={p.id} role="radio" aria-checked={on} onClick={() => setCl((c) => ({ ...c, policy: p.id }))} className={clsx("rounded-xl border p-3 text-left transition-colors", on ? "border-gold/60 bg-gold/10" : "border-line bg-bg/50 hover:border-line-2")}>
                      <div className="mono text-[12.5px] font-semibold text-fg">{p.id}</div>
                      <div className="mt-0.5 text-[12px] text-fg-1">{p.label}</div>
                      <div className="mt-0.5 text-[11.5px] leading-snug text-fg-2">{p.description}</div>
                    </button>
                  );
                })}
              </div>
            </div>
            <Field label={<>Steps: <span className="num text-fg">{cl.steps}</span></>} hint="10 Hz policy ticks; the world pauses while Cosmos generates the next chunk.">
              <Slider label="Closed-loop steps" value={cl.steps} min={20} max={1000} step={10} onChange={(steps) => setCl((c) => ({ ...c, steps }))} />
            </Field>
          </Section>
        </div>

        <div className="col-span-12 xl:col-span-4">
          <div className="space-y-5 xl:sticky xl:top-[92px]">
            <Panel eyebrow="Summary" title="Run plan" accent>
              <ol className="m-0 list-none space-y-2 p-0 text-[13px]">
                {[
                  ["Lift", "LLM emits TSIR", true],
                  ["Verify", "ladder L0 to L5 with repair loop", true],
                  ["Simulate", "TanitSim 2D + metrics", true],
                  ["Render", renOn ? `${ren.views} views, ${ren.res}, ${ren.frames} frames${ren.lidar ? ", LiDAR" : ""}` : "off", renOn],
                  ["Augment", augOn && aug.strategies.length ? `${aug.n} variants (${aug.strategies.length} strategies)` : "off", augOn && aug.strategies.length > 0],
                  ["Closed loop", clOn ? `${cl.policy}, ${cl.steps} steps` : "off", clOn],
                ].map(([name, detail, on]) => (
                  <li key={String(name)} className={clsx("flex items-start gap-2.5", !on && "opacity-50")}>
                    <span className={clsx("mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full", on ? "bg-accent-2" : "bg-fg-3")} />
                    <span>
                      <span className="font-semibold text-fg">{String(name)}</span>
                      <span className="text-fg-2"> · {String(detail)}</span>
                    </span>
                  </li>
                ))}
              </ol>
              {renOn && (
                <div className="mt-4 rounded-xl border border-line bg-bg/50 p-3 text-[12.5px] text-fg-1">
                  Cosmos estimate: <span className="num font-semibold text-fg">{duration(renderEstimate)}</span> at {sPerFrame.toFixed(2)} s per frame and view.
                </div>
              )}
              <button className="btn btn-primary mt-4 h-11 w-full text-[14px]" disabled={!body || create.isPending} onClick={submit}>
                <Rocket size={17} /> {create.isPending ? "Submitting..." : "Start run"}
              </button>
              {!body && <p className="mt-2 mb-0 text-[12px] text-fg-2">Enter an input to enable the run.</p>}
              {create.isError && (
                <p className="mt-2 mb-0 rounded-lg border border-bad/40 bg-bad/10 p-2.5 text-[12.5px] text-bad" role="alert">
                  {(create.error as Error).message}
                </p>
              )}
            </Panel>
            <Panel eyebrow="POST /api/runs" title="Request preview">
              <CodeBlock code={JSON.stringify(body ?? { uc, input: {}, options }, null, 2)} lang="json" maxHeight={340} label="Request body" />
            </Panel>
          </div>
        </div>
      </div>
    </div>
  );
}
