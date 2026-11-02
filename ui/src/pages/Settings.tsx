import { Check, Plug, RotateCcw, Server, Upload, Wifi, WifiOff } from "lucide-react";
import { useState } from "react";
import clsx from "clsx";
import { Chip, Field, PageHeader, Panel, Segmented, Slider, Toggle } from "../components/ui/primitives";
import { useConfig, usePutConfig } from "../lib/hooks";
import { AUGMENT_STRATEGIES, POLICIES, MAX_RENDER_VIEWS, RENDER_CAMERAS, RESOLUTIONS } from "../lib/meta";
import { probe } from "../lib/live";
import { useApi } from "../lib/provider";
import { DEFAULT_SETTINGS, THOR_URL, resetSettings, useSettings, type ApiMode } from "../lib/settings";

function ConnectionPanel() {
  const [s, update] = useSettings();
  const { kind, probe: info, reprobe } = useApi();
  const [draft, setDraft] = useState(s.baseUrl);
  const [test, setTest] = useState<{ ok: boolean; text: string } | null>(null);
  const [testing, setTesting] = useState(false);
  const dirty = draft.trim() !== s.baseUrl;

  const runTest = async () => {
    setTesting(true);
    const r = await probe(draft, 4000);
    setTest(r.ok ? { ok: true, text: `Reachable in ${r.ms} ms` } : { ok: false, text: r.error });
    setTesting(false);
  };

  return (
    <Panel eyebrow="Backend" title="Flywheel API" accent>
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-[1fr_300px]">
        <div className="space-y-5">
          <Field label="API base URL" hint={<>FastAPI on the Thor. Stored in this browser only. Use <span className="mono text-fg-1">/thor</span> to go through the Vite dev proxy (same origin, no CORS).</>}>
            <div className="flex gap-2">
              <input className="field mono" value={draft} onChange={(e) => setDraft(e.target.value)} placeholder={THOR_URL} spellCheck={false} aria-label="API base URL" />
              <button className="btn btn-primary" disabled={!dirty} onClick={() => update({ baseUrl: draft.trim().replace(/\/+$/, "") })}>
                <Check size={15} /> Apply
              </button>
            </div>
            <div className="mt-2 flex flex-wrap gap-2">
              <button className="btn btn-sm" onClick={() => setDraft(THOR_URL)}>
                Thor LAN default
              </button>
              <button className="btn btn-sm" onClick={() => setDraft("/thor")}>
                Dev proxy (/thor)
              </button>
              <button className="btn btn-sm" onClick={() => setDraft("http://localhost:8080")}>
                localhost:8080
              </button>
              <button className="btn btn-sm" onClick={runTest} disabled={testing}>
                <Plug size={14} /> {testing ? "Testing..." : "Test connection"}
              </button>
            </div>
            {test && (
              <p className={clsx("mt-2 mb-0 flex items-center gap-1.5 text-[12.5px]", test.ok ? "text-ok" : "text-bad")} role="status">
                {test.ok ? <Wifi size={14} /> : <WifiOff size={14} />} {test.text}
              </p>
            )}
          </Field>
          <Field label="Data source" hint="Auto probes GET /api/health and falls back to built-in mock data whenever the Thor does not answer.">
            <Segmented<ApiMode>
              label="Data source"
              value={s.mode}
              onChange={(mode) => update({ mode })}
              options={[
                { value: "auto", label: "Auto-detect" },
                { value: "mock", label: "Always mock" },
                { value: "live", label: "Always live" },
              ]}
            />
          </Field>
        </div>
        <div className="rounded-2xl border border-line bg-bg/60 p-5">
          <div className="eyebrow mb-3">Current source</div>
          <div className="flex items-center gap-3">
            <div className={clsx("grid h-12 w-12 place-items-center rounded-2xl border", kind === "live" ? "border-ok/40 bg-ok/10 text-ok" : "border-warn/45 bg-warn/10 text-warn")}>
              <Server size={22} />
            </div>
            <div>
              <div className={clsx("font-display text-[22px] font-semibold", kind === "live" ? "text-ok" : "text-warn")}>{kind === "live" ? "LIVE" : "MOCK"}</div>
              <div className="text-[12px] text-fg-2">{kind === "live" ? "Data from the Thor" : "Built-in demo data"}</div>
            </div>
          </div>
          <dl className="m-0 mt-4 space-y-1.5 text-[12.5px]">
            <div className="flex justify-between gap-3">
              <dt className="text-fg-2">Probe</dt>
              <dd className="m-0 text-fg">{info.checking ? "checking..." : info.ok ? `ok, ${info.ms} ms` : info.error ?? "n/a"}</dd>
            </div>
            <div className="flex justify-between gap-3">
              <dt className="text-fg-2">Target</dt>
              <dd className="mono m-0 truncate text-fg">{kind === "live" ? s.baseUrl : "in-browser"}</dd>
            </div>
            <div className="flex justify-between gap-3">
              <dt className="text-fg-2">Last check</dt>
              <dd className="m-0 text-fg">{info.at ? new Date(info.at).toLocaleTimeString() : "n/a"}</dd>
            </div>
          </dl>
          <button className="btn btn-sm mt-4 w-full" onClick={reprobe}>
            <RotateCcw size={13} /> Probe now
          </button>
        </div>
      </div>
    </Panel>
  );
}

function DefaultsPanel() {
  const [s, update] = useSettings();
  const config = useConfig();
  const put = usePutConfig();
  const pushed = put.isSuccess;
  const push = () => {
    const base = config.data ?? {};
    put.mutate({
      ...base,
      render: { ...base.render, ...s.render },
      augmentation: { ...base.augmentation, default_n: s.augment.n },
      closed_loop: { ...base.closed_loop, default_steps: s.closedLoop.steps },
    });
  };
  return (
    <Panel eyebrow="Studio" title="Run defaults" actions={<Chip tone="neutral">saved in this browser</Chip>}>
      <div className="grid grid-cols-1 gap-8 xl:grid-cols-3">
        <div className="space-y-5">
          <div className="eyebrow">Rendering</div>
          <Field label={<>Camera views: <span className="num text-fg">{s.render.views}</span></>}>
            <Slider label="Camera views" value={s.render.views} min={1} max={MAX_RENDER_VIEWS} onChange={(views) => update({ render: { ...s.render, views } })} />
            <div className="mt-2 flex flex-wrap gap-1.5">
              {RENDER_CAMERAS.map((c, i) => (
                <Chip key={c} tone={i < s.render.views ? "info" : "neutral"} className={clsx(i >= s.render.views && "opacity-50")}>
                  {c}
                </Chip>
              ))}
            </div>
          </Field>
          <Field label="Resolution">
            <Segmented label="Resolution" value={s.render.res} onChange={(res) => update({ render: { ...s.render, res } })} options={RESOLUTIONS.map((r) => ({ value: r, label: r }))} />
          </Field>
          <Field label={<>Frames: <span className="num text-fg">{s.render.frames}</span></>}>
            <Slider label="Frames" value={s.render.frames} min={25} max={241} step={4} onChange={(frames) => update({ render: { ...s.render, frames } })} />
          </Field>
          <div className="flex items-center justify-between">
            <span className="text-[13px] text-fg-1">LiDAR range maps</span>
            <Toggle label="LiDAR" checked={s.render.lidar} onChange={(lidar) => update({ render: { ...s.render, lidar } })} />
          </div>
        </div>
        <div className="space-y-5">
          <div className="eyebrow">Augmentation</div>
          <Field label={<>Variants: <span className="num text-fg">{s.augment.n}</span></>}>
            <Slider label="Variants" value={s.augment.n} min={1} max={32} onChange={(n) => update({ augment: { ...s.augment, n } })} />
          </Field>
          <Field label="Strategies">
            <div className="flex flex-wrap gap-2">
              {AUGMENT_STRATEGIES.map((a) => {
                const on = s.augment.strategies.includes(a.id);
                return (
                  <button
                    key={a.id}
                    role="checkbox"
                    aria-checked={on}
                    onClick={() => update({ augment: { ...s.augment, strategies: on ? s.augment.strategies.filter((x) => x !== a.id) : [...s.augment.strategies, a.id] } })}
                    className={clsx("h-8 rounded-full border px-3 text-[12.5px] font-medium transition-colors", on ? "border-accent/60 bg-accent/15 text-fg" : "border-line bg-bg-1/70 text-fg-1 hover:border-line-2")}
                  >
                    {a.label}
                  </button>
                );
              })}
            </div>
          </Field>
        </div>
        <div className="space-y-5">
          <div className="eyebrow">Closed loop</div>
          <Field label="Policy under test">
            <div className="space-y-2" role="radiogroup" aria-label="Policy">
              {POLICIES.map((p) => (
                <button key={p.id} role="radio" aria-checked={s.closedLoop.policy === p.id} onClick={() => update({ closedLoop: { ...s.closedLoop, policy: p.id } })} className={clsx("block w-full rounded-xl border px-3 py-2 text-left transition-colors", s.closedLoop.policy === p.id ? "border-gold/60 bg-gold/10" : "border-line bg-bg/50 hover:border-line-2")}>
                  <span className="mono text-[12.5px] font-semibold text-fg">{p.id}</span>
                  <span className="ml-2 text-[12px] text-fg-2">{p.label}</span>
                </button>
              ))}
            </div>
          </Field>
          <Field label={<>Steps: <span className="num text-fg">{s.closedLoop.steps}</span></>}>
            <Slider label="Steps" value={s.closedLoop.steps} min={20} max={1000} step={10} onChange={(steps) => update({ closedLoop: { ...s.closedLoop, steps } })} />
          </Field>
        </div>
      </div>
      <hr className="divider my-6" />
      <div className="flex flex-wrap items-center gap-3">
        <button className="btn" onClick={push} disabled={put.isPending}>
          <Upload size={15} /> {put.isPending ? "Pushing..." : "Push to pipeline config"}
        </button>
        <span className="text-[12.5px] text-fg-2">PUT /api/config with these render, augmentation and closed-loop defaults.</span>
        {pushed && (
          <span className="flex items-center gap-1 text-[12.5px] text-ok">
            <Check size={14} /> updated
          </span>
        )}
        {put.isError && <span className="text-[12.5px] text-bad">{(put.error as Error).message}</span>}
      </div>
    </Panel>
  );
}

export default function Settings() {
  const [s, update] = useSettings();
  return (
    <div>
      <PageHeader eyebrow="Console" title="Settings" description="Where the console gets its data, and the defaults new runs start from." />
      <div className="space-y-5">
        <ConnectionPanel />
        <DefaultsPanel />
        <Panel eyebrow="Interface" title="Appearance and data">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div>
              <div className="text-[13.5px] font-medium text-fg">Reduce motion</div>
              <div className="text-[12.5px] text-fg-2">Stops the flywheel animation and page transitions. The operating system preference is respected automatically.</div>
            </div>
            <Toggle label="Reduce motion" checked={s.reduceMotion} onChange={(reduceMotion) => update({ reduceMotion })} />
          </div>
          <hr className="divider my-5" />
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div>
              <div className="text-[13.5px] font-medium text-fg">Reset console settings</div>
              <div className="text-[12.5px] text-fg-2">Restores the API URL ({DEFAULT_SETTINGS.baseUrl}), data source and run defaults.</div>
            </div>
            <button className="btn btn-bad" onClick={resetSettings}>
              <RotateCcw size={14} /> Reset
            </button>
          </div>
        </Panel>
      </div>
    </div>
  );
}
