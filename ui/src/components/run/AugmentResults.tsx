import { Check, CloudRain, Download, GitCompare, Route, ShieldCheck, TriangleAlert, X } from "lucide-react";
import type { ReactNode } from "react";
import clsx from "clsx";
import { duration } from "../../lib/format";
import { useApi } from "../../lib/provider";
import type { AugmentDimension, AugmentInfo, AugmentVariant, Finding } from "../../lib/types";
import { Chip, EmptyState, type Tone } from "../ui/primitives";
import { CodeBlock } from "./CodeBlock";

export const DIM_META: Record<AugmentDimension, { label: string; tone: Tone; Icon: typeof Route }> = {
  weather: { label: "Weather & time", tone: "info", Icon: CloudRain },
  behavior: { label: "Behaviour", tone: "warn", Icon: Route },
  environment: { label: "Environment", tone: "accent", Icon: GitCompare },
};

const LEVELS = ["L0", "L1", "L2", "L3"] as const;

function Verdict({ ok, children, title }: { ok: boolean | null | undefined; children: ReactNode; title?: string }) {
  const tone: Tone = ok === true ? "ok" : ok === false ? "bad" : "neutral";
  return (
    <Chip tone={tone} title={title}>
      {ok === true ? <Check size={11} strokeWidth={3} /> : ok === false ? <X size={11} strokeWidth={3} /> : null}
      {children}
    </Chip>
  );
}

function fmtVal(v: unknown): string {
  if (v === null || v === undefined) return "";
  if (typeof v === "number") return Number.isInteger(v) ? String(v) : v.toFixed(3);
  return String(v);
}

function Diff({ v }: { v: AugmentVariant }) {
  const rows = v.diff ?? [];
  if (!rows.length) return <p className="m-0 text-[12.5px] text-fg-2">No TSIR change recorded.</p>;
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[520px] border-collapse text-[12px]">
        <thead>
          <tr className="text-left text-fg-2">
            <th className="w-14 py-1 pr-2 font-medium">Op</th>
            <th className="py-1 pr-2 font-medium">TSIR path</th>
            <th className="py-1 font-medium">Before / after</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((c, i) => (
            <tr key={i} className="border-t border-line align-top">
              <td className="py-1 pr-2">
                <Chip tone={c.op === "add" ? "ok" : c.op === "remove" ? "bad" : "info"}>{c.op}</Chip>
              </td>
              <td className="mono py-1 pr-2 break-all text-fg-1">{c.path}</td>
              <td className="mono py-1 break-all text-fg">
                {c.op !== "add" && <span className="text-bad/90">{fmtVal(c.before)}</span>}
                {c.op === "replace" && <span className="text-fg-3"> → </span>}
                {c.op !== "remove" && <span className="text-ok">{fmtVal(c.after)}</span>}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Ladder({ findings }: { findings: Finding[] }) {
  return (
    <div className="flex flex-wrap gap-x-4 gap-y-1.5">
      {LEVELS.map((l) => {
        const fs = findings.filter((f) => f.level === l);
        if (!fs.length) return null;
        return (
          <div key={l} className="flex items-center gap-1.5">
            <span className="mono text-[11px] font-semibold text-fg-2">{l}</span>
            {fs.map((f) => (
              <Chip key={f.check} tone={f.ok ? "ok" : "bad"} title={f.message || f.check}>
                {f.check}
              </Chip>
            ))}
          </div>
        );
      })}
    </div>
  );
}

function Media({ runId, v, parent }: { runId: string; v: AugmentVariant; parent?: AugmentInfo["parent_media"] }) {
  const { api } = useApi();
  const clips: { key: string; label: string; src: string }[] = [];
  if (parent?.mosaic && v.media.mosaic) clips.push({ key: "p", label: "Parent · cameras + LiDAR + BEV", src: parent.mosaic });
  if (v.media.mosaic) clips.push({ key: "m", label: "Variant · cameras + LiDAR + BEV", src: v.media.mosaic });
  if (v.media.lidar_bev) clips.push({ key: "l", label: "Variant · LiDAR bird's-eye view", src: v.media.lidar_bev });
  if (!clips.length) {
    const why = v.render?.error ?? v.render?.skipped;
    return <p className="m-0 text-[12.5px] text-fg-2">{why ? `No sensor data: ${why}` : "No sensor data rendered (Render was off)."}</p>;
  }
  return (
    <div className={clsx("grid gap-3", clips.length > 2 ? "lg:grid-cols-3" : "lg:grid-cols-2")}>
      {clips.map((c) => (
        <figure key={c.key} className="m-0">
          <video className="w-full rounded-lg border border-line bg-black" src={api.artifactUrl(runId, c.src)} controls loop muted playsInline preload="metadata" />
          <figcaption className="mt-1 text-[11.5px] text-fg-2">{c.label}</figcaption>
        </figure>
      ))}
    </div>
  );
}

const FILE_LABEL: Record<string, string> = { tsir: "TSIR", xodr: "OpenDRIVE", xosc: "OpenSCENARIO", osc2: "OSC 2 DSL", package: "Scenario_Structure", imdad_record: "IMDAD v2 record", sensors: "sensors.json" };

function Judge({ v }: { v: AugmentVariant }) {
  const j = v.judge;
  if (!j || j.skipped) return <span className="text-[12.5px] text-fg-2">{j?.skipped ?? "not judged"}</span>;
  const cam = j.camera;
  const lid = j.lidar;
  return (
    <div className="flex flex-wrap items-center gap-1.5">
      <Verdict ok={cam?.ok} title={cam?.expected ? `expected ${cam.expected.weather}, ${cam.expected.time_band}` : cam?.skipped}>
        camera{cam?.answer ? ` sees ${cam.answer.weather}, ${cam.answer.time_of_day.replace("_", "/")}` : cam?.skipped ? ` (${cam.skipped})` : ""}
      </Verdict>
      {cam?.objects_visible_ratio != null && <Verdict ok={cam.objects_visible_ratio >= 0.5}>objects visible {Math.round(cam.objects_visible_ratio * 100)}%</Verdict>}
      <Verdict ok={lid?.ok} title={lid?.warnings?.join("; ") || lid?.error || lid?.skipped}>
        LiDAR{lid?.numbers?.mean_points_per_sweep != null ? ` ${Math.round(Number(lid.numbers.mean_points_per_sweep))} pts/sweep` : ""}
        {lid?.numbers?.median_abs_range_err_m != null ? `, range err ${lid.numbers.median_abs_range_err_m} m` : ""}
      </Verdict>
      {j.rerender != null && <Chip tone="warn">re-rendered with another seed</Chip>}
    </div>
  );
}

function VariantCard({ runId, v, info }: { runId: string; v: AugmentVariant; info: AugmentInfo }) {
  const { api } = useApi();
  const crit = v.criticality;
  const band = crit?.band ?? [1, 2.5];
  const ttc = (v.verifier?.metrics?.min_ttc as number | null | undefined) ?? crit?.min_ttc ?? null;
  const failed = v.status !== "ok";
  return (
    <article className={clsx("rounded-2xl border bg-bg/40 p-4", failed ? "border-bad/45" : "border-line")} aria-label={`Variant ${v.index}`}>
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="mb-1.5 flex flex-wrap items-center gap-1.5">
            <span className="mono text-[11px] font-semibold text-fg-2">V{v.index}</span>
            {v.dimensions.map((d) => {
              const m = DIM_META[d];
              return (
                <Chip key={d} tone={m.tone}>
                  <m.Icon size={11} /> {m.label}
                </Chip>
              );
            })}
            {v.lever && v.lever !== "custom" && <Chip title="lever chosen by the editor">{v.lever}</Chip>}
            {failed && <Chip tone="bad">failed</Chip>}
          </div>
          <h3 className="m-0 text-[15.5px] leading-snug font-semibold text-fg">{v.title ?? v.id}</h3>
          <div className="mono mt-0.5 text-[11px] text-fg-3">{v.id}</div>
        </div>
        <div className="flex flex-wrap items-center gap-1.5 text-[11.5px] text-fg-2">
          {v.timings.total_s != null && <span className="num">{duration(v.timings.total_s)}</span>}
          {v.tokens && <span className="num">{(v.tokens.prompt_tokens + v.tokens.completion_tokens).toLocaleString()} tok</span>}
        </div>
      </div>

      {v.description && <p className="mt-2.5 mb-0 text-[13.5px] leading-relaxed text-fg-1">{v.description}</p>}
      {v.rationale && <p className="mt-1.5 mb-0 text-[12.5px] text-fg-2">Why: {v.rationale}</p>}
      {v.limitations && (
        <p className="mt-1.5 mb-0 flex items-start gap-1.5 text-[12.5px] text-warn">
          <TriangleAlert size={13} className="mt-0.5 shrink-0" /> Simulator limitation: {v.limitations}
        </p>
      )}
      {failed && v.error && <p className="mt-2 mb-0 rounded-lg border border-bad/40 bg-bad/[0.06] p-2.5 text-[12.5px] text-bad">{v.error}</p>}

      <div className="mt-3 flex flex-wrap items-center gap-1.5">
        <Verdict ok={v.verifier?.passed} title={`ladder score ${v.verifier?.score}`}>
          verifier L0-L3 {v.verifier ? (v.verifier.passed ? "passed" : "failed") : "n/a"}
        </Verdict>
        <Verdict ok={v.first_pass_valid} title="valid on the first editor proposal">
          {v.first_pass_valid ? "first-pass valid" : "needed repair"}
        </Verdict>
        <Chip title="verifier-guided repair rounds before acceptance">{v.repairs} repair{v.repairs === 1 ? "" : "s"}</Chip>
        <Verdict ok={v.isolation?.ok} title={v.isolation?.violations?.map((x) => `${x.op} ${x.path}`).join("; ")}>
          isolation {v.isolation?.ok ? "ok" : v.isolation ? "violated" : "n/a"}
        </Verdict>
        <Verdict ok={v.description_check?.ok} title={v.description_check?.missing?.length ? `missing: ${v.description_check.missing.join(", ")}` : "the description names every changed attribute"}>
          description {v.description_check?.ok ? "names changes" : v.description_check ? "misses attributes" : "n/a"}
        </Verdict>
        {crit && (
          <Verdict ok={crit.landed} title={crit.reason ?? `searched ${crit.sims ?? 0} sims over ${crit.params?.join(", ") ?? "no"} levers`}>
            min TTC {ttc != null ? `${Number(ttc).toFixed(2)} s` : "n/a"} · band {band[0]}-{band[1]} s
          </Verdict>
        )}
        {v.package && (
          <Verdict ok={v.package.imdad_record_valid} title={v.package.imdad_record_errors?.join("; ")}>
            IMDAD v2 record {v.package.imdad_record_valid ? "valid" : "invalid"}
          </Verdict>
        )}
      </div>

      <div className="mt-3 rounded-xl border border-line bg-bg-1/50 p-3">
        <div className="mb-2 text-[12px] font-medium text-fg-1">Sensor judgement</div>
        <Judge v={v} />
      </div>

      <div className="mt-3">
        <Media runId={runId} v={v} parent={info.parent_media} />
      </div>

      <div className="mt-3 space-y-2">
        <details className="rounded-xl border border-line bg-bg-1/40 p-3 open:bg-bg-1/70">
          <summary className="cursor-pointer text-[12.5px] font-medium text-fg-1">TSIR diff against the parent ({v.diff?.length ?? 0} changed paths)</summary>
          <div className="mt-2.5">
            <Diff v={v} />
            {v.isolation?.violations?.length ? (
              <ul className="mt-2 mb-0 list-none space-y-1 p-0 text-[12px] text-bad">
                {v.isolation.violations.map((x, i) => (
                  <li key={i} className="mono">{x.op} {x.path}: {x.why.slice(0, 160)}</li>
                ))}
              </ul>
            ) : null}
          </div>
        </details>
        {v.verifier && (
          <details className="rounded-xl border border-line bg-bg-1/40 p-3 open:bg-bg-1/70">
            <summary className="cursor-pointer text-[12.5px] font-medium text-fg-1">
              Verifier ladder and edit attempts ({v.attempts.length} attempt{v.attempts.length === 1 ? "" : "s"})
            </summary>
            <div className="mt-2.5 space-y-3">
              <Ladder findings={v.verifier.findings} />
              <ol className="m-0 list-none space-y-1.5 p-0 text-[12px]">
                {v.attempts.map((a) => (
                  <li key={a.index} className="rounded-lg border border-line p-2">
                    <div className="flex flex-wrap items-center gap-1.5">
                      <span className="mono text-fg-2">attempt {a.index}</span>
                      <Chip tone={a.passed ? "ok" : "bad"}>{a.passed ? "accepted" : a.stage}</Chip>
                      {a.lever && <Chip>{a.lever}</Chip>}
                      {a.seconds != null && <span className="num text-fg-2">{duration(a.seconds)}</span>}
                    </div>
                    {a.issues.length > 0 && <ul className="mt-1.5 mb-0 list-disc space-y-0.5 pl-4 text-fg-1">{a.issues.map((x, i) => <li key={i}>{x.slice(0, 320)}</li>)}</ul>}
                  </li>
                ))}
              </ol>
            </div>
          </details>
        )}
        {v.patch && (
          <details className="rounded-xl border border-line bg-bg-1/40 p-3 open:bg-bg-1/70">
            <summary className="cursor-pointer text-[12.5px] font-medium text-fg-1">JSON patch ({v.patch.length} ops) and timings</summary>
            <div className="mt-2.5 space-y-3">
              <CodeBlock code={JSON.stringify(v.patch, null, 2)} lang="json" maxHeight={260} label="RFC 6902 patch" />
              <div className="flex flex-wrap gap-x-4 gap-y-1 text-[12px] text-fg-2">
                {Object.entries(v.timings).map(([k, t]) => (
                  <span key={k}>
                    {k.replace(/_s$/, "")} <span className="num text-fg">{duration(t)}</span>
                  </span>
                ))}
              </div>
            </div>
          </details>
        )}
      </div>

      {Object.keys(v.media).length > 0 && (
        <div className="mt-3 flex flex-wrap gap-1.5">
          {Object.entries(FILE_LABEL).map(([k, label]) =>
            v.media[k] ? (
              <a key={k} className="btn btn-sm" href={api.artifactUrl(runId, v.media[k])} target="_blank" rel="noreferrer" download>
                <Download size={12} /> {label}
              </a>
            ) : null,
          )}
        </div>
      )}
    </article>
  );
}

export function AugmentResults({ runId, info, running }: { runId: string; info: AugmentInfo; running?: boolean }) {
  const prog = Object.entries(info.progress ?? {});
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-2 text-[12.5px] text-fg-1">
        <Chip tone="accent">{info.mode}</Chip>
        <span>seed <span className="num text-fg">{info.seed}</span></span>
        {info.separate && <Chip>separate variants</Chip>}
        <Chip tone={info.render ? "info" : "neutral"}>{info.render ? "with sensor render" : "no render"}</Chip>
        {info.wall_s != null && <span>wall <span className="num text-fg">{duration(info.wall_s)}</span></span>}
        {info.requests && Object.entries(info.requests).map(([k, t]) => (
          <span key={k} className="text-fg-2" title={String(t)}>
            <span className="text-fg-1">{DIM_META[k as AugmentDimension]?.label ?? k}:</span> “{String(t).slice(0, 70)}”
          </span>
        ))}
      </div>
      {running && prog.length > 0 && (
        <ul className="m-0 list-none space-y-1 rounded-xl border border-line bg-bg/50 p-3 text-[12px]">
          {prog.map(([vid, p]) => (
            <li key={vid} className="flex flex-wrap items-center gap-2">
              <span className="live-dot !h-1.5 !w-1.5" />
              <span className="mono text-fg-1">{vid.split("_aug_").pop()}</span>
              <Chip tone="info">{p.stage}</Chip>
              <span className="truncate text-fg-2">{p.detail}</span>
            </li>
          ))}
        </ul>
      )}
      {info.variants.length === 0 ? (
        <EmptyState icon={<ShieldCheck size={26} />} title={running ? "Variants are being built" : "No variants"}>
          {running ? "Each variant appears here once it is verified, described and packaged." : "The job produced no variant."}
        </EmptyState>
      ) : (
        info.variants.map((v) => <VariantCard key={v.id} runId={runId} v={v} info={info} />)
      )}
    </div>
  );
}
