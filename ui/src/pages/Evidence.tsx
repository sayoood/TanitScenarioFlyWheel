import { Film, Lock } from "lucide-react";
import { useMemo, useState } from "react";
import { Link } from "react-router";
import { Chip, EmptyState, PageHeader, Panel, Segmented, Skeleton, type Tone } from "../components/ui/primitives";
import { ucMeta } from "../lib/meta";
import { useApi } from "../lib/provider";
import { useEvidence } from "../lib/hooks";
import type { EvidenceItem } from "../lib/types";

const KIND: Record<EvidenceItem["kind"], { label: string; tone: Tone; what: string }> = {
  uc5: { label: ucMeta("uc5").short, tone: "ok", what: "Lifted NEGATIVE log | POSITIVE twin (same road users, new ego plan)" },
  uc3: { label: ucMeta("uc3").short, tone: "ok", what: "Verified positive twin of a real incident, with closed loop" },
  uc2: { label: ucMeta("uc2").short, tone: "info", what: "Dash-cam clip | scenario re-created from ViPE ego motion + VLM scene" },
  closed_loop: { label: "Closed loop", tone: "accent", what: "Test policies driving the verified scenario" },
  ablation: { label: "Ablation", tone: "warn", what: "One-shot prompt (left) | agentic + verifier repair (right)" },
};

type Filter = "all" | EvidenceItem["kind"];

function title(it: EvidenceItem): string {
  if (it.kind === "ablation") return it.query ?? `query ${it.q}`;
  return it.source ?? it.run ?? it.file;
}

function Card({ it }: { it: EvidenceItem }) {
  const { api } = useApi();
  const k = KIND[it.kind];
  return (
    <Panel
      eyebrow={k.what}
      title={<span className="line-clamp-2 text-[14px]">{title(it)}</span>}
      actions={
        <div className="flex items-center gap-1.5">
          <Chip tone={k.tone}>{k.label}</Chip>
          {it.split && <Chip>{it.split.replace("_", "-")}</Chip>}
          {!it.public && (
            <Chip title="Derived from a dataset, incident or video clip: stays on the Thor, streamed only">
              <Lock size={11} /> Thor only
            </Chip>
          )}
        </div>
      }
    >
      <video className="w-full rounded-lg bg-black" src={api.evidenceUrl(it.file)} controls loop muted playsInline preload="metadata" />
      {it.run && (
        <div className="mt-2 text-[12px] text-fg-2">
          run <Link className="text-accent-2 hover:underline" to={`/runs/${it.run}`}>{it.run}</Link>
        </div>
      )}
    </Panel>
  );
}

export default function Evidence() {
  const q = useEvidence();
  const [filter, setFilter] = useState<Filter>("all");
  const items = useMemo(() => (q.data ?? []).filter((it) => filter === "all" || it.kind === filter), [q.data, filter]);
  const kinds = useMemo(() => Array.from(new Set((q.data ?? []).map((it) => it.kind))), [q.data]);
  return (
    <div>
      <PageHeader
        eyebrow="Visual evidence"
        title="Evidence"
        description="Rendered side-by-side videos of the flywheel's positive results. Each panel's title bar states its verdict and the key metric."
        actions={
          kinds.length > 1 && (
            <Segmented<Filter>
              value={filter}
              onChange={setFilter}
              label="Filter evidence by kind"
              options={[{ value: "all", label: "All" }, ...kinds.map((k) => ({ value: k, label: KIND[k].label }))]}
            />
          )
        }
      />
      {q.isLoading ? (
        <div className="grid gap-5 xl:grid-cols-2">
          {[0, 1].map((i) => <Skeleton key={i} className="h-72" />)}
        </div>
      ) : items.length === 0 ? (
        <EmptyState icon={<Film size={28} />} title="No evidence rendered yet">
          Run <code>python -m tsf.evidence all</code> on the Thor after the runs finish.
        </EmptyState>
      ) : (
        <div className="grid gap-5 xl:grid-cols-2">
          {items.map((it) => <Card key={it.file} it={it} />)}
        </div>
      )}
    </div>
  );
}
