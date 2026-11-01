import { CircleCheck, CircleDashed, CircleX, Clock, Loader, MinusCircle } from "lucide-react";
import type { RunStatus, StageStatus } from "../lib/types";
import { ucMeta } from "../lib/meta";
import { Chip } from "./ui/primitives";

export function StatusChip({ status }: { status: RunStatus | StageStatus }) {
  switch (status) {
    case "ok":
      return (
        <Chip tone="ok" dot>
          Passed
        </Chip>
      );
    case "failed":
      return (
        <Chip tone="bad" dot>
          Failed
        </Chip>
      );
    case "running":
      return (
        <Chip tone="info">
          <span className="live-dot !h-1.5 !w-1.5" /> Running
        </Chip>
      );
    case "queued":
      return (
        <Chip tone="warn" dot>
          Queued
        </Chip>
      );
    case "skipped":
      return <Chip>Skipped</Chip>;
    default:
      return <Chip>Pending</Chip>;
  }
}

export function StatusIcon({ status, size = 16 }: { status: StageStatus | RunStatus; size?: number }) {
  switch (status) {
    case "ok":
      return <CircleCheck size={size} className="text-ok" aria-label="passed" />;
    case "failed":
      return <CircleX size={size} className="text-bad" aria-label="failed" />;
    case "running":
      return <Loader size={size} className="animate-spin text-accent-2" aria-label="running" />;
    case "queued":
      return <Clock size={size} className="text-warn" aria-label="queued" />;
    case "skipped":
      return <MinusCircle size={size} className="text-fg-3" aria-label="skipped" />;
    default:
      return <CircleDashed size={size} className="text-fg-3" aria-label="pending" />;
  }
}

export function UcBadge({ uc }: { uc: string }) {
  const m = ucMeta(uc);
  return (
    <span
      className="inline-flex h-[22px] min-w-[42px] items-center justify-center rounded-md px-1.5 text-[11px] font-bold tracking-wider"
      style={{ color: m.color, background: `color-mix(in srgb, ${m.color} 14%, transparent)`, border: `1px solid color-mix(in srgb, ${m.color} 40%, transparent)` }}
      title={m.name}
    >
      {m.short}
    </span>
  );
}
