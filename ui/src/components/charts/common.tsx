import type { ReactNode } from "react";

export const AXIS = {
  tick: { fill: "#94a3c0", fontSize: 11, fontFamily: "var(--font-sans)" },
  tickLine: false as const,
  axisLine: { stroke: "#1d2840" },
};

export const GRID_STROKE = "#1d2840";

interface TipItem {
  name?: string | number;
  value?: string | number | readonly (string | number)[];
  color?: string;
  dataKey?: string | number;
}

/** Dark glass tooltip shared by all charts. */
export function ChartTooltip({
  active,
  payload,
  label,
  title,
  format,
}: {
  active?: boolean;
  payload?: readonly TipItem[];
  label?: string | number;
  title?: (label: string | number | undefined) => ReactNode;
  format?: (key: string, v: number) => string;
}) {
  if (!active || !payload || payload.length === 0) return null;
  return (
    <div className="rounded-xl border border-line-2 bg-bg-2/95 px-3 py-2 shadow-2xl backdrop-blur" role="status">
      <div className="mb-1 text-[11.5px] font-semibold text-fg-1">{title ? title(label) : label}</div>
      {payload.map((p, i) => (
        <div key={i} className="flex items-center gap-2 text-[12px]">
          <span className="inline-block h-2 w-2 rounded-full" style={{ background: p.color }} />
          <span className="text-fg-2">{p.name}</span>
          <span className="num ml-auto pl-4 text-fg">{typeof p.value === "number" ? (format ? format(String(p.dataKey ?? p.name), p.value) : p.value.toFixed(3)) : String(p.value)}</span>
        </div>
      ))}
    </div>
  );
}
