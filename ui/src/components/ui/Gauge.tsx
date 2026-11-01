import { motion } from "framer-motion";
import { useId, type ReactNode } from "react";
import { clamp } from "../../lib/format";

/** 240 degree radial gauge with animated fill. `value` and `max` in the same unit. */
export function RadialGauge({
  value,
  max,
  label,
  display,
  sub,
  color = "var(--color-accent)",
  warnAt = 0.85,
  size = 128,
}: {
  value: number;
  max: number;
  label: string;
  display: ReactNode;
  sub?: ReactNode;
  color?: string;
  warnAt?: number;
  size?: number;
}) {
  const id = useId().replace(/:/g, "");
  const frac = clamp(value / (max || 1), 0, 1);
  const r = 50;
  const c = 2 * Math.PI * r;
  const arc = c * (240 / 360);
  const stroke = frac >= warnAt ? "var(--color-warn)" : color;
  const start = 150; // degrees; the gauge opens at the bottom
  return (
    <div className="flex flex-col items-center" role="img" aria-label={`${label}: ${Math.round(frac * 100)} percent`}>
      <div className="relative" style={{ width: size, height: size * 0.86 }}>
        <svg viewBox="0 0 120 104" width={size} height={size * 0.867} className="overflow-visible">
          <defs>
            <linearGradient id={`gg${id}`} x1="0" y1="0" x2="1" y2="1">
              <stop offset="0" stopColor={stroke} stopOpacity="0.55" />
              <stop offset="1" stopColor={stroke} />
            </linearGradient>
            <filter id={`gl${id}`} x="-30%" y="-30%" width="160%" height="160%">
              <feGaussianBlur stdDeviation="3" />
            </filter>
          </defs>
          <g transform={`rotate(${start} 60 60)`}>
            <circle cx="60" cy="60" r={r} fill="none" stroke="var(--color-bg-3)" strokeWidth="9" strokeLinecap="round" strokeDasharray={`${arc} ${c}`} />
            <motion.circle
              cx="60"
              cy="60"
              r={r}
              fill="none"
              stroke={`url(#gg${id})`}
              strokeWidth="9"
              strokeLinecap="round"
              strokeDasharray={`${arc} ${c}`}
              initial={{ strokeDashoffset: arc }}
              animate={{ strokeDashoffset: arc * (1 - frac) }}
              transition={{ type: "spring", stiffness: 70, damping: 18 }}
            />
            <motion.circle
              cx="60"
              cy="60"
              r={r}
              fill="none"
              stroke={stroke}
              strokeWidth="9"
              strokeLinecap="round"
              opacity="0.35"
              filter={`url(#gl${id})`}
              strokeDasharray={`${arc} ${c}`}
              initial={{ strokeDashoffset: arc }}
              animate={{ strokeDashoffset: arc * (1 - frac) }}
              transition={{ type: "spring", stiffness: 70, damping: 18 }}
            />
          </g>
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center pt-1">
          <div className="num text-[24px] leading-none font-semibold text-fg">{display}</div>
          {sub && <div className="mt-1 text-[11px] text-fg-2">{sub}</div>}
        </div>
      </div>
      <div className="-mt-1 text-[12px] font-medium text-fg-1">{label}</div>
    </div>
  );
}

/** Tiny SVG sparkline with gradient area. */
export function Sparkline({ data, color = "var(--color-accent)", height = 36, className }: { data: number[]; color?: string; height?: number; className?: string }) {
  const id = useId().replace(/:/g, "");
  if (data.length < 2) return <div style={{ height }} />;
  const w = 120;
  const min = Math.min(...data);
  const max = Math.max(...data);
  const span = max - min || 1;
  const pts = data.map((v, i) => [(i / (data.length - 1)) * w, height - 3 - ((v - min) / span) * (height - 8)] as const);
  const line = pts.map((p, i) => `${i ? "L" : "M"}${p[0].toFixed(1)} ${p[1].toFixed(1)}`).join(" ");
  const area = `${line} L${w} ${height} L0 ${height} Z`;
  return (
    <svg viewBox={`0 0 ${w} ${height}`} preserveAspectRatio="none" className={className} style={{ width: "100%", height }} aria-hidden>
      <defs>
        <linearGradient id={`sp${id}`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor={color} stopOpacity="0.35" />
          <stop offset="1" stopColor={color} stopOpacity="0" />
        </linearGradient>
      </defs>
      <path d={area} fill={`url(#sp${id})`} />
      <path d={line} fill="none" stroke={color} strokeWidth="1.6" strokeLinejoin="round" strokeLinecap="round" vectorEffect="non-scaling-stroke" />
    </svg>
  );
}

export function MiniBar({ frac, color = "var(--color-accent)", className }: { frac: number; color?: string; className?: string }) {
  return (
    <div className={className} style={{ height: 6, borderRadius: 999, background: "var(--color-bg-3)", overflow: "hidden" }}>
      <motion.div
        style={{ height: "100%", borderRadius: 999, background: color }}
        initial={{ width: 0 }}
        animate={{ width: `${clamp(frac, 0, 1) * 100}%` }}
        transition={{ type: "spring", stiffness: 80, damping: 20 }}
      />
    </div>
  );
}
