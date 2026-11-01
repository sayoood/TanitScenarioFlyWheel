import { motion } from "framer-motion";
import type { CSSProperties, ReactNode } from "react";
import { useId } from "react";
import clsx from "clsx";

export function Panel({
  title,
  eyebrow,
  actions,
  children,
  className,
  bodyClassName,
  accent,
  style,
  flush,
  id,
}: {
  id?: string;
  title?: ReactNode;
  eyebrow?: ReactNode;
  actions?: ReactNode;
  children?: ReactNode;
  className?: string;
  bodyClassName?: string;
  accent?: boolean;
  style?: CSSProperties;
  flush?: boolean;
}) {
  return (
    <section id={id} className={clsx("panel", accent && "panel-accent", className)} style={style}>
      {(title || eyebrow || actions) && (
        <header className="panel-hd">
          <div className="min-w-0">
            {eyebrow && <div className="eyebrow">{eyebrow}</div>}
            {title && <h2 className="truncate text-[15px] font-semibold text-fg">{title}</h2>}
          </div>
          {actions && <div className="flex shrink-0 items-center gap-2">{actions}</div>}
        </header>
      )}
      <div className={clsx(flush ? "pt-3" : "panel-body", bodyClassName)}>{children}</div>
    </section>
  );
}

export type Tone = "ok" | "warn" | "bad" | "info" | "accent" | "neutral";

export function Chip({
  tone = "neutral",
  children,
  dot,
  className,
  title,
  style,
}: {
  tone?: Tone;
  children: ReactNode;
  dot?: boolean;
  className?: string;
  title?: string;
  style?: CSSProperties;
}) {
  return (
    <span className={clsx("chip", tone !== "neutral" && `chip-${tone}`, className)} title={title} style={style}>
      {dot && <span className="inline-block h-1.5 w-1.5 rounded-full bg-current" />}
      {children}
    </span>
  );
}

export function PageHeader({
  title,
  description,
  actions,
  eyebrow,
}: {
  title: ReactNode;
  description?: ReactNode;
  actions?: ReactNode;
  eyebrow?: ReactNode;
}) {
  return (
    <div className="mb-6 flex flex-wrap items-end justify-between gap-4">
      <div className="min-w-0">
        {eyebrow && <div className="eyebrow mb-1.5">{eyebrow}</div>}
        <h1 className="text-[28px] leading-tight font-semibold text-fg">{title}</h1>
        {description && <p className="mt-1 max-w-2xl text-[13.5px] text-fg-1">{description}</p>}
      </div>
      {actions && <div className="flex items-center gap-2">{actions}</div>}
    </div>
  );
}

export function Segmented<T extends string>({
  value,
  onChange,
  options,
  size = "md",
  label,
}: {
  value: T;
  onChange: (v: T) => void;
  options: { value: T; label: ReactNode; title?: string; disabled?: boolean }[];
  size?: "sm" | "md";
  label: string;
}) {
  const id = useId();
  return (
    <div role="tablist" aria-label={label} className="inline-flex items-center gap-0.5 rounded-[11px] border border-line bg-bg/60 p-0.5">
      {options.map((o) => {
        const active = o.value === value;
        return (
          <button
            key={o.value}
            role="tab"
            aria-selected={active}
            disabled={o.disabled}
            title={o.title}
            onClick={() => onChange(o.value)}
            className={clsx(
              "relative rounded-[9px] font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-40",
              size === "sm" ? "h-7 px-2.5 text-[12px]" : "h-8 px-3.5 text-[13px]",
              active ? "text-fg" : "text-fg-2 hover:text-fg",
            )}
          >
            {active && (
              <motion.span
                layoutId={`seg-${id}`}
                className="absolute inset-0 rounded-[9px] border border-line-2 bg-bg-3"
                transition={{ type: "spring", stiffness: 500, damping: 38 }}
              />
            )}
            <span className="relative z-10 inline-flex items-center gap-1.5">{o.label}</span>
          </button>
        );
      })}
    </div>
  );
}

export function Toggle({
  checked,
  onChange,
  label,
  disabled,
}: {
  checked: boolean;
  onChange: (v: boolean) => void;
  label: string;
  disabled?: boolean;
}) {
  return (
    <button
      role="switch"
      aria-checked={checked}
      aria-label={label}
      disabled={disabled}
      onClick={() => onChange(!checked)}
      className={clsx(
        "relative h-[22px] w-10 shrink-0 rounded-full border transition-colors disabled:opacity-40",
        checked ? "border-transparent bg-[linear-gradient(135deg,#8b7cf6,#22d3ee)]" : "border-line-2 bg-bg-3",
      )}
    >
      <span className={clsx("absolute top-[2px] h-4 w-4 rounded-full bg-white shadow transition-all", checked ? "left-[21px]" : "left-[2px]")} />
    </button>
  );
}

export function Slider({
  value,
  min,
  max,
  step = 1,
  onChange,
  label,
}: {
  value: number;
  min: number;
  max: number;
  step?: number;
  onChange: (v: number) => void;
  label: string;
}) {
  const fill = `${((value - min) / (max - min || 1)) * 100}%`;
  return (
    <input
      type="range"
      aria-label={label}
      min={min}
      max={max}
      step={step}
      value={value}
      onChange={(e) => onChange(Number(e.target.value))}
      style={{ "--fill": fill } as CSSProperties}
    />
  );
}

export function Field({ label, hint, children, right }: { label: ReactNode; hint?: ReactNode; children: ReactNode; right?: ReactNode }) {
  return (
    <div className="block">
      <div className="mb-1.5 flex items-center justify-between gap-2">
        <span className="text-[12.5px] font-medium text-fg-1">{label}</span>
        {right}
      </div>
      {children}
      {hint && <div className="mt-1.5 text-[12px] text-fg-2">{hint}</div>}
    </div>
  );
}

export function Skeleton({ className, style }: { className?: string; style?: CSSProperties }) {
  return <div className={clsx("skeleton", className)} style={style} aria-hidden />;
}

export function EmptyState({ icon, title, children }: { icon?: ReactNode; title: string; children?: ReactNode }) {
  return (
    <div className="flex flex-col items-center justify-center gap-2 rounded-xl border border-dashed border-line-2 px-6 py-10 text-center">
      {icon && <div className="text-fg-3">{icon}</div>}
      <div className="font-medium text-fg-1">{title}</div>
      {children && <div className="max-w-md text-[13px] text-fg-2">{children}</div>}
    </div>
  );
}

export function KV({ k, v, mono }: { k: ReactNode; v: ReactNode; mono?: boolean }) {
  return (
    <div className="flex items-baseline justify-between gap-4 py-1.5">
      <dt className="text-[12.5px] text-fg-2">{k}</dt>
      <dd className={clsx("m-0 text-right text-[13px] text-fg", mono && "mono")}>{v}</dd>
    </div>
  );
}
