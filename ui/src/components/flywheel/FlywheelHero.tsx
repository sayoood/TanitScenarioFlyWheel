import { motion, useReducedMotion } from "framer-motion";
import { Braces, Camera, Gauge, MessageSquareText, RefreshCcw, ShieldCheck, Siren, TriangleAlert, Video, Radar } from "lucide-react";
import type { LucideIcon } from "lucide-react";
import { useId, useMemo } from "react";
import { ucMeta } from "../../lib/meta";
import type { StageName } from "../../lib/types";

export type NodeId = "ir" | "verify" | "sim" | "render" | "loop";

/** Which ring node a pipeline stage lights up. */
export const STAGE_NODE: Record<StageName, NodeId> = {
  analyse: "ir",
  propose: "ir",
  repair: "ir",
  augment: "ir",
  verify: "verify",
  document: "verify",
  simulate: "sim",
  wsm: "render",
  cosmos_transfer: "render",
  closed_loop: "loop",
  // stages of the other use cases and of the augmentation job
  brief: "ir",
  replay_negative: "ir",
  twin: "ir",
  lift: "ir",
  vipe: "ir",
  scene: "ir",
  parent: "verify",
  plan: "ir",
  edit: "ir",
  criticality: "sim",
  package: "verify",
  sensors: "render",
  render: "render",
  judge: "verify",
};

const W = 1000;
const H = 440;
const CX = 620;
const CY = 220;
const R = 148;

const RING: { id: NodeId; label: string; sub: string; angle: number; Icon: LucideIcon; color: string; lx: number; ly: number; anchor: "start" | "middle" | "end" }[] = [
  { id: "ir", label: "TSIR", sub: "lift · compile", angle: 180, Icon: Braces, color: "#a78bfa", lx: 0, ly: -46, anchor: "middle" },
  { id: "verify", label: "Verify", sub: "ladder L0 to L5", angle: 252, Icon: ShieldCheck, color: "#60a5fa", lx: 0, ly: -46, anchor: "middle" },
  { id: "sim", label: "Sim", sub: "TanitSim 2D", angle: 324, Icon: Gauge, color: "#22d3ee", lx: 40, ly: 4, anchor: "start" },
  { id: "render", label: "Render", sub: "WSM · Cosmos 3", angle: 36, Icon: Camera, color: "#34d399", lx: 40, ly: 4, anchor: "start" },
  { id: "loop", label: "Closed loop", sub: "policy under test", angle: 108, Icon: RefreshCcw, color: "#fbbf24", lx: 0, ly: 54, anchor: "middle" },
];

const SOURCES: { id: string; label: string; sub: string; Icon: LucideIcon; color: string }[] = [
  { id: "uc1", label: ucMeta("uc1").short, sub: "Text query", Icon: MessageSquareText, color: "var(--color-uc1)" },
  { id: "uc2", label: ucMeta("uc2").short, sub: "Dashcam video", Icon: Video, color: "var(--color-uc2)" },
  { id: "uc3", label: ucMeta("uc3").short, sub: "Incident (negative)", Icon: Siren, color: "var(--color-uc3)" },
  { id: "uc4", label: ucMeta("uc4").short, sub: "Sensor snippet", Icon: Radar, color: "var(--color-uc4)" },
  { id: "uc5", label: ucMeta("uc5").short, sub: "Negative snippet", Icon: TriangleAlert, color: "var(--color-uc5)" },
];

const pos = (deg: number, r = R): [number, number] => {
  const a = (deg * Math.PI) / 180;
  return [CX + r * Math.cos(a), CY + r * Math.sin(a)];
};
const f1 = (n: number) => n.toFixed(1);
/** Clockwise arc between two angles on the ring. */
const arc = (a0: number, a1: number, r = R): string => {
  const [x0, y0] = pos(a0, r);
  const [x1, y1] = pos(a1, r);
  const sweep = (((a1 - a0) % 360) + 360) % 360;
  return `M${f1(x0)} ${f1(y0)} A${r} ${r} 0 ${sweep > 180 ? 1 : 0} 1 ${f1(x1)} ${f1(y1)}`;
};

interface Props {
  activeNode?: NodeId;
  activeUc?: string;
  runLabel?: string;
  generation?: { gen: number; score: number };
  busy: boolean;
  queue: number;
  onNodeClick?: (id: NodeId) => void;
  reduceMotion?: boolean;
}

export function FlywheelHero({ activeNode, activeUc, runLabel, generation, busy, queue, reduceMotion }: Props) {
  const uid = useId().replace(/:/g, "");
  const prefersReduced = useReducedMotion();
  const still = !!(reduceMotion || prefersReduced);

  const segments = useMemo(
    () =>
      RING.map((n, i) => {
        const next = RING[(i + 1) % RING.length];
        const a1 = next.angle <= n.angle ? next.angle + 360 : next.angle;
        // trim so the segment starts/ends at the node outlines
        const trim = 11;
        return { from: n.id, to: next.id, d: arc(n.angle + trim, a1 - trim), color: next.color };
      }),
    [],
  );
  const irPos = pos(180);
  const sourceY = (i: number) => 56 + i * 82;
  const hubPulse = busy ? 2.6 : 5.2;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="block h-auto w-full" role="img" aria-label="Flywheel diagram: five use cases feed the TSIR, then verification, simulation, rendering and the closed loop, which feeds the experience library back into the lifter.">
      <defs>
        <radialGradient id={`hub${uid}`} cx="50%" cy="50%" r="50%">
          <stop offset="0" stopColor="#8b7cf6" stopOpacity="0.28" />
          <stop offset="0.6" stopColor="#22d3ee" stopOpacity="0.07" />
          <stop offset="1" stopColor="#22d3ee" stopOpacity="0" />
        </radialGradient>
        <linearGradient id={`ringg${uid}`} x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="#8b7cf6" />
          <stop offset="1" stopColor="#22d3ee" />
        </linearGradient>
        <filter id={`glow${uid}`} x="-60%" y="-60%" width="220%" height="220%">
          <feGaussianBlur stdDeviation="4" result="b" />
          <feMerge>
            <feMergeNode in="b" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
        <marker id={`arrow${uid}`} viewBox="0 0 8 8" refX="6" refY="4" markerWidth="7" markerHeight="7" orient="auto">
          <path d="M0 0 L8 4 L0 8 z" fill="#8896b3" />
        </marker>
      </defs>

      {/* decorative rotating ring */}
      <g style={{ transformOrigin: `${CX}px ${CY}px`, animation: still ? "none" : "spin 90s linear infinite" }}>
        <circle cx={CX} cy={CY} r={R + 38} fill="none" stroke="var(--color-line)" strokeWidth="1" strokeDasharray="2 7" />
        {Array.from({ length: 60 }).map((_, i) => {
          const a = (i / 60) * 360;
          const [x0, y0] = pos(a, R + 52);
          const [x1, y1] = pos(a, R + (i % 5 === 0 ? 44 : 48));
          return <line key={i} x1={f1(x0)} y1={f1(y0)} x2={f1(x1)} y2={f1(y1)} stroke="var(--color-line-2)" strokeWidth={i % 5 === 0 ? 1.4 : 0.8} />;
        })}
      </g>
      <circle cx={CX} cy={CY} r={R - 28} fill={`url(#hub${uid})`} />

      {/* sources to IR */}
      {SOURCES.map((s, i) => {
        const y = sourceY(i);
        const d = `M262 ${y} C 352 ${y}, 372 ${CY}, ${f1(irPos[0] - 30)} ${CY}`;
        const hot = activeUc === s.id;
        return (
          <g key={s.id}>
            <path d={d} fill="none" stroke={s.color} strokeOpacity={hot ? 0.85 : 0.28} strokeWidth={hot ? 2 : 1.3} />
            {!still &&
              [0, 1].map((k) => (
                <circle key={k} r={hot ? 3.6 : 2.4} fill={s.color} opacity={hot ? 1 : 0.55} filter={hot ? `url(#glow${uid})` : undefined}>
                  <animateMotion dur={`${hot ? 1.8 : 5.5 + i * 0.35}s`} begin={`-${k * (hot ? 0.9 : 2.7) + i * 0.4}s`} repeatCount="indefinite" path={d} />
                </circle>
              ))}
          </g>
        );
      })}

      {/* ring segments */}
      {segments.map((s) => {
        const hot = activeNode === s.to;
        return (
          <g key={s.from}>
            <path d={s.d} fill="none" stroke="var(--color-line-2)" strokeWidth="2" strokeLinecap="round" />
            <path d={s.d} fill="none" stroke={s.color} strokeWidth={hot ? 3 : 1.6} strokeOpacity={hot ? 0.95 : 0.45} strokeLinecap="round" strokeDasharray="5 9" style={{ animation: still ? "none" : `dash-flow ${hot ? 0.9 : 3.2}s linear infinite` }} />
            {!still && (
              <circle r={hot ? 4 : 2.6} fill={s.color} opacity={hot ? 1 : 0.7} filter={`url(#glow${uid})`}>
                <animateMotion dur={`${hot ? 1.6 : 4.2}s`} repeatCount="indefinite" path={s.d} begin={`-${RING.findIndex((n) => n.id === s.from) * 0.7}s`} />
              </circle>
            )}
          </g>
        );
      })}

      {/* hub: Harness2 experience loop feeding the lifter */}
      <g>
        <path d={`M${f1(CX - 54)} ${CY} L${f1(irPos[0] + 30)} ${CY}`} stroke="#f5c04d" strokeOpacity="0.55" strokeWidth="1.5" strokeDasharray="3 5" markerEnd={`url(#arrow${uid})`} fill="none" />
        {!still && (
          <circle r="2.8" fill="#f5c04d" filter={`url(#glow${uid})`}>
            <animateMotion dur={`${hubPulse}s`} repeatCount="indefinite" path={`M${f1(CX - 54)} ${CY} L${f1(irPos[0] + 30)} ${CY}`} />
          </circle>
        )}
        <circle cx={CX} cy={CY} r="52" fill="rgb(10 15 27 / 0.86)" stroke={`url(#ringg${uid})`} strokeWidth="1.5" />
        <circle cx={CX} cy={CY} r="60" fill="none" stroke="#f5c04d" strokeOpacity="0.25" strokeWidth="1" strokeDasharray="1 5" style={{ transformOrigin: `${CX}px ${CY}px`, animation: still ? "none" : "spin 40s linear infinite reverse" }} />
        <text x={CX} y={CY - 16} textAnchor="middle" className="eyebrow" fill="#94a3c0" fontSize="9.5" letterSpacing="1.6">HARNESS²</text>
        <text x={CX} y={CY + 10} textAnchor="middle" fill="#eef2fb" fontSize="26" fontWeight="600" fontFamily="var(--font-display)">
          {generation ? generation.score.toFixed(2) : "n/a"}
        </text>
        <text x={CX} y={CY + 28} textAnchor="middle" fill="#94a3c0" fontSize="10.5">{generation ? `generation ${generation.gen}` : "experience library"}</text>
      </g>

      {/* output to training data */}
      {(() => {
        const [rx, ry] = pos(36);
        const d = `M${f1(rx + 30)} ${f1(ry + 6)} C ${f1(rx + 70)} ${f1(ry + 40)}, ${f1(rx + 90)} ${f1(ry + 46)}, ${W - 98} ${f1(ry + 52)}`;
        return (
          <g>
            <path d={d} fill="none" stroke="#34d399" strokeOpacity="0.45" strokeWidth="1.4" markerEnd={`url(#arrow${uid})`} />
            {!still && (
              <circle r="2.6" fill="#34d399" filter={`url(#glow${uid})`}>
                <animateMotion dur={activeNode === "render" ? "1.6s" : "4.8s"} repeatCount="indefinite" path={d} />
              </circle>
            )}
            <rect x={W - 98} y={ry + 36} width="94" height="34" rx="9" fill="rgb(10 15 27 / 0.9)" stroke="var(--color-line-2)" />
            <text x={W - 51} y={ry + 53} textAnchor="middle" fontSize="11" fill="#eef2fb" fontWeight="600">Training data</text>
            <text x={W - 51} y={ry + 65} textAnchor="middle" fontSize="9.5" fill="#94a3c0">store</text>
          </g>
        );
      })()}

      {/* ring nodes */}
      {RING.map((n, i) => {
        const [x, y] = pos(n.angle);
        const hot = activeNode === n.id;
        return (
          <motion.g key={n.id} initial={{ opacity: 0, scale: 0.7 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: 0.15 + i * 0.08, type: "spring", stiffness: 220, damping: 18 }} style={{ transformOrigin: `${x}px ${y}px` }}>
            {hot && !still && <circle className="fw-node-ring" cx={x} cy={y} r="26" fill="none" stroke={n.color} strokeWidth="2" />}
            <circle cx={x} cy={y} r="27" fill="rgb(10 15 27 / 0.92)" stroke={n.color} strokeWidth={hot ? 2.4 : 1.4} strokeOpacity={hot ? 1 : 0.65} filter={hot ? `url(#glow${uid})` : undefined} />
            <n.Icon x={x - 11} y={y - 11} width={22} height={22} color={n.color} strokeWidth={1.8} />
            <text x={x + n.lx} y={y + n.ly} textAnchor={n.anchor} fill={hot ? "#fff" : "#eef2fb"} fontSize="14" fontWeight="650" fontFamily="var(--font-display)">{n.label}</text>
            <text x={x + n.lx} y={y + n.ly + 15} textAnchor={n.anchor} fill="#94a3c0" fontSize="10.5">{n.sub}</text>
          </motion.g>
        );
      })}

      {/* source pills */}
      {SOURCES.map((s, i) => {
        const y = sourceY(i);
        const hot = activeUc === s.id;
        return (
          <motion.g key={s.id} initial={{ opacity: 0, x: -14 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.05 + i * 0.07, duration: 0.4 }}>
            <rect x="14" y={y - 22} width="248" height="44" rx="12" fill="rgb(10 15 27 / 0.9)" stroke={hot ? s.color : "var(--color-line-2)"} strokeWidth={hot ? 1.6 : 1} />
            <rect x="14" y={y - 22} width="5" height="44" rx="2.5" fill={s.color} />
            <s.Icon x={34} y={y - 10} width={20} height={20} color={s.color} strokeWidth={1.8} />
            <text x="66" y={y - 3} fill="#eef2fb" fontSize="13" fontWeight="650" fontFamily="var(--font-display)">{s.label}</text>
            <text x="66" y={y + 12} fill="#94a3c0" fontSize="11">{s.sub}</text>
            {hot && (
              <text x="250" y={y + 4} textAnchor="end" fill={s.color} fontSize="10" fontWeight="700" letterSpacing="1.2">ACTIVE</text>
            )}
          </motion.g>
        );
      })}

      {/* status line */}
      <text x="18" y={H - 12} fill="#94a3c0" fontSize="11.5">
        {busy ? (
          <>
            <tspan fill="#22d3ee" fontWeight="700">RUNNING</tspan>
            <tspan> {runLabel ?? ""}</tspan>
            <tspan fill="#7684a2">  ·  queue {queue}</tspan>
          </>
        ) : (
          <tspan>Idle · queue {queue} · waiting for the next job</tspan>
        )}
      </text>
    </svg>
  );
}
