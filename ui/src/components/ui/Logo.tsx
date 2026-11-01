import { useId } from "react";

/**
 * Brand mark: a flywheel ring split into four arcs (use cases feeding the loop) around the sign of
 * Tanit (circle, crossbar, triangle). Original artwork.
 */
export function LogoMark({ size = 32, className }: { size?: number; className?: string }) {
  const id = useId().replace(/:/g, "");
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" className={className} role="img" aria-label="Tanit Scenario Flywheel">
      <defs>
        <linearGradient id={`g${id}`} x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="#9b8cff" />
          <stop offset="1" stopColor="#22d3ee" />
        </linearGradient>
      </defs>
      <g fill="none" stroke={`url(#g${id})`} strokeWidth="2" strokeLinecap="round">
        <path d="M16 3.5a12.5 12.5 0 0 1 11.9 8.7" opacity=".95" />
        <path d="M28.5 17.5a12.5 12.5 0 0 1-8.2 10" opacity=".75" />
        <path d="M13 28.2a12.5 12.5 0 0 1-9.3-9.6" opacity=".55" />
        <path d="M4.6 11.8A12.5 12.5 0 0 1 11 4.6" opacity=".4" />
      </g>
      <circle cx="16" cy="9.6" r="2.2" fill={`url(#g${id})`} />
      <path d="M10 14.2h12" stroke={`url(#g${id})`} strokeWidth="2" strokeLinecap="round" />
      <path d="M16 15.5 21.4 24.5h-10.8z" fill={`url(#g${id})`} opacity=".95" />
    </svg>
  );
}
