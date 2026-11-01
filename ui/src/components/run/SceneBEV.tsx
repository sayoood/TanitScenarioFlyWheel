import { Crosshair, Maximize2 } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState, type PointerEvent as ReactPointerEvent } from "react";
import clsx from "clsx";
import type { Scene } from "../../lib/types";
import { Toggle } from "../ui/primitives";
import { MS_TO_KMH, actorColor, nearestSignal, samplePose, signalColor, signalState, usePlayback, usePlaybackState } from "./playback";

interface View {
  cx: number;
  cy: number;
  scale: number; // px per metre
}

const BG = "#050912";
const ROAD = "#131b2d";

function roundRect(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number) {
  const rr = Math.min(r, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + rr, y);
  ctx.arcTo(x + w, y, x + w, y + h, rr);
  ctx.arcTo(x + w, y + h, x, y + h, rr);
  ctx.arcTo(x, y + h, x, y, rr);
  ctx.arcTo(x, y, x + w, y, rr);
  ctx.closePath();
}

export function SceneBEV({ scene, className }: { scene: Scene; className?: string }) {
  const pb = usePlayback();
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const view = useRef<View>({ cx: 0, cy: 0, scale: 5 });
  const size = useRef({ w: 0, h: 0, dpr: 1 });
  const raf = useRef(0);
  const [follow, setFollow] = useState(false);
  const followRef = useRef(false);
  const [labels, setLabels] = useState(true);
  const labelsRef = useRef(true);
  const [trails, setTrails] = useState(true);
  const trailsRef = useRef(true);
  const [hover, setHover] = useState<{ id: string; x: number; y: number } | null>(null);
  const drag = useRef<{ x: number; y: number; cx: number; cy: number } | null>(null);
  const ps = usePlaybackState(pb);

  const egoId = useMemo(() => scene.actors.find((a) => a.role === "ego")?.id ?? scene.actors[0]?.id, [scene]);
  const sigOfLine = useMemo(() => scene.stop_lines.map((l) => nearestSignal(scene, l)), [scene]);
  const junctionRect = useMemo(() => {
    const pts = scene.lanes.filter((l) => l.junction).flatMap((l) => l.center);
    if (pts.length === 0) return null;
    const xs = pts.map((p) => p[0]);
    const ys = pts.map((p) => p[1]);
    return { x0: Math.min(...xs) - 1.75, x1: Math.max(...xs) + 1.75, y0: Math.min(...ys) - 1.75, y1: Math.max(...ys) + 1.75 };
  }, [scene]);

  const fit = useCallback(() => {
    const { w, h } = size.current;
    if (!w || !h) return;
    // frame the ego's whole journey plus the junction, with a margin
    let x0 = Infinity;
    let y0 = Infinity;
    let x1 = -Infinity;
    let y1 = -Infinity;
    const grow = (x: number, y: number) => {
      x0 = Math.min(x0, x);
      x1 = Math.max(x1, x);
      y0 = Math.min(y0, y);
      y1 = Math.max(y1, y);
    };
    const tr = egoId ? scene.tracks[egoId] : undefined;
    if (tr && tr.length > 0) {
      grow(tr[0][0], tr[0][1]);
      grow(tr[Math.floor(tr.length * 0.8)][0], tr[Math.floor(tr.length * 0.8)][1]);
    }
    if (junctionRect) {
      grow(junctionRect.x0, junctionRect.y0);
      grow(junctionRect.x1, junctionRect.y1);
    }
    if (!Number.isFinite(x0)) {
      [x0, y0, x1, y1] = scene.bounds;
    }
    const pad = 26;
    const bw = Math.max(x1 - x0 + 2 * pad, 60);
    const bh = Math.max(y1 - y0 + 2 * pad, 60);
    view.current = { cx: (x0 + x1) / 2, cy: (y0 + y1) / 2, scale: Math.max(1.2, Math.min(w / bw, h / bh)) };
  }, [scene, junctionRect, egoId]);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const { w, h, dpr } = size.current;
    if (!w || !h) return;
    const t = pb.t;
    const ts = scene.t;
    const v = view.current;

    if (followRef.current && egoId) {
      const e = samplePose(ts, scene.tracks[egoId], t);
      v.cx = e.x;
      v.cy = e.y;
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = BG;
    ctx.fillRect(0, 0, w, h);
    const X = (x: number) => w / 2 + (x - v.cx) * v.scale;
    const Y = (y: number) => h / 2 - (y - v.cy) * v.scale;

    // grid (10 m minor, 50 m major)
    const left = v.cx - w / 2 / v.scale;
    const right = v.cx + w / 2 / v.scale;
    const bottom = v.cy - h / 2 / v.scale;
    const top = v.cy + h / 2 / v.scale;
    for (const [step, alpha] of [
      [10, 0.035],
      [50, 0.08],
    ] as const) {
      if (step * v.scale < 8) continue;
      ctx.strokeStyle = `rgba(148,163,184,${alpha})`;
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let gx = Math.ceil(left / step) * step; gx < right; gx += step) {
        ctx.moveTo(X(gx), 0);
        ctx.lineTo(X(gx), h);
      }
      for (let gy = Math.ceil(bottom / step) * step; gy < top; gy += step) {
        ctx.moveTo(0, Y(gy));
        ctx.lineTo(w, Y(gy));
      }
      ctx.stroke();
    }

    // road surfaces
    ctx.fillStyle = ROAD;
    if (junctionRect) {
      ctx.fillRect(X(junctionRect.x0), Y(junctionRect.y1), (junctionRect.x1 - junctionRect.x0) * v.scale, (junctionRect.y1 - junctionRect.y0) * v.scale);
    }
    for (const lane of scene.lanes) {
      if (lane.left.length < 2 || lane.right.length < 2) continue;
      ctx.beginPath();
      ctx.moveTo(X(lane.left[0][0]), Y(lane.left[0][1]));
      for (let i = 1; i < lane.left.length; i++) ctx.lineTo(X(lane.left[i][0]), Y(lane.left[i][1]));
      for (let i = lane.right.length - 1; i >= 0; i--) ctx.lineTo(X(lane.right[i][0]), Y(lane.right[i][1]));
      ctx.closePath();
      ctx.fillStyle = lane.junction ? "#161f33" : ROAD;
      ctx.fill();
    }
    // lane edges and centre lines
    ctx.lineWidth = 1;
    ctx.strokeStyle = "rgba(125,145,190,0.34)";
    for (const lane of scene.lanes) {
      if (lane.junction) continue;
      for (const side of [lane.left, lane.right]) {
        ctx.beginPath();
        side.forEach((p, i) => (i ? ctx.lineTo(X(p[0]), Y(p[1])) : ctx.moveTo(X(p[0]), Y(p[1]))));
        ctx.stroke();
      }
    }
    ctx.setLineDash([5, 7]);
    ctx.strokeStyle = "rgba(139,124,246,0.28)";
    for (const lane of scene.lanes) {
      if (!lane.junction) continue;
      ctx.beginPath();
      lane.center.forEach((p, i) => (i ? ctx.lineTo(X(p[0]), Y(p[1])) : ctx.moveTo(X(p[0]), Y(p[1]))));
      ctx.stroke();
    }
    ctx.setLineDash([]);

    // stop lines coloured by the live signal state
    scene.stop_lines.forEach((l, k) => {
      if (l.length < 2) return;
      const si = sigOfLine[k];
      const st = si >= 0 ? signalState(scene.signals[si].states, ts, t) : "off";
      ctx.strokeStyle = signalColor(st);
      ctx.lineWidth = Math.max(2.5, v.scale * 0.45);
      ctx.lineCap = "butt";
      ctx.beginPath();
      l.forEach((p, i) => (i ? ctx.lineTo(X(p[0]), Y(p[1])) : ctx.moveTo(X(p[0]), Y(p[1]))));
      ctx.stroke();
    });

    // signals
    for (const s of scene.signals) {
      const st = signalState(s.states, ts, t);
      const c = signalColor(st);
      const r = Math.max(4, v.scale * 0.9);
      const g = ctx.createRadialGradient(X(s.x), Y(s.y), 0, X(s.x), Y(s.y), r * 3);
      g.addColorStop(0, c + "99");
      g.addColorStop(1, c + "00");
      ctx.fillStyle = g;
      ctx.beginPath();
      ctx.arc(X(s.x), Y(s.y), r * 3, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = c;
      ctx.strokeStyle = "#0b1020";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(X(s.x), Y(s.y), r, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    }

    // trails
    if (trailsRef.current) {
      for (const a of scene.actors) {
        const tr = scene.tracks[a.id];
        if (!tr) continue;
        const i1 = samplePose(ts, tr, t).i;
        if (i1 < 1) continue;
        const ego = a.role === "ego";
        ctx.lineWidth = ego ? 2.4 : 1.2;
        ctx.strokeStyle = ego ? "rgba(34,211,238,0.9)" : actorColor(a.color) + "66";
        ctx.beginPath();
        for (let i = 0; i <= i1; i++) (i ? ctx.lineTo(X(tr[i][0]), Y(tr[i][1])) : ctx.moveTo(X(tr[i][0]), Y(tr[i][1])));
        const cur = samplePose(ts, tr, t);
        ctx.lineTo(X(cur.x), Y(cur.y));
        ctx.stroke();
      }
    }

    // actors (npcs first, ego on top)
    const order = [...scene.actors].sort((a, b) => (a.role === "ego" ? 1 : 0) - (b.role === "ego" ? 1 : 0));
    for (const a of order) {
      const p = samplePose(ts, scene.tracks[a.id], t);
      const [L, Wd] = a.lwh;
      const ego = a.role === "ego";
      ctx.save();
      ctx.translate(X(p.x), Y(p.y));
      ctx.rotate(-p.yaw);
      const l = L * v.scale;
      const wd = Wd * v.scale;
      if (ego) {
        ctx.shadowColor = "rgba(34,211,238,0.85)";
        ctx.shadowBlur = 16;
      }
      roundRect(ctx, -l / 2, -wd / 2, l, wd, Math.min(wd * 0.25, 6));
      ctx.fillStyle = actorColor(a.color);
      ctx.fill();
      ctx.shadowBlur = 0;
      ctx.lineWidth = ego ? 2 : 1;
      ctx.strokeStyle = ego ? "#22d3ee" : "rgba(5,9,18,0.8)";
      ctx.stroke();
      // windshield + heading tick
      ctx.fillStyle = "rgba(5,9,18,0.55)";
      ctx.fillRect(l * 0.1, -wd * 0.38, l * 0.2, wd * 0.76);
      ctx.fillStyle = ego ? "#22d3ee" : "rgba(255,255,255,0.85)";
      ctx.beginPath();
      ctx.moveTo(l / 2 + 2, 0);
      ctx.lineTo(l / 2 - Math.min(7, l * 0.25), -Math.min(4, wd * 0.25));
      ctx.lineTo(l / 2 - Math.min(7, l * 0.25), Math.min(4, wd * 0.25));
      ctx.closePath();
      ctx.fill();
      ctx.restore();

      if (labelsRef.current && (ego || v.scale > 2.6)) {
        ctx.font = "600 10.5px 'JetBrains Mono Variable', monospace";
        ctx.textAlign = "center";
        const txt = `${a.id} ${(p.v * MS_TO_KMH).toFixed(0)}`;
        const tx = X(p.x);
        const ty = Y(p.y) - Math.max(wd, l) / 2 - 7;
        ctx.fillStyle = "rgba(5,9,18,0.72)";
        const tw = ctx.measureText(txt).width + 8;
        roundRect(ctx, tx - tw / 2, ty - 11, tw, 15, 4);
        ctx.fill();
        ctx.fillStyle = ego ? "#67e8f9" : "#cbd5e1";
        ctx.fillText(txt, tx, ty);
      }
    }

    // scale bar and north arrow
    const bar = [10, 20, 50, 100].find((m) => m * v.scale > 60) ?? 100;
    ctx.strokeStyle = "rgba(203,213,225,0.9)";
    ctx.fillStyle = "rgba(203,213,225,0.9)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(14, h - 16);
    ctx.lineTo(14 + bar * v.scale, h - 16);
    ctx.moveTo(14, h - 21);
    ctx.lineTo(14, h - 11);
    ctx.moveTo(14 + bar * v.scale, h - 21);
    ctx.lineTo(14 + bar * v.scale, h - 11);
    ctx.stroke();
    ctx.font = "600 10.5px 'JetBrains Mono Variable', monospace";
    ctx.textAlign = "left";
    ctx.fillText(`${bar} m`, 20 + bar * v.scale, h - 12);
    ctx.textAlign = "center";
    ctx.fillText("N", w - 22, 26);
    ctx.beginPath();
    ctx.moveTo(w - 22, 30);
    ctx.lineTo(w - 27, 44);
    ctx.lineTo(w - 22, 40);
    ctx.lineTo(w - 17, 44);
    ctx.closePath();
    ctx.fill();
  }, [scene, pb, egoId, junctionRect, sigOfLine]);

  const schedule = useCallback(() => {
    if (raf.current) return;
    raf.current = requestAnimationFrame(() => {
      raf.current = 0;
      draw();
    });
  }, [draw]);

  // resize
  useEffect(() => {
    const el = wrapRef.current;
    const canvas = canvasRef.current;
    if (!el || !canvas) return;
    const apply = () => {
      const r = el.getBoundingClientRect();
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const first = size.current.w === 0;
      size.current = { w: Math.max(1, r.width), h: Math.max(1, r.height), dpr };
      canvas.width = Math.round(r.width * dpr);
      canvas.height = Math.round(r.height * dpr);
      if (first) fit();
      schedule();
    };
    apply();
    const ro = new ResizeObserver(apply);
    ro.observe(el);
    return () => ro.disconnect();
  }, [fit, schedule]);

  useEffect(() => pb.subscribe(schedule), [pb, schedule]);
  useEffect(() => {
    followRef.current = follow;
    labelsRef.current = labels;
    trailsRef.current = trails;
    schedule();
  }, [follow, labels, trails, schedule]);
  useEffect(() => {
    fit();
    schedule();
  }, [scene, fit, schedule]);
  useEffect(
    () => () => {
      cancelAnimationFrame(raf.current);
      raf.current = 0; // StrictMode re-mounts: a stale handle would block every later schedule()
    },
    [],
  );

  // interaction
  const onWheel = useCallback(
    (e: WheelEvent) => {
      e.preventDefault();
      const el = canvasRef.current;
      if (!el) return;
      const r = el.getBoundingClientRect();
      const mx = e.clientX - r.left;
      const my = e.clientY - r.top;
      const v = view.current;
      const { w, h } = size.current;
      const wx = v.cx + (mx - w / 2) / v.scale;
      const wy = v.cy - (my - h / 2) / v.scale;
      const k = Math.exp(-e.deltaY * 0.0015);
      v.scale = Math.min(60, Math.max(0.6, v.scale * k));
      if (!followRef.current) {
        v.cx = wx - (mx - w / 2) / v.scale;
        v.cy = wy + (my - h / 2) / v.scale;
      }
      schedule();
    },
    [schedule],
  );
  useEffect(() => {
    const el = canvasRef.current;
    if (!el) return;
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, [onWheel]);

  const onPointerDown = (e: ReactPointerEvent) => {
    (e.target as Element).setPointerCapture(e.pointerId);
    drag.current = { x: e.clientX, y: e.clientY, cx: view.current.cx, cy: view.current.cy };
  };
  const onPointerMove = (e: ReactPointerEvent) => {
    const el = canvasRef.current;
    if (!el) return;
    if (drag.current && !followRef.current) {
      const v = view.current;
      v.cx = drag.current.cx - (e.clientX - drag.current.x) / v.scale;
      v.cy = drag.current.cy + (e.clientY - drag.current.y) / v.scale;
      schedule();
      return;
    }
    const r = el.getBoundingClientRect();
    const mx = e.clientX - r.left;
    const my = e.clientY - r.top;
    const v = view.current;
    const { w, h } = size.current;
    const wx = v.cx + (mx - w / 2) / v.scale;
    const wy = v.cy - (my - h / 2) / v.scale;
    let best: string | null = null;
    let bd = Infinity;
    for (const a of scene.actors) {
      const p = samplePose(scene.t, scene.tracks[a.id], pb.t);
      const d = Math.hypot(p.x - wx, p.y - wy);
      if (d < Math.max(a.lwh[0] / 2 + 1, 12 / v.scale) && d < bd) {
        bd = d;
        best = a.id;
      }
    }
    setHover(best ? { id: best, x: mx, y: my } : null);
  };
  const onPointerUp = () => {
    drag.current = null;
  };

  const hovered = hover ? scene.actors.find((a) => a.id === hover.id) : undefined;
  const hp = hovered ? samplePose(scene.t, scene.tracks[hovered.id], ps.t) : undefined;

  return (
    <div ref={wrapRef} className={clsx("relative overflow-hidden rounded-xl border border-line", className)} style={{ background: BG }}>
      <canvas
        ref={canvasRef}
        className="absolute inset-0 h-full w-full cursor-grab touch-none active:cursor-grabbing"
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerLeave={() => setHover(null)}
        onDoubleClick={() => {
          fit();
          schedule();
        }}
        role="img"
        aria-label="Bird's-eye view of the scene. Scroll to zoom, drag to pan, double-click to reset."
      />
      <div className="pointer-events-none absolute top-2.5 left-3 flex flex-col gap-1">
        <span className="rounded-md bg-bg/70 px-2 py-0.5 text-[11px] font-semibold tracking-wider text-fg-1 backdrop-blur">BEV</span>
      </div>
      <div className="absolute top-2 right-2.5 flex items-center gap-1.5 rounded-lg border border-line bg-bg/75 p-1 backdrop-blur">
        <button className={clsx("btn btn-sm btn-icon !h-7 !w-7", follow && "!border-accent-2/60 !text-accent-2")} onClick={() => setFollow((f) => !f)} aria-pressed={follow} title="Follow ego">
          <Crosshair size={14} />
        </button>
        <button
          className="btn btn-sm btn-icon !h-7 !w-7"
          onClick={() => {
            setFollow(false);
            followRef.current = false;
            fit();
            schedule();
          }}
          title="Reset view"
        >
          <Maximize2 size={14} />
        </button>
        <label className="flex items-center gap-1.5 px-1 text-[11px] text-fg-1">
          Labels <Toggle checked={labels} onChange={setLabels} label="Show labels" />
        </label>
        <label className="flex items-center gap-1.5 px-1 text-[11px] text-fg-1">
          Trails <Toggle checked={trails} onChange={setTrails} label="Show trails" />
        </label>
      </div>
      {hovered && hp && hover && (
        <div className="pointer-events-none absolute z-10 rounded-lg border border-line-2 bg-bg-2/95 px-2.5 py-1.5 text-[11.5px] shadow-xl" style={{ left: Math.min(hover.x + 14, size.current.w - 170), top: Math.max(hover.y - 8, 8) }}>
          <div className="font-semibold text-fg">
            {hovered.id} <span className="font-normal text-fg-2">{hovered.category}</span>
          </div>
          <div className="mono text-fg-1">{(hp.v * MS_TO_KMH).toFixed(1)} km/h</div>
          <div className="mono text-fg-2">
            x {hp.x.toFixed(1)} y {hp.y.toFixed(1)}
          </div>
        </div>
      )}
    </div>
  );
}
