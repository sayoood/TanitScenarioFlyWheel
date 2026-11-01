import { Camera, Radar, VideoOff } from "lucide-react";
import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import clsx from "clsx";
import { useApi } from "../../lib/provider";
import { CAMERA_NAMES, type CameraName, type RunDetail, type Scene } from "../../lib/types";
import { Chip } from "../ui/primitives";
import { actorColor, samplePose, signalColor, signalState, usePlayback, type Playback } from "./playback";

interface Rig {
  yaw: number; // degrees, left positive
  fov: number;
  tele?: boolean;
}

export const RIG: Record<CameraName, Rig> = {
  front_wide: { yaw: 0, fov: 110 },
  front_tele: { yaw: 0, fov: 34, tele: true },
  cross_left: { yaw: 65, fov: 110 },
  cross_right: { yaw: -65, fov: 110 },
  rear_left: { yaw: 145, fov: 110 },
  rear_right: { yaw: -145, fov: 110 },
  rear_tele: { yaw: 180, fov: 34, tele: true },
};

const CAM_H = 1.55;
const NEAR = 0.4;
type V3 = [number, number, number];

function clipNear(poly: V3[]): V3[] {
  const out: V3[] = [];
  for (let i = 0; i < poly.length; i++) {
    const a = poly[i];
    const b = poly[(i + 1) % poly.length];
    const ina = a[0] >= NEAR;
    const inb = b[0] >= NEAR;
    if (ina) out.push(a);
    if (ina !== inb) {
      const f = (NEAR - a[0]) / (b[0] - a[0]);
      out.push([NEAR, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f]);
    }
  }
  return out;
}

function drawCamera(canvas: HTMLCanvasElement, scene: Scene, pb: Playback, rig: Rig) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const W = canvas.width;
  const H = canvas.height;
  const t = pb.t;
  const ego = scene.actors.find((a) => a.role === "ego") ?? scene.actors[0];
  const e = samplePose(scene.t, ego && scene.tracks[ego.id], t);
  const yaw = e.yaw + (rig.yaw * Math.PI) / 180;
  const cx = e.x + Math.cos(e.yaw) * 0.8;
  const cy = e.y + Math.sin(e.yaw) * 0.8;
  const f = W / 2 / Math.tan((rig.fov * Math.PI) / 360);
  const cos = Math.cos(yaw);
  const sin = Math.sin(yaw);
  const toCam = (x: number, y: number, z: number): V3 => {
    const dx = x - cx;
    const dy = y - cy;
    return [dx * cos + dy * sin, -dx * sin + dy * cos, z - CAM_H];
  };
  const proj = (p: V3): [number, number] => [W / 2 - (p[1] / p[0]) * f, H / 2 - (p[2] / p[0]) * f];
  const horizon = H / 2;

  const sky = ctx.createLinearGradient(0, 0, 0, horizon);
  sky.addColorStop(0, "#27406e");
  sky.addColorStop(1, "#8fb0d8");
  ctx.fillStyle = sky;
  ctx.fillRect(0, 0, W, horizon);
  const ground = ctx.createLinearGradient(0, horizon, 0, H);
  ground.addColorStop(0, "#1b2433");
  ground.addColorStop(1, "#0d121c");
  ctx.fillStyle = ground;
  ctx.fillRect(0, horizon, W, H - horizon);

  const poly = (pts: V3[], fill?: string, stroke?: string, lw = 1) => {
    const c = clipNear(pts);
    if (c.length < 3) return;
    ctx.beginPath();
    c.forEach((p, i) => {
      const [sx, sy] = proj(p);
      if (i) ctx.lineTo(sx, sy);
      else ctx.moveTo(sx, sy);
    });
    ctx.closePath();
    if (fill) {
      ctx.fillStyle = fill;
      ctx.fill();
    }
    if (stroke) {
      ctx.strokeStyle = stroke;
      ctx.lineWidth = lw;
      ctx.stroke();
    }
  };
  const polyline = (pts: number[][], z: number, stroke: string, lw: number) => {
    ctx.strokeStyle = stroke;
    ctx.lineWidth = lw;
    ctx.beginPath();
    let pen = false;
    for (let i = 0; i < pts.length; i++) {
      const p = toCam(pts[i][0], pts[i][1], z);
      if (p[0] < NEAR) {
        pen = false;
        continue;
      }
      const [sx, sy] = proj(p);
      if (pen) ctx.lineTo(sx, sy);
      else ctx.moveTo(sx, sy);
      pen = true;
    }
    ctx.stroke();
  };

  // road
  for (const lane of scene.lanes) {
    const n = Math.min(lane.left.length, lane.right.length);
    for (let i = 0; i < n - 1; i++) {
      const a = toCam(lane.left[i][0], lane.left[i][1], 0);
      const b = toCam(lane.left[i + 1][0], lane.left[i + 1][1], 0);
      const c = toCam(lane.right[i + 1][0], lane.right[i + 1][1], 0);
      const d = toCam(lane.right[i][0], lane.right[i][1], 0);
      if (a[0] < NEAR && b[0] < NEAR && c[0] < NEAR && d[0] < NEAR) continue;
      poly([a, b, c, d], "#2a3347");
    }
  }
  for (const lane of scene.lanes) {
    if (lane.junction) continue;
    polyline(lane.left, 0, "rgba(220,228,245,0.55)", 1);
    polyline(lane.right, 0, "rgba(220,228,245,0.55)", 1);
  }
  // stop lines
  scene.stop_lines.forEach((l) => {
    if (l.length < 2) return;
    let best = 0;
    let bd = Infinity;
    scene.signals.forEach((s, k) => {
      const d = (s.x - l[0][0]) ** 2 + (s.y - l[0][1]) ** 2;
      if (d < bd) {
        bd = d;
        best = k;
      }
    });
    const st = scene.signals[best] ? signalState(scene.signals[best].states, scene.t, t) : "off";
    polyline(l, 0, signalColor(st), 3);
  });

  // actors as shaded boxes, far to near
  interface Box {
    d: number;
    corners: V3[];
    color: string;
    id: string;
  }
  const boxes: Box[] = [];
  for (const a of scene.actors) {
    if (a.role === "ego") continue;
    const p = samplePose(scene.t, scene.tracks[a.id], t);
    const [L, Wd, Hh] = a.lwh;
    const c = Math.cos(p.yaw);
    const s = Math.sin(p.yaw);
    const corners: V3[] = [];
    for (const z of [0.25, 0.25 + Hh]) {
      for (const [lx, ly] of [
        [L / 2, Wd / 2],
        [L / 2, -Wd / 2],
        [-L / 2, -Wd / 2],
        [-L / 2, Wd / 2],
      ]) {
        corners.push(toCam(p.x + lx * c - ly * s, p.y + lx * s + ly * c, z));
      }
    }
    const d = corners.reduce((m, q) => m + q[0], 0) / 8;
    if (d < NEAR || d > 220) continue;
    boxes.push({ d, corners, color: actorColor(a.color), id: a.id });
  }
  boxes.sort((a, b) => b.d - a.d);
  const faces = [
    [0, 1, 2, 3, 0.62], // bottom (rarely visible)
    [4, 5, 6, 7, 1.0], // top
    [0, 1, 5, 4, 0.9], // front
    [2, 3, 7, 6, 0.7], // rear
    [1, 2, 6, 5, 0.78], // right
    [3, 0, 4, 7, 0.78], // left
  ] as const;
  for (const b of boxes) {
    const base = b.color;
    for (const fc of faces) {
      const pts = [b.corners[fc[0]], b.corners[fc[1]], b.corners[fc[2]], b.corners[fc[3]]];
      // backface culling: face normal vs. camera ray
      const u: V3 = [pts[1][0] - pts[0][0], pts[1][1] - pts[0][1], pts[1][2] - pts[0][2]];
      const v: V3 = [pts[3][0] - pts[0][0], pts[3][1] - pts[0][1], pts[3][2] - pts[0][2]];
      const nrm: V3 = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
      const facing = nrm[0] * pts[0][0] + nrm[1] * pts[0][1] + nrm[2] * pts[0][2];
      if (facing >= 0 && fc[4] !== 1.0) continue;
      if (fc[4] === 1.0 && facing < 0) continue;
      ctx.globalAlpha = 1;
      poly(pts, shade(base, fc[4]), "rgba(5,9,18,0.7)", 1);
    }
  }
  // signal heads
  for (const sg of scene.signals) {
    const p = toCam(sg.x, sg.y, 5);
    if (p[0] < NEAR || p[0] > 160) continue;
    const [sx, sy] = proj(p);
    const r = Math.max(1.5, (f * 0.16) / p[0]);
    const col = signalColor(signalState(sg.states, scene.t, t));
    ctx.fillStyle = col;
    ctx.shadowColor = col;
    ctx.shadowBlur = 8;
    ctx.beginPath();
    ctx.arc(sx, sy, r, 0, Math.PI * 2);
    ctx.fill();
    ctx.shadowBlur = 0;
  }
  // vignette
  const vg = ctx.createRadialGradient(W / 2, H / 2, H * 0.35, W / 2, H / 2, H * 0.95);
  vg.addColorStop(0, "rgba(0,0,0,0)");
  vg.addColorStop(1, "rgba(0,0,0,0.45)");
  ctx.fillStyle = vg;
  ctx.fillRect(0, 0, W, H);
}

function shade(hex: string, k: number): string {
  const m = /^#?([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i.exec(hex);
  if (!m) return hex;
  const c = (i: number) => Math.round(Math.min(255, parseInt(m[i], 16) * k));
  return `rgb(${c(1)},${c(2)},${c(3)})`;
}

function drawLidar(canvas: HTMLCanvasElement, scene: Scene, pb: Playback) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const W = canvas.width;
  const H = canvas.height;
  const ego = scene.actors.find((a) => a.role === "ego") ?? scene.actors[0];
  const e = samplePose(scene.t, ego && scene.tracks[ego.id], pb.t);
  ctx.fillStyle = "#04070d";
  ctx.fillRect(0, 0, W, H);
  const R = 60;
  const sc = Math.min(W, H) / 2 / R;
  const c = Math.cos(-e.yaw + Math.PI / 2);
  const s = Math.sin(-e.yaw + Math.PI / 2);
  const P = (x: number, y: number): [number, number] => {
    const dx = x - e.x;
    const dy = y - e.y;
    return [W / 2 + (dx * c - dy * s) * sc, H / 2 - (dx * s + dy * c) * sc];
  };
  ctx.strokeStyle = "rgba(34,211,238,0.13)";
  ctx.lineWidth = 1;
  for (const r of [15, 30, 45, 60]) {
    ctx.beginPath();
    ctx.arc(W / 2, H / 2, r * sc, 0, Math.PI * 2);
    ctx.stroke();
  }
  const dot = (x: number, y: number, col: string) => {
    const [px, py] = P(x, y);
    if (px < 0 || py < 0 || px > W || py > H) return;
    ctx.fillStyle = col;
    ctx.fillRect(px, py, 1.6, 1.6);
  };
  for (const lane of scene.lanes) {
    if (lane.junction) continue;
    for (const side of [lane.left, lane.right]) {
      for (let i = 0; i < side.length - 1; i++) {
        const a = side[i];
        const b = side[i + 1];
        const L = Math.hypot(b[0] - a[0], b[1] - a[1]);
        for (let d = 0; d < L; d += 1.2) {
          const x = a[0] + ((b[0] - a[0]) * d) / L;
          const y = a[1] + ((b[1] - a[1]) * d) / L;
          const dist = Math.hypot(x - e.x, y - e.y);
          if (dist < R) dot(x, y, `rgba(139,124,246,${0.75 - dist / 120})`);
        }
      }
    }
  }
  for (const a of scene.actors) {
    const p = samplePose(scene.t, scene.tracks[a.id], pb.t);
    const [L, Wd] = a.lwh;
    const co = Math.cos(p.yaw);
    const si = Math.sin(p.yaw);
    for (let k = 0; k < 36; k++) {
      const u = (k / 36) * 4;
      const side = Math.floor(u);
      const f = u - side;
      const pts: [number, number][] = [
        [L / 2, -Wd / 2 + f * Wd],
        [L / 2 - f * L, Wd / 2],
        [-L / 2, Wd / 2 - f * Wd],
        [-L / 2 + f * L, -Wd / 2],
      ];
      const [lx, ly] = pts[side];
      dot(p.x + lx * co - ly * si, p.y + lx * si + ly * co, a.role === "ego" ? "#22d3ee" : "#fbbf24");
    }
  }
  ctx.fillStyle = "#22d3ee";
  ctx.beginPath();
  ctx.moveTo(W / 2, H / 2 - 6);
  ctx.lineTo(W / 2 - 4, H / 2 + 4);
  ctx.lineTo(W / 2 + 4, H / 2 + 4);
  ctx.closePath();
  ctx.fill();
}

function PreviewCanvas({ scene, draw, className }: { scene: Scene; draw: (c: HTMLCanvasElement, scene: Scene, pb: Playback) => void; className?: string }) {
  const pb = usePlayback();
  const ref = useRef<HTMLCanvasElement>(null);
  const visible = useRef(true);
  useEffect(() => {
    const c = ref.current;
    if (!c) return;
    const io = new IntersectionObserver(([en]) => {
      visible.current = en.isIntersecting;
    });
    io.observe(c);
    let raf = 0;
    let lastDraw = 0;
    const paint = () => {
      raf = 0;
      if (!visible.current) return;
      const now = performance.now();
      if (now - lastDraw < 45) {
        raf = requestAnimationFrame(paint);
        return;
      }
      lastDraw = now;
      draw(c, scene, pb);
    };
    const schedule = () => {
      if (!raf) raf = requestAnimationFrame(paint);
    };
    schedule();
    const off = pb.subscribe(schedule);
    return () => {
      off();
      io.disconnect();
      cancelAnimationFrame(raf);
    };
  }, [scene, pb, draw]);
  return <canvas ref={ref} width={480} height={270} className={clsx("block h-full w-full object-cover", className)} />;
}

function SyncedVideo({ src, onFail }: { src: string; onFail: () => void }) {
  const pb = usePlayback();
  const ref = useRef<HTMLVideoElement>(null);
  useEffect(() => {
    const v = ref.current;
    if (!v) return;
    let lastCheck = 0;
    const sync = () => {
      const s = pb.getSnapshot();
      const dur = Number.isFinite(v.duration) ? v.duration : Infinity;
      const target = Math.min(s.t, Math.max(0, dur - 0.05));
      v.playbackRate = Math.min(16, Math.max(0.0625, s.speed));
      const now = performance.now();
      if (!s.playing || now - lastCheck > 400) {
        lastCheck = now;
        if (Math.abs(v.currentTime - target) > (s.playing ? 0.3 : 0.02)) v.currentTime = target;
      }
      if (s.playing && s.t < dur && v.paused) void v.play().catch(() => undefined);
      if ((!s.playing || s.t >= dur) && !v.paused) v.pause();
    };
    sync();
    return pb.subscribe(sync);
  }, [pb]);
  return <video ref={ref} src={src} muted playsInline preload="auto" className="h-full w-full object-cover" onError={onFail} aria-label="Generated camera video" />;
}

function Tile({ label, big, onClick, children, badge }: { label: string; big?: boolean; onClick?: () => void; children: ReactNode; badge?: ReactNode }) {
  return (
    <button
      onClick={onClick}
      className={clsx("group relative overflow-hidden rounded-xl border border-line bg-bg text-left transition-colors hover:border-line-2 focus-visible:border-accent-2", big ? "col-span-2 row-span-2" : "")}
      aria-label={`${label}${big ? " (enlarged)" : ", click to enlarge"}`}
    >
      <div className="aspect-video w-full">{children}</div>
      <div className="pointer-events-none absolute top-2 left-2 flex items-center gap-1.5">
        <span className="rounded-md bg-bg/80 px-1.5 py-0.5 text-[10.5px] font-bold tracking-wider text-fg backdrop-blur">{label}</span>
        {badge}
      </div>
    </button>
  );
}

export function CameraGrid({ run, scene }: { run: RunDetail; scene: Scene | undefined }) {
  const { api, kind } = useApi();
  const [focus, setFocus] = useState<CameraName | "lidar">("front_wide");
  const [failed, setFailed] = useState<Record<string, boolean>>({});
  const available = useMemo(() => new Map((run.cameras ?? []).map((c) => [c.name, c.artifact])), [run.cameras]);
  const lidarArt = run.artifacts.find((a) => a.name.includes("lidar"));
  const drawFns = useMemo(() => Object.fromEntries(CAMERA_NAMES.map((n) => [n, (c: HTMLCanvasElement, s: Scene, pb: Playback) => drawCamera(c, s, pb, RIG[n])])) as Record<CameraName, (c: HTMLCanvasElement, s: Scene, pb: Playback) => void>, []);

  const generated = available.size;
  const stageRunning = run.stages.some((s) => (s.name === "cosmos_transfer" || s.name === "wsm") && s.status === "running");
  const skipped = run.stages.find((s) => s.name === "cosmos_transfer")?.status === "skipped";
  const why = stageRunning ? "Generation in progress" : skipped ? "Render stage skipped for this run" : run.status === "failed" ? "Run failed before rendering" : "Not generated yet";

  return (
    <div>
      <div className="mb-3 flex flex-wrap items-center gap-2 text-[12.5px] text-fg-1">
        <Chip tone={generated ? "ok" : "neutral"}>
          <Camera size={12} /> {generated} of {CAMERA_NAMES.length} cameras
        </Chip>
        {kind === "mock" && generated > 0 && <Chip tone="warn">mock: previews are projected from the Scene, not Cosmos output</Chip>}
        {kind === "live" && generated > 0 && <Chip tone="info">synced to the timeline above</Chip>}
        <span className="text-fg-2">Click a tile to enlarge it.</span>
      </div>
      <div className="grid grid-cols-2 gap-3 md:grid-cols-4" style={{ gridAutoFlow: "dense" }}>
        {CAMERA_NAMES.map((n) => {
          const art = available.get(n);
          const big = focus === n;
          const rig = RIG[n];
          const badge = rig.tele ? <span className="rounded-md bg-bg/80 px-1.5 py-0.5 text-[10px] text-fg-1 backdrop-blur">tele {rig.fov} deg</span> : null;
          let body: ReactNode;
          if (art && scene) {
            const useVideo = kind === "live" && !failed[n];
            body = useVideo ? (
              <SyncedVideo src={api.artifactUrl(run.id, art)} onFail={() => setFailed((f) => ({ ...f, [n]: true }))} />
            ) : (
              <>
                <PreviewCanvas scene={scene} draw={drawFns[n]} />
                {kind === "live" && <span className="pointer-events-none absolute right-2 bottom-2 rounded-md bg-bg/80 px-1.5 py-0.5 text-[10px] text-warn backdrop-blur">video file missing, preview from Scene</span>}
              </>
            );
          } else {
            body = (
              <div className="grid h-full w-full place-items-center bg-[repeating-linear-gradient(135deg,rgb(255_255_255/0.025)_0_10px,transparent_10px_20px)] text-center">
                <div className="px-3">
                  <VideoOff size={big ? 30 : 20} className="mx-auto text-fg-3" />
                  <div className="mt-1.5 text-[12px] text-fg-2">{why}</div>
                </div>
              </div>
            );
          }
          return (
            <Tile key={n} label={n} big={big} onClick={() => setFocus(n)} badge={badge}>
              {body}
            </Tile>
          );
        })}
        <Tile label="lidar" big={focus === "lidar"} onClick={() => setFocus("lidar")} badge={<span className="rounded-md bg-bg/80 px-1.5 py-0.5 text-[10px] text-fg-1 backdrop-blur">128 beams</span>}>
          {lidarArt && scene ? (
            <PreviewCanvas scene={scene} draw={drawLidar} />
          ) : (
            <div className="grid h-full w-full place-items-center bg-[repeating-linear-gradient(135deg,rgb(255_255_255/0.025)_0_10px,transparent_10px_20px)] text-center">
              <div className="px-3">
                <Radar size={20} className="mx-auto text-fg-3" />
                <div className="mt-1.5 text-[12px] text-fg-2">{lidarArt ? "No scene" : "LiDAR not generated"}</div>
              </div>
            </div>
          )}
        </Tile>
      </div>
    </div>
  );
}
