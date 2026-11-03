import { Canvas, useThree } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import { Pause, Play } from "lucide-react";
import { useEffect, useMemo, useRef, useState, type ComponentRef } from "react";
import * as THREE from "three";
import { useApi } from "../../lib/provider";
import { Field, Segmented, Slider, Toggle } from "../ui/primitives";

type ColorBy = "height" | "intensity" | "range" | "flat";
type View = "chase" | "top" | "side";
const GROUND = -1.9; // sensor frame: the roof LiDAR sits about 1.9 m above the road

// sensor (x forward, y left, z up) -> three (x right, y up, z backwards)
const S2T = (x: number, y: number, z: number): [number, number, number] => [-y, z, -x];

/** Turbo colour map (polynomial fit, Google AI 2019), t in 0..1. */
function turbo(t: number): [number, number, number] {
  const c = (v: number) => Math.min(1, Math.max(0, v));
  t = c(t);
  return [
    c(0.13572138 + t * (4.6153926 + t * (-42.66032258 + t * (132.13108234 + t * (-152.94239396 + t * 59.28637943))))),
    c(0.09140261 + t * (2.19418839 + t * (4.84296658 + t * (-14.18503333 + t * (4.27729857 + t * 2.82956604))))),
    c(0.1066733 + t * (12.64194608 + t * (-60.58204836 + t * (110.36276771 + t * (-89.90310912 + t * 27.34824973))))),
  ];
}

function buildGeometry(pts: Float32Array, colorBy: ColorBy, maxRange: number, hideGround: boolean) {
  const n = pts.length / 4;
  const pos = new Float32Array(n * 3);
  const col = new Float32Array(n * 3);
  let k = 0;
  for (let i = 0; i < n; i++) {
    const x = pts[4 * i], y = pts[4 * i + 1], z = pts[4 * i + 2], it = pts[4 * i + 3];
    const r = Math.hypot(x, y);
    if (r > maxRange || (hideGround && z < GROUND + 0.25)) continue;
    const [tx, ty, tz] = S2T(x, y, z);
    pos.set([tx, ty, tz], 3 * k);
    let rgb: [number, number, number];
    if (colorBy === "height") rgb = turbo((z - GROUND) / 4);
    else if (colorBy === "intensity") rgb = turbo(Math.log1p(it * 60) / Math.log1p(60)); // most returns are dark: log scale
    else if (colorBy === "range") rgb = turbo(r / maxRange);
    else rgb = [0.55, 0.75, 1];
    col.set(rgb, 3 * k);
    k++;
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(pos.subarray(0, 3 * k), 3));
  g.setAttribute("color", new THREE.BufferAttribute(col.subarray(0, 3 * k), 3));
  return { geometry: g, count: k };
}

function Cloud({ geometry, size }: { geometry: THREE.BufferGeometry; size: number }) {
  useEffect(() => () => geometry.dispose(), [geometry]);
  return (
    <points geometry={geometry}>
      <pointsMaterial size={size} vertexColors sizeAttenuation />
    </points>
  );
}

function Rings({ maxRange }: { maxRange: number }) {
  const rings = [];
  for (let r = 10; r <= maxRange; r += 10) rings.push(r);
  return (
    <group position={[0, GROUND, 0]} rotation={[-Math.PI / 2, 0, 0]}>
      {rings.map((r) => (
        <mesh key={r}>
          <ringGeometry args={[r - 0.04, r + 0.04, 128]} />
          <meshBasicMaterial color={r % 50 === 0 ? "#3b5a8c" : "#1d2c47"} side={THREE.DoubleSide} />
        </mesh>
      ))}
    </group>
  );
}

function Ego() {
  // Hyperion rig: LiDAR above the rear axle; a 4.9 x 1.9 x 1.5 m box, nose 3.9 m ahead of the sensor
  return (
    <mesh position={S2T(1.45, 0, GROUND + 0.75)}>
      <boxGeometry args={[1.9, 1.5, 4.9]} />
      <meshBasicMaterial color="#22d3ee" wireframe />
    </mesh>
  );
}

const VIEWS: Record<View, { pos: [number, number, number]; target: [number, number, number] }> = {
  chase: { pos: [0, 14, 26], target: [0, 0, -12] },
  top: { pos: [0, 95, 0.01], target: [0, 0, 0] },
  side: { pos: [48, 12, -6], target: [0, 0, -6] },
};

function ViewRig({ view, nonce }: { view: View; nonce: number }) {
  const controls = useRef<ComponentRef<typeof OrbitControls>>(null);
  const { camera } = useThree();
  useEffect(() => {
    camera.position.set(...VIEWS[view].pos);
    controls.current?.target.set(...VIEWS[view].target);
    controls.current?.update();
  }, [camera, view, nonce]);
  return <OrbitControls ref={controls} makeDefault enableDamping dampingFactor={0.1} minDistance={2} maxDistance={400} />;
}

/** Generated LiDAR sweeps in 3D: orbit (drag), pan (right-drag), zoom (wheel); sweep playback and display options. */
export default function LidarViewer({ runId, sweeps, hz, t0 }: { runId: string; sweeps: number; hz: number; t0: number }) {
  const { api } = useApi();
  const [i, setI] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [colorBy, setColorBy] = useState<ColorBy>("height");
  const [size, setSize] = useState(0.12);
  const [maxRange, setMaxRange] = useState(80);
  const [every, setEvery] = useState<"1" | "2" | "4">("1");
  const [hideGround, setHideGround] = useState(false);
  const [rings, setRings] = useState(true);
  const [ego, setEgo] = useState(true);
  const [view, setView] = useState<View>("chase");
  const [nonce, setNonce] = useState(0);
  const cache = useRef(new Map<string, Float32Array>());
  const [pts, setPts] = useState<Float32Array | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    let live = true;
    const load = async (k: number) => {
      const key = `${k}/${every}`;
      if (!cache.current.has(key)) cache.current.set(key, await api.lidarSweep(runId, k, Number(every)));
      return cache.current.get(key)!;
    };
    load(i).then((p) => {
      if (live) {
        setPts(p);
        setErr(null);
      }
      if (i + 1 < sweeps) void load(i + 1).catch(() => undefined); // prefetch for playback
    }, (e: Error) => live && setErr(e.message));
    return () => {
      live = false;
    };
  }, [api, runId, i, every, sweeps]);

  useEffect(() => {
    if (!playing) return;
    const id = setInterval(() => setI((k) => (k + 1) % sweeps), 1000 / hz);
    return () => clearInterval(id);
  }, [playing, hz, sweeps]);

  const built = useMemo(() => (pts ? buildGeometry(pts, colorBy, maxRange, hideGround) : null), [pts, colorBy, maxRange, hideGround]);

  return (
    <div className="grid gap-4 xl:grid-cols-[1fr_280px]">
      <div className="relative h-[560px] overflow-hidden rounded-xl border border-line bg-[#04070f]">
        <Canvas dpr={[1, 1.75]} camera={{ fov: 50, near: 0.1, far: 1200, position: VIEWS.chase.pos }} aria-label="3D LiDAR point cloud">
          <color attach="background" args={["#04070f"]} />
          {built && <Cloud geometry={built.geometry} size={size} />}
          {rings && <Rings maxRange={maxRange} />}
          {ego && <Ego />}
          <ViewRig view={view} nonce={nonce} />
        </Canvas>
        <div className="pointer-events-none absolute top-2 left-3 mono text-[11.5px] text-fg-1">
          sweep {i + 1}/{sweeps} · t = {(t0 + i / hz).toFixed(2)} s · {built ? built.count.toLocaleString() : "…"} points
        </div>
        <div className="pointer-events-none absolute bottom-2 left-3 text-[11px] text-fg-3">Drag to orbit · right-drag to pan · wheel to zoom</div>
        {err && <div className="absolute inset-0 grid place-items-center text-[13px] text-bad">LiDAR sweep could not be loaded: {err}</div>}
      </div>

      <div className="space-y-4">
        <div className="flex items-center gap-2">
          <button className="btn btn-sm !px-2.5" onClick={() => setPlaying((p) => !p)} aria-label={playing ? "Pause sweeps" : "Play sweeps"}>
            {playing ? <Pause size={14} /> : <Play size={14} />}
          </button>
          <div className="flex-1">
            <Slider label="Sweep" value={i} min={0} max={Math.max(0, sweeps - 1)} onChange={(v) => { setPlaying(false); setI(v); }} />
          </div>
        </div>
        <Field label="View">
          <Segmented label="View" size="sm" value={view} onChange={(v) => { setView(v); setNonce((n) => n + 1); }}
            options={[{ value: "chase", label: "Chase" }, { value: "top", label: "Top" }, { value: "side", label: "Side" }]} />
        </Field>
        <Field label="Colour by">
          <Segmented label="Colour by" size="sm" value={colorBy} onChange={setColorBy}
            options={[{ value: "height", label: "Height" }, { value: "intensity", label: "Intensity" }, { value: "range", label: "Range" }, { value: "flat", label: "Flat" }]} />
        </Field>
        <Field label={<>Point size: <span className="num text-fg">{size.toFixed(2)} m</span></>}>
          <Slider label="Point size" value={size} min={0.02} max={0.4} step={0.02} onChange={setSize} />
        </Field>
        <Field label={<>Max range: <span className="num text-fg">{maxRange} m</span></>}>
          <Slider label="Max range" value={maxRange} min={10} max={100} step={5} onChange={setMaxRange} />
        </Field>
        <Field label="Density">
          <Segmented label="Density" size="sm" value={every} onChange={setEvery}
            options={[{ value: "1", label: "All" }, { value: "2", label: "1/2" }, { value: "4", label: "1/4" }]} />
        </Field>
        <div className="space-y-2.5 text-[12.5px] text-fg-1">
          <label className="flex items-center justify-between">Hide ground <Toggle checked={hideGround} onChange={setHideGround} label="Hide ground" /></label>
          <label className="flex items-center justify-between">Range rings (10 m) <Toggle checked={rings} onChange={setRings} label="Range rings" /></label>
          <label className="flex items-center justify-between">Ego vehicle <Toggle checked={ego} onChange={setEgo} label="Ego vehicle" /></label>
        </div>
        <p className="text-[11.5px] leading-snug text-fg-3">
          Cosmos3 Sim-Auto LiDAR, 128 beams at {hz} Hz, in the sensor frame. Ground and ego box are approximate (roof mount about 1.9 m above the road).
        </p>
      </div>
    </div>
  );
}
