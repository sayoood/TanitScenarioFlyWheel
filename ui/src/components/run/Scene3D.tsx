import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import { useEffect, useMemo, useRef, type ComponentRef } from "react";
import * as THREE from "three";
import type { Scene } from "../../lib/types";
import { actorColor, nearestSignal, samplePose, signalColor, signalState, usePlayback, type Playback } from "./playback";

export type CamMode = "orbit" | "chase" | "top";

// world (x east, y north, z up) -> three (x, y up, z = -north)
const W2T = (x: number, y: number, h = 0): [number, number, number] => [x, h, -y];

// ----------------------------------------------------------------------------- geometry helpers
function resample(pts: number[][], n: number): number[][] {
  if (pts.length === n || pts.length < 2) return pts;
  const cum = [0];
  for (let i = 1; i < pts.length; i++) cum.push(cum[i - 1] + Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]));
  const total = cum[cum.length - 1] || 1;
  const out: number[][] = [];
  let j = 0;
  for (let i = 0; i < n; i++) {
    const d = (i / (n - 1)) * total;
    while (j < cum.length - 2 && cum[j + 1] < d) j++;
    const f = (d - cum[j]) / (cum[j + 1] - cum[j] || 1);
    out.push([pts[j][0] + (pts[j + 1][0] - pts[j][0]) * f, pts[j][1] + (pts[j + 1][1] - pts[j][1]) * f]);
  }
  return out;
}

class Builder {
  pos: number[] = [];
  idx: number[] = [];
  quad(a: number[], b: number[], c: number[], d: number[], y: number) {
    const base = this.pos.length / 3;
    for (const p of [a, b, c, d]) this.pos.push(p[0], y, -p[1]);
    this.idx.push(base, base + 1, base + 2, base + 1, base + 3, base + 2);
  }
  build(): THREE.BufferGeometry {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(this.pos, 3));
    g.setIndex(this.idx);
    g.computeVertexNormals();
    return g;
  }
}

function strip(b: Builder, left: number[][], right: number[][], y: number) {
  const n = Math.max(left.length, right.length);
  const L = resample(left, n);
  const R = resample(right, n);
  for (let i = 0; i < n - 1; i++) b.quad(L[i], R[i], L[i + 1], R[i + 1], y);
}

function offsetLine(c: number[][], half: number): { l: number[][]; r: number[][] } {
  const l: number[][] = [];
  const r: number[][] = [];
  for (let i = 0; i < c.length; i++) {
    const a = c[Math.max(0, i - 1)];
    const bb = c[Math.min(c.length - 1, i + 1)];
    let tx = bb[0] - a[0];
    let ty = bb[1] - a[1];
    const m = Math.hypot(tx, ty) || 1;
    tx /= m;
    ty /= m;
    l.push([c[i][0] - ty * half, c[i][1] + tx * half]);
    r.push([c[i][0] + ty * half, c[i][1] - tx * half]);
  }
  return { l, r };
}

function lineStrip(b: Builder, poly: number[][], width: number, y: number) {
  const { l, r } = offsetLine(poly, width / 2);
  strip(b, l, r, y);
}

function dashed(b: Builder, poly: number[][], width: number, on: number, off: number, y: number) {
  let acc = 0;
  let drawing = true;
  let seg: number[][] = [poly[0]];
  for (let i = 1; i < poly.length; i++) {
    const p = poly[i - 1];
    const q = poly[i];
    const L = Math.hypot(q[0] - p[0], q[1] - p[1]);
    let pos = 0;
    while (pos < L) {
      const lim = drawing ? on : off;
      const step = Math.min(L - pos, lim - acc);
      pos += step;
      acc += step;
      const pt = [p[0] + ((q[0] - p[0]) * pos) / L, p[1] + ((q[1] - p[1]) * pos) / L];
      if (drawing) seg.push(pt);
      if (acc >= lim - 1e-6) {
        if (drawing && seg.length > 1) lineStrip(b, seg, width, y);
        drawing = !drawing;
        acc = 0;
        seg = drawing ? [pt] : [];
      }
    }
  }
  if (drawing && seg.length > 1) lineStrip(b, seg, width, y);
}

// ----------------------------------------------------------------------------- scene parts
function Roads({ scene }: { scene: Scene }) {
  const { surface, edges, centres, patch } = useMemo(() => {
    const s = new Builder();
    const e = new Builder();
    const c = new Builder();
    for (const lane of scene.lanes) {
      strip(s, lane.left, lane.right, lane.junction ? 0.012 : 0.01);
      if (!lane.junction) {
        lineStrip(e, lane.left, 0.14, 0.03);
        lineStrip(e, lane.right, 0.14, 0.03);
      } else {
        dashed(c, lane.center, 0.1, 1.2, 2.2, 0.025);
      }
    }
    const jp = scene.lanes.filter((l) => l.junction).flatMap((l) => l.center);
    let patchGeo: THREE.BufferGeometry | null = null;
    if (jp.length) {
      const xs = jp.map((p) => p[0]);
      const ys = jp.map((p) => p[1]);
      const x0 = Math.min(...xs) - 1.75;
      const x1 = Math.max(...xs) + 1.75;
      const y0 = Math.min(...ys) - 1.75;
      const y1 = Math.max(...ys) + 1.75;
      const pb = new Builder();
      pb.quad([x0, y1], [x1, y1], [x0, y0], [x1, y0], 0.008);
      patchGeo = pb.build();
    }
    return { surface: s.build(), edges: e.build(), centres: c.build(), patch: patchGeo };
  }, [scene]);
  return (
    <group>
      <mesh geometry={surface} receiveShadow>
        <meshStandardMaterial color="#1a2438" roughness={0.95} metalness={0} side={THREE.DoubleSide} polygonOffset polygonOffsetFactor={-1} polygonOffsetUnits={-1} />
      </mesh>
      {patch && (
        <mesh geometry={patch} receiveShadow>
          <meshStandardMaterial color="#1a2438" roughness={0.95} side={THREE.DoubleSide} />
        </mesh>
      )}
      <mesh geometry={edges}>
        <meshBasicMaterial color="#56668c" side={THREE.DoubleSide} polygonOffset polygonOffsetFactor={-2} polygonOffsetUnits={-2} />
      </mesh>
      <mesh geometry={centres}>
        <meshBasicMaterial color="#8b7cf6" transparent opacity={0.45} side={THREE.DoubleSide} />
      </mesh>
    </group>
  );
}

function StopLines({ scene, pb }: { scene: Scene; pb: Playback }) {
  const refs = useRef<(THREE.MeshBasicMaterial | null)[]>([]);
  const geos = useMemo(
    () =>
      scene.stop_lines.map((l) => {
        const b = new Builder();
        lineStrip(b, l, 0.55, 0.04);
        return b.build();
      }),
    [scene],
  );
  const sig = useMemo(() => scene.stop_lines.map((l) => nearestSignal(scene, l)), [scene]);
  useFrame(() => {
    scene.stop_lines.forEach((_, i) => {
      const m = refs.current[i];
      if (!m) return;
      const s = sig[i] >= 0 ? signalState(scene.signals[sig[i]].states, scene.t, pb.t) : "off";
      m.color.set(signalColor(s));
    });
  });
  return (
    <group>
      {geos.map((g, i) => (
        <mesh key={i} geometry={g}>
          <meshBasicMaterial ref={(m) => void (refs.current[i] = m)} color="#4b5877" side={THREE.DoubleSide} polygonOffset polygonOffsetFactor={-3} polygonOffsetUnits={-3} />
        </mesh>
      ))}
    </group>
  );
}

function Signals({ scene, pb }: { scene: Scene; pb: Playback }) {
  const lamps = useRef<THREE.MeshStandardMaterial[][]>([]);
  useFrame(() => {
    scene.signals.forEach((s, i) => {
      const st = signalState(s.states, scene.t, pb.t).toLowerCase();
      const on = st.startsWith("red") ? 0 : st.startsWith("yellow") || st.startsWith("amber") ? 1 : st.startsWith("green") ? 2 : -1;
      const m = lamps.current[i];
      if (!m) return;
      const cols = ["#fb7185", "#fbbf24", "#34d399"];
      m.forEach((mat, k) => {
        mat.color.set(k === on ? cols[k] : "#1b2234");
        mat.emissive.set(k === on ? cols[k] : "#000000");
        mat.emissiveIntensity = k === on ? 2.4 : 0;
      });
    });
  });
  return (
    <group>
      {scene.signals.map((s, i) => {
        const yaw = Math.atan2(s.y, s.x); // faces outwards along the arm, towards approaching traffic
        return (
          <group key={i} position={W2T(s.x, s.y)} rotation={[0, yaw, 0]}>
            <mesh position={[0, 2.4, 0]} castShadow>
              <cylinderGeometry args={[0.08, 0.1, 4.8, 8]} />
              <meshStandardMaterial color="#2b3858" roughness={0.6} metalness={0.4} />
            </mesh>
            <mesh position={[0.05, 5.05, 0]} castShadow>
              <boxGeometry args={[0.38, 1.25, 0.46]} />
              <meshStandardMaterial color="#0c1220" roughness={0.5} />
            </mesh>
            {[0, 1, 2].map((k) => (
              <mesh key={k} position={[0.27, 5.45 - k * 0.4, 0]}>
                <sphereGeometry args={[0.14, 16, 12]} />
                <meshStandardMaterial
                  ref={(m) => {
                    if (!m) return;
                    (lamps.current[i] ??= [])[k] = m;
                  }}
                  color="#1b2234"
                />
              </mesh>
            ))}
          </group>
        );
      })}
    </group>
  );
}

// ----------------------------------------------------------------------------- vehicles
function Vehicle({ category, lwh, color, ego }: { category: string; lwh: [number, number, number]; color: string; ego: boolean }) {
  const [L, Wd, H] = lwh;
  const body = actorColor(color);
  const glass = "#0a1020";
  const clearance = 0.28;
  const mat = (c: string, rough = 0.5, metal = 0.2) => <meshStandardMaterial color={c} roughness={rough} metalness={metal} />;
  const wheels = [
    [L * 0.31, Wd / 2 - 0.12],
    [L * 0.31, -(Wd / 2 - 0.12)],
    [-L * 0.31, Wd / 2 - 0.12],
    [-L * 0.31, -(Wd / 2 - 0.12)],
  ];
  const isTruck = category === "truck";
  const isBus = category === "bus";
  const isVan = category === "van";
  return (
    <group>
      {isTruck ? (
        <>
          <mesh position={[L * 0.34, clearance + H * 0.36, 0]} castShadow>
            <boxGeometry args={[L * 0.28, H * 0.72, Wd]} />
            {mat(body)}
          </mesh>
          <mesh position={[L * 0.4, clearance + H * 0.55, 0]}>
            <boxGeometry args={[L * 0.14, H * 0.3, Wd * 0.92]} />
            {mat(glass, 0.2, 0.5)}
          </mesh>
          <mesh position={[-L * 0.14, clearance + H * 0.46, 0]} castShadow>
            <boxGeometry args={[L * 0.68, H * 0.84, Wd]} />
            {mat("#c9d2e3", 0.8, 0.05)}
          </mesh>
        </>
      ) : isBus ? (
        <>
          <mesh position={[0, clearance + H * 0.48, 0]} castShadow>
            <boxGeometry args={[L, H * 0.94, Wd]} />
            {mat(body, 0.55)}
          </mesh>
          <mesh position={[0, clearance + H * 0.66, 0]}>
            <boxGeometry args={[L * 0.96, H * 0.28, Wd * 1.01]} />
            {mat(glass, 0.2, 0.5)}
          </mesh>
        </>
      ) : (
        <>
          <mesh position={[0, clearance + H * 0.27, 0]} castShadow>
            <boxGeometry args={[L, H * (isVan ? 0.62 : 0.5), Wd]} />
            {mat(body)}
          </mesh>
          <mesh position={[isVan ? L * 0.04 : -L * 0.06, clearance + H * (isVan ? 0.72 : 0.7), 0]} castShadow>
            <boxGeometry args={[L * (isVan ? 0.8 : 0.52), H * (isVan ? 0.38 : 0.4), Wd * 0.9]} />
            {mat(glass, 0.25, 0.5)}
          </mesh>
          <mesh position={[isVan ? L * 0.04 : -L * 0.06, clearance + H * (isVan ? 0.72 : 0.7) + H * 0.2 + 0.01, 0]}>
            <boxGeometry args={[L * (isVan ? 0.78 : 0.4), 0.04, Wd * 0.84]} />
            {mat(body)}
          </mesh>
        </>
      )}
      {wheels.map(([x, z], i) => (
        <mesh key={i} position={[x, 0.34, z]} rotation={[Math.PI / 2, 0, 0]}>
          <cylinderGeometry args={[0.34, 0.34, 0.26, 14]} />
          <meshStandardMaterial color="#05070d" roughness={0.9} />
        </mesh>
      ))}
      {[1, -1].map((s) => (
        <mesh key={s} position={[L / 2 + 0.01, clearance + H * 0.22, s * Wd * 0.33]}>
          <boxGeometry args={[0.05, 0.14, 0.3]} />
          <meshStandardMaterial color="#fff6d8" emissive="#fff0b0" emissiveIntensity={1.4} />
        </mesh>
      ))}
      {[1, -1].map((s) => (
        <mesh key={`r${s}`} position={[-L / 2 - 0.01, clearance + H * 0.22, s * Wd * 0.33]}>
          <boxGeometry args={[0.05, 0.12, 0.3]} />
          <meshStandardMaterial color="#7a1020" emissive="#ff2038" emissiveIntensity={0.9} />
        </mesh>
      ))}
      {ego && (
        <mesh position={[0, 0.05, 0]} rotation={[-Math.PI / 2, 0, 0]}>
          <ringGeometry args={[Math.max(L, Wd) * 0.72, Math.max(L, Wd) * 0.8, 48]} />
          <meshBasicMaterial color="#22d3ee" transparent opacity={0.85} side={THREE.DoubleSide} />
        </mesh>
      )}
    </group>
  );
}

function Actors({ scene, pb }: { scene: Scene; pb: Playback }) {
  const groups = useRef<Record<string, THREE.Group | null>>({});
  useFrame(() => {
    for (const a of scene.actors) {
      const g = groups.current[a.id];
      if (!g) continue;
      const p = samplePose(scene.t, scene.tracks[a.id], pb.t);
      g.position.set(p.x, 0, -p.y);
      g.rotation.y = p.yaw;
    }
  });
  return (
    <group>
      {scene.actors.map((a) => (
        <group key={a.id} ref={(g) => void (groups.current[a.id] = g)}>
          <Vehicle category={a.category} lwh={a.lwh} color={a.color} ego={a.role === "ego"} />
        </group>
      ))}
    </group>
  );
}

function EgoTrail({ scene, pb }: { scene: Scene; pb: Playback }) {
  const egoId = scene.actors.find((a) => a.role === "ego")?.id ?? scene.actors[0]?.id;
  const line = useMemo(() => {
    const tr = (egoId && scene.tracks[egoId]) || [];
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(tr.flatMap((p) => [p[0], 0.08, -p[1]]), 3));
    g.setDrawRange(0, 0);
    const l = new THREE.Line(g, new THREE.LineBasicMaterial({ color: "#22d3ee" }));
    return l;
  }, [scene, egoId]);
  useFrame(() => {
    if (!egoId) return;
    const i = samplePose(scene.t, scene.tracks[egoId], pb.t).i;
    line.geometry.setDrawRange(0, i + 1);
  });
  return <primitive object={line} />;
}

// ----------------------------------------------------------------------------- camera
const ORBIT_START = new THREE.Vector3(-46, 36, 50);

function CameraDirector({ scene, pb, mode }: { scene: Scene; pb: Playback; mode: CamMode }) {
  const { camera } = useThree();
  const controls = useRef<ComponentRef<typeof OrbitControls>>(null);
  const saved = useRef<{ p: THREE.Vector3; t: THREE.Vector3 } | null>(null);
  const prev = useRef<CamMode | null>(null);
  const egoId = scene.actors.find((a) => a.role === "ego")?.id ?? scene.actors[0]?.id;
  const tmp = useMemo(() => ({ want: new THREE.Vector3(), look: new THREE.Vector3(), cur: new THREE.Vector3() }), []);

  useEffect(() => {
    const c = controls.current;
    if (prev.current === "orbit" && mode !== "orbit" && c) saved.current = { p: camera.position.clone(), t: c.target.clone() };
    if (mode === "orbit") {
      const s = saved.current;
      camera.position.copy(s ? s.p : ORBIT_START);
      c?.target.copy(s ? s.t : new THREE.Vector3(0, 0, 0));
      c?.update();
    }
    if (mode === "top") {
      camera.position.set(0, 190, 0.01);
      camera.lookAt(0, 0, 0);
    }
    prev.current = mode;
  }, [mode, camera]);

  useFrame((_, dt) => {
    if (!egoId) return;
    if (mode === "chase") {
      const p = samplePose(scene.t, scene.tracks[egoId], pb.t);
      const fx = Math.cos(p.yaw);
      const fy = Math.sin(p.yaw);
      tmp.want.set(p.x - fx * 13, 5.2, -(p.y - fy * 13));
      tmp.look.set(p.x + fx * 12, 1.2, -(p.y + fy * 12));
      const k = 1 - Math.exp(-dt * 5);
      camera.position.lerp(tmp.want, k);
      tmp.cur.copy(tmp.look);
      camera.lookAt(tmp.cur);
    }
  });
  return <OrbitControls ref={controls} makeDefault enabled={mode === "orbit"} enableDamping dampingFactor={0.08} maxPolarAngle={Math.PI / 2.05} minDistance={6} maxDistance={420} />;
}

export default function Scene3D({ scene, mode }: { scene: Scene; mode: CamMode }) {
  const pb = usePlayback();
  return (
    <Canvas
      shadows="percentage"
      dpr={[1, 1.75]}
      camera={{ fov: 48, near: 0.5, far: 1400, position: ORBIT_START.toArray() }}
      gl={{ antialias: true, powerPreference: "high-performance" }}
      aria-label="3D view of the scene with orbit, chase and top cameras"
    >
      <color attach="background" args={["#060b16"]} />
      <fog attach="fog" args={["#060b16", 140, 520]} />
      <hemisphereLight args={["#a9baff", "#0a0f1e", 0.9]} />
      <directionalLight
        position={[70, 130, 50]}
        intensity={2.1}
        castShadow
        shadow-mapSize={[2048, 2048]}
        shadow-camera-left={-90}
        shadow-camera-right={90}
        shadow-camera-top={90}
        shadow-camera-bottom={-90}
        shadow-camera-near={10}
        shadow-camera-far={320}
        shadow-bias={-0.0004}
      />
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.02, 0]} receiveShadow>
        <planeGeometry args={[1600, 1600]} />
        <meshStandardMaterial color="#0a1120" roughness={1} />
      </mesh>
      <gridHelper args={[800, 80, "#16213a", "#101a30"]} position={[0, -0.01, 0]} />
      <Roads scene={scene} />
      <StopLines scene={scene} pb={pb} />
      <Signals scene={scene} pb={pb} />
      <EgoTrail scene={scene} pb={pb} />
      <Actors scene={scene} pb={pb} />
      <CameraDirector scene={scene} pb={pb} mode={mode} />
    </Canvas>
  );
}
