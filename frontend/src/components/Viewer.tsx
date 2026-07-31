import { OrbitControls } from "@react-three/drei";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { useEffect, useMemo, useRef } from "react";
import * as THREE from "three";

import type { Frame, Geom, SceneDesc } from "../types";

/** Mutable playback state. Kept in a ref so advancing a frame costs no React render. */
export interface PlaybackState {
  index: number;
  playing: boolean;
  speed: number;
  loop: boolean;
}

/**
 * MuJoCo primitives, in MuJoCo's size conventions.
 * Capsules and cylinders extend along local +Z; Three builds them along +Y, so
 * the geometry is rotated once at construction rather than per instance.
 */
function makeGeometry(g: Geom): THREE.BufferGeometry {
  const s = g.size;
  switch (g.type) {
    case "sphere":
      return new THREE.SphereGeometry(s[0], 24, 16);
    case "capsule": {
      const geo = new THREE.CapsuleGeometry(s[0], 2 * s[1], 8, 20);
      geo.rotateX(Math.PI / 2);
      return geo;
    }
    case "cylinder": {
      const geo = new THREE.CylinderGeometry(s[0], s[0], 2 * s[1], 28);
      geo.rotateX(Math.PI / 2);
      return geo;
    }
    case "ellipsoid": {
      const geo = new THREE.SphereGeometry(1, 24, 16);
      geo.scale(s[0], s[1], s[2]);
      return geo;
    }
    case "box":
      return new THREE.BoxGeometry(2 * s[0], 2 * s[1], 2 * s[2]);
    case "plane": {
      // A zero half-extent means "infinite" in MJCF; pick something big enough
      // to read as ground without blowing out the depth buffer.
      const hx = s[0] > 0 ? s[0] : 40;
      const hy = s[1] > 0 ? s[1] : 40;
      return new THREE.PlaneGeometry(2 * hx, 2 * hy);
    }
    default:
      return new THREE.SphereGeometry(0.02, 8, 6);
  }
}

function GeomMesh({ g }: { g: Geom }) {
  const geometry = useMemo(() => makeGeometry(g), [g]);
  useEffect(() => () => geometry.dispose(), [geometry]);

  const [qw, qx, qy, qz] = g.quat;
  const [r, gr, b, a] = g.rgba;
  const isFloor = g.type === "plane";

  return (
    <mesh
      geometry={geometry}
      position={g.pos}
      quaternion={[qx, qy, qz, qw]}
      castShadow={!isFloor}
      receiveShadow
    >
      <meshStandardMaterial
        color={new THREE.Color(r, gr, b)}
        transparent={a < 1}
        opacity={a}
        roughness={isFloor ? 0.95 : 0.5}
        metalness={isFloor ? 0.0 : 0.15}
      />
    </mesh>
  );
}

interface AnimatedProps {
  scene: SceneDesc;
  frames: Frame[];
  fps: number;
  playback: React.RefObject<PlaybackState>;
  onIndex?: (i: number) => void;
  follow: boolean;
  controlsRef: React.RefObject<{ target: THREE.Vector3; update: () => void } | null>;
}

function Animated({ scene, frames, fps, playback, onIndex, follow, controlsRef }: AnimatedProps) {
  const groups = useRef<(THREE.Group | null)[]>([]);
  const clock = useRef(0);
  const lastNotified = useRef({ index: -1, at: 0 });
  const lastRoot = useRef<THREE.Vector3 | null>(null);
  const { camera } = useThree();

  // A new rollout restarts the clock and drops the camera-follow anchor.
  useEffect(() => {
    clock.current = 0;
    lastRoot.current = null;
  }, [frames]);

  useFrame((_, delta) => {
    const n = frames.length;
    if (n === 0) return;
    const pb = playback.current;

    if (pb.playing) {
      clock.current += delta * pb.speed;
      let idx = Math.floor(clock.current * fps);
      if (idx >= n) {
        if (pb.loop) {
          clock.current = 0;
          idx = 0;
        } else {
          idx = n - 1;
          pb.playing = false;
        }
      }
      pb.index = idx;
    } else {
      // Keep the clock consistent with a scrub, so resuming doesn't jump.
      clock.current = pb.index / fps;
    }

    const f = frames[Math.min(Math.max(pb.index, 0), n - 1)];
    if (!f) return;

    for (let i = 0; i < groups.current.length; i++) {
      const grp = groups.current[i];
      const row = f[i];
      if (!grp || !row) continue;
      grp.position.set(row[0], row[1], row[2]);
      // MuJoCo serializes w,x,y,z; Three wants x,y,z,w.
      grp.quaternion.set(row[4], row[5], row[6], row[3]);
    }

    if (follow && f[0]) {
      const root = new THREE.Vector3(f[0][0], f[0][1], f[0][2]);
      if (lastRoot.current) {
        const shift = root.clone().sub(lastRoot.current);
        camera.position.add(shift);
        controlsRef.current?.target.add(shift);
        controlsRef.current?.update();
      }
      lastRoot.current = root;
    }

    // The scrubber only needs ~12 Hz; re-rendering React at 60 fps would
    // undo the point of driving the scene through refs.
    const now = performance.now();
    if (onIndex && pb.index !== lastNotified.current.index && now - lastNotified.current.at > 80) {
      lastNotified.current = { index: pb.index, at: now };
      onIndex(pb.index);
    }
  });

  return (
    <>
      {scene.bodies.map((body, i) => (
        <group
          key={`${body.name}-${i}`}
          ref={(el) => {
            groups.current[i] = el;
          }}
        >
          {body.geoms.map((g, j) => (
            <GeomMesh key={`${g.name}-${j}`} g={g} />
          ))}
        </group>
      ))}
    </>
  );
}

function ZUp() {
  const { camera } = useThree();
  useEffect(() => {
    // MuJoCo is Z-up. Setting this before OrbitControls reads it keeps orbiting
    // intuitive instead of rolling the horizon.
    camera.up.set(0, 0, 1);
  }, [camera]);
  return null;
}

export interface ViewerProps {
  scene: SceneDesc;
  frames: Frame[];
  fps: number;
  playback: React.RefObject<PlaybackState>;
  onIndex?: (i: number) => void;
  camera?: { distance: number; elevation: number; azimuth?: number };
  /** Track the root body — without it, the rover drives out of frame. */
  follow?: boolean;
}

export function Viewer({ scene, frames, fps, playback, onIndex, camera, follow = false }: ViewerProps) {
  const controlsRef = useRef<never>(null);

  const initial = useMemo<[number, number, number]>(() => {
    const d = camera?.distance ?? 1.8;
    const el = (-(camera?.elevation ?? -20) * Math.PI) / 180;
    const az = ((camera?.azimuth ?? 130) * Math.PI) / 180;
    return [d * Math.cos(el) * Math.cos(az), d * Math.cos(el) * Math.sin(az), d * Math.sin(el)];
  }, [camera]);

  return (
    <Canvas
      shadows
      dpr={[1, 2]}
      camera={{ fov: 45, near: 0.01, far: 300, position: initial }}
      gl={{ antialias: true }}
    >
      <ZUp />
      <color attach="background" args={["#0d1117"]} />
      <fog attach="fog" args={["#0d1117", 8, 45]} />

      <ambientLight intensity={0.55} />
      <hemisphereLight args={["#93b5ff", "#20242c", 0.7]} />
      <directionalLight
        position={[2.5, -3, 5]}
        intensity={2.2}
        castShadow
        shadow-mapSize={[1024, 1024]}
        shadow-camera-near={0.1}
        shadow-camera-far={30}
        shadow-camera-left={-4}
        shadow-camera-right={4}
        shadow-camera-top={4}
        shadow-camera-bottom={-4}
      />

      {scene.static_geoms.map((g, i) => (
        <GeomMesh key={`static-${g.name}-${i}`} g={g} />
      ))}
      {/* GridHelper lies in XZ; rotate it into MuJoCo's XY ground plane. */}
      <gridHelper
        args={[60, 60, "#2b3444", "#1a212c"]}
        rotation={[Math.PI / 2, 0, 0]}
        position={[0, 0, 0.002]}
      />

      <Animated
        scene={scene}
        frames={frames}
        fps={fps}
        playback={playback}
        onIndex={onIndex}
        follow={follow}
        controlsRef={controlsRef}
      />

      <OrbitControls
        ref={controlsRef}
        target={[0, 0, 0.22]}
        enableDamping
        dampingFactor={0.12}
        minDistance={0.3}
        maxDistance={40}
        maxPolarAngle={Math.PI / 2 - 0.02}
      />
    </Canvas>
  );
}
