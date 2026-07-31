/** Shapes mirrored from the backend's JSON. */

export interface ParamSpec {
  key: string;
  label: string;
  default: number;
  min: number;
  max: number;
  step: number;
  unit: string;
  help: string;
  group: string;
}

export interface TemplateInfo {
  key: string;
  name: string;
  description: string;
  params: ParamSpec[];
  tasks: string[];
  camera: { distance: number; elevation: number; azimuth?: number };
  /** "builtin" | "extension" | "upload" — only uploads can be deleted. */
  source: string;
}

export interface RobotReport {
  ok: boolean;
  errors: string[];
  warnings: string[];
  info: {
    bodies?: number;
    geoms?: number;
    actuators?: number;
    dof?: number;
    total_mass?: number;
    compatible_tasks?: string[];
    task_problems?: Record<string, string[]>;
  };
  preview?: {
    scene: SceneDesc;
    frame: Frame;
    camera: { distance: number; elevation: number; azimuth?: number };
  };
}

export interface UploadedRobot {
  key: string;
  name: string;
  description: string;
  tasks: string[];
  uploaded: string;
  info: RobotReport["info"];
  warnings: string[];
}

export interface TaskInfo {
  key: string;
  name: string;
  description: string;
  success_metric: string;
  typical_steps: number;
}

export interface Catalog {
  templates: TemplateInfo[];
  tasks: TaskInfo[];
  limits: { max_steps: number };
  policies: {
    format: string;
    max_upload_bytes: number;
    public_seeds: number[];
  };
}

export interface PolicyManifestInfo {
  task: string;
  template: string;
  obs_dim: number;
  act_dim: number;
  obs_layout: string;
  params: Record<string, number>;
  algo: string;
  author: string;
  notes: string;
  created: string;
  version: number;
}

export interface SeedScore {
  episodes: number;
  mean_return: number;
  std_return: number;
  mean_length: number;
  success_rate?: number;
  mean_final_distance?: number;
}

export interface PolicyResult {
  policy_id: string;
  uploaded: string;
  size_bytes: number;
  manifest: PolicyManifestInfo;
  public: SeedScore;
  held_out: SeedScore;
  gap: { mean_return: number; success_rate?: number };
  rollout?: Rollout;
}

export interface Starter {
  key: string;
  name: string;
  description: string;
  level: string;
  code: string;
}

export interface SavedScript {
  name: string;
  chars: number;
}

export interface CodeStatus {
  enabled: boolean;
  reason: string | null;
  running: boolean;
  run: { run_id: string; alive: boolean; elapsed: number; exit_code: number | null } | null;
}

export interface ConsoleLine {
  id: number;
  text: string;
  kind: "out" | "exit" | "sys";
}

/** An attached run is someone else's training loop, streaming in from outside. */
export interface AttachedRun {
  run_id: string;
  task: string;
  template: string;
  algo: string;
  author: string;
  total_steps: number;
  started: string;
}

export type GeomType =
  | "plane" | "sphere" | "capsule" | "ellipsoid" | "cylinder" | "box";

export interface Geom {
  name: string;
  type: GeomType;
  size: number[];
  pos: [number, number, number];
  /** MuJoCo order: w, x, y, z. */
  quat: [number, number, number, number];
  rgba: [number, number, number, number];
}

export interface SceneDesc {
  bodies: { name: string; geoms: Geom[] }[];
  static_geoms: Geom[];
  up_axis: string;
}

/** One frame is [x, y, z, qw, qx, qy, qz] per body, in `scene.bodies` order. */
export type Frame = number[][];

export interface RolloutStats {
  return: number;
  length: number;
  success?: boolean;
  final_distance?: number;
  mean_distance?: number;
  final_forward_velocity?: number;
  mean_upright?: number;
  [key: string]: number | boolean | undefined;
}

export interface Rollout {
  scene: SceneDesc;
  fps: number;
  frames: Frame[];
  n_frames: number;
  stats: RolloutStats;
}

export interface ProgressEvent {
  type: "progress";
  iteration: number;
  step: number;
  total_steps: number;
  progress: number;
  elapsed: number;
  fps: number;
  mean_reward?: number;
  mean_length?: number;
  episodes?: number;
  is_success?: number;
  distance?: number;
  forward_velocity?: number;
}

export interface EvalResult {
  episodes: number;
  mean_return: number;
  std_return: number;
  success_rate?: number;
  mean_final_distance?: number;
}

export type TrainEvent =
  | { type: "attach_state"; active: AttachedRun[] }
  | { type: "console"; run_id: string; line: string }
  | { type: "code_exit"; run_id: string; exit_code: number; elapsed: number }
  | { type: "accepted"; run_id: string; config: Record<string, unknown> }
  | {
      type: "start";
      config: Record<string, unknown>;
      scene: SceneDesc;
      obs_dim: number;
      action_dim: number;
      control_hz: number;
    }
  | ProgressEvent
  | ({ type: "rollout"; iteration: number; step: number; final?: boolean } & Rollout)
  | {
      type: "done";
      wall_time: number;
      steps_completed: number;
      stopped_early: boolean;
      iterations: number;
      final_stats: RolloutStats;
    }
  | ({ type: "eval"; run_id: string } & EvalResult)
  | { type: "saved"; run_id: string }
  | { type: "error"; message: string }
  | { type: "closed"; run_id: string };

export interface TrainRequest {
  task: string;
  template: string;
  params: Record<string, number>;
  total_steps?: number;
  seed?: number;
  rollout_every?: number;
  save?: boolean;
  eval_episodes?: number;
}

export interface PreviewResponse {
  template: string;
  params: Record<string, number>;
  scene: SceneDesc;
  frame: Frame;
  camera: { distance: number; elevation: number; azimuth?: number };
  dof: number;
}

export interface RunSummary {
  run_id: string;
  task: string;
  template: string;
  params: Record<string, number>;
  created: string;
  wall_time: number;
  eval: EvalResult | null;
  steps_completed: number;
  has_policy: boolean;
}

export interface RunDetail extends RunSummary {
  config: Record<string, unknown>;
  history: ProgressEvent[];
  final_rollout?: Rollout;
  stopped_early?: boolean;
}
