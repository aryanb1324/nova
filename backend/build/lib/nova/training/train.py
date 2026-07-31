"""PPO training with a streaming event feed.

`train()` is deliberately transport-agnostic: it pushes plain dicts at an
`on_event` callback. The CLI prints them, the API server forwards them down a
websocket, and tests assert on them.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Callable

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from .. import envs
from ..envs import record as record_rollout

EventFn = Callable[[dict], None]
StopFn = Callable[[], bool]


@dataclass
class TrainConfig:
    task: str = "reach"
    template: str = "reach_arm"
    params: dict = field(default_factory=dict)

    total_steps: int = 250_000
    #: Measured on a 10-core M5: throughput keeps climbing past core count
    #: because each worker is mostly blocked on IPC, and flattens around 16.
    n_envs: int = 16
    seed: int = 0
    #: "subproc" spreads envs across cores; "dummy" avoids IPC overhead and can
    #: win for very cheap envs. Benchmarked per task, not assumed.
    vec: str = "subproc"

    #: Large batches with few, big gradient steps: the nets are tiny, so
    #: per-step Python/torch overhead dominates the actual matmuls.
    n_steps: int = 256
    batch_size: int = 1024
    #: Tuned by sweep on the reach task: at 3e-4 the policy collapses to a
    #: single posture and never conditions on the target (0% success); 1e-3
    #: with a little entropy reaches 100%. Small nets tolerate a hot LR.
    learning_rate: float = 1e-3
    gamma: float = 0.98
    gae_lambda: float = 0.95
    #: Enough exploration pressure to stop early mode-collapse, not enough to
    #: stop the policy committing. 0.0 costs ~40 points of success rate.
    ent_coef: float = 0.01
    clip_range: float = 0.2
    n_epochs: int = 10
    net_arch: tuple[int, ...] = (64, 64)
    #: Normalize the reward stream during training only. Observations are left
    #: alone deliberately: the envs already scale their own features, so nothing
    #: stateful has to be saved alongside the policy for playback to be correct.
    norm_reward: bool = True

    #: Capture a playable rollout every N PPO iterations (0 disables).
    rollout_every: int = 10

    def to_dict(self) -> dict:
        d = asdict(self)
        d["net_arch"] = list(self.net_arch)
        return d


def _env_factory(cfg: TrainConfig):
    def _make():
        return envs.make(cfg.task, cfg.template, cfg.params)
    return _make


class StreamCallback(BaseCallback):
    """Emits progress after every PPO iteration, and rollouts periodically."""

    def __init__(self, cfg: TrainConfig, on_event: EventFn, should_stop: StopFn | None):
        super().__init__()
        self.cfg = cfg
        self.on_event = on_event
        self.should_stop = should_stop
        self.iteration = 0
        self.started = time.perf_counter()
        self.history: list[dict] = []
        self.best: dict | None = None
        self._eval_env = envs.make(cfg.task, cfg.template, cfg.params, seed=cfg.seed + 9999)

    def _on_step(self) -> bool:
        # Honouring a stop request between iterations is close enough: an
        # iteration is a fraction of a second.
        return not (self.should_stop and self.should_stop())

    def _episode_stats(self) -> dict:
        buf = list(self.model.ep_info_buffer or [])
        if not buf:
            return {}
        stats = {
            "mean_reward": round(float(np.mean([e["r"] for e in buf])), 4),
            "mean_length": round(float(np.mean([e["l"] for e in buf])), 2),
            "episodes": len(buf),
        }
        for key in envs.get_task(self.cfg.task).get("monitor_keys", ()):
            vals = [e[key] for e in buf if key in e]
            if vals:
                stats[key] = round(float(np.mean(vals)), 4)
        return stats

    def _on_rollout_end(self) -> None:
        self.iteration += 1
        elapsed = time.perf_counter() - self.started
        record = {
            "type": "progress",
            "iteration": self.iteration,
            "step": int(self.num_timesteps),
            "total_steps": self.cfg.total_steps,
            "progress": round(self.num_timesteps / max(1, self.cfg.total_steps), 4),
            "elapsed": round(elapsed, 2),
            "fps": int(self.num_timesteps / elapsed) if elapsed > 0 else 0,
            **self._episode_stats(),
        }
        self.history.append(record)
        self.on_event(record)

        if self.best is None or record.get("mean_reward", -1e9) > self.best.get("mean_reward", -1e9):
            self.best = record

        if self.cfg.rollout_every and self.iteration % self.cfg.rollout_every == 0:
            self._emit_rollout()

    def _emit_rollout(self) -> None:
        roll = record_rollout(self._eval_env, self.model, seed=self.cfg.seed + 1234)
        self.on_event({
            "type": "rollout",
            "iteration": self.iteration,
            "step": int(self.num_timesteps),
            **roll,
        })


def configure_threads() -> None:
    """Pin torch to one thread.

    With 16 env worker processes already saturating the cores, letting torch
    also spawn a thread pool halves throughput - measured 9.8k fps at one
    thread against 5.1k at two.
    """
    import torch
    torch.set_num_threads(1)


def build_model(cfg: TrainConfig, vec_env) -> PPO:
    configure_threads()
    return PPO(
        "MlpPolicy",
        vec_env,
        n_steps=cfg.n_steps,
        batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        ent_coef=cfg.ent_coef,
        clip_range=cfg.clip_range,
        n_epochs=cfg.n_epochs,
        policy_kwargs={"net_arch": list(cfg.net_arch)},
        seed=cfg.seed,
        # Small MLPs are latency-bound, not throughput-bound: CPU beats MPS here
        # because every kernel launch costs more than the matmul saves.
        device="cpu",
        verbose=0,
    )


def train(
    cfg: TrainConfig,
    on_event: EventFn | None = None,
    should_stop: StopFn | None = None,
) -> dict:
    """Run a full training job. Returns a summary dict including the final rollout."""
    emit = on_event or (lambda _e: None)

    vec_cls = SubprocVecEnv if cfg.vec == "subproc" and cfg.n_envs > 1 else DummyVecEnv
    monitor_keys = envs.get_task(cfg.task).get("monitor_keys", ())
    vec_env = make_vec_env(
        _env_factory(cfg),
        n_envs=cfg.n_envs,
        seed=cfg.seed,
        vec_env_cls=vec_cls,
        monitor_kwargs={"info_keywords": tuple(monitor_keys)},
    )
    if cfg.norm_reward:
        vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True, gamma=cfg.gamma)

    try:
        probe = envs.make(cfg.task, cfg.template, cfg.params, seed=cfg.seed)
        emit({
            "type": "start",
            "config": cfg.to_dict(),
            "scene": envs.describe(probe.model),
            "obs_dim": int(probe.observation_space.shape[0]),
            "action_dim": int(probe.action_space.shape[0]),
            "control_hz": round(1.0 / probe.dt, 2),
        })

        model = build_model(cfg, vec_env)
        cb = StreamCallback(cfg, emit, should_stop)
        started = time.perf_counter()
        model.learn(total_timesteps=cfg.total_steps, callback=cb, progress_bar=False)
        wall = time.perf_counter() - started

        final = record_rollout(cb._eval_env, model, seed=cfg.seed + 4321)
        emit({"type": "rollout", "iteration": cb.iteration, "step": int(model.num_timesteps),
              "final": True, **final})

        summary = {
            "type": "done",
            "wall_time": round(wall, 2),
            "steps_completed": int(model.num_timesteps),
            "stopped_early": model.num_timesteps < cfg.total_steps,
            "iterations": cb.iteration,
            "final_stats": final["stats"],
            "best": cb.best,
            "history": cb.history,
        }
        emit(summary)
        return {**summary, "model": model, "final_rollout": final}
    finally:
        vec_env.close()


def evaluate(model, cfg: TrainConfig, n_episodes: int = 30) -> dict:
    """Honest post-training score: deterministic policy, fresh seeds."""
    env = envs.make(cfg.task, cfg.template, cfg.params, seed=cfg.seed + 777)
    returns, successes, extras = [], [], []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=10_000 + ep)
        total, info = 0.0, {}
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            if terminated or truncated:
                break
        returns.append(total)
        if "is_success" in info:
            successes.append(bool(info["is_success"]))
        if "distance" in info:
            extras.append(float(info["distance"]))

    out = {
        "episodes": n_episodes,
        "mean_return": round(float(np.mean(returns)), 3),
        "std_return": round(float(np.std(returns)), 3),
    }
    if successes:
        out["success_rate"] = round(float(np.mean(successes)), 4)
    if extras:
        out["mean_final_distance"] = round(float(np.mean(extras)), 4)
    return out
