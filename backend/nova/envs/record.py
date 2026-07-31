"""Record a policy acting in an env into something the browser can replay.

A rollout is a static scene description plus a body-pose matrix per frame. No
MuJoCo, no kinematics and no physics on the client - just transforms.
"""

from __future__ import annotations

import numpy as np

from .base import RobotEnv
from . import scene


def record(
    env: RobotEnv,
    policy=None,
    *,
    seed: int | None = None,
    deterministic: bool = True,
    max_frames: int | None = None,
) -> dict:
    """Run one episode and capture it.

    `policy` is anything with SB3's `.predict(obs, deterministic=)` signature;
    None means random actions, which is what "before training" looks like.
    """
    obs, _ = env.reset(seed=seed)
    frames = [scene.frame(env.data)]
    rewards: list[float] = []
    infos: list[dict] = []

    limit = max_frames or env.max_steps
    for _ in range(limit):
        if policy is None:
            action = env.action_space.sample()
        else:
            action, _ = policy.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        frames.append(scene.frame(env.data))
        rewards.append(float(reward))
        infos.append(info)
        if terminated or truncated:
            break

    return {
        "scene": scene.describe(env.model),
        "fps": round(1.0 / env.dt, 3),
        # (n_frames, n_bodies, 7) -> nested lists, rounded to keep JSON small.
        "frames": np.round(np.asarray(frames), 4).tolist(),
        "n_frames": len(frames),
        "stats": _summarize(rewards, infos),
    }


def _summarize(rewards: list[float], infos: list[dict]) -> dict:
    stats: dict = {
        "return": round(float(np.sum(rewards)), 4),
        "length": len(rewards),
    }
    if not infos:
        return stats

    # Whatever numeric keys the task chose to report, summarized the way that
    # key actually means something: successes are "did it ever", the rest is
    # a final value plus an average.
    if "is_success" in infos[-1]:
        stats["success"] = bool(any(i.get("is_success") for i in infos))
    for key in ("distance", "forward_velocity", "upright", "height"):
        vals = [i[key] for i in infos if key in i]
        if vals:
            stats[f"final_{key}"] = round(float(vals[-1]), 4)
            stats[f"mean_{key}"] = round(float(np.mean(vals)), 4)
    return stats
