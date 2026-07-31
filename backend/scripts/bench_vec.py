#!/usr/bin/env python
"""Find the fastest PPO throughput configuration on this machine.

MuJoCo envs this small are cheap enough that IPC and BLAS thread contention can
cost more than parallelism gains, so the right settings are measured, not
assumed.
"""

from __future__ import annotations

import os
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

STEPS = 20_480


def run_case(task, template, vec, n_envs, threads, n_steps=128,
             n_epochs=10, batch_size=256):
    import torch
    torch.set_num_threads(threads)
    from nova.training import TrainConfig
    from nova.training.train import build_model, _env_factory
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from nova import envs as envs_mod

    cfg = TrainConfig(task=task, template=template, n_envs=n_envs, vec=vec,
                      n_steps=n_steps, n_epochs=n_epochs, batch_size=batch_size,
                      total_steps=STEPS, rollout_every=0)
    vec_cls = SubprocVecEnv if vec == "subproc" and n_envs > 1 else DummyVecEnv
    keys = tuple(envs_mod.get_task(task).get("monitor_keys", ()))
    venv = make_vec_env(_env_factory(cfg), n_envs=n_envs, seed=0,
                        vec_env_cls=vec_cls, monitor_kwargs={"info_keywords": keys})
    try:
        model = build_model(cfg, venv)
        t0 = time.perf_counter()
        model.learn(total_timesteps=STEPS, progress_bar=False)
        el = time.perf_counter() - t0
        return int(model.num_timesteps / el), round(el, 2)
    finally:
        venv.close()


def main():
    task, template = sys.argv[1], sys.argv[2]
    vec, n_envs, threads, n_steps = sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
    n_epochs = int(sys.argv[7]) if len(sys.argv) > 7 else 10
    batch_size = int(sys.argv[8]) if len(sys.argv) > 8 else 256
    fps, el = run_case(task, template, vec, n_envs, threads, n_steps, n_epochs, batch_size)
    print(f"{task:11s} {vec:7s} envs={n_envs:<3d} thr={threads} "
          f"n_steps={n_steps:<4d} epochs={n_epochs:<3d} batch={batch_size:<5d} "
          f"-> {fps:>7,} fps  ({el}s)")


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
