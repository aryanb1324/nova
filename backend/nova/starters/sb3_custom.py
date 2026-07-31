"""Your architecture, someone else's PPO.

Testing a new network is usually not the same as testing a new algorithm. This
keeps a known-good PPO and swaps in a custom feature extractor, so anything that
changes is attributable to the architecture rather than to a hyperparameter you
also touched.

Replace `MyExtractor` with whatever you are actually testing — a residual stack,
an attention block, a recurrent core (see SB3's RecurrentPPO), a hand-designed
feature transform. The rest of this file can stay as it is.
"""

import numpy as np
import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv

import nova
from nova import bench

TASK, TEMPLATE = "reach", "reach_arm"
TOTAL_STEPS = 400_000
N_ENVS = 16     # measured sweet spot on a 10-core machine

# Lets the benchmark harness run this file once per seed.
SEED = bench.seed_from_env()


class MyExtractor(BaseFeaturesExtractor):
    """<- This is the part to replace.

    A residual MLP, as an example of something you cannot express with
    `net_arch` alone.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 64):
        super().__init__(observation_space, features_dim)
        obs_dim = int(np.prod(observation_space.shape))
        self.stem = nn.Linear(obs_dim, features_dim)
        self.block = nn.Sequential(
            nn.Tanh(), nn.Linear(features_dim, features_dim),
            nn.Tanh(), nn.Linear(features_dim, features_dim),
        )

    def forward(self, obs: th.Tensor) -> th.Tensor:
        h = self.stem(obs)
        return th.tanh(h + self.block(h))


def make_env():
    return nova.make(TASK, TEMPLATE)


class StreamToNova(BaseCallback):
    """Pushes SB3's own metrics onto the reward chart in the browser."""

    def __init__(self, run):
        super().__init__()
        self.run = run

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        buf = list(self.model.ep_info_buffer or [])
        if not buf:
            return
        self.run.log(
            step=int(self.num_timesteps),
            mean_reward=float(np.mean([e["r"] for e in buf])),
            is_success=float(np.mean([e.get("is_success", 0) for e in buf])),
        )


def main() -> None:
    th.set_num_threads(1)   # the env workers already own the cores

    vec = make_vec_env(make_env, n_envs=N_ENVS, seed=SEED, vec_env_cls=SubprocVecEnv,
                       monitor_kwargs={"info_keywords": ("is_success",)})
    model = PPO(
        "MlpPolicy", vec,
        policy_kwargs={
            "features_extractor_class": MyExtractor,
            "features_extractor_kwargs": {"features_dim": 64},
            "net_arch": [64],
        },
        n_steps=256, batch_size=1024, n_epochs=10,
        learning_rate=1e-3, ent_coef=0.01, gamma=0.98,
        device="cpu", seed=SEED, verbose=0,
    )
    print(model.policy)

    with nova.attach(TASK, TEMPLATE, algo="PPO + custom extractor",
                     total_steps=TOTAL_STEPS) as run:
        model.learn(total_timesteps=TOTAL_STEPS, callback=StreamToNova(run))
    vec.close()

    nova.export(model, "custom_net.onnx", task=TASK, template=TEMPLATE,
                algo="PPO + custom extractor")
    scores = nova.evaluate("custom_net.onnx", episodes=30)
    scores.pop("rollout", None)
    print(f"\npublic   : {scores['public']}")
    print(f"held_out : {scores['held_out']}")
    bench.record("sb3_custom", scores, steps=TOTAL_STEPS)


if __name__ == "__main__":
    main()
