"""REINFORCE — vanilla policy gradient, from scratch.

The simplest thing that actually learns. Collect episodes, weight the
log-probability of each action by how good the future turned out to be, step the
optimizer. No value network, no clipping, no replay.

It is also much worse than PPO, which is the point of having it. At an identical
400k-step budget on the same task and network size, measured on an M5:

    REINFORCE          3% success
    PPO (from scratch) 87% success

Both files are about the same length. The difference is entirely in how the
update uses the data — a learned baseline, importance ratios, several epochs per
batch. That is a floor worth beating, and every part of it is visible here.

Things worth trying: swap the reward-to-go normalization for a learned value
baseline, anneal the learning rate, make log_std state-dependent, add entropy to
the loss, or reuse each batch for more than one gradient step.
"""

import numpy as np
import torch as th
import torch.nn as nn

import nova
from nova import bench

TASK, TEMPLATE = "reach", "reach_arm"
ITERATIONS = 250          # 400k env steps — the same budget ppo_minimal.py uses
EPISODES_PER_BATCH = 16
LEARNING_RATE = 3e-3
GAMMA = 0.99

# Lets the benchmark harness run this file once per seed.
SEED = bench.seed_from_env()


class Policy(nn.Module):
    """Gaussian policy: a network for the mean, one free parameter for spread."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 64):
        super().__init__()
        self.mean = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, act_dim),
        )
        self.log_std = nn.Parameter(th.full((act_dim,), -0.5))

    def forward(self, obs: th.Tensor) -> th.Tensor:
        """Deterministic action. This is what gets exported."""
        return th.tanh(self.mean(obs))

    def distribution(self, obs: th.Tensor) -> th.distributions.Normal:
        return th.distributions.Normal(self.mean(obs), self.log_std.exp())

    def predict(self, obs, deterministic: bool = True):
        with th.no_grad():
            return self(th.as_tensor(obs, dtype=th.float32)).numpy(), None


def discounted_to_go(rewards: list[float], gamma: float) -> np.ndarray:
    """How good the future turned out, from each step onward."""
    out = np.zeros(len(rewards), dtype=np.float32)
    running = 0.0
    for i in reversed(range(len(rewards))):
        running = rewards[i] + gamma * running
        out[i] = running
    return out


def main() -> None:
    th.manual_seed(SEED)
    env = nova.make(TASK, TEMPLATE, seed=SEED)
    policy = Policy(env.observation_space.shape[0], env.action_space.shape[0])
    optimizer = th.optim.Adam(policy.parameters(), lr=LEARNING_RATE)

    total = ITERATIONS * EPISODES_PER_BATCH * env.max_steps
    steps = 0

    with nova.attach(TASK, TEMPLATE, algo="REINFORCE (from scratch)",
                     total_steps=total) as run:
        for iteration in range(ITERATIONS):
            batch_obs, batch_actions, batch_weights, returns = [], [], [], []

            for _ in range(EPISODES_PER_BATCH):
                obs, _ = env.reset()
                ep_obs, ep_actions, ep_rewards = [], [], []
                while True:
                    obs_t = th.as_tensor(obs, dtype=th.float32)
                    with th.no_grad():
                        action = policy.distribution(obs_t).sample()
                    ep_obs.append(obs)
                    ep_actions.append(action.numpy())

                    obs, reward, terminated, truncated, _ = env.step(
                        np.clip(action.numpy(), -1, 1))
                    ep_rewards.append(reward)
                    steps += 1
                    if terminated or truncated:
                        break

                batch_obs += ep_obs
                batch_actions += ep_actions
                batch_weights.append(discounted_to_go(ep_rewards, GAMMA))
                returns.append(sum(ep_rewards))

            obs_t = th.as_tensor(np.array(batch_obs), dtype=th.float32)
            act_t = th.as_tensor(np.array(batch_actions), dtype=th.float32)
            weights = th.as_tensor(np.concatenate(batch_weights))
            # Centring and scaling is what keeps the gradient from exploding;
            # without it REINFORCE on this task diverges almost immediately.
            weights = (weights - weights.mean()) / (weights.std() + 1e-8)

            log_prob = policy.distribution(obs_t).log_prob(act_t).sum(axis=-1)
            loss = -(log_prob * weights).mean()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

            mean_return = float(np.mean(returns))
            print(f"it {iteration + 1:>3}/{ITERATIONS}  return {mean_return:>8.2f}  "
                  f"loss {loss.item():>8.3f}  steps {steps:,}")
            run.log(step=steps, mean_reward=mean_return)
            if (iteration + 1) % 10 == 0:
                run.rollout(policy, seed=10_000, step=steps)

    nova.export(policy, "reinforce.onnx", task=TASK, template=TEMPLATE,
                algo="REINFORCE (from scratch)")
    scores = nova.evaluate("reinforce.onnx", episodes=30)
    scores.pop("rollout", None)
    print(f"\npublic   : {scores['public']}")
    print(f"held_out : {scores['held_out']}")
    bench.record("reinforce", scores, steps=steps)


if __name__ == "__main__":
    main()
