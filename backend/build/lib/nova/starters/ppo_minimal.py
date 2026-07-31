"""PPO from scratch — no RL library, ~150 lines.

This is the usual baseline people modify, so it is written to be read: rollout
buffer, GAE, clipped surrogate, value loss, entropy bonus, all in one file with
nothing hidden behind a framework.

One detail that is wrong in a lot of small PPO implementations and correct here:
a truncated episode is *not* a terminal state. These tasks end by hitting the
step limit, not by failing, so the value of the final state has to be bootstrapped
back in. Treat truncation as termination and you teach the agent that time
running out is a catastrophe.

Things worth trying: change the network, anneal the clip range, add observation
normalization, or replace GAE with plain n-step returns.
"""

import numpy as np
import torch as th
import torch.nn as nn

import nova

TASK, TEMPLATE = "reach", "reach_arm"

TOTAL_STEPS = 400_000
ROLLOUT = 2048          # steps collected before each update
EPOCHS = 10             # passes over each batch
MINIBATCH = 256
LEARNING_RATE = 1e-3
GAMMA = 0.98
GAE_LAMBDA = 0.95
CLIP = 0.2
ENT_COEF = 0.01
VF_COEF = 0.5


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 64):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, act_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        self.log_std = nn.Parameter(th.zeros(act_dim))

    def forward(self, obs: th.Tensor) -> th.Tensor:
        """Deterministic action — this is the graph that gets exported."""
        return self.actor(obs)

    def dist(self, obs: th.Tensor) -> th.distributions.Normal:
        return th.distributions.Normal(self.actor(obs), self.log_std.exp())

    def value(self, obs: th.Tensor) -> th.Tensor:
        return self.critic(obs).squeeze(-1)

    def predict(self, obs, deterministic: bool = True):
        with th.no_grad():
            return np.clip(self(th.as_tensor(obs, dtype=th.float32)).numpy(), -1, 1), None


def main() -> None:
    th.manual_seed(0)
    env = nova.make(TASK, TEMPLATE, seed=0)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    net = ActorCritic(obs_dim, act_dim)
    optimizer = th.optim.Adam(net.parameters(), lr=LEARNING_RATE, eps=1e-5)

    obs, _ = env.reset(seed=0)
    episode_return, recent = 0.0, []
    steps = 0

    with nova.attach(TASK, TEMPLATE, algo="PPO (from scratch)",
                     total_steps=TOTAL_STEPS) as run:
        while steps < TOTAL_STEPS:
            # ---- collect ------------------------------------------------
            buf_obs = np.zeros((ROLLOUT, obs_dim), dtype=np.float32)
            buf_act = np.zeros((ROLLOUT, act_dim), dtype=np.float32)
            buf_logp = np.zeros(ROLLOUT, dtype=np.float32)
            buf_rew = np.zeros(ROLLOUT, dtype=np.float32)
            buf_val = np.zeros(ROLLOUT, dtype=np.float32)
            buf_done = np.zeros(ROLLOUT, dtype=np.float32)

            for t in range(ROLLOUT):
                obs_t = th.as_tensor(obs, dtype=th.float32)
                with th.no_grad():
                    dist = net.dist(obs_t)
                    action = dist.sample()
                    buf_logp[t] = dist.log_prob(action).sum().item()
                    buf_val[t] = net.value(obs_t).item()

                buf_obs[t] = obs
                buf_act[t] = action.numpy()

                obs, reward, terminated, truncated, _ = env.step(
                    np.clip(action.numpy(), -1, 1))
                buf_rew[t] = reward
                episode_return += reward
                steps += 1

                if terminated or truncated:
                    # Only a real failure ends the value chain. Running out of
                    # time does not, so bootstrap through it.
                    if truncated and not terminated:
                        with th.no_grad():
                            buf_rew[t] += GAMMA * net.value(
                                th.as_tensor(obs, dtype=th.float32)).item()
                    buf_done[t] = 1.0
                    recent.append(episode_return)
                    episode_return = 0.0
                    obs, _ = env.reset()

            # ---- advantages (GAE) ---------------------------------------
            with th.no_grad():
                last_value = net.value(th.as_tensor(obs, dtype=th.float32)).item()

            advantages = np.zeros(ROLLOUT, dtype=np.float32)
            running = 0.0
            for t in reversed(range(ROLLOUT)):
                next_value = last_value if t == ROLLOUT - 1 else buf_val[t + 1]
                mask = 1.0 - buf_done[t]
                delta = buf_rew[t] + GAMMA * next_value * mask - buf_val[t]
                running = delta + GAMMA * GAE_LAMBDA * mask * running
                advantages[t] = running
            returns = advantages + buf_val

            # ---- update -------------------------------------------------
            obs_t = th.as_tensor(buf_obs)
            act_t = th.as_tensor(buf_act)
            old_logp = th.as_tensor(buf_logp)
            adv_t = th.as_tensor(advantages)
            ret_t = th.as_tensor(returns)

            for _ in range(EPOCHS):
                for start in range(0, ROLLOUT, MINIBATCH):
                    idx = slice(start, start + MINIBATCH)
                    dist = net.dist(obs_t[idx])
                    logp = dist.log_prob(act_t[idx]).sum(axis=-1)
                    ratio = (logp - old_logp[idx]).exp()

                    mb_adv = adv_t[idx]
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                    unclipped = ratio * mb_adv
                    clipped = th.clamp(ratio, 1 - CLIP, 1 + CLIP) * mb_adv
                    policy_loss = -th.min(unclipped, clipped).mean()
                    value_loss = ((net.value(obs_t[idx]) - ret_t[idx]) ** 2).mean()
                    entropy = dist.entropy().sum(axis=-1).mean()

                    loss = policy_loss + VF_COEF * value_loss - ENT_COEF * entropy
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                    optimizer.step()

            mean_return = float(np.mean(recent[-20:])) if recent else 0.0
            print(f"{steps:>7,}/{TOTAL_STEPS:,}  return {mean_return:>8.2f}  "
                  f"value_loss {value_loss.item():>8.3f}")
            run.log(step=steps, mean_reward=mean_return)
            if (steps // ROLLOUT) % 20 == 0:
                run.rollout(net, seed=10_000, step=steps)

    nova.export(net, "ppo_scratch.onnx", task=TASK, template=TEMPLATE,
                algo="PPO (from scratch)")
    scores = nova.evaluate("ppo_scratch.onnx", episodes=30)
    scores.pop("rollout", None)
    print(f"\npublic   : {scores['public']}")
    print(f"held_out : {scores['held_out']}")


if __name__ == "__main__":
    main()
