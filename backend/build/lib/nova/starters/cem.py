"""Cross-entropy method — no gradients at all.

Sample a population of parameter vectors, keep the best few, refit a Gaussian to
them, repeat. There is no backpropagation anywhere in this file, which is the
useful thing about having it here: the environment does not care what kind of
algorithm you bring.

Two details that are easy to get wrong and matter more than the hyperparameters:

  1. Every candidate in a generation must be scored on the SAME episodes.
     Give each its own random targets and you are ranking luck, not policies.
  2. Those episodes must stay fixed across generations too, or the elites chosen
     at step t are no longer elite at t+1 and the distribution never converges.

Together those make this a fixed training set — so it will fit those specific
episodes. That is why the run ends by scoring on episodes it has never seen.
"""

import numpy as np
import torch as th
import torch.nn as nn

import nova

TASK, TEMPLATE = "reach", "reach_arm"
ITERATIONS = 30
POPULATION = 32
ELITE = 6
TRAIN_EPISODES = 8      # the fixed set every candidate is scored on
SIGMA = 0.5
HIDDEN = 0              # 0 = linear policy; CEM struggles as dimension grows


class Policy(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 0):
        super().__init__()
        if hidden:
            self.net = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(),
                                     nn.Linear(hidden, act_dim), nn.Tanh())
        else:
            self.net = nn.Sequential(nn.Linear(obs_dim, act_dim), nn.Tanh())

    def forward(self, obs: th.Tensor) -> th.Tensor:
        return self.net(obs)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def set_flat(self, flat: np.ndarray) -> None:
        i = 0
        with th.no_grad():
            for p in self.parameters():
                n = p.numel()
                p.copy_(th.from_numpy(flat[i:i + n]).float().view_as(p))
                i += n

    def act(self, obs) -> np.ndarray:
        with th.no_grad():
            return self.net(th.as_tensor(obs, dtype=th.float32)).numpy()

    def predict(self, obs, deterministic: bool = True):
        return self.act(obs), None


def episode_return(env, policy: Policy, seed: int) -> float:
    obs, _ = env.reset(seed=seed)
    total = 0.0
    while True:
        obs, reward, terminated, truncated, _ = env.step(policy.act(obs))
        total += reward
        if terminated or truncated:
            return total


def main() -> None:
    env = nova.make(TASK, TEMPLATE)
    policy = Policy(env.observation_space.shape[0], env.action_space.shape[0], HIDDEN)
    n = policy.n_params
    print(f"searching {n} parameters directly")

    rng = np.random.default_rng(0)
    mean, sigma = np.zeros(n), np.full(n, SIGMA)
    train_seeds = [7_000 + k for k in range(TRAIN_EPISODES)]

    steps_each = env.max_steps * TRAIN_EPISODES
    total = ITERATIONS * POPULATION * steps_each
    steps = 0
    best_flat, best = mean.copy(), -np.inf

    with nova.attach(TASK, TEMPLATE, algo="cross-entropy method",
                     total_steps=total) as run:
        for it in range(ITERATIONS):
            population = rng.normal(mean, sigma, size=(POPULATION, n))
            scores = np.empty(POPULATION)

            for i, candidate in enumerate(population):
                policy.set_flat(candidate)
                scores[i] = np.mean([episode_return(env, policy, s) for s in train_seeds])
                steps += steps_each

            elite_idx = np.argsort(scores)[-ELITE:]
            elite = population[elite_idx]
            mean = elite.mean(axis=0)
            # Decaying floor: elite spread alone collapses to a point too fast.
            floor = 0.02 + 0.12 * (1.0 - it / max(1, ITERATIONS - 1))
            sigma = np.maximum(elite.std(axis=0), floor)

            if scores[elite_idx[-1]] > best:
                best = float(scores[elite_idx[-1]])
                best_flat = population[elite_idx[-1]].copy()

            print(f"it {it + 1:>3}/{ITERATIONS}  mean {scores.mean():>8.2f}  "
                  f"elite {scores[elite_idx].mean():>8.2f}  best {best:>8.2f}")
            run.log(step=steps, mean_reward=float(scores.mean()))
            if (it + 1) % 5 == 0:
                policy.set_flat(mean)
                run.rollout(policy, seed=10_000, step=steps)

    policy.set_flat(best_flat)
    nova.export(policy, "cem.onnx", task=TASK, template=TEMPLATE,
                algo="cross-entropy method")
    train = nova.evaluate(policy, task=TASK, template=TEMPLATE, seeds=train_seeds)
    scores = nova.evaluate("cem.onnx", episodes=30)
    scores.pop("rollout", None)
    print(f"\ntrain    : {train}")
    print(f"held_out : {scores['held_out']}")
    print(f"\nThe drop between those two is the whole lesson.")


if __name__ == "__main__":
    main()
