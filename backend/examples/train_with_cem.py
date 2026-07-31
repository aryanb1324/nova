#!/usr/bin/env python
"""Bring-your-own-algorithm example: the cross-entropy method, not PPO.

CEM is gradient-free - it samples a population of parameter vectors, keeps the
best few, and refits a Gaussian to them. Nothing here touches stable-baselines3,
and the only role torch plays is holding the weights so they can be exported.

Run the UI first (`./dev.sh`), then:

    python examples/train_with_cem.py

and watch the reward curve appear in the browser next to the built-in runs.

A word on the result, because this example does not "win": measured on an M5,
CEM drives its training return to about -9 but scores about -27 on targets it
never saw, for 0% success, where PPO on the same task reaches 100%. That is not
a bug in the example - CEM optimizes a fixed set of episodes, so it memorizes
those targets instead of learning to reach an arbitrary one. Showing that
clearly, on seeds the search never touched, is the entire reason uploads are
scored against held-out episodes.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import torch as th

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import nova  # noqa: E402

class TinyPolicy(th.nn.Module):
    """A tanh policy, optionally with one hidden layer.

    Default is linear (`hidden=0`): 57 parameters for the reach task against 707
    for a 32-unit MLP. CEM is a global search over the parameter vector, so its
    sample cost grows sharply with dimension - and because the observation
    already contains the target-minus-fingertip error, a linear map on it is
    close to a proportional controller, which is most of the job.
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 0):
        super().__init__()
        if hidden > 0:
            self.net = th.nn.Sequential(
                th.nn.Linear(obs_dim, hidden), th.nn.Tanh(),
                th.nn.Linear(hidden, act_dim), th.nn.Tanh(),
            )
        else:
            self.net = th.nn.Sequential(th.nn.Linear(obs_dim, act_dim), th.nn.Tanh())

    def forward(self, obs: th.Tensor) -> th.Tensor:
        return self.net(obs)

    # ---- flat-vector view, which is what CEM actually optimizes ----

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

    def act(self, obs: np.ndarray) -> np.ndarray:
        with th.no_grad():
            return self.net(th.from_numpy(np.asarray(obs, np.float32))).numpy()

    def predict(self, obs, deterministic: bool = True):  # noqa: ARG002
        return self.act(obs), None


def episode_return(env, policy: TinyPolicy, seed: int) -> float:
    obs, _ = env.reset(seed=seed)
    total = 0.0
    while True:
        obs, reward, terminated, truncated, _ = env.step(policy.act(obs))
        total += reward
        if terminated or truncated:
            return total


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="reach")
    ap.add_argument("--template", default="reach_arm")
    # Defaults chosen for a ~100-second run on 10 cores.
    ap.add_argument("--iterations", type=int, default=30)
    ap.add_argument("--population", type=int, default=32)
    ap.add_argument("--elite", type=int, default=6)
    ap.add_argument("--episodes-per-candidate", type=int, default=8,
                    help="size of the fixed training set of episodes")
    ap.add_argument("--sigma", type=float, default=0.5)
    ap.add_argument("--hidden", type=int, default=0,
                    help="hidden units; 0 (default) means a linear policy")
    ap.add_argument("--out", default="cem_policy.onnx")
    ap.add_argument("--no-stream", action="store_true", help="skip the live UI")
    args = ap.parse_args()

    env = nova.make(args.task, args.template)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    policy = TinyPolicy(obs_dim, act_dim, args.hidden)
    n = policy.n_params

    steps_per_candidate = env.max_steps * args.episodes_per_candidate
    total_steps = args.iterations * args.population * steps_per_candidate
    print(f"CEM over {n} parameters | {args.task}/{args.template} "
          f"| budget {total_steps:,} env steps")

    rng = np.random.default_rng(0)
    mean = np.zeros(n)
    sigma = np.full(n, args.sigma)
    # Distinct from the seeds any evaluation uses, so the final score is honest.
    train_seeds = [7_000 + k for k in range(args.episodes_per_candidate)]

    stream = (
        nova.attach(args.task, args.template, algo="cross-entropy method",
                       total_steps=total_steps)
        if not args.no_stream else _NullRun()
    )

    steps = 0
    best_flat, best_score = mean.copy(), -np.inf
    with stream as run:
        for it in range(args.iterations):
            population = rng.normal(mean, sigma, size=(args.population, n))
            scores = np.empty(args.population)

            # Common random numbers, and the same ones every generation.
            #
            # Two separate things are going on here. Within a generation, every
            # candidate must face identical targets or the ranking measures luck
            # instead of skill. Across generations, the set must ALSO stay fixed:
            # resample it and the elites chosen at step t are no longer elite at
            # t+1, so the distribution never converges. That makes this a fixed
            # training set, which is a real risk of memorizing those targets -
            # and precisely what scoring against unseen seeds afterwards detects.
            seeds = train_seeds

            for i, candidate in enumerate(population):
                policy.set_flat(candidate)
                scores[i] = np.mean([episode_return(env, policy, s) for s in seeds])
                steps += steps_per_candidate

            elite_idx = np.argsort(scores)[-args.elite:]
            elite = population[elite_idx]
            mean = elite.mean(axis=0)
            # Elite spread alone shrinks fast enough that the search collapses to
            # a point and stops improving. Hold a noise floor that decays over the
            # run, so early generations keep exploring and late ones refine.
            floor = 0.02 + 0.12 * (1.0 - it / max(1, args.iterations - 1))
            sigma = np.maximum(elite.std(axis=0), floor)

            if scores[elite_idx[-1]] > best_score:
                best_score = float(scores[elite_idx[-1]])
                best_flat = population[elite_idx[-1]].copy()

            print(f"  it{it + 1:>3}/{args.iterations}  mean={scores.mean():>8.2f}  "
                  f"elite={scores[elite_idx].mean():>8.2f}  best={best_score:>8.2f}  "
                  f"steps={steps:,}")

            run.log(step=steps, mean_reward=float(scores.mean()))
            if (it + 1) % 5 == 0:
                policy.set_flat(mean)
                run.rollout(policy, seed=10_000, step=steps)

    policy.set_flat(best_flat)
    out = pathlib.Path(args.out)
    nova.export(policy, out, task=args.task, template=args.template,
                   algo="cross-entropy method", notes=f"{args.iterations} CEM iterations")
    print(f"\nexported -> {out}")

    train = nova.evaluate(policy, task=args.task, template=args.template,
                             seeds=train_seeds)
    scores = nova.evaluate(out, episodes=30)
    scores.pop("rollout", None)
    print(f"train    : {train}")
    print(f"public   : {scores['public']}")
    print(f"held_out : {scores['held_out']}")

    drop = train["mean_return"] - scores["held_out"]["mean_return"]
    print(f"\ntrain -> unseen drop: {drop:.1f} return")
    if drop > 5:
        print("  That drop is the point of scoring on seeds the search never saw.\n"
              f"  CEM fitted {len(train_seeds)} fixed targets well and generalized poorly;\n"
              "  compare `python scripts/train_cli.py --task reach`, where PPO sees a\n"
              "  fresh target every episode and reaches ~100% on unseen ones.")
    return 0


class _NullRun:
    """Stand-in when streaming is off, so the loop reads the same either way."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None

    def log(self, **kwargs) -> None:
        pass

    def rollout(self, *args, **kwargs) -> None:
        pass


if __name__ == "__main__":
    raise SystemExit(main())
