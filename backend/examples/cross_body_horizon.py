#!/usr/bin/env python
"""Does cross-body transfer survive a shorter episode budget?

    python -m examples.cross_body_horizon --mode slack
    python -m examples.cross_body_horizon --mode sanity --horizon 40
    python -m examples.cross_body_horizon --mode study  --horizon 40 --seeds 3
    python -m examples.cross_body_horizon --mode study  --horizon 100 --seeds 3

The published cross-body study found that actuation is nearly free: motors at
-50%, -30% and +50% all retained ~100% of baseline success, even though motor
strength appears nowhere in the observation. The proposed explanation was slack:
the reach episode is 100 steps (2 s at 50 Hz) with dense shaping, so a weaker arm
still arrives in time. Under time pressure, weak motors should start to hurt.

This file tests that. It does *not* touch the built-in `reach` task - the
benchmark suite and other studies depend on its 100-step horizon. Instead it
registers a separate task, `reach_short`, whose env is a `ReachEnv` subclass with
a smaller `max_steps`. The observation is untouched, so it keeps `reach.v1`.

Three modes:

  slack   Trains one policy on the *standard* 100-step task and records, per
          held-out episode, the first step at which the fingertip is inside the
          success radius. That distribution is the slack, measured directly:
          it says how much of the two seconds the policy actually needs.

  sanity  Trains on the baseline body at the short horizon and reports held-out
          success. If this is near zero the horizon is not a stress test, it is
          an impossible task, and nothing measured on the ladder would mean
          anything.

  study   The full ladder: train on the baseline body at the short horizon, then
          score zero-shot on every body in `cross_body.REACH_BODIES`. Passing
          --horizon 100 runs the same code path at the original budget, which is
          the control the short-horizon numbers get compared against.

Two things are inherited from the original study and matter as much here:
evaluation uses held-out seeds, and one body in the ladder is deliberately
under-actuated and is labelled rather than scored (see `cross_body.feasible`).
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

import nova  # noqa: E402
from nova.envs.reach import ReachEnv  # noqa: E402
from nova.policies.evaluate import held_out_seeds, score  # noqa: E402
from nova.study import cross_body  # noqa: E402
from nova.training import TrainConfig, train  # noqa: E402

#: This module registers its own task, so a spawned training worker - a fresh
#: interpreter that only imports `nova` - has to be told to re-import it. NOVA
#: recovers unknown tasks from NOVA_EXTENSIONS on the first failed lookup, and
#: children inherit the environment, so setting it in main() is enough.
MODULE = "examples.cross_body_horizon"

TASK = "reach_short"
TEMPLATE = "reach_arm"
LONG_TASK = "reach"

#: Set by main() before training starts, and read here on re-import in each
#: worker. A class attribute alone would not survive the process boundary.
HORIZON = int(os.environ.get("NOVA_SHORT_HORIZON", "40"))


class ShortReachEnv(ReachEnv):
    """The reach task with less time on the clock.

    Only `max_steps` changes. The observation formula is inherited verbatim, so
    the layout tag stays `reach.v1` - a policy trained here is interchangeable
    with one trained on the built-in task, which is the whole point: the
    comparison must be about the time budget and nothing else.
    """

    task_key = TASK
    max_steps = HORIZON


def register(horizon: int) -> None:
    """(Re-)register `reach_short` with a given clock."""
    ShortReachEnv.max_steps = horizon
    nova.register_task(
        TASK,
        env=ShortReachEnv,
        name=f"Reach the target ({horizon} steps)",
        description=(
            f"The reach task truncated to {horizon} steps "
            f"({horizon / 50:.2f} s at 50 Hz), to remove the slack in the budget."
        ),
        success_metric="is_success",
        monitor_keys=("is_success", "distance"),
        typical_steps=400_000,
        # Same interface as the built-in reach task, because it is the same task
        # with a shorter clock. Stated explicitly rather than inherited:
        # `requires` is what matches an uploaded robot to a task, and a task that
        # silently omitted it would appear to accept arms with no fingertip site.
        requires={
            "sites": ("ee",),
            "mocap_bodies": ("target",),
            "free_joints": (0, 0),
            "note": "needs a site named 'ee' at the fingertip and a mocap body "
                    "named 'target'; the arm itself must be bolted down",
        },
        # The parent runs this file as __main__; a worker imports it under its
        # real name. If both happen in one interpreter, overwrite rather than
        # raise - and main() re-registers once the horizon is known.
        replace=True,
    )


register(HORIZON)


# ---------------------------------------------------------------------------
# Measuring the slack directly
# ---------------------------------------------------------------------------

def first_arrival(policy, task: str, params: dict, seeds) -> list[int | None]:
    """Per episode, the first step whose info says the fingertip is on target.

    `score()` reads only the final step, which answers "did it end up there".
    That is the right success definition, but it cannot distinguish a policy
    that arrives at step 12 and parks from one that arrives at step 95. The
    difference between those two is exactly the slack under test.
    """
    env = nova.make(task, TEMPLATE, params)
    out: list[int | None] = []
    for seed in seeds:
        obs, _ = env.reset(seed=int(seed))
        arrived = None
        while True:
            action, _ = policy.predict(obs, deterministic=True)
            obs, _reward, terminated, truncated, info = env.step(action)
            if arrived is None and info.get("is_success"):
                arrived = env._steps
            if terminated or truncated:
                break
        out.append(arrived)
    return out


def describe_arrivals(arrivals: list[int | None]) -> str:
    hit = [a for a in arrivals if a is not None]
    if not hit:
        return "never arrived in any episode"
    q = np.percentile(hit, [50, 75, 90, 100])
    return (f"arrived in {len(hit)}/{len(arrivals)} episodes; "
            f"first-arrival step: median {q[0]:.0f}, p75 {q[1]:.0f}, "
            f"p90 {q[2]:.0f}, max {q[3]:.0f}")


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def run_slack(args) -> int:
    """How much of the 100-step budget does the standard policy actually use?"""
    print(f"training 1 seed on the standard {LONG_TASK} task "
          f"({ReachEnv.max_steps} steps) to measure arrival times\n")
    cfg = TrainConfig(task=LONG_TASK, template=TEMPLATE, total_steps=args.steps,
                      seed=0, rollout_every=0)
    outcome = train(cfg)

    params = nova.params_for(TEMPLATE)
    seeds = held_out_seeds(args.episodes)
    arrivals = first_arrival(outcome["model"], LONG_TASK, params, seeds)
    print(f"baseline body: {describe_arrivals(arrivals)}")

    # The same question asked of the bodies the invariance claim rests on. If a
    # -50% motor arm arrives much later than the baseline, the slack is real and
    # a shorter horizon should expose it.
    for body in cross_body.REACH_BODIES:
        if not body.delta or "gear" not in body.delta:
            continue
        p = body.params_for(TEMPLATE)
        print(f"{body.name:<14}: "
              f"{describe_arrivals(first_arrival(outcome['model'], LONG_TASK, p, seeds))}")
    return 0


def run_sanity(args) -> int:
    """Is the short horizon learnable at all on the training body?"""
    print(f"sanity check: {TASK} at {HORIZON} steps ({HORIZON / 50:.2f} s), "
          f"{args.seeds} seed(s), {args.steps:,} steps each\n")
    params = nova.params_for(TEMPLATE)
    seeds = held_out_seeds(args.episodes)
    rates = []
    for seed in range(args.seeds):
        cfg = TrainConfig(task=TASK, template=TEMPLATE, total_steps=args.steps,
                          seed=seed, rollout_every=0)
        outcome = train(cfg)
        result = score(outcome["model"], TASK, TEMPLATE, params, seeds)
        rates.append(result.get("success_rate", 0.0))
        print(f"  seed {seed}: baseline success {rates[-1]:.0%}  "
              f"final distance {result.get('mean_final_distance', 0):.3f} m")
    print(f"\nbaseline success at horizon {HORIZON}: "
          f"{np.mean(rates):.0%} ± {np.std(rates) * 100:.0f}")
    if np.mean(rates) < 0.70:
        print("-> too hard to be informative; lengthen the horizon and retry")
    return 0


def run_study(args) -> int:
    bodies = cross_body.bodies_for(TEMPLATE)
    clamped = [b.name for b in bodies if b.is_clamped(TEMPLATE)]
    if clamped:
        print(f"note: range limits clamped these perturbations: {clamped}")

    print(f"task {TASK} at {HORIZON} steps ({HORIZON / 50:.2f} s)  |  "
          f"{args.seeds} training seeds x {len(bodies)} bodies "
          f"x {args.episodes} held-out episodes")

    results: list[dict] = []
    started = time.perf_counter()

    for seed in range(args.seeds):
        cfg = TrainConfig(task=TASK, template=TEMPLATE, total_steps=args.steps,
                          seed=seed, rollout_every=0)
        outcome = train(cfg)

        rows = cross_body.evaluate_across_bodies(
            outcome["model"], TASK, TEMPLATE, bodies, episodes=args.episodes)
        for row in rows:
            row["train_seed"] = seed
            row["horizon"] = HORIZON
        results.extend(rows)

        home = next(r for r in rows if r["body"] == "baseline")
        print(f"  seed {seed}: baseline success {home.get('success_rate', 0):.0%}")

    summary = cross_body.summarize(results, "success_rate")
    out = args.out or f"cross_body_h{HORIZON}.json"
    cross_body.save(out, {
        "task": TASK,
        "template": TEMPLATE,
        "horizon": HORIZON,
        "control_hz": 50,
        "seeds": list(range(args.seeds)),
        "steps": args.steps,
        "episodes": args.episodes,
        "train_params": nova.params_for(TEMPLATE),
        "results": results,
        "summary": summary,
    })

    print(f"\nZero-shot transfer at horizon {HORIZON}, {args.seeds} training "
          f"seeds, held-out episodes:\n")
    print(cross_body.format_table(summary))
    print(f"\nwrote {out}  ({time.perf_counter() - started:.0f}s total)")
    return 0


MODES = {"slack": run_slack, "sanity": run_sanity, "study": run_study}


def main(argv=None) -> int:
    global HORIZON
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=sorted(MODES), default="study")
    ap.add_argument("--horizon", type=int, default=HORIZON,
                    help="agent steps per episode for the short task (50 Hz)")
    ap.add_argument("--seeds", type=int, default=3, help="training seeds")
    ap.add_argument("--steps", type=int, default=400_000)
    ap.add_argument("--episodes", type=int, default=30,
                    help="held-out episodes per body")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    HORIZON = args.horizon
    register(args.horizon)
    # Both variables have to be in the environment before the vec env spawns:
    # NOVA_EXTENSIONS so the worker knows to import this module, and the horizon
    # so the class it re-creates has the same clock as the one here.
    os.environ["NOVA_SHORT_HORIZON"] = str(args.horizon)
    existing = [m for m in os.environ.get("NOVA_EXTENSIONS", "").split(",") if m]
    if MODULE not in existing:
        os.environ["NOVA_EXTENSIONS"] = ",".join([*existing, MODULE])

    return MODES[args.mode](args)


if __name__ == "__main__":
    raise SystemExit(main())
