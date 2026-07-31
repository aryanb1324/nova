#!/usr/bin/env python
"""Cross-embodiment transfer: train on one arm, test on arms it never saw.

    python -m examples.cross_body_study --seeds 3

Trains a policy N times on the default reaching arm, then scores each trained
policy - with no retraining and no fine-tuning - on a ladder of modified arms:
longer links, weaker motors, heavier limbs, more damping.

What makes the result readable is that geometry and actuation are separated. The
policy's observation includes the fingertip and the target, so a longer arm is
partly *visible* to it. Motor strength is not in the observation at all. If
geometry transfers better than actuation, that is the reason - and it points at
what a body-conditioned policy would need to be told.

Everything is measured on held-out evaluation seeds and averaged across training
seeds, so neither the episodes nor a single lucky run can flatter the number.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import nova  # noqa: E402
from nova.study import cross_body  # noqa: E402
from nova.training import TrainConfig, train  # noqa: E402

TASK, TEMPLATE = "reach", "reach_arm"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=3, help="training seeds")
    ap.add_argument("--steps", type=int, default=400_000)
    ap.add_argument("--episodes", type=int, default=30,
                    help="held-out episodes per body")
    ap.add_argument("--out", default="cross_body.json")
    args = ap.parse_args()

    bodies = cross_body.bodies_for(TEMPLATE)
    clamped = [b.name for b in bodies if b.is_clamped(TEMPLATE)]
    if clamped:
        # A perturbation the parameter range silently absorbed would overstate
        # how well the policy transfers. Say so rather than quietly reporting it.
        print(f"note: range limits clamped these perturbations: {clamped}")

    print(f"{args.seeds} training seeds x {len(bodies)} bodies "
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
        results.extend(rows)

        home = next(r for r in rows if r["body"] == "baseline")
        print(f"  seed {seed}: trained in {outcome['wall_time']:>5.1f}s, "
              f"baseline success {home.get('success_rate', 0):.0%}")

    summary = cross_body.summarize(results, "success_rate")
    cross_body.save(args.out, {
        "task": TASK,
        "template": TEMPLATE,
        "seeds": list(range(args.seeds)),
        "steps": args.steps,
        "episodes": args.episodes,
        "train_params": nova.params_for(TEMPLATE),
        "results": results,
        "summary": summary,
    })

    print(f"\nZero-shot transfer, {args.seeds} training seeds, held-out episodes:\n")
    print(cross_body.format_table(summary))
    print(f"\nwrote {args.out}  ({time.perf_counter() - started:.0f}s total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
