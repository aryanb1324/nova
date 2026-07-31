#!/usr/bin/env python
"""Cross-embodiment transfer for a robot that carries its own weight.

    python -m examples.cross_body_locomotion --seeds 3

The reach study found mass to be the expensive axis, but a bolted-down arm only
has to swing its limbs. A quadruped has to hold its whole body up on them, so
this is the sharper version of the same question.

ONE THING THIS SCRIPT HAS TO DO THAT THE REACH STUDY DID NOT
-----------------------------------------------------------
`nova.robots.contract.feasible()` is arm-specific - it compares holding torque
against available torque for a two-link arm and returns True for everything
else. So for a quadruped there is no static check, and a body that scores
catastrophically here could be either a genuine transfer failure or a robot too
weak to stand up at all. The reach study made exactly that mistake once.

So instead of a static check this uses the empirical control: any body that
transfers badly gets a policy trained natively ON it. If the native policy also
fails, the body is the problem and the transfer number means nothing. If the
native policy succeeds, the transfer loss is real.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import nova  # noqa: E402
from nova.study import cross_body  # noqa: E402
from nova.study.cross_body import Body  # noqa: E402
from nova.training import TrainConfig, train  # noqa: E402

TASK, TEMPLATE = "locomotion", "quadruped"
METRIC = "mean_final_distance"

#: Separates the same two families as the reach ladder - geometry the policy can
#: partly infer from its own proprioception, versus actuation and inertia it
#: cannot observe at all.
BODIES: tuple[Body, ...] = (
    Body("baseline"),
    Body("legs +20%", {"upper_leg": 1.20, "lower_leg": 1.20}),
    Body("legs -20%", {"upper_leg": 0.80, "lower_leg": 0.80}),
    Body("torso longer +30%", {"torso_length": 1.30}),
    Body("torso wider +30%", {"torso_width": 1.30}),
    Body("heavy legs +40%", {"leg_radius": 1.40}),
    Body("motors -30%", {"gear": 0.70}),
    Body("motors -50%", {"gear": 0.50}),
    Body("motors +50%", {"gear": 1.50}),
    Body("damping x3", {"joint_damping": 3.0}),
)

#: Below this fraction of baseline distance, run the native-training control.
SUSPECT_RETENTION = 0.55


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=800_000)
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--out", default="cross_body_locomotion.json")
    ap.add_argument("--no-control", action="store_true",
                    help="skip native-training controls for weak bodies")
    args = ap.parse_args()

    clamped = [b.name for b in BODIES if b.is_clamped(TEMPLATE)]
    if clamped:
        print(f"note: range limits clamped these perturbations: {clamped}")

    print(f"{args.seeds} training seeds x {len(BODIES)} bodies "
          f"x {args.episodes} held-out episodes\n")

    results: list[dict] = []
    started = time.perf_counter()

    for seed in range(args.seeds):
        outcome = train(TrainConfig(task=TASK, template=TEMPLATE,
                                    total_steps=args.steps, seed=seed,
                                    rollout_every=0))
        rows = cross_body.evaluate_across_bodies(
            outcome["model"], TASK, TEMPLATE, BODIES, episodes=args.episodes)
        for row in rows:
            row["train_seed"] = seed
        results.extend(rows)
        home = next(r for r in rows if r["body"] == "baseline")
        print(f"  seed {seed}: baseline {home.get(METRIC, 0):.2f} m", flush=True)

    summary = cross_body.summarize(results, METRIC)
    base = summary.get("baseline", {}).get("mean", 0.0)

    # ---- the control that keeps this honest ----------------------------
    controls: dict[str, float] = {}
    if not args.no_control and base > 0:
        suspects = [name for name, e in summary.items()
                    if name != "baseline" and e["mean"] / base < SUSPECT_RETENTION]
        if suspects:
            print(f"\ntraining natively on {len(suspects)} weak "
                  f"{'body' if len(suspects) == 1 else 'bodies'}, to tell "
                  "'the policy failed' from 'no policy could':", flush=True)
        for name in suspects:
            body = next(b for b in BODIES if b.name == name)
            native = train(TrainConfig(task=TASK, template=TEMPLATE,
                                       params=body.params_for(TEMPLATE),
                                       total_steps=args.steps, seed=0,
                                       rollout_every=0))
            scored = cross_body.evaluate_across_bodies(
                native["model"], TASK, TEMPLATE, (body,), episodes=args.episodes)
            controls[name] = float(scored[0].get(METRIC, 0.0))
            verdict = ("the body is the problem"
                       if controls[name] < base * SUSPECT_RETENTION
                       else "genuine transfer loss")
            print(f"  {name:<20} transferred {summary[name]['mean']:.2f} m, "
                  f"native {controls[name]:.2f} m  -> {verdict}", flush=True)

    cross_body.save(args.out, {
        "task": TASK, "template": TEMPLATE, "metric": METRIC,
        "seeds": list(range(args.seeds)), "steps": args.steps,
        "episodes": args.episodes, "train_params": nova.params_for(TEMPLATE),
        "results": results, "summary": summary, "native_controls": controls,
    })

    print(f"\nZero-shot transfer, {args.seeds} training seeds, held-out episodes:\n")
    print("| body | distance (m) | retained | native control |")
    print("|---|---|---|---|")
    for name, entry in summary.items():
        dist = f"{entry['mean']:.2f} ± {entry['std']:.2f}"
        retained = "—" if name == "baseline" else (
            f"{entry['mean'] / base * 100:.0f}%" if base else "—")
        control = f"{controls[name]:.2f} m" if name in controls else "—"
        print(f"| {name} | {dist} | {retained} | {control} |")

    print(f"\nwrote {args.out}  ({time.perf_counter() - started:.0f}s total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
