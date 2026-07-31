#!/usr/bin/env python
"""Headless training, straight from the terminal.

    python scripts/train_cli.py --task reach --template reach_arm --steps 250000

Exists so the whole simulate-and-learn core can be exercised without a browser
anywhere in the loop.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from nova import envs, robots  # noqa: E402
from nova.api import store  # noqa: E402
from nova.training import TrainConfig, evaluate, train  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", default="reach", choices=sorted(envs.TASKS))
    p.add_argument("--template", default=None,
                   help="robot template key (default: first compatible with the task)")
    p.add_argument("--steps", type=int, default=None,
                   help="total env steps (default: the task's typical budget)")
    p.add_argument("--n-envs", type=int, default=TrainConfig.n_envs)
    p.add_argument("--vec", default="subproc", choices=["subproc", "dummy"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ent-coef", type=float, default=TrainConfig.ent_coef)
    p.add_argument("--lr", type=float, default=TrainConfig.learning_rate)
    p.add_argument("--no-norm-reward", action="store_true")
    p.add_argument("--param", action="append", default=[], metavar="KEY=VALUE",
                   help="override a template parameter, repeatable")
    p.add_argument("--eval-episodes", type=int, default=30)
    p.add_argument("--rollout-every", type=int, default=10)
    p.add_argument("--save", default=None, metavar="NAME",
                   help="write the run to runs/NAME/ (model, rollout, history)")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def resolve_template(task: str, requested: str | None) -> str:
    if requested:
        compatible = robots.get(requested).tasks
        if task not in compatible:
            raise SystemExit(
                f"template {requested!r} supports {list(compatible)}, not {task!r}"
            )
        return requested
    for key, tmpl in robots.TEMPLATES.items():
        if task in tmpl.tasks:
            return key
    raise SystemExit(f"no template supports task {task!r}")


def parse_params(pairs: list[str]) -> dict:
    out = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"--param expects KEY=VALUE, got {pair!r}")
        key, _, value = pair.partition("=")
        try:
            out[key.strip()] = float(value)
        except ValueError:
            raise SystemExit(f"--param value must be a number, got {value!r}") from None
    return out


def main(argv=None) -> int:
    args = parse_args(argv)
    template = resolve_template(args.task, args.template)
    task_meta = envs.get_task(args.task)

    cfg = TrainConfig(
        task=args.task,
        template=template,
        params=parse_params(args.param),
        total_steps=args.steps or task_meta["typical_steps"],
        n_envs=args.n_envs,
        vec=args.vec,
        seed=args.seed,
        rollout_every=args.rollout_every,
        ent_coef=args.ent_coef,
        learning_rate=args.lr,
        norm_reward=not args.no_norm_reward,
    )

    rollouts: list[dict] = []

    def on_event(ev: dict) -> None:
        kind = ev["type"]
        if kind == "start" and not args.quiet:
            print(f"task={cfg.task} template={cfg.template} "
                  f"obs={ev['obs_dim']} act={ev['action_dim']} @{ev['control_hz']}Hz")
            print(f"budget={cfg.total_steps:,} steps  envs={cfg.n_envs} ({cfg.vec})")
            print("-" * 72)
        elif kind == "progress" and not args.quiet:
            metric = ""
            if "is_success" in ev:
                metric = f"  success={ev['is_success']:.0%}"
            elif "distance" in ev:
                metric = f"  distance={ev['distance']:.2f}m"
            print(f"  it{ev['iteration']:>4}  {ev['step']:>8,}/{ev['total_steps']:,} "
                  f"({ev['progress']:>5.1%})  reward={ev.get('mean_reward', float('nan')):>9.3f}"
                  f"{metric}  {ev['fps']:>6,} fps  {ev['elapsed']:>5.1f}s")
        elif kind == "rollout":
            rollouts.append(ev)
        elif kind == "done" and not args.quiet:
            print("-" * 72)
            print(f"trained {ev['steps_completed']:,} steps in {ev['wall_time']}s "
                  f"({ev['iterations']} iterations)")

    result = train(cfg, on_event=on_event)

    print("\nevaluating (deterministic, unseen seeds)...")
    t0 = time.perf_counter()
    scores = evaluate(result["model"], cfg, n_episodes=args.eval_episodes)
    print(json.dumps(scores, indent=2))
    print(f"eval took {time.perf_counter() - t0:.1f}s")

    if args.save:
        # Same writer the server uses, so CLI runs show up in the UI's run list
        # with the same shape as anything trained from the browser.
        out = store.save(
            args.save,
            meta={
                "run_id": args.save,
                "task": cfg.task,
                "template": cfg.template,
                "params": robots.get(template).resolve(cfg.params),
                "config": cfg.to_dict(),
                "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "wall_time": result["wall_time"],
                "steps_completed": result["steps_completed"],
                "stopped_early": result["stopped_early"],
                "history": result["history"],
                "eval": scores,
            },
            rollout=result["final_rollout"],
            model=result["model"],
        )
        if rollouts:
            (out / "rollouts.json").write_text(json.dumps([
                {"iteration": r["iteration"], "step": r["step"], "stats": r["stats"]}
                for r in rollouts
            ], indent=2))
        print(f"saved -> {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
