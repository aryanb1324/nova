"""`nova` command line entry point."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys


_INIT_README = """# {name}

An RL experiment against the NOVA simulator, started from the `{starter}`
reference implementation.

```bash
python train.py
```

Start the NOVA app first (`./dev.sh` in the NOVA checkout) and the run streams
into the browser: reward curve, live 3D rollouts, the lot. Without it, training
still works and just prints to the terminal.

## What you get

```python
import nova

env = nova.make("reach", "reach_arm")   # a normal Gymnasium environment
nova.templates()                         # robots, and their tunable parameters
nova.tasks()                             # tasks
```

Observations are already scaled; actions are continuous in `[-1, 1]`.

## Scoring

```python
nova.export(policy, "mine.onnx", task="reach", template="reach_arm")
print(nova.evaluate("mine.onnx"))
```

`evaluate` scores on a published seed set and on held-out seeds you can't tune
against. The gap between them is usually the interesting part.
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="nova", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("catalog", help="list robots and tasks")

    p_demo = sub.add_parser("demo", help="run the whole loop end to end")
    p_demo.add_argument("--task", default="reach")
    p_demo.add_argument("--template", default="reach_arm")
    p_demo.add_argument("--steps", type=int, default=None)
    p_demo.add_argument("--out", default="demo_policy.onnx")
    p_demo.add_argument("--url", default="http://127.0.0.1:8000")

    p_eval = sub.add_parser("eval", help="score an .onnx policy")
    p_eval.add_argument("policy", help="path to a .onnx exported by nova.export()")
    p_eval.add_argument("--episodes", type=int, default=30)
    p_eval.add_argument("--public-only", action="store_true",
                        help="skip the held-out seed set")

    p_init = sub.add_parser("init", help="scaffold a project to work on in your own editor")
    p_init.add_argument("directory")
    p_init.add_argument("--starter", default="ppo_minimal",
                        help="which reference implementation to start from")

    p_serve = sub.add_parser("serve", help="run the API server")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)

    args = parser.parse_args(argv)

    if args.command == "demo":
        import os

        # The env workers saturate the cores on their own; an extra BLAS pool
        # inside each one halves throughput.
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        from nova.demo import run_demo

        return run_demo(task=args.task, template=args.template, steps=args.steps,
                        out=args.out, url=args.url)

    if args.command == "init":
        from nova import starters

        target = pathlib.Path(args.directory)
        if target.exists() and any(target.iterdir()):
            print(f"{target} exists and is not empty", file=sys.stderr)
            return 2
        try:
            code = starters.source(args.starter)
        except KeyError:
            names = [s["key"] for s in starters.STARTERS]
            print(f"unknown starter {args.starter!r}; have {names}", file=sys.stderr)
            return 2

        target.mkdir(parents=True, exist_ok=True)
        (target / "train.py").write_text(code)
        (target / "requirements.txt").write_text("nova\ntorch\nnumpy\n")
        (target / "README.md").write_text(_INIT_README.format(
            name=target.name, starter=args.starter))
        print(f"created {target}/")
        print("  train.py          your training script (edit this)")
        print("  requirements.txt")
        print("  README.md")
        print(f"\nnext:\n  cd {target} && python train.py")
        print("  (start the NOVA app first and it will stream into the browser)")
        return 0

    if args.command == "catalog":
        import nova

        for t in nova.templates():
            print(f"{t['key']:<14} {t['name']:<16} tasks={t['tasks']}")
        print()
        for t in nova.tasks():
            print(f"{t['key']:<14} {t['name']:<20} typical_steps={t['typical_steps']:,}")
        return 0

    if args.command == "eval":
        import nova
        from nova.policies.evaluate import PUBLIC_SEEDS
        from nova.policies.loader import PolicyRejected

        path = pathlib.Path(args.policy)
        if not path.exists():
            print(f"no such file: {path}", file=sys.stderr)
            return 2
        try:
            if args.public_only:
                result = nova.evaluate(path, seeds=PUBLIC_SEEDS[:args.episodes])
            else:
                result = nova.evaluate(path, episodes=args.episodes)
        except PolicyRejected as exc:
            print(f"rejected: {exc}", file=sys.stderr)
            return 1
        result.pop("rollout", None)
        print(json.dumps(result, indent=2))
        return 0

    if args.command == "serve":
        import os

        os.environ.setdefault("OMP_NUM_THREADS", "1")
        import uvicorn

        uvicorn.run("nova.api.server:app", host=args.host, port=args.port)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
