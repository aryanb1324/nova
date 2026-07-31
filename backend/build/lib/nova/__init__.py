"""nova - browser-based robot design and policy training sandbox.

Bring your own algorithm:

    import nova

    env = nova.make("reach", "reach_arm")
    ...                                        # your training loop, your library
    nova.export(policy, "mine.onnx", task="reach", template="reach_arm")
    print(nova.evaluate("mine.onnx"))

To watch it learn in the browser while it runs, wrap the loop in `attach()`.

Every import here is lazy. Importing nova costs nothing; touching `export`
pulls in torch, and touching `evaluate` pulls in onnxruntime, but neither one
drags in the other.
"""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING, Any, Iterable

if TYPE_CHECKING:  # pragma: no cover
    from .envs import RobotEnv
    from .robots import ParamSpec, RobotTemplate

__version__ = "0.2.0"

__all__ = [
    "make", "templates", "tasks", "params_for",
    "export", "evaluate", "record", "attach",
    "register_template", "register_task", "load_extensions",
    "RobotEnv", "RobotTemplate", "ParamSpec",
    "PUBLIC_SEEDS", "__version__",
]


def register_template(template, *, replace: bool = False):
    """Add your own robot without forking this package.

        import nova
        nova.register_template(nova.RobotTemplate(key="my_arm", ..., builder=build))

    Everything downstream reads the registry, so the robot appears in the API
    and gets sliders in the browser generated from its own ParamSpec list.
    """
    from .robots import register
    return register(template, replace=replace)


def register_task(key: str, **kwargs):
    """Add your own task. `env` must be a RobotEnv subclass with an obs_layout."""
    from .envs import register_task as _register
    return _register(key, **kwargs)


def load_extensions(modules: Iterable[str] | None = None) -> list[str]:
    """Import modules that register robots or tasks.

    Called by the server at startup with whatever is in the `NOVA_EXTENSIONS`
    environment variable (comma-separated), so a fork or a separate package can
    add content without touching this one:

        NOVA_EXTENSIONS=my_robots ./dev.sh
    """
    import importlib
    import os

    names = list(modules) if modules is not None else [
        m.strip() for m in os.environ.get("NOVA_EXTENSIONS", "").split(",") if m.strip()
    ]
    loaded = []
    for name in names:
        try:
            importlib.import_module(name)
            loaded.append(name)
        except Exception as exc:
            # A broken extension shouldn't take the server down with it; say so
            # loudly and carry on with whatever else registered.
            print(f"[NOVA] could not load extension {name!r}: "
                  f"{type(exc).__name__}: {exc}")
    return loaded


def make(task: str = "reach", template: str = "reach_arm",
         params: dict | None = None, seed: int | None = None):
    """A Gymnasium environment for one robot on one task."""
    from .envs import make as _make
    return _make(task, template, params, seed)


def templates() -> list[dict]:
    """Available robots, with their tunable parameters."""
    from . import robots
    return robots.catalog()


def tasks() -> list[dict]:
    """Available tasks."""
    from . import envs
    return envs.catalog()


def params_for(template: str, overrides: dict | None = None) -> dict:
    """Full parameter set for a robot, with any overrides clamped to range."""
    from . import robots
    return robots.get(template).resolve(overrides)


def record(env, policy=None, *, seed: int | None = None, deterministic: bool = True) -> dict:
    """Run one episode and capture it in the format the browser replays."""
    from .envs import record as _record
    return _record(env, policy, seed=seed, deterministic=deterministic)


def export(source: Any, path: str | pathlib.Path, *, task: str, template: str,
           params: dict | None = None, algo: str = "", author: str = "",
           notes: str = ""):
    """Write a trained policy to a portable .onnx with a nova manifest.

    `source` may be a stable-baselines3 model or any torch module mapping
    (batch, obs_dim) -> (batch, act_dim). The export is checked against the
    original before the file is written.
    """
    from .policies.export import export_policy
    return export_policy(source, path, task=task, template=template, params=params,
                         algo=algo, author=author, notes=notes)


def evaluate(policy: Any, *, task: str | None = None, template: str | None = None,
             params: dict | None = None, seeds: Iterable[int] | None = None,
             episodes: int = 30) -> dict:
    """Score a policy.

    `policy` may be a path to an .onnx file, its raw bytes, or any object with
    `.predict(obs, deterministic=True)`. For an .onnx the task and robot come
    from its embedded manifest, so no other arguments are needed.

    Without `seeds`, scores against both the published seed set and the held-out
    one, and reports the gap between them.
    """
    from .policies.evaluate import PUBLIC_SEEDS, evaluate_bytes, held_out_seeds, score

    data: bytes | None = None
    if isinstance(policy, (str, pathlib.Path)):
        data = pathlib.Path(policy).read_bytes()
    elif isinstance(policy, (bytes, bytearray)):
        data = bytes(policy)

    if data is not None and seeds is None:
        return evaluate_bytes(data, episodes=episodes)

    if data is not None:
        from .policies.loader import load
        loaded, manifest = load(data)
        policy, task, template = loaded, manifest.task, manifest.template
        params = manifest.params

    if not task or not template:
        raise ValueError(
            "task and template are required when scoring a live policy object"
        )
    if seeds is not None:
        return score(policy, task, template, params or {}, list(seeds))

    public = score(policy, task, template, params or {}, PUBLIC_SEEDS[:episodes])
    private = score(policy, task, template, params or {}, held_out_seeds(episodes))
    return {"public": public, "held_out": private}


def attach(task: str = "reach", template: str = "reach_arm",
           params: dict | None = None, *, algo: str = "custom", author: str = "",
           total_steps: int = 0, url: str | None = None):
    """Stream a training run into a running nova UI.

        with nova.attach(algo="my-CEM", total_steps=300_000) as run:
            for it in range(30):
                ...
                run.log(step=steps_done, mean_reward=score)
                run.rollout(policy)

    Never fatal: if the UI isn't running, training carries on regardless.
    """
    from .attach_client import DEFAULT_URL, AttachRun
    return AttachRun(task, template, params, algo=algo, author=author,
                     total_steps=total_steps, url=url or DEFAULT_URL)


def __getattr__(name: str):
    """Lazy re-exports, so `import nova` stays cheap."""
    if name == "PUBLIC_SEEDS":
        from .policies.evaluate import PUBLIC_SEEDS
        return PUBLIC_SEEDS
    if name in ("RobotTemplate", "ParamSpec"):
        from . import robots
        return getattr(robots, name)
    if name == "RobotEnv":
        from .envs import RobotEnv
        return RobotEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
