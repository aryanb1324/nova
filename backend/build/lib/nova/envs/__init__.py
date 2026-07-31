"""Task environments.

Like the robot registry, this one is open: subclass `RobotEnv`, call
`nova.register_task`, and your task appears in the API and the browser with no
changes to this package. See `examples/custom_robot.py`.
"""

from __future__ import annotations

from .base import RobotEnv
from .locomotion import LocomotionEnv
from .reach import ReachEnv
from .record import record
from .scene import describe, frame

TASKS: dict[str, dict] = {}


def register_task(
    key: str,
    *,
    env: type[RobotEnv],
    name: str,
    description: str,
    success_metric: str = "",
    monitor_keys: tuple[str, ...] = (),
    typical_steps: int = 400_000,
    replace: bool = False,
) -> dict:
    """Add a task to the catalog.

    `monitor_keys` are per-episode `info` keys to average and surface on the
    training chart. Every key listed must be present in the info dict of the
    final step of an episode, or the Monitor wrapper will raise mid-training.
    """
    if not (isinstance(env, type) and issubclass(env, RobotEnv)):
        raise TypeError(f"task {key!r}: env must be a RobotEnv subclass")
    if key in TASKS and not replace:
        raise ValueError(
            f"a task named {key!r} is already registered; "
            "pass replace=True if you meant to override it"
        )
    if getattr(env, "obs_layout", "unversioned") == "unversioned":
        raise ValueError(
            f"task {key!r}: {env.__name__} must set obs_layout (e.g. "
            f"'{key}.v1'). Policies record it, and without a version an "
            "observation change silently invalidates every saved policy."
        )
    entry = {
        "key": key,
        "name": name,
        "description": description,
        "env": env,
        "success_metric": success_metric,
        "monitor_keys": tuple(monitor_keys),
        "typical_steps": int(typical_steps),
    }
    TASKS[key] = entry
    return entry


register_task(
    "reach",
    env=ReachEnv,
    name="Reach the target",
    description="Put the fingertip on the green ball, wherever it appears.",
    success_metric="is_success",
    monitor_keys=("is_success", "distance"),
    # Measured: 100% success at 400k, ~90% at 200k. ~30s on a 10-core M5.
    typical_steps=400_000,
)
register_task(
    "locomotion",
    env=LocomotionEnv,
    name="Walk forward",
    description="Travel as far as possible along +X without falling over.",
    success_metric="distance",
    monitor_keys=("distance", "forward_velocity"),
    typical_steps=800_000,
)


def get_task(key: str) -> dict:
    try:
        return TASKS[key]
    except KeyError:
        raise KeyError(f"unknown task {key!r}; have {sorted(TASKS)}") from None


def make(task: str, template: str, params: dict | None = None,
         seed: int | None = None) -> RobotEnv:
    return get_task(task)["env"](template_key=template, params=params, seed=seed)


def catalog() -> list[dict]:
    """JSON-serializable task list for the frontend."""
    return [{k: v for k, v in t.items() if k != "env"} for t in TASKS.values()]


__all__ = [
    "TASKS",
    "RobotEnv",
    "ReachEnv",
    "LocomotionEnv",
    "register_task",
    "get_task",
    "make",
    "catalog",
    "describe",
    "frame",
    "record",
]
