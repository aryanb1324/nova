"""Robot templates: fixed topologies with tunable numbers.

The three built-ins are registered here, but the registry is open — a robot
defined in your own package works the same way, with no fork required. See
`nova.register_template` and `examples/custom_robot.py`.
"""

from __future__ import annotations

from . import arm, quadruped, wheeled
from .spec import ParamSpec, RobotTemplate

TEMPLATES: dict[str, RobotTemplate] = {}


def register(template: RobotTemplate, *, replace: bool = False) -> RobotTemplate:
    """Add a robot to the catalog.

    Everything downstream is schema-driven, so a registered template appears in
    the API and gets its own sliders in the browser with no frontend change.
    """
    if not isinstance(template, RobotTemplate):
        raise TypeError(f"expected a RobotTemplate, got {type(template).__name__}")
    if template.key in TEMPLATES and not replace:
        raise ValueError(
            f"a robot named {template.key!r} is already registered; "
            "pass replace=True if you meant to override it"
        )
    if not template.params:
        raise ValueError(
            f"{template.key!r} declares no parameters, so the editor would render "
            "an empty panel"
        )
    TEMPLATES[template.key] = template
    return template


for _template in (arm.TEMPLATE, wheeled.TEMPLATE, quadruped.TEMPLATE):
    register(_template)


def get(key: str) -> RobotTemplate:
    try:
        return TEMPLATES[key]
    except KeyError:
        raise KeyError(
            f"unknown robot template {key!r}; have {sorted(TEMPLATES)}"
        ) from None


def build_mjcf(key: str, params: dict | None = None) -> str:
    return get(key).build(params)


def standing_height(key: str, params: dict) -> float:
    fn = get(key).standing_height
    return fn(params) if fn else 0.0


def catalog() -> list[dict]:
    """JSON-serializable template list for the frontend's picker and editor."""
    return [t.to_dict() for t in TEMPLATES.values()]


__all__ = [
    "TEMPLATES",
    "ParamSpec",
    "RobotTemplate",
    "register",
    "get",
    "build_mjcf",
    "standing_height",
    "catalog",
]
