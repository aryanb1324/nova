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
    # No parameters is legitimate: an uploaded robot is fixed geometry. The
    # editor says so rather than showing an empty panel.
    TEMPLATES[template.key] = template
    return template


def unregister(key: str) -> bool:
    return TEMPLATES.pop(key, None) is not None


for _template in (arm.TEMPLATE, wheeled.TEMPLATE, quadruped.TEMPLATE):
    register(_template)

# Imported after register() exists: uploads calls back into it.
from . import contract, uploads  # noqa: E402


_recovered = False


def _recover() -> bool:
    """Re-run the registrations a fresh interpreter wouldn't have.

    Training spawns worker processes, and a spawned worker is a new interpreter
    that only imports `nova` — it never saw the uploads registered at runtime by
    the server, nor whatever NOVA_EXTENSIONS the server loaded. Rather than
    thread that state through every worker, recover it on the first failed
    lookup: uploads come off disk, extensions from the environment variable
    (which children do inherit). Runs at most once per process.
    """
    global _recovered
    if _recovered:
        return False
    _recovered = True

    import nova

    nova.load_extensions()
    uploads.register_stored()
    return True


def get(key: str) -> RobotTemplate:
    try:
        return TEMPLATES[key]
    except KeyError:
        if _recover() and key in TEMPLATES:
            return TEMPLATES[key]
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
    "contract",
    "uploads",
    "register",
    "unregister",
    "get",
    "build_mjcf",
    "standing_height",
    "catalog",
]
