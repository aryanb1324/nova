"""Parameter schemas for robot templates.

A template declares its tunable parameters up front so the frontend can render
an editor for *any* template without knowing what it is. Topology is fixed per
template; only the numbers move. That keeps every design the user can express
guaranteed-simulatable, which is the whole reason training can't be broken by a
bad edit.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Callable


@dataclass(frozen=True)
class ParamSpec:
    """One tunable knob, with everything the UI needs to render a slider."""

    key: str
    label: str
    default: float
    min: float
    max: float
    step: float = 0.01
    unit: str = ""
    help: str = ""
    group: str = "Body"

    def clamp(self, value: float) -> float:
        return max(self.min, min(self.max, float(value)))

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class RobotTemplate:
    """A named robot topology plus the knobs that reshape it."""

    key: str
    name: str
    description: str
    params: tuple[ParamSpec, ...]
    builder: Callable[[dict], str]
    tasks: tuple[str, ...]
    camera: dict = field(default_factory=lambda: {"distance": 1.6, "elevation": -20})
    #: Root height at spawn, given parameters. Locomotion uses it to tell
    #: "fell over" from "standing". Carried on the template rather than in a
    #: lookup table so a robot defined outside this package still works.
    standing_height: Callable[[dict], float] | None = None

    def defaults(self) -> dict[str, float]:
        return {p.key: p.default for p in self.params}

    def resolve(self, overrides: dict | None = None) -> dict[str, float]:
        """Merge user overrides onto defaults, clamping to declared ranges.

        Unknown keys are dropped rather than raising: the frontend and backend
        can drift by a release without breaking every saved robot.
        """
        values = self.defaults()
        by_key = {p.key: p for p in self.params}
        for key, raw in (overrides or {}).items():
            spec = by_key.get(key)
            if spec is None:
                continue
            try:
                values[key] = spec.clamp(raw)
            except (TypeError, ValueError):
                continue
        return values

    def build(self, overrides: dict | None = None) -> str:
        """Return MJCF XML for this template at the given parameter values."""
        return self.builder(self.resolve(overrides))

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "name": self.name,
            "description": self.description,
            "params": [p.to_dict() for p in self.params],
            "tasks": list(self.tasks),
            "camera": dict(self.camera),
        }
