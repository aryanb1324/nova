"""Self-describing metadata carried inside the ONNX file itself.

A policy is meaningless without knowing which observation vector it expects.
Storing that in the model's own `metadata_props` keeps a submission to exactly
one file - no sidecar JSON to lose, no archive to unpack.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field

#: Bumped only if the manifest layout itself changes.
MANIFEST_VERSION = 1

_PREFIX = "nova."
#: Policies exported before the project was renamed carry this prefix. Read it,
#: never write it, so an .onnx someone already downloaded still loads.
_LEGACY_PREFIXES = ("robosim.",)


@dataclass
class PolicyManifest:
    task: str
    template: str
    obs_dim: int
    act_dim: int
    obs_layout: str
    params: dict = field(default_factory=dict)
    algo: str = ""
    author: str = ""
    notes: str = ""
    created: str = ""
    version: int = MANIFEST_VERSION

    def __post_init__(self) -> None:
        if not self.created:
            self.created = time.strftime("%Y-%m-%dT%H:%M:%S")

    def to_props(self) -> dict[str, str]:
        out = {}
        for key, value in asdict(self).items():
            out[_PREFIX + key] = (
                json.dumps(value) if isinstance(value, (dict, list)) else str(value)
            )
        return out

    @classmethod
    def from_props(cls, props: dict[str, str]) -> "PolicyManifest":
        props = _normalize(props)
        get = lambda k, d="": props.get(_PREFIX + k, d)  # noqa: E731

        missing = [k for k in ("task", "template", "obs_dim", "act_dim", "obs_layout")
                   if not props.get(_PREFIX + k)]
        if missing:
            raise ValueError(
                "this .onnx has no nova manifest "
                f"(missing {', '.join(missing)}). Export it with nova.export() "
                "so it records which task and robot it was trained for."
            )

        try:
            params = json.loads(get("params", "{}"))
        except json.JSONDecodeError:
            params = {}
        if not isinstance(params, dict):
            params = {}

        try:
            return cls(
                task=get("task"),
                template=get("template"),
                obs_dim=int(get("obs_dim")),
                act_dim=int(get("act_dim")),
                obs_layout=get("obs_layout"),
                params={k: float(v) for k, v in params.items()},
                algo=get("algo"),
                author=get("author"),
                notes=get("notes"),
                created=get("created"),
                version=int(get("version", "1")),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"malformed nova manifest: {exc}") from None

    def to_dict(self) -> dict:
        return asdict(self)


def is_manifest_key(key: str) -> bool:
    return key.startswith(_PREFIX) or key.startswith(_LEGACY_PREFIXES)


def _normalize(props: dict[str, str]) -> dict[str, str]:
    """Rewrite pre-rename manifest keys onto the current prefix."""
    out = {}
    for key, value in props.items():
        for legacy in _LEGACY_PREFIXES:
            if key.startswith(legacy):
                key = _PREFIX + key[len(legacy):]
                break
        out[key] = value
    return out
