"""Portable policies: export, validate, and score.

Imports are deliberately lazy. `export` needs torch; `load`/`evaluate` need only
onnxruntime, so an evaluation worker never pays for the training stack.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .manifest import MANIFEST_VERSION, PolicyManifest

if TYPE_CHECKING:  # pragma: no cover
    from .evaluate import evaluate_bytes, evaluate_isolated, held_out_seeds, score
    from .export import export_policy
    from .loader import OnnxPolicy, PolicyRejected, load, load_file, validate

_LAZY = {
    "export_policy": ".export",
    "load": ".loader",
    "load_file": ".loader",
    "validate": ".loader",
    "inspect": ".loader",
    "OnnxPolicy": ".loader",
    "PolicyRejected": ".loader",
    "evaluate_bytes": ".evaluate",
    "evaluate_isolated": ".evaluate",
    "score": ".evaluate",
    "held_out_seeds": ".evaluate",
    "PUBLIC_SEEDS": ".evaluate",
}


def __getattr__(name: str) -> Any:
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    return getattr(importlib.import_module(module, __name__), name)


__all__ = ["PolicyManifest", "MANIFEST_VERSION", *_LAZY]
