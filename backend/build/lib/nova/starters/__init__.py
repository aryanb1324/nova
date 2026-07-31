"""Reference algorithm implementations, meant to be edited.

Every one of these is a real file you can run directly:

    python -m nova.starters.reinforce

They are also what the in-app editor loads, so what you read here is exactly
what you get in the browser. They are written to be modified - short, commented,
and deliberately not factored into a shared framework, because the first thing
someone testing their own architecture does is change the middle of the loop.
"""

from __future__ import annotations

import pathlib

_DIR = pathlib.Path(__file__).parent

STARTERS: tuple[dict, ...] = (
    {
        "key": "blank",
        "name": "Blank skeleton",
        "description": "The training loop with the algorithm left out. Start here.",
        "level": "start",
    },
    {
        "key": "reinforce",
        "name": "REINFORCE from scratch",
        "description": "Vanilla policy gradient in ~90 lines of torch. The simplest "
                       "thing that learns.",
        "level": "reference",
    },
    {
        "key": "ppo_minimal",
        "name": "PPO from scratch",
        "description": "A readable PPO with GAE and clipping - no RL library. "
                       "The usual baseline to modify.",
        "level": "reference",
    },
    {
        "key": "cem",
        "name": "Cross-entropy method",
        "description": "Gradient-free. Shows that the env doesn't care what "
                       "your algorithm is.",
        "level": "reference",
    },
    {
        "key": "sb3_custom",
        "name": "Custom network with SB3",
        "description": "Keep PPO, swap in your own architecture. Fastest way to "
                       "test a new network.",
        "level": "reference",
    },
)


def source(key: str) -> str:
    path = _DIR / f"{key}.py"
    if not path.exists():
        raise KeyError(f"unknown starter {key!r}")
    return path.read_text()


def catalog() -> list[dict]:
    """Starter metadata plus the code itself, for the editor's template picker."""
    out = []
    for meta in STARTERS:
        try:
            out.append({**meta, "code": source(meta["key"])})
        except KeyError:
            continue
    return out
