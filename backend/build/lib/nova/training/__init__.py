"""Policy training."""

from __future__ import annotations

from ..envs import record
from .train import TrainConfig, evaluate, train

__all__ = ["TrainConfig", "train", "evaluate", "record"]
