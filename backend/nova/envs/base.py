"""Shared MuJoCo/Gymnasium plumbing for every task.

Subclasses supply three things - an observation vector, a reward, and a
termination test. Everything else (model construction from template parameters,
action scaling, episode bookkeeping) lives here.
"""

from __future__ import annotations

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from .. import robots


class RobotEnv(gym.Env):
    """Base env: one parametric robot template, one task."""

    metadata = {"render_modes": []}

    #: Simulator substeps per agent action. Control rate = 1 / (timestep * frame_skip).
    frame_skip: int = 5
    #: Agent steps before truncation.
    max_steps: int = 100

    #: Identifies the exact meaning of the observation vector. An uploaded policy
    #: records this; if the formula changes, bump the version so old policies are
    #: rejected loudly instead of silently acting on numbers that moved under them.
    #: Dimension checks alone would not catch a reordering.
    obs_layout: str = "unversioned"

    def __init__(
        self,
        template_key: str = "reach_arm",
        params: dict | None = None,
        seed: int | None = None,
    ):
        super().__init__()
        self.template_key = template_key
        self.template = robots.get(template_key)
        self.params = self.template.resolve(params)

        self.mjcf = self.template.builder(self.params)
        self.model = mujoco.MjModel.from_xml_string(self.mjcf)
        self.data = mujoco.MjData(self.model)

        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._steps = 0

        self.action_space = spaces.Box(-1.0, 1.0, (self.model.nu,), dtype=np.float32)
        # Subclasses resolve model-dependent ids here: the first reset happens
        # below, and _randomize() may already need them.
        self._setup()
        obs = self._reset_model()
        self.observation_space = spaces.Box(
            -np.inf, np.inf, (obs.size,), dtype=np.float32
        )

    # ---- subclass hooks -------------------------------------------------

    def _setup(self) -> None:
        """Called once, after the model exists and before the first reset."""

    def _observe(self) -> np.ndarray:
        raise NotImplementedError

    def _reward(self, action: np.ndarray) -> tuple[float, dict]:
        """Return (reward, info). `info` is merged into the step info dict."""
        raise NotImplementedError

    def _terminated(self) -> bool:
        return False

    def _randomize(self) -> None:
        """Per-episode randomization. Called after mj_resetData."""

    # ---- gymnasium API --------------------------------------------------

    @property
    def dt(self) -> float:
        return self.model.opt.timestep * self.frame_skip

    def _reset_model(self) -> np.ndarray:
        mujoco.mj_resetData(self.model, self.data)
        self._randomize()
        mujoco.mj_forward(self.model, self.data)
        self._steps = 0
        return self._observe()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        return self._reset_model(), {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)
        # Site/body world positions are stale after mj_step's final integration
        # unless we re-run forward kinematics; observations read them.
        mujoco.mj_forward(self.model, self.data)

        self._steps += 1
        reward, info = self._reward(action)
        terminated = self._terminated()
        truncated = self._steps >= self.max_steps
        return self._observe(), float(reward), bool(terminated), bool(truncated), info

    # ---- helpers --------------------------------------------------------

    def site_xpos(self, name: str) -> np.ndarray:
        return self.data.site(name).xpos.copy()

    def body_xpos(self, name: str) -> np.ndarray:
        return self.data.body(name).xpos.copy()

    def body_poses(self) -> np.ndarray:
        """World pose of every body as [x, y, z, qw, qx, qy, qz] rows.

        This is what gets streamed to the browser: the viewer sets a transform
        per body and needs no kinematics of its own.
        """
        return np.concatenate(
            [self.data.xpos[1:], self.data.xquat[1:]], axis=1
        ).astype(np.float32)
