"""Locomotion task: travel as far forward as possible without falling over.

Works for any template with a free-floating root - the rover and the quadruped
share this env and differ only in their actuators.
"""

from __future__ import annotations

import mujoco
import numpy as np

from .. import robots
from .base import RobotEnv


class LocomotionEnv(RobotEnv):
    """Move along +x. Terminates early on a fall, which is what makes the
    alive bonus meaningful."""

    task_key = "locomotion"
    frame_skip = 4   # 0.005s timestep -> 50 Hz control
    max_steps = 250  # 5 seconds per episode

    # [height, projected_gravity x3, local_linvel x3, angvel x3,
    #  cos(jq), sin(jq), jvel*0.1] - trailing width depends on the template's DoF
    obs_layout = "locomotion.v1"

    #: Reward stops growing past this, so the policy pursues a gait rather than
    #: a single ballistic launch.
    max_speed = 2.0
    #: Fall test, as a fraction of the robot's spawn height.
    fall_ratio = 0.55

    def __init__(self, template_key: str = "wheeled_bot", params=None, seed=None):
        super().__init__(template_key, params, seed)

    def _setup(self) -> None:
        free = [
            i for i in range(self.model.njnt)
            if self.model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE
        ]
        if not free:
            raise ValueError(
                f"template {self.template_key!r} has no free joint; "
                "locomotion needs a robot that can move through the world"
            )
        self._root_body = int(self.model.jnt_bodyid[free[0]])
        self._stand_h = robots.standing_height(self.template_key, self.params)
        # Everything after the root's 7 qpos / 6 qvel entries is an actuated joint.
        self._jnt_q = slice(7, self.model.nq)
        self._jnt_v = slice(6, self.model.nv)

    def _randomize(self) -> None:
        self.data.qpos[self._jnt_q] += self._rng.uniform(
            -0.05, 0.05, self.model.nq - 7
        )

    # ---- observations ---------------------------------------------------

    def _root_frame(self) -> np.ndarray:
        return self.data.xmat[self._root_body].reshape(3, 3)

    def _observe(self) -> np.ndarray:
        rot = self._root_frame()
        # Gravity expressed in the robot's own frame: tells it which way is down
        # without ever telling it which way it is facing.
        projected_gravity = rot.T @ np.array([0.0, 0.0, -1.0])
        local_linvel = rot.T @ self.data.qvel[0:3]
        angvel = self.data.qvel[3:6]
        jq = self.data.qpos[self._jnt_q]

        return np.concatenate([
            [self.data.qpos[2]],          # height above ground
            projected_gravity,
            local_linvel,
            angvel,
            np.cos(jq),                   # wrap-safe: wheel angles grow forever
            np.sin(jq),
            self.data.qvel[self._jnt_v] * 0.1,
        ]).astype(np.float32)

    # ---- reward ---------------------------------------------------------

    def _upright(self) -> float:
        """+1 when the robot's own up-axis points at the sky, 0 when horizontal."""
        return float(self._root_frame()[2, 2])

    def _reward(self, action):
        vx = float(self.data.qvel[0])
        vy = float(self.data.qvel[1])
        forward = min(vx, self.max_speed)

        reward = (
            1.5 * forward                            # the actual objective
            + 0.5                                    # alive bonus, forfeited by falling
            - 0.05 * abs(vy)                         # keep it tracking straight
            - 0.005 * float(np.square(action).sum())  # energy
        )
        return reward, {
            "forward_velocity": vx,
            "height": float(self.data.qpos[2]),
            "upright": self._upright(),
            "distance": float(self.data.qpos[0]),
        }

    def _terminated(self) -> bool:
        fell = self.data.qpos[2] < self.fall_ratio * self._stand_h
        tipped = self._upright() < 0.2
        return bool(fell or tipped)
