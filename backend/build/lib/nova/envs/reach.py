"""Reach task: drive the fingertip to a randomly placed target.

This is the MVP demo. It's chosen deliberately over locomotion because it trains
to a convincing policy in seconds rather than minutes, which is what lets the
user sit and watch the reward curve climb instead of walking away.
"""

from __future__ import annotations

import numpy as np

from .base import RobotEnv

#: Fingertip within this distance counts as a hit.
SUCCESS_RADIUS = 0.05


class ReachEnv(RobotEnv):
    """3-DoF arm reaching a target sampled in its workspace."""

    task_key = "reach"
    frame_skip = 5   # 0.004s timestep -> 50 Hz control
    max_steps = 100  # 2 seconds per episode

    # [cos(q)x3, sin(q)x3, qvel*0.1 x3, ee*s x3, target*s x3, (target-ee)*s x3]
    obs_layout = "reach.v1"

    #: Shoulder sits atop the base plinth; targets are sampled around it.
    _shoulder = np.array([0.0, 0.0, 0.08])

    #: Positions are order-0.3 m while the joint cos/sin features are order 1.
    #: Left unscaled, the target signal is the quietest input to the network and
    #: the policy collapses to a single posture that ignores it.
    pos_scale = 3.0

    def _setup(self) -> None:
        self._mocap_id = self.model.body("target").mocapid[0]
        self._reach = self.params["upper_length"] + self.params["fore_length"]

    # ---- task definition ------------------------------------------------

    def _sample_target(self) -> np.ndarray:
        """A point the arm can plausibly get to: not folded in on itself, not
        beyond full extension, and never below the floor."""
        reach = self._reach
        for _ in range(20):
            radius = self._rng.uniform(0.40, 0.85) * reach
            azimuth = self._rng.uniform(-np.pi, np.pi)
            # Bias upward: targets under the base are unreachable through the floor.
            elevation = self._rng.uniform(-0.35, 1.15)
            offset = radius * np.array([
                np.cos(elevation) * np.cos(azimuth),
                np.cos(elevation) * np.sin(azimuth),
                np.sin(elevation),
            ])
            target = self._shoulder + offset
            if target[2] > 0.07:
                return target
        # Fall back to a point we know is always valid rather than looping forever.
        return self._shoulder + np.array([0.5 * reach, 0.0, 0.5 * reach])

    def _randomize(self) -> None:
        # Small joint jitter so the policy can't memorize one opening move.
        self.data.qpos[:] = self._rng.uniform(-0.15, 0.15, self.model.nq)
        self.data.qvel[:] = 0.0
        self.data.mocap_pos[self._mocap_id] = self._sample_target()

    @property
    def target(self) -> np.ndarray:
        return self.data.mocap_pos[self._mocap_id].copy()

    def _distance(self) -> float:
        return float(np.linalg.norm(self.site_xpos("ee") - self.target))

    def _observe(self) -> np.ndarray:
        ee = self.site_xpos("ee")
        target = self.target
        return np.concatenate([
            np.cos(self.data.qpos),   # joint angles, wrap-safe
            np.sin(self.data.qpos),
            self.data.qvel * 0.1,     # scaled into roughly the same range as the rest
            ee * self.pos_scale,
            target * self.pos_scale,
            (target - ee) * self.pos_scale,  # the error vector, handed over explicitly
        ]).astype(np.float32)

    def _reward(self, action):
        dist = self._distance()
        success = dist < SUCCESS_RADIUS
        reward = (
            -dist                                  # dense shaping: always know which way is better
            + (1.0 if success else 0.0)            # crisp bonus for actually arriving
            - 0.01 * float(np.square(action).sum())  # don't thrash the motors
        )
        return reward, {"distance": dist, "is_success": success}
