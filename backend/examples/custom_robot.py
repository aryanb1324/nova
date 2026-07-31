"""Add your own robot and your own task, from outside the package.

Nothing here edits NOVA. This file registers a two-joint pendulum and a
"swing-up" task, and from that moment they behave exactly like the built-ins:
they appear in the API catalog, the browser renders sliders for the parameters
declared below, and every trainer works against them unchanged.

Try it standalone:

    python examples/custom_robot.py

Or load it into the app, so the robot shows up in the picker:

    NOVA_EXTENSIONS=examples.custom_robot ./dev.sh
"""

from __future__ import annotations

import numpy as np

import nova

# ---------------------------------------------------------------------------
# 1. The robot — a function from parameters to MJCF
# ---------------------------------------------------------------------------

PARAMS = (
    nova.ParamSpec("pole_length", "Pole length", 0.5, 0.2, 1.0, 0.05, "m",
                   "Longer poles swing slower and are harder to catch."),
    nova.ParamSpec("pole_radius", "Pole thickness", 0.03, 0.015, 0.06, 0.005, "m"),
    nova.ParamSpec("gear", "Motor strength", 3.0, 0.5, 12.0, 0.25, "N·m",
                   "Deliberately weak by default — too weak to lift the pole "
                   "directly, so it has to be swung.",
                   group="Actuation"),
    nova.ParamSpec("damping", "Joint damping", 0.05, 0.0, 1.0, 0.01, "N·m·s",
                   group="Actuation"),
)

_MJCF = """<mujoco model="pendulum">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.004" integrator="implicitfast" gravity="0 0 -9.81"/>
  <visual><headlight diffuse="0.6 0.6 0.6" ambient="0.35 0.35 0.35"/></visual>
  <default>
    <joint damping="{damping:.4f}" armature="0.01"/>
    <geom rgba="0.55 0.60 0.70 1"/>
    <motor ctrlrange="-1 1" ctrllimited="true"/>
  </default>
  <worldbody>
    <light pos="0 0 2" dir="0 0 -1" directional="true"/>
    <geom name="floor" type="plane" size="3 3 0.05" rgba="0.20 0.22 0.26 1"/>
    <body name="post" pos="0 0 {hub:.4f}">
      <geom name="post_geom" type="cylinder" size="0.04 {hub:.4f}"
            pos="0 0 -{half_hub:.4f}" rgba="0.30 0.32 0.38 1"/>
      <body name="pole" pos="0 0 0">
        <joint name="hinge" type="hinge" axis="0 1 0"/>
        <geom name="pole_geom" type="capsule"
              fromto="0 0 0 0 0 {length:.4f}" size="{radius:.4f}"/>
        <site name="tip" pos="0 0 {length:.4f}" size="0.03" rgba="0.95 0.35 0.2 1"/>
      </body>
    </body>
  </worldbody>
  <actuator><motor name="hinge_m" joint="hinge" gear="{gear:.4f}"/></actuator>
</mujoco>
"""


def build(params: dict) -> str:
    hub = params["pole_length"] + 0.15   # clearance so the pole can hang down
    return _MJCF.format(
        length=params["pole_length"],
        radius=params["pole_radius"],
        gear=params["gear"],
        damping=params["damping"],
        hub=hub,
        half_hub=hub / 2,
    )


PENDULUM = nova.RobotTemplate(
    key="pendulum",
    name="Pendulum",
    description="One weak motor at the hub. Has to swing the pole up, not lift it.",
    params=PARAMS,
    builder=build,
    tasks=("swingup",),
    camera={"distance": 2.2, "elevation": -10, "azimuth": 90},
)


# ---------------------------------------------------------------------------
# 2. The task — a RobotEnv subclass
# ---------------------------------------------------------------------------

class SwingUpEnv(nova.RobotEnv):
    """Get the pole upright and keep it there."""

    frame_skip = 5    # 50 Hz control
    max_steps = 200   # 4 seconds
    # Version the observation. If you change _observe(), bump this, and every
    # policy exported against the old layout is rejected instead of misbehaving.
    obs_layout = "swingup.v1"

    def _setup(self) -> None:
        self._length = self.params["pole_length"]

    def _randomize(self) -> None:
        # Start hanging down, give or take — otherwise it never learns to swing.
        self.data.qpos[0] = np.pi + self._rng.uniform(-0.3, 0.3)
        self.data.qvel[0] = self._rng.uniform(-0.5, 0.5)

    def _observe(self) -> np.ndarray:
        angle = float(self.data.qpos[0])
        return np.array([
            np.cos(angle), np.sin(angle), float(self.data.qvel[0]) * 0.1,
            self.site_xpos("tip")[2] / max(self._length, 1e-6),
        ], dtype=np.float32)

    def _reward(self, action):
        # Height of the tip above the hub, normalized: +1 upright, -1 hanging.
        uprightness = float(np.cos(self.data.qpos[0]))
        spin = abs(float(self.data.qvel[0]))
        reward = uprightness - 0.01 * spin - 0.001 * float(np.square(action).sum())
        # No is_success here, deliberately. A task does not need a binary metric,
        # and forcing one on this one was actively misleading. PPO reaches ~0.89
        # mean uprightness at 300k steps: it swings up and stays up most of the
        # time without ever fully settling, so a pass/fail line read from the
        # final step reported 0% and looked like total failure. The continuous
        # number describes what the policy actually does.
        return reward, {"uprightness": uprightness}


nova.register_template(PENDULUM, replace=True)
nova.register_task(
    "swingup",
    env=SwingUpEnv,
    name="Swing up",
    description="Swing the pole upright and balance it there.",
    success_metric="uprightness",
    monitor_keys=("uprightness",),
    typical_steps=300_000,
    # State the interface, or an uploaded robot with no free joint would look
    # compatible with swing-up even though _observe() reads a site it lacks.
    requires={
        "sites": ("tip",),
        "free_joints": (0, 0),
        "note": "needs a site named 'tip' at the end of the pole, on a fixed base",
    },
    replace=True,
)


# ---------------------------------------------------------------------------
# 3. Use it like anything else
# ---------------------------------------------------------------------------

def main() -> int:
    from nova.training import TrainConfig, train

    print("registered robots:", [t["key"] for t in nova.templates()])
    print("registered tasks :", [t["key"] for t in nova.tasks()])

    env = nova.make("swingup", "pendulum")
    print(f"observations {env.observation_space.shape[0]}  "
          f"actions {env.action_space.shape[0]}")

    steps = 300_000
    with nova.attach("swingup", "pendulum", algo="PPO (built-in)",
                     total_steps=steps) as run:
        def on_event(ev: dict) -> None:
            if ev["type"] in ("progress", "rollout"):
                run.send(ev)
            if ev["type"] == "progress" and ev["iteration"] % 10 == 0:
                print(f"  {ev['step']:>7,}/{steps:,}  reward={ev.get('mean_reward', 0):>7.2f}"
                      f"  uprightness={ev.get('uprightness', 0):>5.2f}")

        result = train(
            TrainConfig(task="swingup", template="pendulum",
                        total_steps=steps, rollout_every=10),
            on_event=on_event,
        )

    nova.export(result["model"], "pendulum.onnx", task="swingup", template="pendulum",
                algo="PPO (built-in)")
    scores = nova.evaluate("pendulum.onnx", episodes=30)
    scores.pop("rollout", None)
    print(f"\npublic   : {scores['public']}")
    print(f"held_out : {scores['held_out']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
