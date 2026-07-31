"""Parametric 3-DoF reaching arm.

This is the MVP demo robot: it trains to a visibly competent reach in well under
a minute on CPU, which is what makes the hit-train-and-watch loop feel live.
"""

from __future__ import annotations

from .spec import ParamSpec, RobotTemplate

PARAMS = (
    ParamSpec("upper_length", "Upper arm length", 0.28, 0.15, 0.45, 0.01, "m",
              "Longer reaches further but is harder to control precisely."),
    ParamSpec("fore_length", "Forearm length", 0.24, 0.12, 0.40, 0.01, "m",
              "The second link, from elbow to fingertip."),
    ParamSpec("link_radius", "Link thickness", 0.030, 0.015, 0.050, 0.002, "m",
              "Thicker links are heavier, so they need more torque to swing."),
    ParamSpec("gear", "Actuator strength", 8.0, 2.0, 20.0, 0.5, "N·m",
              "Peak motor torque. Too little and the arm sags; too much and it flails.",
              group="Actuation"),
    ParamSpec("joint_damping", "Joint damping", 0.8, 0.05, 3.0, 0.05, "N·m·s",
              "Resistance to motion. Higher is steadier but slower.",
              group="Actuation"),
    ParamSpec("elbow_limit", "Elbow range", 2.6, 1.0, 3.0, 0.05, "rad",
              "How far the elbow can bend in each direction.",
              group="Joints"),
)

_TEMPLATE = """<mujoco model="reach_arm">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.004" integrator="implicitfast" gravity="0 0 -9.81"/>
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.35 0.35 0.35" specular="0.1 0.1 0.1"/>
    <global azimuth="130" elevation="-20"/>
  </visual>

  <default>
    <joint damping="{damping:.4f}" armature="0.05"/>
    <geom rgba="0.55 0.60 0.70 1" friction="1 0.05 0.01"/>
    <motor ctrlrange="-1 1" ctrllimited="true"/>
  </default>

  <worldbody>
    <light pos="0.6 -0.6 1.6" dir="-0.3 0.3 -1" directional="true"/>
    <geom name="floor" type="plane" size="1.5 1.5 0.05" rgba="0.20 0.22 0.26 1"/>

    <body name="base" pos="0 0 0.04">
      <geom name="base_geom" type="cylinder" size="0.070 0.040" rgba="0.30 0.32 0.38 1"/>

      <body name="upper" pos="0 0 0.04">
        <joint name="pan" type="hinge" axis="0 0 1" range="-3.1416 3.1416"/>
        <joint name="shoulder" type="hinge" axis="0 1 0" range="-2.2 2.2"/>
        <geom name="upper_geom" type="capsule"
              fromto="0 0 0 0 0 {upper:.4f}" size="{radius:.4f}"/>

        <body name="fore" pos="0 0 {upper:.4f}">
          <joint name="elbow" type="hinge" axis="0 1 0"
                 range="-{elbow:.4f} {elbow:.4f}"/>
          <geom name="fore_geom" type="capsule"
                fromto="0 0 0 0 0 {fore:.4f}" size="{fore_radius:.4f}"/>
          <site name="ee" pos="0 0 {fore:.4f}" size="0.025" rgba="0.95 0.35 0.20 1"/>
          <!-- Visual fingertip. Sites aren't exported to the viewer, and mass=0
               keeps it from perturbing the dynamics the policy was tuned on. -->
          <geom name="tip" type="sphere" size="0.028" pos="0 0 {fore:.4f}"
                rgba="0.95 0.35 0.20 1" contype="0" conaffinity="0" mass="0"/>
        </body>
      </body>
    </body>

    <body name="target" mocap="true" pos="0.30 0 0.35">
      <geom name="target_geom" type="sphere" size="0.035" rgba="0.20 0.85 0.45 0.55"
            contype="0" conaffinity="0"/>
    </body>
  </worldbody>

  <actuator>
    <motor name="pan_m" joint="pan" gear="{gear:.4f}"/>
    <motor name="shoulder_m" joint="shoulder" gear="{gear:.4f}"/>
    <motor name="elbow_m" joint="elbow" gear="{elbow_gear:.4f}"/>
  </actuator>
</mujoco>
"""


def build(params: dict) -> str:
    upper = params["upper_length"]
    fore = params["fore_length"]
    radius = params["link_radius"]
    gear = params["gear"]
    return _TEMPLATE.format(
        upper=upper,
        fore=fore,
        radius=radius,
        # The forearm is slightly slimmer so the arm reads as tapered rather than
        # as two identical sticks.
        fore_radius=radius * 0.85,
        gear=gear,
        # The elbow carries less inertia than the shoulder, so it needs less torque.
        elbow_gear=gear * 0.7,
        damping=params["joint_damping"],
        elbow=params["elbow_limit"],
    )


def reach_radius(params: dict) -> float:
    """Fully-extended tip distance from the shoulder. Used to place targets."""
    return params["upper_length"] + params["fore_length"]


TEMPLATE = RobotTemplate(
    key="reach_arm",
    name="Reaching Arm",
    description=(
        "A 3-joint arm bolted to the floor. Trains fastest of all the templates - "
        "a good first run."
    ),
    params=PARAMS,
    builder=build,
    tasks=("reach",),
    camera={"distance": 1.5, "elevation": -18, "azimuth": 130},
)
