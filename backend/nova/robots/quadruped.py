"""Parametric quadruped: box torso, four 2-joint legs.

Legs are generated in Python rather than spelled out in the XML template so that
changing the topology later means editing one loop, not four near-identical
blocks that drift apart.
"""

from __future__ import annotations

from .spec import ParamSpec, RobotTemplate

PARAMS = (
    ParamSpec("torso_length", "Torso length", 0.30, 0.18, 0.50, 0.01, "m",
              "Longer torsos are more stable fore-aft but harder to turn."),
    ParamSpec("torso_width", "Torso width", 0.16, 0.08, 0.30, 0.01, "m",
              "Wider stance resists rolling over sideways."),
    ParamSpec("upper_leg", "Thigh length", 0.16, 0.08, 0.28, 0.01, "m"),
    ParamSpec("lower_leg", "Shin length", 0.18, 0.08, 0.30, 0.01, "m"),
    ParamSpec("leg_radius", "Leg thickness", 0.018, 0.010, 0.035, 0.002, "m"),
    ParamSpec("gear", "Actuator strength", 18.0, 5.0, 45.0, 1.0, "N·m",
              "Peak joint torque. Weak legs collapse; strong legs launch the robot.",
              group="Actuation"),
    ParamSpec("joint_damping", "Joint damping", 0.5, 0.05, 3.0, 0.05, "N·m·s",
              group="Actuation"),
)

_LEGS = (
    # name, x-sign, y-sign
    ("fl", 1, 1),
    ("fr", 1, -1),
    ("bl", -1, 1),
    ("br", -1, -1),
)

_SHELL = """<mujoco model="quadruped">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.005" integrator="implicitfast" gravity="0 0 -9.81"/>
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.35 0.35 0.35" specular="0.1 0.1 0.1"/>
    <global azimuth="130" elevation="-20"/>
  </visual>

  <default>
    <joint damping="{damping:.4f}" armature="0.01"/>
    <geom rgba="0.55 0.60 0.70 1" friction="0.9 0.1 0.1"/>
    <motor ctrlrange="-1 1" ctrllimited="true"/>
  </default>

  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" directional="true"/>
    <geom name="floor" type="plane" size="40 40 0.05" rgba="0.20 0.22 0.26 1"/>

    <body name="torso" pos="0 0 {start_height:.4f}">
      <freejoint name="root"/>
      <geom name="torso_geom" type="box"
            size="{half_len:.4f} {half_wid:.4f} 0.040" rgba="0.30 0.34 0.42 1"/>
      <site name="imu" pos="0 0 0" size="0.01"/>
{legs}    </body>
  </worldbody>

  <actuator>
{actuators}  </actuator>
</mujoco>
"""

_LEG = """      <body name="{n}_thigh" pos="{x:.4f} {y:.4f} -0.030">
        <joint name="{n}_hip" type="hinge" axis="0 1 0" range="-1.2 1.2"/>
        <geom name="{n}_thigh_geom" type="capsule"
              fromto="0 0 0 0 0 -{upper:.4f}" size="{radius:.4f}"/>
        <body name="{n}_shin" pos="0 0 -{upper:.4f}">
          <joint name="{n}_knee" type="hinge" axis="0 1 0" range="-2.4 0.2"/>
          <geom name="{n}_shin_geom" type="capsule"
                fromto="0 0 0 0 0 -{lower:.4f}" size="{shin_radius:.4f}"/>
          <site name="{n}_foot" pos="0 0 -{lower:.4f}" size="0.012"/>
        </body>
      </body>
"""


def build(params: dict) -> str:
    half_len = params["torso_length"] / 2.0
    half_wid = params["torso_width"] / 2.0
    upper = params["upper_leg"]
    lower = params["lower_leg"]
    radius = params["leg_radius"]
    gear = params["gear"]

    legs, actuators = [], []
    for name, sx, sy in _LEGS:
        legs.append(_LEG.format(
            n=name,
            x=sx * half_len * 0.85,
            y=sy * half_wid,
            upper=upper,
            lower=lower,
            radius=radius,
            shin_radius=radius * 0.85,
        ))
        actuators.append(f'    <motor name="{name}_hip_m" joint="{name}_hip" gear="{gear:.4f}"/>\n')
        actuators.append(f'    <motor name="{name}_knee_m" joint="{name}_knee" gear="{gear * 0.8:.4f}"/>\n')

    return _SHELL.format(
        damping=params["joint_damping"],
        # Spawn with legs partly folded rather than locked straight, so the very
        # first contact isn't a fall from full extension.
        start_height=(upper + lower) * 0.72 + 0.04,
        half_len=half_len,
        half_wid=half_wid,
        legs="".join(legs),
        actuators="".join(actuators),
    )


def standing_height(params: dict) -> float:
    return (params["upper_leg"] + params["lower_leg"]) * 0.72 + 0.04


TEMPLATE = RobotTemplate(
    key="quadruped",
    name="Quadruped",
    description=(
        "Four legs, two joints each. The hardest of the templates to train - "
        "expect a scramble before it walks."
    ),
    params=PARAMS,
    builder=build,
    tasks=("locomotion",),
    standing_height=standing_height,
    camera={"distance": 2.4, "elevation": -18, "azimuth": 130},
)
