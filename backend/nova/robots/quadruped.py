"""Parametric quadruped: box torso, four 2-joint legs.

Legs are generated in Python rather than spelled out in the XML template so that
changing the topology later means editing one loop, not four near-identical
blocks that drift apart.
"""

from __future__ import annotations

import math

from .spec import ParamSpec, RobotTemplate

#: Joint travel, measured from a straight leg.
HIP_RANGE = (-1.2, 1.2)
KNEE_RANGE = (-2.4, 0.2)
#: Thigh joints hang this far below the torso's centre.
HIP_DROP = 0.030

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

_LEG = """      <body name="{n}_thigh" pos="{x:.4f} {y:.4f} -{hip_drop:.4f}">
        <joint name="{n}_hip" type="hinge" axis="0 1 0" range="{hip_lo:.3f} {hip_hi:.3f}"
               ref="{hip:.5f}"/>
        <geom name="{n}_thigh_geom" type="capsule"
              fromto="0 0 0 {thigh_x:.4f} 0 {thigh_tip:.4f}" size="{radius:.4f}"/>
        <body name="{n}_shin" pos="{thigh_x:.4f} 0 {thigh_tip:.4f}">
          <joint name="{n}_knee" type="hinge" axis="0 1 0" range="{knee_lo:.3f} {knee_hi:.3f}"
                 ref="{knee:.5f}"/>
          <geom name="{n}_shin_geom" type="capsule"
                fromto="0 0 0 {shin_x:.4f} 0 {shin_tip:.4f}" size="{shin_radius:.4f}"/>
          <site name="{n}_foot" pos="{shin_x:.4f} 0 {shin_tip:.4f}" size="0.012"/>
        </body>
      </body>
"""


def _stance(params: dict) -> dict:
    """The pose the robot spawns in, and the torso height that goes with it.

    The legs are folded so the feet rest *on* the floor. Spawning with them
    straight put the feet 8.5 cm underground at the default sizes, and the
    contact solver answered by launching the robot at ~1.9 m/s - every episode
    began with a somersault the policy had to recover from before it could even
    try to walk.

    `ref` keeps the joint coordinates measured from a straight leg, so the
    ranges above still mean what they say and the knee retains its travel
    towards extension - which is the half a gait pushes off with.
    """
    upper, lower = params["upper_leg"], params["lower_leg"]
    foot_r = params["leg_radius"] * 0.85

    # Fold to the height the sliders have always described, less the foot's own
    # radius so the capsule's underside lands on z=0 rather than its centreline.
    reach = 0.72 * (upper + lower) + 0.01 - foot_r
    # Two links cannot span more than their sum nor less than their difference;
    # the sliders can ask for either, so clamp before trusting the geometry.
    reach = min(max(reach, abs(upper - lower) + 1e-4), upper + lower - 1e-4)

    cos_knee = (reach ** 2 - upper ** 2 - lower ** 2) / (2 * upper * lower)
    knee = -math.acos(min(1.0, max(-1.0, cos_knee)))
    knee = min(max(knee, KNEE_RANGE[0]), KNEE_RANGE[1])
    # Hip angle that puts the foot directly below the joint it hangs from.
    hip = -math.atan2(lower * math.sin(knee), upper + lower * math.cos(knee))
    hip = min(max(hip, HIP_RANGE[0]), HIP_RANGE[1])

    # Forward kinematics of whatever survived the clamps, so the spawn height
    # matches the pose exactly even at the extremes of the sliders.
    thigh_x, thigh_z = -upper * math.sin(hip), upper * math.cos(hip)
    shin_x = -lower * math.sin(hip + knee)
    shin_z = lower * math.cos(hip + knee)
    return {
        "hip": hip, "knee": knee,
        # Tip positions are stored already negated: a clamped pose can put a
        # link's tip above its own joint, and "-{z}" in the template would then
        # emit "--0.01".
        "thigh_x": thigh_x, "thigh_tip": -thigh_z,
        "shin_x": shin_x, "shin_tip": -shin_z,
        "height": HIP_DROP + thigh_z + shin_z + foot_r,
    }


def build(params: dict) -> str:
    half_len = params["torso_length"] / 2.0
    half_wid = params["torso_width"] / 2.0
    radius = params["leg_radius"]
    gear = params["gear"]
    pose = _stance(params)

    legs, actuators = [], []
    for name, sx, sy in _LEGS:
        legs.append(_LEG.format(
            n=name,
            x=sx * half_len * 0.85,
            y=sy * half_wid,
            hip_drop=HIP_DROP,
            hip_lo=HIP_RANGE[0], hip_hi=HIP_RANGE[1],
            knee_lo=KNEE_RANGE[0], knee_hi=KNEE_RANGE[1],
            radius=radius,
            shin_radius=radius * 0.85,
            **{k: v for k, v in pose.items() if k != "height"},
        ))
        actuators.append(f'    <motor name="{name}_hip_m" joint="{name}_hip" gear="{gear:.4f}"/>\n')
        actuators.append(f'    <motor name="{name}_knee_m" joint="{name}_knee" gear="{gear * 0.8:.4f}"/>\n')

    return _SHELL.format(
        damping=params["joint_damping"],
        start_height=pose["height"],
        half_len=half_len,
        half_wid=half_wid,
        legs="".join(legs),
        actuators="".join(actuators),
    )


def standing_height(params: dict) -> float:
    return _stance(params)["height"]


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
