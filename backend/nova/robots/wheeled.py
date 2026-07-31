"""Parametric two-wheeled rover with a front caster.

Differential drive: two independently-motored wheels at the back, one low-friction
skid at the front. Trains fast, and its failure modes (spinning in place, tipping
forward under hard acceleration) are legible to a beginner.
"""

from __future__ import annotations

from .spec import ParamSpec, RobotTemplate

PARAMS = (
    ParamSpec("chassis_length", "Chassis length", 0.28, 0.15, 0.50, 0.01, "m"),
    ParamSpec("chassis_width", "Chassis width", 0.18, 0.10, 0.35, 0.01, "m",
              "Wheels sit outboard of this, so wider means a more stable track."),
    ParamSpec("wheel_radius", "Wheel radius", 0.060, 0.030, 0.120, 0.005, "m",
              "Bigger wheels travel further per turn but need more torque."),
    ParamSpec("wheel_width", "Wheel width", 0.024, 0.010, 0.050, 0.002, "m"),
    ParamSpec("gear", "Motor strength", 3.0, 0.5, 12.0, 0.25, "N·m",
              "Peak wheel torque. Too much just spins the wheels.",
              group="Actuation"),
    ParamSpec("wheel_friction", "Tyre grip", 1.2, 0.20, 2.50, 0.05, "",
              "Sliding friction of the tyres against the ground.",
              group="Actuation"),
)

_TEMPLATE = """<mujoco model="wheeled_bot">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.005" integrator="implicitfast" gravity="0 0 -9.81"/>
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.35 0.35 0.35" specular="0.1 0.1 0.1"/>
    <global azimuth="130" elevation="-20"/>
  </visual>

  <default>
    <geom rgba="0.55 0.60 0.70 1"/>
    <motor ctrlrange="-1 1" ctrllimited="true"/>
  </default>

  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" directional="true"/>
    <geom name="floor" type="plane" size="40 40 0.05" rgba="0.20 0.22 0.26 1"/>

    <body name="chassis" pos="0 0 {wheel_r:.4f}">
      <freejoint name="root"/>
      <geom name="chassis_geom" type="box"
            size="{half_len:.4f} {half_wid:.4f} 0.030" rgba="0.30 0.34 0.42 1"
            friction="0.4 0.05 0.01"/>
      <site name="imu" pos="0 0 0" size="0.01"/>

      <geom name="caster" type="sphere" size="{caster_r:.4f}"
            pos="{caster_x:.4f} 0 {caster_z:.4f}" rgba="0.25 0.27 0.32 1"
            friction="0.02 0.005 0.0001"/>

      <body name="left_wheel" pos="{wheel_x:.4f} {wheel_y:.4f} 0">
        <joint name="left_wheel_j" type="hinge" axis="0 1 0" damping="0.02"/>
        <geom name="left_wheel_geom" type="cylinder" zaxis="0 1 0"
              size="{wheel_r:.4f} {half_wheel_w:.4f}" rgba="0.18 0.19 0.22 1"
              friction="{friction:.4f} 0.02 0.001"/>
      </body>

      <body name="right_wheel" pos="{wheel_x:.4f} -{wheel_y:.4f} 0">
        <joint name="right_wheel_j" type="hinge" axis="0 1 0" damping="0.02"/>
        <geom name="right_wheel_geom" type="cylinder" zaxis="0 1 0"
              size="{wheel_r:.4f} {half_wheel_w:.4f}" rgba="0.18 0.19 0.22 1"
              friction="{friction:.4f} 0.02 0.001"/>
      </body>
    </body>
  </worldbody>

  <actuator>
    <motor name="left_m" joint="left_wheel_j" gear="{gear:.4f}"/>
    <motor name="right_m" joint="right_wheel_j" gear="{gear:.4f}"/>
  </actuator>
</mujoco>
"""


def build(params: dict) -> str:
    half_len = params["chassis_length"] / 2.0
    half_wid = params["chassis_width"] / 2.0
    wheel_r = params["wheel_radius"]
    half_wheel_w = params["wheel_width"] / 2.0
    # A small skid, sized so it clears the ground by nothing at all: its bottom
    # sits exactly on z=0 when the wheels do.
    caster_r = max(0.015, wheel_r * 0.35)

    return _TEMPLATE.format(
        half_len=half_len,
        half_wid=half_wid,
        wheel_r=wheel_r,
        half_wheel_w=half_wheel_w,
        caster_r=caster_r,
        caster_x=half_len * 0.75,
        caster_z=caster_r - wheel_r,
        # Drive wheels sit at the back, just outboard of the chassis shell.
        wheel_x=-half_len * 0.55,
        wheel_y=half_wid + half_wheel_w + 0.005,
        friction=params["wheel_friction"],
        gear=params["gear"],
    )


def standing_height(params: dict) -> float:
    return params["wheel_radius"]


TEMPLATE = RobotTemplate(
    key="wheeled_bot",
    name="Wheeled Rover",
    description=(
        "Differential-drive rover with two motors. The gentlest introduction - "
        "it learns to drive straight almost immediately."
    ),
    params=PARAMS,
    builder=build,
    tasks=("locomotion",),
    standing_height=standing_height,
    camera={"distance": 2.0, "elevation": -18, "azimuth": 130},
)
