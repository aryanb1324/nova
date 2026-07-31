"""Turn a compiled MuJoCo model into a description the browser can draw.

The viewer never runs physics. It builds one group per body from `bodies`, hangs
primitive meshes off it, and then just sets a transform per body per frame. That
keeps MuJoCo entirely on the server and the frontend to plain Three.js.
"""

from __future__ import annotations

import mujoco
import numpy as np

_GEOM_NAMES = {
    mujoco.mjtGeom.mjGEOM_PLANE: "plane",
    mujoco.mjtGeom.mjGEOM_SPHERE: "sphere",
    mujoco.mjtGeom.mjGEOM_CAPSULE: "capsule",
    mujoco.mjtGeom.mjGEOM_ELLIPSOID: "ellipsoid",
    mujoco.mjtGeom.mjGEOM_CYLINDER: "cylinder",
    mujoco.mjtGeom.mjGEOM_BOX: "box",
}


def _geom_dict(model: mujoco.MjModel, i: int) -> dict | None:
    kind = _GEOM_NAMES.get(mujoco.mjtGeom(model.geom_type[i]))
    if kind is None:
        return None  # meshes/hfields aren't produced by our templates
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
    return {
        "name": name or f"geom{i}",
        "type": kind,
        # MuJoCo's size semantics per type: sphere [r], capsule/cylinder
        # [r, half_length], box [hx, hy, hz], plane [half_x, half_y, spacing].
        "size": [round(float(v), 5) for v in model.geom_size[i]],
        "pos": [round(float(v), 5) for v in model.geom_pos[i]],
        "quat": [round(float(v), 5) for v in model.geom_quat[i]],  # w, x, y, z
        "rgba": [round(float(v), 4) for v in model.geom_rgba[i]],
    }


def describe(model: mujoco.MjModel) -> dict:
    """Static scene graph: one entry per movable body, plus the fixed world."""
    bodies: list[dict] = []
    for b in range(1, model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b)
        geoms = [
            g for g in (
                _geom_dict(model, i)
                for i in range(model.ngeom)
                if model.geom_bodyid[i] == b
            ) if g is not None
        ]
        bodies.append({"name": name or f"body{b}", "geoms": geoms})

    static = [
        g for g in (
            _geom_dict(model, i)
            for i in range(model.ngeom)
            if model.geom_bodyid[i] == 0
        ) if g is not None
    ]

    return {
        # Index i here lines up with row i of every frame in a rollout.
        "bodies": bodies,
        "static_geoms": static,
        "up_axis": "z",
    }


def frame(data: mujoco.MjData) -> np.ndarray:
    """One frame: [x, y, z, qw, qx, qy, qz] per body, world frame."""
    return np.concatenate(
        [data.xpos[1:], data.xquat[1:]], axis=1
    ).astype(np.float32)
