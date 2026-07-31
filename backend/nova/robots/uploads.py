"""Robots people upload, as opposed to the ones this package generates.

An upload is fixed geometry — a single MJCF file, no parameters and therefore no
sliders. Everything else about it is identical to a built-in: it goes in the same
registry, renders through the same pose-streaming path, and trains with the same
code. The task it can be trained on is *detected*, not declared, by checking the
model against each task's interface.
"""

from __future__ import annotations

import json
import pathlib
import re
import time

import mujoco

from .contract import Report, validate_mjcf
from .spec import RobotTemplate

UPLOADS_DIR = pathlib.Path(__file__).resolve().parents[2] / "user_robots"

_SAFE = re.compile(r"[^a-z0-9_]+")
#: Uploaded keys are prefixed so they can never shadow a built-in or an
#: extension, whatever the user calls their robot.
KEY_PREFIX = "up_"


class UploadRejected(ValueError):
    """The model failed the contract; the report says why."""

    def __init__(self, report: Report):
        self.report = report
        super().__init__("; ".join(report.errors) or "invalid robot")


def make_key(name: str) -> str:
    slug = _SAFE.sub("_", name.strip().lower()).strip("_") or "robot"
    return f"{KEY_PREFIX}{slug[:40]}"


def _dir(key: str) -> pathlib.Path:
    path = (UPLOADS_DIR / key).resolve()
    if path.parent != UPLOADS_DIR.resolve():
        raise ValueError(f"invalid robot key {key!r}")
    return path


def spawn_height(xml: str) -> float:
    """Where the root sits at rest, read off the compiled model.

    Locomotion needs this to tell "fell over" from "standing". A built-in
    computes it from its parameters; an upload has none, so take it from the
    model the author actually wrote.
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    free = [
        j for j in range(model.njnt)
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
    ]
    if not free:
        return 0.0
    root = int(model.jnt_bodyid[free[0]])
    return float(data.xpos[root][2])


def camera_for(xml: str) -> dict:
    """Frame the robot by its actual size.

    A built-in knows how big it is; an upload could be a 20 cm gripper or a 3 m
    gantry. Guessing one distance crops the big ones and strands the small ones,
    so measure the model's extent and back off from that.
    """
    import numpy as np

    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    reach = 0.0
    for i in range(model.ngeom):
        # Skip the ground plane; its half-extent is the world, not the robot.
        if mujoco.mjtGeom(model.geom_type[i]) == mujoco.mjtGeom.mjGEOM_PLANE:
            continue
        centre = data.geom_xpos[i]
        radius = float(np.max(model.geom_size[i]))
        reach = max(reach, float(np.linalg.norm(centre)) + radius)

    return {
        "distance": round(max(0.6, reach * 3.0), 3),
        "elevation": -18,
        "azimuth": 130,
    }


def build_template(key: str, name: str, xml: str, tasks: tuple[str, ...],
                   description: str = "") -> RobotTemplate:
    height = spawn_height(xml)
    return RobotTemplate(
        key=key,
        name=name,
        description=description or "Uploaded robot (fixed geometry).",
        params=(),                      # fixed geometry: nothing to tune
        builder=lambda _params: xml,    # same XML whatever it's called with
        tasks=tasks,
        camera=camera_for(xml),
        standing_height=(lambda _params, h=height: h) if height else None,
        source="upload",
    )


def validate(xml: str) -> Report:
    """Full contract check, including which tasks the model can do."""
    return validate_mjcf(xml)


def save(name: str, xml: str, description: str = "") -> dict:
    """Validate, persist, and return the record. Raises UploadRejected."""
    report = validate(xml)
    if not report.ok:
        raise UploadRejected(report)

    key = make_key(name)
    tasks = tuple(report.info.get("compatible_tasks", ()))
    out = _dir(key)
    out.mkdir(parents=True, exist_ok=True)
    (out / "robot.xml").write_text(xml)
    meta = {
        "key": key,
        "name": name.strip() or key,
        "description": description.strip(),
        "tasks": list(tasks),
        "uploaded": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "info": report.info,
        "warnings": report.warnings,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def load(key: str) -> dict | None:
    path = _dir(key)
    meta_path = path / "meta.json"
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text())
    meta["xml"] = (path / "robot.xml").read_text()
    return meta


def listing() -> list[dict]:
    if not UPLOADS_DIR.exists():
        return []
    out = []
    for child in sorted(UPLOADS_DIR.iterdir()):
        if not child.is_dir():
            continue
        meta_path = child / "meta.json"
        if meta_path.exists():
            out.append(json.loads(meta_path.read_text()))
    return out


def delete(key: str) -> bool:
    import shutil

    from . import unregister

    path = _dir(key)
    if not path.exists():
        return False
    shutil.rmtree(path)
    unregister(key)
    return True


def register_stored() -> list[str]:
    """Re-register every saved upload. Called at server startup.

    A stored robot that no longer compiles is skipped rather than fatal - the
    contract may have tightened since it was uploaded, and one bad file should
    not stop the server.
    """
    from . import register

    loaded = []
    for meta in listing():
        try:
            xml = (_dir(meta["key"]) / "robot.xml").read_text()
            register(
                build_template(meta["key"], meta["name"], xml,
                               tuple(meta.get("tasks", ())),
                               meta.get("description", "")),
                replace=True,
            )
            loaded.append(meta["key"])
        except Exception as exc:  # noqa: BLE001
            print(f"[NOVA] skipping stored robot {meta.get('key')!r}: "
                  f"{type(exc).__name__}: {exc}")
    return loaded
