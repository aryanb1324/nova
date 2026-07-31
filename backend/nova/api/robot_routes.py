"""Upload your own robot.

Unlike an uploaded *policy*, an uploaded robot is not executed — it is XML
describing shapes and joints, compiled by MuJoCo on the server. The risks are
therefore different: not "what will this code do" but "can this be drawn, can it
be trained, and will it wedge the simulator". The contract in
`nova/robots/contract.py` checks exactly those.
"""

from __future__ import annotations

import mujoco
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from .. import robots
from ..envs import scene as scene_mod
from ..robots import contract, uploads

router = APIRouter()

MAX_UPLOAD_CHARS = contract.MAX_XML_CHARS


class RobotPayload(BaseModel):
    xml: str
    name: str = Field(default="", max_length=80)
    description: str = Field(default="", max_length=400)


def _preview(xml: str) -> dict:
    """Scene plus a rest pose, so the viewer can show it before it is saved."""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return {
        "scene": scene_mod.describe(model),
        "frame": scene_mod.frame(data).tolist(),
        "camera": uploads.camera_for(xml),
    }


# Declared before /{robot_key} so the literal path wins the match.
@router.get("/api/robots/requirements")
def robot_requirements(task: str | None = None) -> dict:
    """What a custom robot must satisfy to be rendered and trained."""
    return contract.requirements(task)


@router.post("/api/robots/validate")
def validate_robot(payload: RobotPayload) -> dict:
    """Dry run: check a model without storing it.

    Cheap enough to call while someone is still editing, which is the point -
    they should see what's wrong before they commit to it.
    """
    if len(payload.xml) > MAX_UPLOAD_CHARS:
        raise HTTPException(413, f"model exceeds {MAX_UPLOAD_CHARS:,} characters")
    if not payload.xml.strip():
        raise HTTPException(422, "no model supplied")

    report = uploads.validate(payload.xml)
    result = report.to_dict()
    if report.ok:
        try:
            result["preview"] = _preview(payload.xml)
        except Exception as exc:  # noqa: BLE001
            # Passed the contract but won't render: report it rather than 500.
            result["ok"] = False
            result["errors"] = [*result["errors"], f"could not build a preview: {exc}"]
    return result


@router.post("/api/robots")
def upload_robot(payload: RobotPayload) -> dict:
    if len(payload.xml) > MAX_UPLOAD_CHARS:
        raise HTTPException(413, f"model exceeds {MAX_UPLOAD_CHARS:,} characters")

    name = payload.name.strip() or "my robot"
    key = uploads.make_key(name)
    if key in robots.TEMPLATES and robots.get(key).source != "upload":
        raise HTTPException(409, f"{name!r} collides with a built-in robot")

    try:
        meta = uploads.save(name, payload.xml, payload.description)
    except uploads.UploadRejected as exc:
        raise HTTPException(422, detail=exc.report.to_dict()) from None

    robots.register(
        uploads.build_template(meta["key"], meta["name"], payload.xml,
                               tuple(meta["tasks"]), meta["description"]),
        replace=True,
    )
    return {**meta, "preview": _preview(payload.xml)}


@router.get("/api/robots")
def list_robots() -> dict:
    """Uploaded robots only. Built-ins come from /api/catalog."""
    return {"robots": [
        {k: v for k, v in meta.items() if k != "xml"} for meta in uploads.listing()
    ]}


@router.get("/api/robots/{robot_key}")
def get_robot(robot_key: str) -> dict:
    try:
        meta = uploads.load(robot_key)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None
    if meta is None:
        raise HTTPException(404, f"no uploaded robot {robot_key!r}")
    return meta


@router.delete("/api/robots/{robot_key}")
def delete_robot(robot_key: str) -> dict:
    try:
        existing = robots.TEMPLATES.get(robot_key)
        if existing is not None and existing.source != "upload":
            raise HTTPException(403, "built-in robots cannot be deleted")
        ok = uploads.delete(robot_key)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None
    if not ok:
        raise HTTPException(404, f"no uploaded robot {robot_key!r}")
    return {"deleted": robot_key}
