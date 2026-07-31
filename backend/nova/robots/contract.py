"""What a robot has to satisfy for NOVA to render and train it.

The built-in templates are generated, so they always comply. This exists for the
next step - letting people bring their own bodies - and it is written now so
that feature is a small addition rather than a redesign.

Two separate contracts are checked here, and they fail for different reasons:

  RENDERABLE - the viewer draws primitives it knows, positioned by body pose.
      A mesh, a heightfield or an external texture has nothing to draw, because
      the browser is never sent asset files.

  TRAINABLE  - the task has an interface. `reach` needs a fingertip site and a
      movable target; `locomotion` needs a body that can travel. A model can be
      perfectly valid MuJoCo and still be unusable for a given task.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import mujoco

#: The viewer builds these from Three.js primitives. Anything else would need
#: geometry shipped to the browser, which the pose-only wire format does not do.
RENDERABLE_GEOMS = {
    mujoco.mjtGeom.mjGEOM_PLANE,
    mujoco.mjtGeom.mjGEOM_SPHERE,
    mujoco.mjtGeom.mjGEOM_CAPSULE,
    mujoco.mjtGeom.mjGEOM_ELLIPSOID,
    mujoco.mjtGeom.mjGEOM_CYLINDER,
    mujoco.mjtGeom.mjGEOM_BOX,
}

MAX_XML_CHARS = 500_000
MAX_BODIES = 48
MAX_GEOMS = 192
MAX_ACTUATORS = 32
#: Enough frames to prove the model is stable without being a denial of service.
SETTLE_STEPS = 200

#: Shape of a task's robot interface, and what it defaults to.
DEFAULT_REQUIREMENTS: dict = {
    "sites": (),
    "mocap_bodies": (),
    "free_joints": (0, 99),
    "note": "",
}


def task_requirements() -> dict[str, dict]:
    """Every registered task's robot interface, read from the live registry.

    Not a constant: tasks are registerable, so a task added by an extension has
    to be able to state what a robot must provide for it. Imported lazily
    because `envs` imports `robots`, and this lives under `robots`.
    """
    from ..envs import TASKS

    out = {}
    for key, task in TASKS.items():
        spec = dict(DEFAULT_REQUIREMENTS)
        spec.update(task.get("requires") or {})
        out[key] = spec
    return out


@dataclass
class Report:
    ok: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    info: dict = field(default_factory=dict)

    def fail(self, message: str) -> None:
        self.ok = False
        self.errors.append(message)

    def to_dict(self) -> dict:
        return {"ok": self.ok, "errors": self.errors,
                "warnings": self.warnings, "info": self.info}


def validate_mjcf(xml: str, task: str | None = None) -> Report:
    """Check a user-supplied MJCF against the render and task contracts."""
    report = Report()

    if len(xml) > MAX_XML_CHARS:
        report.fail(f"model is {len(xml):,} characters; the limit is {MAX_XML_CHARS:,}")
        return report
    # <include> pulls in a path of the author's choosing. Nothing legitimate in a
    # single-file robot needs it.
    if re.search(r"<\s*include\b", xml, re.IGNORECASE):
        report.fail("<include> is not allowed; submit a single self-contained file")
    if re.search(r"\b(file|meshdir|texturedir|assetdir)\s*=", xml, re.IGNORECASE):
        report.fail(
            "external asset references are not allowed - NOVA sends the browser "
            "poses, not files, so meshes and textures cannot be drawn"
        )

    try:
        model = mujoco.MjModel.from_xml_string(xml)
    except Exception as exc:
        report.fail(f"MuJoCo could not compile this model: {str(exc)[:300]}")
        return report

    _check_size(model, report)
    _check_renderable(model, report)
    _check_actuation(model, report)
    _check_stability(model, report)

    if task is not None:
        _check_task(model, task, report)

    usable = compatible_tasks(model)
    report.info.update({
        "bodies": int(model.nbody - 1),
        "geoms": int(model.ngeom),
        "actuators": int(model.nu),
        "dof": int(model.nv),
        "qpos": int(model.nq),
        "timestep": float(model.opt.timestep),
        "total_mass": round(float(model.body_mass.sum()), 4),
        "compatible_tasks": usable,
        # Why each unusable task is unusable, so the message is actionable
        # rather than just "unsupported".
        "task_problems": {
            t: _task_problems(model, t)
            for t in task_requirements() if t not in usable
        },
    })
    if task is None and not usable:
        report.fail(
            "this robot doesn't provide any task's interface, so there is nothing "
            "to train it on. " + "; ".join(
                f"{t}: {', '.join(p)}" for t, p in report.info["task_problems"].items()
            )
        )
    return report


def _check_size(model, report: Report) -> None:
    if model.nbody - 1 > MAX_BODIES:
        report.fail(f"{model.nbody - 1} bodies; the limit is {MAX_BODIES}")
    if model.ngeom > MAX_GEOMS:
        report.fail(f"{model.ngeom} geoms; the limit is {MAX_GEOMS}")
    if model.nu == 0:
        report.fail("the model has no actuators, so there is nothing for a policy "
                    "to control")
    elif model.nu > MAX_ACTUATORS:
        report.fail(f"{model.nu} actuators; the limit is {MAX_ACTUATORS}")


def _check_renderable(model, report: Report) -> None:
    bad = sorted({
        mujoco.mjtGeom(model.geom_type[i]).name
        for i in range(model.ngeom)
        if mujoco.mjtGeom(model.geom_type[i]) not in RENDERABLE_GEOMS
    })
    if bad:
        report.fail(
            f"uses geom types the viewer cannot draw: {', '.join(bad)}. "
            "Use plane, sphere, capsule, ellipsoid, cylinder or box."
        )


def _check_actuation(model, report: Report) -> None:
    unlimited = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or f"actuator{i}"
        for i in range(model.nu)
        if not model.actuator_ctrllimited[i]
    ]
    if unlimited:
        report.fail(
            "every actuator needs an explicit ctrlrange (policies emit actions in "
            f"[-1, 1]); missing on: {', '.join(unlimited[:5])}"
        )

    unbounded = []
    for j in range(model.njnt):
        jtype = model.jnt_type[j]
        if jtype in (mujoco.mjtJoint.mjJNT_FREE, mujoco.mjtJoint.mjJNT_BALL):
            continue
        if not model.jnt_limited[j]:
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or f"joint{j}"
            unbounded.append(name)
    if unbounded:
        # A hinge that spins forever is legal and sometimes intended - a wheel.
        report.warnings.append(
            "joints without a range will rotate without limit, which is right for "
            f"wheels and usually wrong for limbs: {', '.join(unbounded[:5])}"
        )


def _check_stability(model, report: Report) -> None:
    """A model that explodes on its own will not train."""
    import numpy as np

    data = mujoco.MjData(model)
    try:
        mujoco.mj_step(model, data, nstep=SETTLE_STEPS)
    except Exception as exc:
        report.fail(f"simulation failed while settling: {str(exc)[:200]}")
        return
    if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)):
        report.fail(
            "the model becomes numerically unstable when simulated with no control "
            "input - check masses, joint ranges and the timestep"
        )
    elif model.nv == 0:
        # Nothing can move, so there is nothing to be unstable - and nothing to
        # learn either, which _check_size already reports via the actuator count.
        report.warnings.append("the model has no degrees of freedom; nothing can move")
    elif float(np.abs(data.qvel).max()) > 500.0:
        report.warnings.append(
            "parts of the model are moving very fast at rest; it may be unstable"
        )


def _task_problems(model, task: str) -> list[str]:
    """Why this model can't do this task. Empty means it can."""
    specs = task_requirements()
    spec = specs.get(task)
    if spec is None:
        return [f"unknown task {task!r}; have {sorted(specs)}"]

    problems = []
    for site in spec["sites"]:
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site) < 0:
            problems.append(f"needs a site named {site!r}")

    for body in spec["mocap_bodies"]:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
        if bid < 0:
            problems.append(f"needs a body named {body!r}")
        elif model.body_mocapid[bid] < 0:
            problems.append(f"body {body!r} must be declared mocap=\"true\"")

    free = sum(
        1 for j in range(model.njnt)
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
    )
    low, high = spec["free_joints"]
    if not (low <= free <= high):
        expected = str(low) if low == high else f"{low}-{high}"
        problems.append(f"expects {expected} free joint(s), found {free}")
    return problems


def compatible_tasks(model) -> list[str]:
    """Which tasks this model can actually be trained on.

    Uploaders declare nothing: the model either provides a task's interface or it
    doesn't, and this is what tells them which.
    """
    return [t for t in task_requirements() if not _task_problems(model, t)]


def _check_task(model, task: str, report: Report) -> None:
    spec = task_requirements().get(task)
    for problem in _task_problems(model, task):
        note = f" - {spec['note']}" if spec else ""
        report.fail(f"task {task!r} {problem}{note}")


def requirements(task: str | None = None) -> dict:
    """Machine-readable contract, for docs and for the eventual upload form."""
    return {
        "renderable_geoms": sorted(g.name.replace("mjGEOM_", "").lower()
                                   for g in RENDERABLE_GEOMS),
        "limits": {
            "max_xml_chars": MAX_XML_CHARS,
            "max_bodies": MAX_BODIES,
            "max_geoms": MAX_GEOMS,
            "max_actuators": MAX_ACTUATORS,
        },
        "rules": [
            "single self-contained MJCF file; no <include>",
            "no meshes, heightfields, textures or any external asset reference",
            "every actuator declares a ctrlrange; actions arrive in [-1, 1]",
            "the model must stay finite when simulated with no control input",
        ],
        "tasks": (
            {task: task_requirements()[task]} if task in task_requirements()
            else task_requirements()
        ),
    }
