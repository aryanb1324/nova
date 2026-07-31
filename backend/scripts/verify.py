#!/usr/bin/env python
"""Check the invariants this project rests on. No pytest, no network, no server.

    python scripts/verify.py

Covers: every template compiles and steps at its parameter extremes, every env
passes Gymnasium's contract checker, rollouts serialize to the shape the viewer
expects, and a run survives a save/load round trip.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import tempfile

import mujoco
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from nova import envs, robots  # noqa: E402
from nova.api import store  # noqa: E402
from nova.training import record  # noqa: E402

failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  ok    {name}")
    else:
        failures.append(f"{name}: {detail}")
        print(f"  FAIL  {name}  {detail}")


def builtins_only() -> dict:
    """Only the templates this package ships.

    Whatever the user has uploaded is their data, not a fixture — including it
    would make the suite's results depend on it.
    """
    return {k: t for k, t in robots.TEMPLATES.items() if t.source == "builtin"}


def check_templates() -> None:
    print("\ntemplates compile and step at every slider extreme")
    for key, tmpl in builtins_only().items():
        cases: dict[str, dict] = {"default": {}}
        for p in tmpl.params:
            cases[f"{p.key}=min"] = {p.key: p.min}
            cases[f"{p.key}=max"] = {p.key: p.max}
        cases["all=min"] = {p.key: p.min for p in tmpl.params}
        cases["all=max"] = {p.key: p.max for p in tmpl.params}

        bad = []
        for name, overrides in cases.items():
            try:
                model = mujoco.MjModel.from_xml_string(tmpl.build(overrides))
                data = mujoco.MjData(model)
                mujoco.mj_step(model, data, nstep=50)
                if not np.all(np.isfinite(data.qpos)):
                    bad.append(f"{name} (non-finite state)")
            except Exception as exc:
                bad.append(f"{name} ({type(exc).__name__}: {exc})")
        check(f"{key} ({len(cases)} cases)", not bad, "; ".join(bad[:3]))

        # Out-of-range input must be clamped, never trusted through to the model.
        first = tmpl.params[0]
        clamped = tmpl.resolve({first.key: first.max * 1000})
        check(f"{key} clamps out-of-range input", clamped[first.key] == first.max,
              f"got {clamped[first.key]}")
        check(f"{key} ignores unknown params", "not_a_param" not in tmpl.resolve({"not_a_param": 1}))


def check_envs() -> None:
    print("\nenvs satisfy the Gymnasium contract")
    from stable_baselines3.common.env_checker import check_env

    for task, template in [("reach", "reach_arm"), ("locomotion", "wheeled_bot"),
                           ("locomotion", "quadruped")]:
        env = envs.make(task, template, seed=0)
        try:
            check_env(env, warn=False, skip_render_check=True)
            ok, detail = True, ""
        except Exception as exc:
            ok, detail = False, str(exc)[:160]
        check(f"{task}/{template} passes check_env", ok, detail)

        obs, _ = env.reset(seed=0)
        check(f"{task}/{template} obs finite", bool(np.all(np.isfinite(obs))))

        # Determinism: same seed, same actions, same trajectory.
        def rollout_qpos(seed: int) -> np.ndarray:
            env.reset(seed=seed)
            rng = np.random.default_rng(0)
            for _ in range(25):
                env.step(rng.uniform(-1, 1, env.action_space.shape))
            return env.data.qpos.copy()

        check(f"{task}/{template} is seed-deterministic",
              np.allclose(rollout_qpos(7), rollout_qpos(7)))

        # A task must only be offered for templates that can actually run it.
        compatible = task in robots.get(template).tasks
        check(f"{task}/{template} declared compatible", compatible)


def check_rollout_format() -> None:
    print("\nrollouts serialize to the shape the viewer expects")
    env = envs.make("reach", "reach_arm", seed=0)
    roll = record(env, policy=None, seed=1)

    n_bodies = env.model.nbody - 1  # the world body is not streamed
    frames = np.asarray(roll["frames"])
    check("frames are (n_frames, n_bodies, 7)",
          frames.ndim == 3 and frames.shape[1] == n_bodies and frames.shape[2] == 7,
          f"got {frames.shape}, expected (*, {n_bodies}, 7)")
    check("n_frames matches the frame count", roll["n_frames"] == frames.shape[0])
    check("scene body count matches frame rows", len(roll["scene"]["bodies"]) == n_bodies)
    check("quaternions are unit length",
          np.allclose(np.linalg.norm(frames[:, :, 3:7], axis=2), 1.0, atol=1e-3))
    check("rollout is JSON-serializable", isinstance(json.dumps(roll), str))

    every_geom = [g for b in roll["scene"]["bodies"] for g in b["geoms"]]
    every_geom += roll["scene"]["static_geoms"]
    known = {"plane", "sphere", "capsule", "ellipsoid", "cylinder", "box"}
    unknown = sorted({g["type"] for g in every_geom} - known)
    check("all geom types are renderable", not unknown, f"unknown: {unknown}")
    check("every geom carries size/pos/quat/rgba",
          all(len(g["size"]) == 3 and len(g["pos"]) == 3 and len(g["quat"]) == 4
              and len(g["rgba"]) == 4 for g in every_geom))


def check_store() -> None:
    print("\nruns survive a save/load round trip")
    env = envs.make("reach", "reach_arm", seed=0)
    roll = record(env, policy=None, seed=2)
    meta = {
        "run_id": "verify-tmp", "task": "reach", "template": "reach_arm",
        "params": robots.get("reach_arm").defaults(), "config": {"total_steps": 1000},
        "created": "2026-01-01T00:00:00", "wall_time": 1.0,
        "steps_completed": 1000, "history": [], "eval": {"success_rate": 1.0},
    }
    original = store.RUNS_DIR
    with tempfile.TemporaryDirectory() as tmp:
        store.RUNS_DIR = pathlib.Path(tmp)
        try:
            store.save("verify-tmp", meta=meta, rollout=roll)
            loaded = store.load("verify-tmp")
            check("run round-trips", loaded is not None and loaded["task"] == "reach")
            check("rollout round-trips",
                  loaded is not None and loaded["final_rollout"]["n_frames"] == roll["n_frames"])
            check("summary is populated",
                  (store.summary("verify-tmp") or {}).get("template") == "reach_arm")
            check("listing finds it", any(r["run_id"] == "verify-tmp" for r in store.listing()))

            # Path traversal must be refused, not silently resolved.
            escaped = False
            try:
                store.run_dir("../../etc/passwd")
            except ValueError:
                escaped = True
            check("rejects path traversal in run ids", escaped)

            check("delete removes it", store.delete("verify-tmp") and store.load("verify-tmp") is None)
        finally:
            store.RUNS_DIR = original


def check_policies() -> None:
    print("\npolicies export, validate, and reject bad uploads")
    import onnx
    import torch as th

    from nova.policies.evaluate import held_out_seeds
    from nova.policies.export import export_policy
    from nova.policies.loader import PolicyRejected, validate

    env = envs.make("reach", "reach_arm", seed=0)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    module = th.nn.Sequential(th.nn.Linear(obs_dim, act_dim), th.nn.Tanh())
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "p.onnx"
        manifest = export_policy(module, path, task="reach", template="reach_arm",
                                 algo="verify")
        check("plain torch module exports", path.exists())
        check("manifest records the observation layout",
              manifest.obs_layout == type(env).obs_layout,
              f"{manifest.obs_layout} != {type(env).obs_layout}")

        data = path.read_bytes()
        check("round-trips through validation", validate(data).task == "reach")

        def rejects(label: str, payload: bytes) -> None:
            try:
                validate(payload)
            except PolicyRejected:
                check(label, True)
            except Exception as exc:  # noqa: BLE001
                check(label, False, f"raised {type(exc).__name__}, not PolicyRejected")
            else:
                check(label, False, "was accepted")

        rejects("rejects non-ONNX bytes", b"definitely not onnx")
        rejects("rejects an empty file", b"")
        rejects("rejects an oversized file", b"x" * 26_000_000)

        def retagged(key: str, value: str) -> bytes:
            model = onnx.load_from_string(data)
            for prop in model.metadata_props:
                if prop.key == key:
                    prop.value = value
            return model.SerializeToString()

        stripped = onnx.load_from_string(data)
        del stripped.metadata_props[:]
        rejects("rejects a policy with no manifest", stripped.SerializeToString())
        rejects("rejects a stale observation layout",
                retagged("nova.obs_layout", "reach.v0"))
        rejects("rejects a mismatched obs_dim", retagged("nova.obs_dim", "99"))
        rejects("rejects an unknown robot", retagged("nova.template", "nonesuch"))
        rejects("rejects a task the robot can't do", retagged("nova.task", "locomotion"))

        from nova.policies.loader import load

        policy, _ = load(data)
        for label, probe in [("NaN", np.nan), ("infinity", np.inf), ("1e30", 1e30)]:
            action, _ = policy.predict(np.full(obs_dim, probe, dtype=np.float32))
            check(f"clamps actions from a {label} observation",
                  bool(np.all(np.isfinite(action)) and np.all(np.abs(action) <= 1.0)),
                  f"got {action}")

    public = held_out_seeds(8, salt="one")
    check("held-out seeds depend on the salt", public != held_out_seeds(8, salt="two"))
    check("held-out seeds are stable for a salt", public == held_out_seeds(8, salt="one"))
    check("held-out seeds avoid the published set",
          not set(public) & set(range(10_000, 10_030)))


def check_starters() -> None:
    print("\nstarter algorithms are real, runnable files")
    import ast

    from nova import starters

    catalog = starters.catalog()
    check("every declared starter has source", len(catalog) == len(starters.STARTERS),
          f"{len(catalog)} of {len(starters.STARTERS)}")
    for meta in catalog:
        code = meta["code"]
        try:
            tree = ast.parse(code)
            ok, detail = True, ""
        except SyntaxError as exc:
            ok, detail = False, f"line {exc.lineno}: {exc.msg}"
        check(f"{meta['key']} parses", ok, detail)
        check(f"{meta['key']} uses the nova API",
              "import nova" in code and "nova.make(" in code)
        # Every starter should show the whole arc, not just a training loop.
        if meta["key"] != "blank":
            check(f"{meta['key']} exports and scores", "nova.export(" in code
                  and "nova.evaluate(" in code)
        check(f"{meta['key']} streams to the UI", "nova.attach(" in code)
        del tree


def check_code_runner() -> None:
    print("\ncode execution is gated to this machine")
    from nova.api.runner import execution_enabled, is_local

    for host in ("127.0.0.1", "::1", "localhost"):
        check(f"allows loopback {host}", is_local(host))
    for host in ("10.0.0.5", "203.0.113.9", "example.com", None):
        check(f"refuses {host}", not is_local(host))

    original = os.environ.get("NOVA_CODE_EXECUTION")
    try:
        for value in ("off", "0", "false", "no"):
            os.environ["NOVA_CODE_EXECUTION"] = value
            check(f"kill switch honours {value!r}", not execution_enabled())
        os.environ["NOVA_CODE_EXECUTION"] = "on"
        check("enabled by default", execution_enabled())
    finally:
        if original is None:
            os.environ.pop("NOVA_CODE_EXECUTION", None)
        else:
            os.environ["NOVA_CODE_EXECUTION"] = original


def check_robot_contract() -> None:
    print("\ncustom-robot contract accepts the built-ins and rejects the rest")
    from nova.robots.contract import validate_mjcf

    for key, tmpl in builtins_only().items():
        for task in tmpl.tasks:
            report = validate_mjcf(tmpl.build(), task=task)
            check(f"{key} satisfies the {task} contract", report.ok,
                  "; ".join(report.errors[:2]))

    cases = [
        ("rejects <include>",
         '<mujoco><include file="/etc/passwd"/><worldbody/></mujoco>'),
        ("rejects external assets",
         '<mujoco><asset><mesh name="m" file="r.stl"/></asset><worldbody/></mujoco>'),
        ("rejects a model with no actuators",
         '<mujoco><worldbody><body><geom type="sphere" size=".1"/></body>'
         '</worldbody></mujoco>'),
        ("rejects uncompilable XML", "<mujoco><not-a-tag/></mujoco>"),
    ]
    for label, xml in cases:
        report = validate_mjcf(xml)
        check(label, not report.ok, "was accepted")

    # A valid model can still be wrong for a task.
    arm = robots.build_mjcf("reach_arm")
    check("refuses the arm for locomotion",
          not validate_mjcf(arm, task="locomotion").ok)
    check("refuses the rover for reach",
          not validate_mjcf(robots.build_mjcf("wheeled_bot"), task="reach").ok)


def check_extensions() -> None:
    print("\nrobots and tasks can be registered from outside the package")
    import nova

    before_robots = len(nova.templates())
    before_tasks = len(nova.tasks())

    import examples.custom_robot  # noqa: F401 - registers on import

    check("an external module adds a robot", len(nova.templates()) > before_robots)
    check("an external module adds a task", len(nova.tasks()) > before_tasks)
    check("the added robot is in the catalog",
          any(t["key"] == "pendulum" for t in nova.templates()))
    check("the added task is in the catalog",
          any(t["key"] == "swingup" for t in nova.tasks()))

    env = nova.make("swingup", "pendulum")
    check("the added env constructs", env.observation_space.shape[0] > 0)
    check("its parameters render as sliders",
          all({"key", "label", "min", "max", "default"} <= set(p)
              for p in next(t for t in nova.templates()
                            if t["key"] == "pendulum")["params"]))

    from nova.robots.contract import validate_mjcf
    check("the added robot satisfies the render contract",
          validate_mjcf(robots.build_mjcf("pendulum")).ok)

    # Registration should refuse the mistakes that produce broken catalogs.
    def refuses(label: str, fn) -> None:
        try:
            fn()
        except (ValueError, TypeError):
            check(label, True)
        else:
            check(label, False, "was accepted")

    refuses("refuses a duplicate robot key",
            lambda: nova.register_template(robots.get("reach_arm")))
    refuses("refuses a non-template", lambda: nova.register_template(object()))
    refuses("refuses a duplicate task key",
            lambda: nova.register_task("reach", env=envs.ReachEnv, name="x",
                                       description="x"))
    refuses("refuses a task whose env isn't a RobotEnv",
            lambda: nova.register_task("bogus", env=dict, name="x", description="x"))

    class Unversioned(envs.RobotEnv):
        pass

    refuses("refuses a task with no obs_layout",
            lambda: nova.register_task("unversioned", env=Unversioned, name="x",
                                       description="x"))

    check("load_extensions survives a missing module",
          nova.load_extensions(["nova_does_not_exist"]) == [])
    check("load_extensions reports what loaded",
          nova.load_extensions(["examples.custom_robot"]) == ["examples.custom_robot"])


_HOPPER = """<mujoco model="hopper">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.005" integrator="implicitfast"/>
  <default><joint damping="0.5" armature="0.01"/><motor ctrlrange="-1 1" ctrllimited="true"/></default>
  <worldbody>
    <geom name="floor" type="plane" size="40 40 .05"/>
    <body name="torso" pos="0 0 0.6">
      <freejoint name="root"/>
      <geom name="torso_g" type="capsule" fromto="0 0 0 0 0 0.2" size="0.05"/>
      <body name="thigh">
        <joint name="hip" type="hinge" axis="0 1 0" range="-1.2 1.2"/>
        <geom name="thigh_g" type="capsule" fromto="0 0 0 0 0 -0.25" size="0.04"/>
      </body>
    </body>
  </worldbody>
  <actuator><motor joint="hip" gear="20"/></actuator>
</mujoco>"""


def check_uploaded_robots() -> None:
    print("\nuploaded robots validate, register, and train")
    import nova
    from nova.robots import uploads
    from nova.robots.contract import compatible_tasks

    report = uploads.validate(_HOPPER)
    check("a well-formed upload passes", report.ok, "; ".join(report.errors[:2]))
    check("its task is detected, not declared",
          report.info["compatible_tasks"] == ["locomotion"],
          str(report.info.get("compatible_tasks")))
    check("it explains why other tasks don't fit",
          bool(report.info["task_problems"].get("reach")))
    check("spawn height comes off the model",
          abs(uploads.spawn_height(_HOPPER) - 0.6) < 1e-6)
    check("camera is sized to the robot",
          uploads.camera_for(_HOPPER)["distance"] > 0.6)

    model = mujoco.MjModel.from_xml_string(robots.build_mjcf("reach_arm"))
    check("a fixed-base arm detects as reach only",
          compatible_tasks(model) == ["reach"], str(compatible_tasks(model)))

    original = uploads.UPLOADS_DIR
    with tempfile.TemporaryDirectory() as tmp:
        uploads.UPLOADS_DIR = pathlib.Path(tmp)
        try:
            meta = uploads.save("Verify Hopper", _HOPPER)
            check("upload persists", (uploads.UPLOADS_DIR / meta["key"]).exists())
            check("key is namespaced away from built-ins",
                  meta["key"].startswith(uploads.KEY_PREFIX))
            check("it round-trips", (uploads.load(meta["key"]) or {}).get("name")
                  == "Verify Hopper")
            check("it appears in the listing",
                  any(r["key"] == meta["key"] for r in uploads.listing()))

            loaded = uploads.register_stored()
            check("register_stored brings it back", meta["key"] in loaded)
            env = nova.make("locomotion", meta["key"])
            check("it makes a working env", env.observation_space.shape[0] > 0)
            check("it renders as primitives",
                  len(envs.describe(env.model)["bodies"]) == 2)
            check("the template reports source=upload",
                  robots.get(meta["key"]).source == "upload")
            check("it has no tunable parameters", robots.get(meta["key"]).params == ())

            check("delete removes it", uploads.delete(meta["key"]))
            check("and unregisters it", meta["key"] not in robots.TEMPLATES)
        finally:
            uploads.UPLOADS_DIR = original

    def rejects(label: str, xml: str) -> None:
        try:
            uploads.save("x", xml)
        except uploads.UploadRejected:
            check(label, True)
        else:
            check(label, False, "was accepted")

    rejects("refuses a robot no task can use",
            '<mujoco><worldbody><body><joint name="j" type="hinge" axis="0 1 0" '
            'range="-1 1"/><geom type="sphere" size=".1"/></body></worldbody>'
            '<actuator><motor joint="j" ctrlrange="-1 1"/></actuator></mujoco>')
    rejects("refuses external assets",
            '<mujoco><asset><mesh name="m" file="r.stl"/></asset><worldbody/></mujoco>')


def main() -> int:
    check_templates()
    check_envs()
    check_rollout_format()
    check_store()
    check_policies()
    check_starters()
    check_code_runner()
    check_robot_contract()
    check_uploaded_robots()
    # Last: it registers extra robots and tasks, which would change the counts
    # every earlier check reasons about.
    check_extensions()

    print()
    if failures:
        print(f"{len(failures)} FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
