"""Validate and run an uploaded policy.

Everything here treats the file as hostile. ONNX is enormously safer than a
pickle - it is data for a fixed operator set, not code - but onnxruntime is still
a large C++ parser with a CVE history, so it is *not* by itself a security
boundary. The checks below reject the obviously wrong early; the real containment
is that inference only ever happens inside a resource-limited worker process
(see `evaluate.py`).
"""

from __future__ import annotations

import pathlib

import numpy as np
import onnx

from .. import envs, robots
from .manifest import PolicyManifest

#: An MLP policy for these envs is tens of KB. This is generous by three orders
#: of magnitude and still small enough that parsing can't be used as a memory bomb.
MAX_BYTES = 25_000_000
#: Standard operators only. A custom domain means custom kernels, which means
#: loading someone else's shared library.
ALLOWED_DOMAINS = {"", "ai.onnx"}
MIN_OPSET, MAX_OPSET = 9, 22

_FLOAT = onnx.TensorProto.FLOAT


class PolicyRejected(ValueError):
    """The upload is not something we are willing to run."""


def _dim(value_info) -> list:
    shape = value_info.type.tensor_type.shape
    return [d.dim_value if d.HasField("dim_value") else None for d in shape.dim]


def inspect(data: bytes) -> tuple[PolicyManifest, onnx.ModelProto]:
    """Structural validation. Raises PolicyRejected with an actionable message."""
    if len(data) > MAX_BYTES:
        raise PolicyRejected(
            f"file is {len(data) / 1e6:.1f} MB; the limit is {MAX_BYTES / 1e6:.0f} MB"
        )
    if not data:
        raise PolicyRejected("file is empty")

    try:
        model = onnx.load_from_string(data)
    except Exception as exc:
        raise PolicyRejected(f"not a readable ONNX file: {type(exc).__name__}") from None

    try:
        onnx.checker.check_model(model)
    except Exception as exc:
        raise PolicyRejected(f"ONNX validity check failed: {str(exc)[:200]}") from None

    for opset in model.opset_import:
        if opset.domain not in ALLOWED_DOMAINS:
            raise PolicyRejected(
                f"uses operator domain {opset.domain!r}; only standard ONNX "
                "operators are accepted"
            )
        if opset.domain in ("", "ai.onnx") and not (MIN_OPSET <= opset.version <= MAX_OPSET):
            raise PolicyRejected(
                f"opset {opset.version} is outside the supported range "
                f"{MIN_OPSET}-{MAX_OPSET}"
            )
    for node in model.graph.node:
        if node.domain not in ALLOWED_DOMAINS:
            raise PolicyRejected(
                f"operator {node.op_type!r} comes from non-standard domain {node.domain!r}"
            )

    inputs = [i for i in model.graph.input
              if i.name not in {init.name for init in model.graph.initializer}]
    if len(inputs) != 1:
        raise PolicyRejected(
            f"expected exactly one input (the observation), found {len(inputs)}"
        )
    if len(model.graph.output) != 1:
        raise PolicyRejected(
            f"expected exactly one output (the action), found {len(model.graph.output)}"
        )
    if inputs[0].type.tensor_type.elem_type != _FLOAT:
        raise PolicyRejected("the observation input must be float32")
    if model.graph.output[0].type.tensor_type.elem_type != _FLOAT:
        raise PolicyRejected("the action output must be float32")

    try:
        manifest = PolicyManifest.from_props({p.key: p.value for p in model.metadata_props})
    except ValueError as exc:
        # Surfaces as a rejected upload, not a server error.
        raise PolicyRejected(str(exc)) from None

    in_shape, out_shape = _dim(inputs[0]), _dim(model.graph.output[0])
    if len(in_shape) != 2 or in_shape[1] not in (None, manifest.obs_dim):
        raise PolicyRejected(
            f"input shape {in_shape} does not match the declared observation size "
            f"(batch, {manifest.obs_dim})"
        )
    if len(out_shape) != 2 or out_shape[1] not in (None, manifest.act_dim):
        raise PolicyRejected(
            f"output shape {out_shape} does not match the declared action size "
            f"(batch, {manifest.act_dim})"
        )

    return manifest, model


def check_compatible(manifest: PolicyManifest) -> None:
    """Confirm the manifest describes something this build can actually run."""
    try:
        task = envs.get_task(manifest.task)
    except KeyError:
        raise PolicyRejected(
            f"unknown task {manifest.task!r}; this build has {sorted(envs.TASKS)}"
        ) from None
    try:
        template = robots.get(manifest.template)
    except KeyError:
        raise PolicyRejected(
            f"unknown robot {manifest.template!r}; this build has "
            f"{sorted(robots.TEMPLATES)}"
        ) from None
    if manifest.task not in template.tasks:
        raise PolicyRejected(
            f"robot {manifest.template!r} cannot perform task {manifest.task!r}"
        )

    env = task["env"](template_key=manifest.template, params=manifest.params)
    expected_layout = type(env).obs_layout
    if manifest.obs_layout != expected_layout:
        raise PolicyRejected(
            f"policy was trained against observation layout {manifest.obs_layout!r}, "
            f"but this build uses {expected_layout!r}. The observation changed "
            "meaning since it was exported - retrain or re-export it."
        )
    if manifest.obs_dim != env.observation_space.shape[0]:
        raise PolicyRejected(
            f"policy expects {manifest.obs_dim} observations, but "
            f"{manifest.task}/{manifest.template} produces "
            f"{env.observation_space.shape[0]}"
        )
    if manifest.act_dim != env.action_space.shape[0]:
        raise PolicyRejected(
            f"policy outputs {manifest.act_dim} actions, but "
            f"{manifest.task}/{manifest.template} needs {env.action_space.shape[0]}"
        )


def validate(data: bytes) -> PolicyManifest:
    manifest, _ = inspect(data)
    check_compatible(manifest)
    return manifest


class OnnxPolicy:
    """Runs an ONNX policy with the `.predict()` signature the rest of the code uses."""

    def __init__(self, data: bytes, manifest: PolicyManifest, *, low: float = -1.0,
                 high: float = 1.0):
        import onnxruntime as ort

        options = ort.SessionOptions()
        # One thread: matches the rest of the project's tuning, and keeps an
        # uploaded graph from spawning a pool inside the worker.
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.log_severity_level = 3

        self.session = ort.InferenceSession(
            data, options, providers=["CPUExecutionProvider"]
        )
        self.manifest = manifest
        self.input_name = self.session.get_inputs()[0].name
        self.low, self.high = low, high

    def predict(self, obs, deterministic: bool = True):  # noqa: ARG002 - signature parity
        batch = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        action = self.session.run(None, {self.input_name: batch})[0][0]
        # Clip rather than trust: an uploaded graph can emit anything, including
        # NaN, and MuJoCo will happily integrate garbage into a broken state.
        action = np.nan_to_num(action, nan=0.0, posinf=self.high, neginf=self.low)
        return np.clip(action, self.low, self.high).astype(np.float64), None


def load(data: bytes) -> tuple[OnnxPolicy, PolicyManifest]:
    manifest = validate(data)
    return OnnxPolicy(data, manifest), manifest


def load_file(path: str | pathlib.Path) -> tuple[OnnxPolicy, PolicyManifest]:
    return load(pathlib.Path(path).read_bytes())
