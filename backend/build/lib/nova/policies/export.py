"""Turn a trained policy into a portable, code-free artifact.

ONNX rather than SB3's `.zip` on purpose: the zip is a pickle, and unpickling an
uploaded file is arbitrary code execution. An ONNX graph is data interpreted by a
fixed operator set, so a policy can be handed between machines without handing
over control of them.
"""

from __future__ import annotations

import pathlib
from typing import Any

import numpy as np
import onnx
import torch as th

from .. import envs, robots
from .manifest import PolicyManifest, is_manifest_key

#: Widely supported and enough for MLP policies.
DEFAULT_OPSET = 17


class _DeterministicActor(th.nn.Module):
    """Strips an SB3 ActorCriticPolicy down to obs -> action.

    Calling the policy's own forward() builds a Normal distribution, which the
    torch exporter refuses to trace. For a Gaussian head the deterministic action
    is exactly the distribution mean, so the sampling machinery is not needed -
    and the value head and log-probs are training-time only.
    """

    def __init__(self, policy: Any):
        super().__init__()
        self.policy = policy
        self.squash = bool(getattr(policy, "squash_output", False))

    def forward(self, obs: th.Tensor) -> th.Tensor:
        features = self.policy.extract_features(obs)
        latent_pi = self.policy.mlp_extractor.forward_actor(features)
        actions = self.policy.action_net(latent_pi)
        return th.tanh(actions) if self.squash else actions


def _as_module(source: Any) -> th.nn.Module:
    """Accept an SB3 algorithm, an SB3 policy, or a plain torch module."""
    policy = getattr(source, "policy", source)
    if hasattr(policy, "mlp_extractor") and hasattr(policy, "action_net"):
        return _DeterministicActor(policy)
    if isinstance(policy, th.nn.Module):
        return policy
    raise TypeError(
        f"cannot export {type(source).__name__}: expected a stable-baselines3 model "
        "or a torch.nn.Module mapping (batch, obs_dim) -> (batch, act_dim)"
    )


def export_policy(
    source: Any,
    path: str | pathlib.Path,
    *,
    task: str,
    template: str,
    params: dict | None = None,
    algo: str = "",
    author: str = "",
    notes: str = "",
    opset: int = DEFAULT_OPSET,
    tolerance: float = 1e-4,
) -> PolicyManifest:
    """Write `source` to `path` as ONNX with a nova manifest embedded.

    The exported graph is checked against the original on random observations
    before the file is written, so an export can never quietly ship a policy that
    behaves differently from the one that was trained.
    """
    probe_env = envs.make(task, template, params)
    obs_dim = int(probe_env.observation_space.shape[0])
    act_dim = int(probe_env.action_space.shape[0])

    module = _as_module(source).eval()
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with th.no_grad():
        th.onnx.export(
            module,
            th.zeros(1, obs_dim),
            str(path),
            input_names=["observation"],
            output_names=["action"],
            dynamic_axes={"observation": {0: "batch"}, "action": {0: "batch"}},
            opset_version=opset,
        )

    _verify_against_source(module, path, obs_dim, act_dim, tolerance)

    manifest = PolicyManifest(
        task=task,
        template=template,
        obs_dim=obs_dim,
        act_dim=act_dim,
        obs_layout=type(probe_env).obs_layout,
        params=robots.get(template).resolve(params),
        algo=algo,
        author=author,
        notes=notes,
    )
    _stamp(path, manifest)
    return manifest


def _verify_against_source(
    module: th.nn.Module, path: pathlib.Path, obs_dim: int, act_dim: int, tolerance: float
) -> None:
    import onnxruntime as ort

    rng = np.random.default_rng(0)
    probe = rng.normal(size=(64, obs_dim)).astype(np.float32)

    with th.no_grad():
        expected = module(th.from_numpy(probe)).numpy()

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    got = sess.run(None, {"observation": probe})[0]

    if got.shape != (64, act_dim):
        raise ValueError(
            f"exported graph produces {got.shape[1:]} values per observation, "
            f"but this task expects {act_dim} actions"
        )
    drift = float(np.abs(expected - got).max())
    if drift > tolerance:
        raise ValueError(
            f"exported policy disagrees with the original by {drift:.2e} "
            f"(tolerance {tolerance:.0e}); refusing to write a policy that would "
            "behave differently from the one you trained"
        )


def _stamp(path: pathlib.Path, manifest: PolicyManifest) -> None:
    model = onnx.load(str(path))
    keep = [p for p in model.metadata_props if not is_manifest_key(p.key)]
    del model.metadata_props[:]
    model.metadata_props.extend(keep)
    for key, value in manifest.to_props().items():
        entry = model.metadata_props.add()
        entry.key = key
        entry.value = value
    model.doc_string = (
        f"NOVA policy - {manifest.task} on {manifest.template} "
        f"({manifest.obs_dim} obs -> {manifest.act_dim} actions)"
    )
    onnx.save(model, str(path))
