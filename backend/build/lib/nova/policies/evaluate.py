"""Score a policy on a published seed set and a held-out one.

Public seeds are printed in the docs so anyone can reproduce their own score
locally. Held-out seeds are derived from a salt the operator controls, so a
submission cannot be tuned against the episodes it will actually be graded on.
A large gap between the two scores is itself the interesting signal.
"""

from __future__ import annotations

import hashlib
import hmac
import multiprocessing as mp
import os
import traceback

import numpy as np

from .. import envs
from ..envs import record as record_rollout

#: Published. Reproduce locally with nova.evaluate(..., seeds=PUBLIC_SEEDS).
PUBLIC_SEEDS: tuple[int, ...] = tuple(range(10_000, 10_030))

#: Only meaningful once NOVA_EVAL_SALT is set to something private. Left at
#: this default, held-out seeds are reproducible too - which is correct for a
#: local install and wrong for a public leaderboard.
_DEFAULT_SALT = "nova-public-default"

DEFAULT_EPISODES = 30
#: An episode is 2-5 simulated seconds; 60 of them cannot legitimately take
#: this long, so exceeding it means the uploaded graph is pathological.
DEFAULT_TIMEOUT_S = 180
DEFAULT_MEMORY_MB = 3072


def held_out_seeds(n: int = DEFAULT_EPISODES, salt: str | None = None) -> tuple[int, ...]:
    key = (salt or os.environ.get("NOVA_EVAL_SALT") or _DEFAULT_SALT).encode()
    out = []
    for i in range(n):
        digest = hmac.new(key, str(i).encode(), hashlib.sha256).digest()
        out.append(int.from_bytes(digest[:4], "big") % (2**31 - 1))
    return tuple(out)


def score(policy, task: str, template: str, params: dict, seeds) -> dict:
    """Run one deterministic episode per seed and summarize."""
    env = envs.make(task, template, params)
    returns, successes, distances, lengths = [], [], [], []

    for seed in seeds:
        obs, _ = env.reset(seed=int(seed))
        total, steps, info = 0.0, 0, {}
        while True:
            action, _ = policy.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            steps += 1
            if terminated or truncated:
                break
        returns.append(total)
        lengths.append(steps)
        if "is_success" in info:
            successes.append(bool(info["is_success"]))
        if "distance" in info:
            distances.append(float(info["distance"]))

    out = {
        "episodes": len(list(seeds)),
        "mean_return": round(float(np.mean(returns)), 3),
        "std_return": round(float(np.std(returns)), 3),
        "mean_length": round(float(np.mean(lengths)), 1),
    }
    if successes:
        out["success_rate"] = round(float(np.mean(successes)), 4)
    if distances:
        out["mean_final_distance"] = round(float(np.mean(distances)), 4)
    return out


def evaluate_bytes(data: bytes, *, episodes: int = DEFAULT_EPISODES,
                   with_rollout: bool = True) -> dict:
    """Validate an uploaded policy and score it. Runs the graph in-process -
    callers handling untrusted input should use `evaluate_isolated` instead."""
    from .loader import load

    policy, manifest = load(data)
    public = score(policy, manifest.task, manifest.template, manifest.params, PUBLIC_SEEDS[:episodes])
    private = score(policy, manifest.task, manifest.template, manifest.params,
                    held_out_seeds(episodes))

    result = {
        "manifest": manifest.to_dict(),
        "public": public,
        "held_out": private,
        "gap": _gap(public, private),
    }
    if with_rollout:
        env = envs.make(manifest.task, manifest.template, manifest.params)
        result["rollout"] = record_rollout(env, policy, seed=int(PUBLIC_SEEDS[0]))
    return result


def _gap(public: dict, private: dict) -> dict:
    """How much worse the policy does on episodes it could not have been tuned on."""
    out = {"mean_return": round(public["mean_return"] - private["mean_return"], 3)}
    if "success_rate" in public and "success_rate" in private:
        out["success_rate"] = round(public["success_rate"] - private["success_rate"], 4)
    return out


# --------------------------------------------------------------------------
# Isolated execution
# --------------------------------------------------------------------------

def _worker(data: bytes, episodes: int, memory_mb: int, queue: "mp.Queue") -> None:
    try:
        import resource
        # Best-effort containment. Not a sandbox - an ONNX graph cannot syscall,
        # but onnxruntime is a large C++ parser, so cap what a bad one can consume.
        try:
            limit = memory_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
        except (ValueError, OSError):
            pass  # macOS is inconsistent about RLIMIT_AS; the wall clock still applies
        queue.put({"ok": True, "result": evaluate_bytes(data, episodes=episodes)})
    except Exception as exc:
        queue.put({
            "ok": False,
            "error": str(exc) or type(exc).__name__,
            "kind": type(exc).__name__,
            "trace": traceback.format_exc(limit=4),
        })


def evaluate_isolated(
    data: bytes,
    *,
    episodes: int = DEFAULT_EPISODES,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    memory_mb: int = DEFAULT_MEMORY_MB,
) -> dict:
    """Evaluate an untrusted policy in a separate, resource-limited process.

    The process boundary is what makes an upload survivable: a graph that hangs,
    explodes in memory, or crashes the runtime takes down a child, not the server.
    """
    ctx = mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue()
    proc = ctx.Process(target=_worker, args=(data, episodes, memory_mb, queue), daemon=True)
    proc.start()
    proc.join(timeout_s)

    if proc.is_alive():
        proc.terminate()
        proc.join(5)
        raise TimeoutError(
            f"evaluation exceeded {timeout_s}s and was stopped; the policy is "
            "far slower than a policy for this task should be"
        )

    if queue.empty():
        raise RuntimeError(
            f"evaluation process died without reporting (exit code {proc.exitcode}). "
            "Either the uploaded model crashed the ONNX runtime, or the caller "
            "started this from a script with no `if __name__ == \"__main__\":` "
            "guard - spawned workers re-import the entry module, and without the "
            "guard the child re-runs the script instead of the worker."
        )

    payload = queue.get()
    if not payload.get("ok"):
        raise ValueError(payload.get("error", "evaluation failed"))
    return payload["result"]
