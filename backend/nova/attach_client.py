"""Client for streaming an external training run into a running nova UI.

Uses urllib rather than requests so that bringing your own algorithm doesn't
mean inheriting our HTTP stack. Every call is best-effort: a training run that
takes minutes must not die because a browser tab was closed.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

#: Scripts launched from the in-app editor get this set for them, so a starter
#: needs no configuration to appear in the UI it was launched from.
DEFAULT_URL = os.environ.get("NOVA_URL", "http://127.0.0.1:8000")
_TIMEOUT_S = 10


class AttachRun:
    """A live run the UI can watch. Use as a context manager."""

    def __init__(self, task: str, template: str, params: dict | None = None, *,
                 algo: str = "custom", author: str = "", total_steps: int = 0,
                 url: str = DEFAULT_URL):
        self.url = url.rstrip("/")
        self.task = task
        self.template = template
        self.algo = algo
        self.total_steps = total_steps
        self.run_id: str | None = None
        self.iteration = 0
        self.started = time.perf_counter()
        self.connected = False
        self._warned = False
        self._env = None

        payload = {
            "task": task, "template": template, "params": params or {},
            "algo": algo, "author": author, "total_steps": total_steps,
        }
        reply = self._post("/api/attach", payload)
        if reply:
            self.run_id = reply.get("run_id")
            self.params = reply.get("params", params or {})
            self.obs_dim = reply.get("obs_dim")
            self.action_dim = reply.get("action_dim")
            self.connected = True
            print(f"[NOVA] streaming to {self.url} as {self.run_id}")
        else:
            self.params = params or {}
            print(f"[NOVA] no UI at {self.url}; training without streaming")

    # ---- lifecycle -------------------------------------------------------

    def __enter__(self) -> "AttachRun":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.finish(failed=exc_type is not None)

    def finish(self, *, failed: bool = False, **summary) -> None:
        if not (self.connected and self.run_id):
            return
        self._post(f"/api/attach/{self.run_id}/finish", {
            "wall_time": round(time.perf_counter() - self.started, 2),
            "iterations": self.iteration,
            "stopped_early": failed,
            **summary,
        })
        self.connected = False

    # ---- events ----------------------------------------------------------

    def log(self, *, step: int, mean_reward: float | None = None, **extra) -> None:
        """Report progress. Iteration, elapsed time and throughput are derived."""
        self.iteration += 1
        if not (self.connected and self.run_id):
            return
        elapsed = time.perf_counter() - self.started
        event = {
            "type": "progress",
            "iteration": self.iteration,
            "step": int(step),
            "total_steps": self.total_steps or int(step),
            "progress": round(step / self.total_steps, 4) if self.total_steps else 0.0,
            "elapsed": round(elapsed, 2),
            "fps": int(step / elapsed) if elapsed > 0 else 0,
        }
        if mean_reward is not None:
            event["mean_reward"] = round(float(mean_reward), 4)
        event.update({k: v for k, v in extra.items() if v is not None})
        self._post(f"/api/attach/{self.run_id}/event", event)

    def send(self, event: dict) -> None:
        """Forward an already-formed event.

        Lets a trainer that already emits progress/rollout dicts in this shape -
        the built-in PPO, for one - stream through without reformatting.
        """
        if not (self.connected and self.run_id):
            return
        if event.get("type") == "progress":
            self.iteration = event.get("iteration", self.iteration + 1)
        self._post(f"/api/attach/{self.run_id}/event", event)

    def rollout(self, policy, *, seed: int | None = None, step: int | None = None) -> dict | None:
        """Record an episode with the current policy and show it in the viewer.

        `policy` may be anything with `.predict(obs, deterministic=)` or a plain
        callable taking an observation and returning an action.
        """
        if not (self.connected and self.run_id):
            return None
        from .envs import make, record

        if self._env is None:
            self._env = make(self.task, self.template, self.params)
        roll = record(self._env, _as_policy(policy), seed=seed)
        self._post(f"/api/attach/{self.run_id}/event", {
            "type": "rollout",
            "iteration": self.iteration,
            "step": int(step if step is not None else 0),
            **roll,
        })
        return roll

    # ---- transport -------------------------------------------------------

    def _post(self, path: str, payload: dict) -> dict | None:
        body = json.dumps(payload).encode()
        req = urllib.request.Request(
            self.url + path, data=body,
            headers={"content-type": "application/json"}, method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:
                return json.loads(resp.read() or b"{}")
        except urllib.error.HTTPError as exc:
            self._warn(f"{path} rejected: {exc.code} {exc.read()[:200].decode(errors='replace')}")
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            self._warn(f"{path} unreachable: {exc}")
        except json.JSONDecodeError:
            return {}
        return None

    def _warn(self, message: str) -> None:
        # Once only: a dropped UI should not scroll the training output away.
        if not self._warned:
            print(f"[NOVA] {message} (further streaming errors suppressed)")
            self._warned = True


class _Callable:
    def __init__(self, fn):
        self.fn = fn

    def predict(self, obs, deterministic: bool = True):  # noqa: ARG002
        return self.fn(obs), None


def _as_policy(policy):
    if policy is None or hasattr(policy, "predict"):
        return policy
    if callable(policy):
        return _Callable(policy)
    raise TypeError(
        "policy must be None, have a .predict(obs, deterministic=) method, "
        "or be a callable taking an observation and returning an action"
    )
