"""Filesystem-backed run storage.

Local-first by design: a run is a directory under `runs/` holding the policy,
the config that produced it, the reward history and one playable rollout. No
database, no accounts - reloading is just reading a directory back.
"""

from __future__ import annotations

import json
import pathlib
import re
import shutil
import time

RUNS_DIR = pathlib.Path(__file__).resolve().parents[2] / "runs"
#: Uploaded policies live apart from trained runs: they have no training history,
#: and keeping them separate means an upload can never be mistaken for a local run.
POLICIES_DIR = pathlib.Path(__file__).resolve().parents[2] / "policies"

_SAFE = re.compile(r"[^a-zA-Z0-9_-]+")


def _slug(text: str) -> str:
    return _SAFE.sub("-", text).strip("-").lower() or "run"


def new_run_id(task: str, template: str, prefix: str = "") -> str:
    stem = f"{_slug(template)}-{_slug(task)}-{time.strftime('%Y%m%d-%H%M%S')}"
    return f"{_slug(prefix)}-{stem}" if prefix else stem


def run_dir(run_id: str) -> pathlib.Path:
    """Resolve a run directory, refusing anything that escapes RUNS_DIR."""
    path = (RUNS_DIR / run_id).resolve()
    if path.parent != RUNS_DIR.resolve():
        raise ValueError(f"invalid run id {run_id!r}")
    return path


def save(run_id: str, *, meta: dict, rollout: dict | None, model=None) -> pathlib.Path:
    out = run_dir(run_id)
    out.mkdir(parents=True, exist_ok=True)
    (out / "run.json").write_text(json.dumps(meta, indent=2))
    if rollout is not None:
        (out / "final_rollout.json").write_text(json.dumps(rollout))
    if model is not None:
        model.save(out / "policy")
    return out


def summary(run_id: str) -> dict | None:
    path = run_dir(run_id) / "run.json"
    if not path.exists():
        return None
    meta = json.loads(path.read_text())
    # A run directory written by an older version may be missing fields; show it
    # rather than rendering a blank row.
    config = meta.get("config", {})
    return {
        "run_id": run_id,
        "task": meta.get("task") or config.get("task") or "unknown",
        "template": meta.get("template") or config.get("template") or "unknown",
        "params": meta.get("params", {}),
        "created": meta.get("created"),
        "wall_time": meta.get("wall_time"),
        "eval": meta.get("eval"),
        "steps_completed": meta.get("steps_completed") or config.get("total_steps") or 0,
        "has_policy": (run_dir(run_id) / "policy.zip").exists(),
    }


def listing() -> list[dict]:
    if not RUNS_DIR.exists():
        return []
    out = []
    for child in sorted(RUNS_DIR.iterdir(), reverse=True):
        if not child.is_dir():
            continue
        try:
            s = summary(child.name)
        except ValueError:
            continue
        if s:
            out.append(s)
    return out


def load(run_id: str) -> dict | None:
    path = run_dir(run_id)
    meta_path = path / "run.json"
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text())
    rollout_path = path / "final_rollout.json"
    if rollout_path.exists():
        meta["final_rollout"] = json.loads(rollout_path.read_text())
    meta["run_id"] = run_id
    return meta


def delete(run_id: str) -> bool:
    path = run_dir(run_id)
    if not path.exists():
        return False
    shutil.rmtree(path)
    return True


# --------------------------------------------------------------------------
# Uploaded policies
# --------------------------------------------------------------------------

def new_policy_id(task: str, template: str) -> str:
    return f"{_slug(template)}-{_slug(task)}-{time.strftime('%Y%m%d-%H%M%S')}"


def policy_dir(policy_id: str) -> pathlib.Path:
    path = (POLICIES_DIR / policy_id).resolve()
    if path.parent != POLICIES_DIR.resolve():
        raise ValueError(f"invalid policy id {policy_id!r}")
    return path


def save_policy(policy_id: str, data: bytes, meta: dict) -> pathlib.Path:
    out = policy_dir(policy_id)
    out.mkdir(parents=True, exist_ok=True)
    (out / "policy.onnx").write_bytes(data)
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return out


def read_policy(policy_id: str) -> bytes | None:
    path = policy_dir(policy_id) / "policy.onnx"
    return path.read_bytes() if path.exists() else None


def policy_meta(policy_id: str) -> dict | None:
    path = policy_dir(policy_id) / "meta.json"
    if not path.exists():
        return None
    meta = json.loads(path.read_text())
    meta["policy_id"] = policy_id
    return meta


def list_policies() -> list[dict]:
    if not POLICIES_DIR.exists():
        return []
    out = []
    for child in sorted(POLICIES_DIR.iterdir(), reverse=True):
        if not child.is_dir():
            continue
        try:
            meta = policy_meta(child.name)
        except ValueError:
            continue
        if meta:
            # The rollout is large and only needed when one policy is opened.
            out.append({k: v for k, v in meta.items() if k != "rollout"})
    return out


def delete_policy(policy_id: str) -> bool:
    path = policy_dir(policy_id)
    if not path.exists():
        return False
    shutil.rmtree(path)
    return True
