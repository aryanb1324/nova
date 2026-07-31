"""Endpoints for getting a policy out, and taking someone else's in."""

from __future__ import annotations

import time

from fastapi import APIRouter, HTTPException, Request, Response
from starlette.concurrency import run_in_threadpool

from .. import policies
from . import store

router = APIRouter()

#: Uploads are read fully into memory before parsing, so cap the read itself
#: rather than relying on the validator to reject afterwards.
MAX_UPLOAD = 25_000_000
#: Episodes per seed set. 30 keeps a full evaluation under ~5 seconds.
EVAL_EPISODES = 30


@router.get("/api/runs/{run_id}/policy.onnx")
def download_run_policy(run_id: str) -> Response:
    """Export a locally trained run as a portable ONNX policy.

    Cached beside the run: the torch export takes a few seconds and the result
    never changes for a finished run.
    """
    try:
        path = store.run_dir(run_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None

    meta = store.load(run_id)
    if meta is None:
        raise HTTPException(404, f"no run {run_id!r}")

    onnx_path = path / "policy.onnx"
    if not onnx_path.exists():
        zip_path = path / "policy.zip"
        if not zip_path.exists():
            raise HTTPException(
                409, f"run {run_id!r} has no saved policy to export"
            )
        try:
            from stable_baselines3 import PPO

            model = PPO.load(zip_path, device="cpu")
            policies.export_policy(
                model,
                onnx_path,
                task=meta["task"],
                template=meta["template"],
                params=meta.get("params", {}),
                algo="PPO (stable-baselines3)",
                notes=f"exported from run {run_id}",
            )
        except Exception as exc:
            raise HTTPException(500, f"export failed: {exc}") from None

    return Response(
        content=onnx_path.read_bytes(),
        media_type="application/octet-stream",
        headers={"content-disposition": f'attachment; filename="{run_id}.onnx"'},
    )


@router.post("/api/policies")
async def upload_policy(request: Request) -> dict:
    """Accept an .onnx upload, validate it, and score it.

    The body is the raw file - the manifest lives inside it, so there is nothing
    else to send and no multipart parsing to do.
    """
    data = await request.body()
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, f"upload exceeds {MAX_UPLOAD / 1e6:.0f} MB")

    # Validation and scoring are blocking and CPU-bound; keep them off the loop.
    return await run_in_threadpool(_validate_and_score, data)


def _validate_and_score(data: bytes) -> dict:
    from ..policies.loader import PolicyRejected

    try:
        result = policies.evaluate_isolated(data, episodes=EVAL_EPISODES)
    except PolicyRejected as exc:
        raise HTTPException(422, str(exc)) from None
    except TimeoutError as exc:
        raise HTTPException(408, str(exc)) from None
    except ValueError as exc:
        # The worker re-raises rejections as plain ValueError across the process
        # boundary, so a bad upload still reads as a client error, not a crash.
        raise HTTPException(422, str(exc)) from None
    except Exception as exc:
        raise HTTPException(500, f"evaluation failed: {exc}") from None

    manifest = result["manifest"]
    policy_id = store.new_policy_id(manifest["task"], manifest["template"])
    meta = {
        "policy_id": policy_id,
        "uploaded": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "size_bytes": len(data),
        **result,
    }
    store.save_policy(policy_id, data, meta)
    return meta


@router.get("/api/policies")
def list_policies() -> dict:
    return {"policies": store.list_policies()}


@router.get("/api/policies/{policy_id}")
def get_policy(policy_id: str) -> dict:
    try:
        meta = store.policy_meta(policy_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None
    if meta is None:
        raise HTTPException(404, f"no policy {policy_id!r}")
    return meta


@router.get("/api/policies/{policy_id}/policy.onnx")
def download_policy(policy_id: str) -> Response:
    try:
        data = store.read_policy(policy_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None
    if data is None:
        raise HTTPException(404, f"no policy {policy_id!r}")
    return Response(
        content=data,
        media_type="application/octet-stream",
        headers={"content-disposition": f'attachment; filename="{policy_id}.onnx"'},
    )


@router.delete("/api/policies/{policy_id}")
def delete_policy(policy_id: str) -> dict:
    try:
        ok = store.delete_policy(policy_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None
    if not ok:
        raise HTTPException(404, f"no policy {policy_id!r}")
    return {"deleted": policy_id}
