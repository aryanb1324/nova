"""Endpoints for writing and running your own training code."""

from __future__ import annotations

import pathlib

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from .. import starters
from .attach import hub
from .runner import execution_enabled, is_local, runner

router = APIRouter()

SAVED = pathlib.Path(__file__).resolve().parents[2] / "user_scripts" / "saved"
MAX_SCRIPT_CHARS = 400_000


class RunRequest(BaseModel):
    script: str
    name: str = "untitled"


class SaveRequest(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    script: str


def _guard(request: Request) -> None:
    """Code execution is a local-machine feature. Keep it that way."""
    if not execution_enabled():
        raise HTTPException(
            403,
            "code execution is disabled (NOVA_CODE_EXECUTION=off). "
            "Upload a trained .onnx policy instead.",
        )
    client = request.client.host if request.client else None
    if not is_local(client):
        raise HTTPException(
            403,
            f"code execution is only available from this machine (request came "
            f"from {client}). Running a script you typed on your own computer is "
            "just running a script; running one sent over a network is not, so "
            "NOVA refuses. Upload a trained .onnx policy instead.",
        )


@router.get("/api/code/starters")
def list_starters() -> dict:
    """Readable reference implementations to start from."""
    return {"starters": starters.catalog()}


@router.get("/api/code/status")
def code_status(request: Request) -> dict:
    client = request.client.host if request.client else None
    current = runner.current
    return {
        "enabled": execution_enabled() and is_local(client),
        "reason": None if execution_enabled() else "NOVA_CODE_EXECUTION=off",
        "running": bool(current and current.alive),
        "run": current.summary() if current else None,
    }


@router.post("/api/code/run")
def run_code(req: RunRequest, request: Request) -> dict:
    _guard(request)
    if not req.script.strip():
        raise HTTPException(422, "script is empty")
    if len(req.script) > MAX_SCRIPT_CHARS:
        raise HTTPException(413, "script is too large")

    port = request.url.port or 8000
    try:
        run = runner.start(req.script, req.name, hub.publish, port=port)
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from None

    hub.publish({
        "type": "console",
        "run_id": run.run_id,
        "line": f"[nova] running {req.name} …",
    })
    return {"run_id": run.run_id, "script": run.path.name}


@router.post("/api/code/stop")
def stop_code(request: Request) -> dict:
    _guard(request)
    return {"stopped": runner.stop()}


# ---- saved scripts -------------------------------------------------------

def _safe_name(name: str) -> str:
    stem = "".join(c for c in name if c.isalnum() or c in "-_ ").strip()
    if not stem:
        raise HTTPException(422, "name must contain letters or numbers")
    return stem.replace(" ", "-").lower()


@router.get("/api/code/saved")
def list_saved() -> dict:
    if not SAVED.exists():
        return {"saved": []}
    scripts = [
        {"name": p.stem, "chars": p.stat().st_size} for p in SAVED.glob("*.py")
    ]
    return {"saved": sorted(scripts, key=lambda d: d["name"])}


@router.get("/api/code/saved/{name}")
def get_saved(name: str) -> dict:
    path = SAVED / f"{_safe_name(name)}.py"
    if not path.exists():
        raise HTTPException(404, f"no saved script {name!r}")
    return {"name": path.stem, "script": path.read_text()}


@router.post("/api/code/saved")
def save_script(req: SaveRequest) -> dict:
    SAVED.mkdir(parents=True, exist_ok=True)
    path = SAVED / f"{_safe_name(req.name)}.py"
    path.write_text(req.script)
    return {"name": path.stem, "chars": len(req.script)}


@router.delete("/api/code/saved/{name}")
def delete_saved(name: str) -> dict:
    path = SAVED / f"{_safe_name(name)}.py"
    if not path.exists():
        raise HTTPException(404, f"no saved script {name!r}")
    path.unlink()
    return {"deleted": path.stem}
