"""Let a training script running anywhere stream into this UI.

The point is that someone testing their own algorithm gets the same reward curve
and 3D replay as the built-in PPO, without their code ever running on the server.
They train in their own process, on their own machine, with whatever library they
like, and post events here.

Events use exactly the shapes `/ws/train` already emits, so the frontend needs no
second code path.
"""

from __future__ import annotations

import asyncio
import time

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from .. import envs, robots
from . import store

router = APIRouter()

#: An attached run with no events for this long is presumed dead and reaped, so
#: a crashed training script doesn't linger in the UI forever.
STALE_AFTER_S = 600
MAX_ACTIVE = 8


class AttachStart(BaseModel):
    task: str = "reach"
    template: str = "reach_arm"
    params: dict[str, float] = Field(default_factory=dict)
    algo: str = "custom"
    author: str = ""
    total_steps: int = 0
    notes: str = ""


class Hub:
    """Fan-out of external run events to any connected browsers."""

    def __init__(self) -> None:
        self.subscribers: set[asyncio.Queue] = set()
        self.runs: dict[str, dict] = {}

    def subscribe(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=256)
        self.subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        self.subscribers.discard(queue)

    def publish(self, event: dict) -> None:
        for queue in list(self.subscribers):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                # A browser that cannot keep up loses frames rather than
                # stalling the trainer that is posting them.
                self.subscribers.discard(queue)

    def reap(self) -> None:
        now = time.time()
        for run_id, meta in list(self.runs.items()):
            if now - meta["last_seen"] > STALE_AFTER_S:
                self.runs.pop(run_id, None)

    def active(self) -> list[dict]:
        self.reap()
        return [
            {k: v for k, v in meta.items() if k != "last_event"}
            for meta in self.runs.values()
        ]


hub = Hub()


@router.post("/api/attach")
def attach_start(req: AttachStart) -> dict:
    """Open an external run. Returns the id to post events against."""
    hub.reap()
    if len(hub.runs) >= MAX_ACTIVE:
        raise HTTPException(429, f"too many active attached runs (max {MAX_ACTIVE})")

    try:
        envs.get_task(req.task)
        template = robots.get(req.template)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from None
    if req.task not in template.tasks:
        raise HTTPException(
            422, f"template {req.template!r} supports {list(template.tasks)}, not {req.task!r}"
        )

    params = template.resolve(req.params)
    run_id = store.new_run_id(req.task, req.template, prefix="ext")
    meta = {
        "run_id": run_id,
        "external": True,
        "task": req.task,
        "template": req.template,
        "params": params,
        "algo": req.algo,
        "author": req.author,
        "notes": req.notes,
        "total_steps": max(0, int(req.total_steps)),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "last_seen": time.time(),
    }
    hub.runs[run_id] = meta

    probe = envs.make(req.task, req.template, params)
    hub.publish({
        "type": "start",
        "run_id": run_id,
        "external": True,
        "config": {
            "task": req.task, "template": req.template, "params": params,
            "total_steps": meta["total_steps"], "algo": req.algo,
        },
        "scene": envs.describe(probe.model),
        "obs_dim": int(probe.observation_space.shape[0]),
        "action_dim": int(probe.action_space.shape[0]),
        "control_hz": round(1.0 / probe.dt, 2),
    })
    return {
        "run_id": run_id,
        "obs_dim": int(probe.observation_space.shape[0]),
        "action_dim": int(probe.action_space.shape[0]),
        "params": params,
    }


@router.post("/api/attach/{run_id}/event")
async def attach_event(run_id: str, event: dict) -> dict:
    meta = hub.runs.get(run_id)
    if meta is None:
        raise HTTPException(404, f"no active attached run {run_id!r}")
    if not isinstance(event, dict) or "type" not in event:
        raise HTTPException(422, "event must be an object with a 'type' field")

    meta["last_seen"] = time.time()
    hub.publish({**event, "run_id": run_id, "external": True})
    return {"ok": True}


@router.post("/api/attach/{run_id}/finish")
async def attach_finish(run_id: str, summary: dict | None = None) -> dict:
    meta = hub.runs.pop(run_id, None)
    if meta is None:
        raise HTTPException(404, f"no active attached run {run_id!r}")
    hub.publish({
        "type": "done",
        "run_id": run_id,
        "external": True,
        **(summary or {}),
    })
    hub.publish({"type": "closed", "run_id": run_id, "external": True})
    return {"ok": True, "run_id": run_id}


@router.get("/api/attach/active")
def attach_active() -> dict:
    return {"active": hub.active()}


@router.websocket("/ws/attach")
async def ws_attach(ws: WebSocket) -> None:
    """Browsers subscribe here to watch externally-run training."""
    await ws.accept()
    queue = hub.subscribe()
    try:
        await ws.send_json({"type": "attach_state", "active": hub.active()})
        while True:
            event = await queue.get()
            await ws.send_json(event)
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        hub.unsubscribe(queue)
