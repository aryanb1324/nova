"""FastAPI app: template catalog, live-preview, and streaming training over websocket.

Training runs in its own process. That keeps the event loop responsive, lets a
run be cancelled mid-flight, and stops a crashed job from taking the server with
it. Events travel worker -> multiprocessing.Queue -> asyncio.Queue -> websocket.
"""

from __future__ import annotations

import asyncio
import multiprocessing as mp
import os
import pathlib
import threading
import time
import traceback

import mujoco
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.staticfiles import StaticFiles

from .. import envs, robots
from ..envs import scene as scene_mod
from ..training import TrainConfig, evaluate, train
from . import store
from .attach import router as attach_router
from .code_routes import router as code_router
from .policy_routes import router as policy_router
from .robot_routes import router as robot_router

#: Ceiling on a single local run, so a stray request can't pin the machine for
#: an hour. Roughly two minutes of wall-clock on a 10-core M-series.
MAX_STEPS = 2_000_000

import nova as _nova

# Robots and tasks from other packages register themselves on import, so pull
# them in before anything reads the catalog.
_LOADED_EXTENSIONS = _nova.load_extensions()
# Uploaded robots live on disk; bring them back after a restart.
_LOADED_UPLOADS = robots.uploads.register_stored()

app = FastAPI(title="nova", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    # Local-first: the Vite dev server is the only expected origin.
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(policy_router)
app.include_router(attach_router)
app.include_router(code_router)
app.include_router(robot_router)


# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------

class PreviewRequest(BaseModel):
    template: str
    params: dict[str, float] = Field(default_factory=dict)
    task: str | None = None


class TrainRequest(BaseModel):
    task: str = "reach"
    template: str = "reach_arm"
    params: dict[str, float] = Field(default_factory=dict)
    total_steps: int | None = None
    seed: int = 0
    n_envs: int | None = None
    rollout_every: int = 10
    save: bool = True
    eval_episodes: int = 30


# --------------------------------------------------------------------------
# Catalog + preview
# --------------------------------------------------------------------------

@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "version": app.version}


@app.get("/api/catalog")
def catalog() -> dict:
    """Everything the frontend needs to render the picker and the editor."""
    from ..policies.evaluate import PUBLIC_SEEDS

    return {
        "templates": robots.catalog(),
        "tasks": envs.catalog(),
        "limits": {"max_steps": MAX_STEPS},
        "policies": {
            "format": "onnx",
            "max_upload_bytes": 25_000_000,
            # Published so anyone can reproduce their own score locally.
            "public_seeds": list(PUBLIC_SEEDS),
        },
        "extensions": _LOADED_EXTENSIONS,
        "uploaded_robots": _LOADED_UPLOADS,
    }


@app.post("/api/preview")
def preview(req: PreviewRequest) -> dict:
    """Static scene + one frame, for the editor's live 3D preview.

    Cheap enough to call on every slider drag: it compiles the model and runs
    forward kinematics once, with no stepping.
    """
    try:
        template = robots.get(req.template)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from None

    params = template.resolve(req.params)
    try:
        model = mujoco.MjModel.from_xml_string(template.builder(params))
    except Exception as exc:
        raise HTTPException(422, f"model failed to compile: {exc}") from None

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return {
        "template": req.template,
        "params": params,
        "scene": scene_mod.describe(model),
        "frame": scene_mod.frame(data).tolist(),
        "camera": dict(template.camera),
        "dof": int(model.nu),
    }


# --------------------------------------------------------------------------
# Saved runs
# --------------------------------------------------------------------------

@app.get("/api/runs")
def list_runs() -> dict:
    return {"runs": store.listing()}


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> dict:
    try:
        run = store.load(run_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None
    if run is None:
        raise HTTPException(404, f"no run {run_id!r}")
    return run


@app.delete("/api/runs/{run_id}")
def delete_run(run_id: str) -> dict:
    try:
        ok = store.delete(run_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None
    if not ok:
        raise HTTPException(404, f"no run {run_id!r}")
    return {"deleted": run_id}


# --------------------------------------------------------------------------
# Training worker
# --------------------------------------------------------------------------

def _train_worker(cfg: TrainConfig, run_id: str, req_save: bool, eval_episodes: int,
                  queue: "mp.Queue", stop_event) -> None:
    """Runs in its own process. Every outcome ends with a None sentinel."""
    try:
        result = train(
            cfg,
            on_event=queue.put,
            should_stop=stop_event.is_set,
        )
        model = result.pop("model")
        scores = evaluate(model, cfg, n_episodes=eval_episodes)
        queue.put({"type": "eval", "run_id": run_id, **scores})

        if req_save:
            store.save(
                run_id,
                meta={
                    "run_id": run_id,
                    "task": cfg.task,
                    "template": cfg.template,
                    "params": robots.get(cfg.template).resolve(cfg.params),
                    "config": cfg.to_dict(),
                    "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "wall_time": result["wall_time"],
                    "steps_completed": result["steps_completed"],
                    "stopped_early": result["stopped_early"],
                    "history": result["history"],
                    "eval": scores,
                },
                rollout=result["final_rollout"],
                model=model,
            )
            queue.put({"type": "saved", "run_id": run_id})
    except Exception:
        queue.put({"type": "error", "message": traceback.format_exc(limit=8)})
    finally:
        queue.put(None)


def _build_config(req: TrainRequest) -> TrainConfig:
    try:
        task_meta = envs.get_task(req.task)
        template = robots.get(req.template)
    except KeyError as exc:
        raise ValueError(str(exc)) from None

    if req.task not in template.tasks:
        raise ValueError(
            f"template {req.template!r} supports {list(template.tasks)}, not {req.task!r}"
        )

    steps = req.total_steps or task_meta["typical_steps"]
    if steps < 1000:
        raise ValueError("total_steps must be at least 1000")

    kwargs = {
        "task": req.task,
        "template": req.template,
        "params": dict(req.params),
        "total_steps": min(int(steps), MAX_STEPS),
        "seed": int(req.seed),
        "rollout_every": max(0, int(req.rollout_every)),
    }
    if req.n_envs:
        kwargs["n_envs"] = max(1, min(int(req.n_envs), 32))
    return TrainConfig(**kwargs)


@app.websocket("/ws/train")
async def ws_train(ws: WebSocket) -> None:
    await ws.accept()
    ctx = mp.get_context("spawn")
    proc = None
    stop_event = ctx.Event()

    try:
        req = TrainRequest(**await ws.receive_json())
    except Exception as exc:
        await ws.send_json({"type": "error", "message": f"bad request: {exc}"})
        await ws.close()
        return

    try:
        cfg = _build_config(req)
    except ValueError as exc:
        await ws.send_json({"type": "error", "message": str(exc)})
        await ws.close()
        return

    run_id = store.new_run_id(cfg.task, cfg.template)
    queue: mp.Queue = ctx.Queue()
    loop = asyncio.get_running_loop()
    inbox: asyncio.Queue = asyncio.Queue()

    def drain() -> None:
        """Blocking reader thread: mp.Queue has no async interface."""
        while True:
            item = queue.get()
            loop.call_soon_threadsafe(inbox.put_nowait, item)
            if item is None:
                return

    async def watch_client() -> None:
        """A stop message - or a disconnect - cancels the run."""
        try:
            while True:
                msg = await ws.receive_json()
                if msg.get("action") == "stop":
                    stop_event.set()
        except Exception:
            stop_event.set()

    # daemon=False: this process spawns its own SubprocVecEnv children, which
    # a daemonic process is not permitted to do.
    proc = ctx.Process(
        target=_train_worker,
        args=(cfg, run_id, req.save, req.eval_episodes, queue, stop_event),
        daemon=False,
    )
    proc.start()
    threading.Thread(target=drain, daemon=True).start()
    watcher = asyncio.create_task(watch_client())

    try:
        await ws.send_json({"type": "accepted", "run_id": run_id, "config": cfg.to_dict()})
        while True:
            item = await inbox.get()
            if item is None:
                break
            await ws.send_json(item)
        await ws.send_json({"type": "closed", "run_id": run_id})
    except WebSocketDisconnect:
        stop_event.set()
    finally:
        watcher.cancel()
        stop_event.set()
        if proc is not None:
            proc.join(timeout=10)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=5)
        try:
            await ws.close()
        except RuntimeError:
            pass


# --------------------------------------------------------------------------
# Built frontend
# --------------------------------------------------------------------------

#: Where `npm run build` puts the bundle. Overridable so an image can lay the
#: two halves out however it likes.
FRONTEND_DIST = pathlib.Path(
    os.environ.get("NOVA_FRONTEND_DIST")
    or pathlib.Path(__file__).resolve().parents[3] / "frontend" / "dist"
)


class _FrontendFiles(StaticFiles):
    """StaticFiles that declines websocket scopes instead of asserting on them.

    A mount at `/` is the last route in the table, so it also catches websocket
    connections to paths no `/ws` route claimed. Plain StaticFiles asserts the
    scope is HTTP and the connection fails with a 500 and a traceback; closing
    before accept restores Starlette's own answer for an unrouted websocket.
    """

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await send({"type": "websocket.close", "code": 1000})
            return
        await super().__call__(scope, receive, send)


def _mount_frontend() -> bool:
    """Serve the built frontend at `/`, if there is one.

    Deliberately the last thing the module does: Starlette matches routes in
    registration order, so every `/api` and `/ws` route above is resolved first
    and this mount only ever sees paths nothing else claimed. In development
    there is no `dist/` and nothing is mounted at all - Vite keeps serving the
    app and proxying to this process, exactly as before.
    """
    if not (FRONTEND_DIST / "index.html").is_file():
        return False
    # html=True: `/` returns index.html rather than a directory listing.
    app.mount("/", _FrontendFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")
    return True


SERVING_FRONTEND = _mount_frontend()
