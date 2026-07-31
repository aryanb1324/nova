#!/usr/bin/env python
"""Render an animated GIF of a trained policy, straight from MuJoCo.

    python scripts/render_gif.py --run gif_run --out ../docs/nova-reach.gif

This is the README's demo loop. It renders offscreen with `mujoco.Renderer` —
no browser, no screen recording — so the output is deterministic and can be
regenerated from any saved run:

    OMP_NUM_THREADS=1 python scripts/train_cli.py --task reach --save gif_run
    python scripts/render_gif.py --run gif_run

Episode seeds are not taken in order. A reach whose target happens to sit
behind the arm, or barely a hand's width from the fingertip, is a successful
episode and a useless frame. `pick_episodes` scores candidates by how much
motion they put *on screen* and keeps only the ones that also succeed, so the
GIF shows the policy doing the thing rather than merely not failing.

macOS renders offscreen through CGL with no configuration. On a headless Linux
box set MUJOCO_GL=egl (GPU) or MUJOCO_GL=osmesa (software) before running.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import mujoco  # noqa: E402
from PIL import Image  # noqa: E402

from nova import envs  # noqa: E402
from nova.api import store  # noqa: E402

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "docs" / "nova-reach.gif"


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--run", default="gif_run",
                   help="run directory under backend/runs/ holding policy.zip")
    p.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT)
    p.add_argument("--episodes", type=int, default=3,
                   help="consecutive episodes to show")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=420)
    p.add_argument("--fps", type=float, default=25.0,
                   help="playback rate; the sim is subsampled to match")
    p.add_argument("--colors", type=int, default=96,
                   help="GIF palette size (fewer = smaller file)")
    p.add_argument("--tail", type=float, default=0.6, metavar="SECONDS",
                   help="keep recording this long after the fingertip arrives, "
                        "then cut; 0 plays the full 2s episode")
    p.add_argument("--hold", type=float, default=0.5, metavar="SECONDS",
                   help="pause on each episode's last frame before the cut")
    p.add_argument("--seeds", default=None, metavar="A,B,C",
                   help="explicit episode seeds; skips automatic selection")
    p.add_argument("--seed-pool", type=int, default=80,
                   help="candidate episodes to score when picking seeds")
    p.add_argument("--azimuth", type=float, default=138.0)
    p.add_argument("--elevation", type=float, default=-24.0)
    p.add_argument("--distance", type=float, default=0.98)
    p.add_argument("--lookat", default="0.0,0.0,0.26")
    p.add_argument("--backdrop", default="26,29,36", metavar="R,G,B",
                   help="fill colour for the empty background; 'none' keeps black")
    p.add_argument("--dither", action="store_true",
                   help="dither the palette; smoother gradients, much larger file")
    return p.parse_args(argv)


def load_run(run_id: str):
    """Return (meta, model) for a saved run, or exit with something actionable."""
    run_dir = store.run_dir(run_id)
    policy_path = run_dir / "policy.zip"
    meta_path = run_dir / "run.json"
    if not policy_path.exists():
        raise SystemExit(
            f"no policy at {policy_path}\n"
            f"train one first:  OMP_NUM_THREADS=1 python scripts/train_cli.py "
            f"--task reach --save {run_id}"
        )
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    from stable_baselines3 import PPO

    # CPU explicitly: this is an MLP on a handful of floats, and the GPU
    # transfer costs more than the forward pass.
    return meta, PPO.load(policy_path, device="cpu")


def make_camera(args) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth = args.azimuth
    cam.elevation = args.elevation
    cam.distance = args.distance
    cam.lookat[:] = [float(v) for v in args.lookat.split(",")]
    return cam


def screen_basis(renderer: mujoco.Renderer, data, cam) -> tuple[np.ndarray, np.ndarray]:
    """The camera's right/up vectors in world coordinates.

    Read off the populated scene rather than re-deriving them from azimuth and
    elevation, so the scoring below stays correct if the camera convention or
    the defaults ever move.
    """
    renderer.update_scene(data, camera=cam)
    scene_cam = renderer.scene.camera[0]
    forward = np.array(scene_cam.forward, dtype=float)
    up = np.array(scene_cam.up, dtype=float)
    right = np.cross(forward, up)
    return right / np.linalg.norm(right), up / np.linalg.norm(up)


def run_episode(env, model, seed: int, renderer=None, cam=None, stride: int = 1,
                tail_steps: int | None = None):
    """Roll one episode out. Renders frames only when a renderer is supplied.

    Returns (frames, stats). The scoring pass calls this without a renderer,
    which makes trying 80 candidate episodes cost about a second.

    `tail_steps` cuts the episode that many steps after the fingertip first
    arrives. The env always runs its full 2 seconds, but a good policy gets
    there in well under one and then just sits — dead weight in a looping GIF.
    """
    obs, _ = env.reset(seed=seed)
    frames = []
    path = [env.site_xpos("ee")]
    hit_at = None

    if renderer is not None:
        renderer.update_scene(env.data, camera=cam)
        frames.append(renderer.render())

    for step in range(env.max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, _reward, terminated, truncated, info = env.step(action)
        path.append(env.site_xpos("ee"))
        if hit_at is None and info.get("is_success"):
            hit_at = step
        if renderer is not None and (step + 1) % stride == 0:
            renderer.update_scene(env.data, camera=cam)
            frames.append(renderer.render())
        if terminated or truncated:
            break
        if tail_steps is not None and hit_at is not None and step - hit_at >= tail_steps:
            break

    path = np.asarray(path)
    return frames, {
        "seed": seed,
        "success": bool(info.get("is_success", False)),
        "distance": float(info.get("distance", np.inf)),
        "steps": step + 1,
        "path": path,
        "target": env.target.copy(),
    }


def pick_episodes(env, model, right, up, n: int, pool: int) -> list[int]:
    """Choose seeds that read well on screen.

    Every candidate that succeeds is eligible; among those, prefer the ones
    where the fingertip sweeps a long path *across the image plane*. A reach
    aimed straight at the camera covers real distance in the world and almost
    none in the picture.

    Travel is a filter rather than the ranking, though. Ranked by travel alone
    the top three are all long swings down to the floor — a policy that only
    ever reaches in one direction, as far as the viewer can tell. So: keep the
    legible half, then spread the picks across the workspace.
    """
    scored = []
    for seed in range(pool):
        _, stats = run_episode(env, model, seed)
        if not stats["success"]:
            continue
        steps = np.diff(stats["path"], axis=0)
        on_screen = np.stack([steps @ right, steps @ up], axis=1)
        travel = float(np.linalg.norm(on_screen, axis=1).sum())
        scored.append({"seed": seed, "travel": travel, "target": stats["target"]})

    if not scored:
        raise SystemExit(
            f"no successful episode in {pool} tries — is this run trained? "
            "Check run.json's eval.success_rate."
        )

    cutoff = float(np.median([c["travel"] for c in scored]))
    candidates = [c for c in scored if c["travel"] >= cutoff]

    # Targets are sampled over a full sphere of azimuth but only about a third
    # of that range in height, so unweighted distance would always find its
    # spread horizontally and every reach would end at floor level.
    weight = np.array([1.0, 1.0, 3.0])
    chosen = [max(candidates, key=lambda c: c["travel"])]
    candidates.remove(chosen[0])
    while candidates and len(chosen) < n:
        best = max(candidates, key=lambda c: min(
            float(np.linalg.norm((c["target"] - k["target"]) * weight))
            for k in chosen
        ))
        chosen.append(best)
        candidates.remove(best)

    # Ascending seed order for the final render: arbitrary but reproducible, and
    # it stops the GIF from front-loading its most dramatic episode.
    return sorted(c["seed"] for c in chosen)


def fill_backdrop(frame: np.ndarray, colour: np.ndarray | None) -> np.ndarray:
    """Paint the empty background behind the scene.

    The arm's floor plane is finite, so roughly a third of this framing is sky —
    and MuJoCo renders sky as pure black with no skybox in the model. Left
    alone it reads as a hole punched in the image. Exact (0,0,0) is an
    unambiguous mask: every lit surface in the scene is above zero, and the
    antialiased horizon keeps its dark fringe, which just looks like a horizon.
    """
    if colour is None:
        return frame
    frame = frame.copy()
    frame[frame.sum(axis=2) == 0] = colour
    return frame


def quantize(frames: list[np.ndarray], colors: int, dither: bool) -> list[Image.Image]:
    """Map every frame onto one shared palette.

    Per-frame palettes make the background shimmer as the quantizer picks
    slightly different greys each frame, which looks like noise and defeats
    GIF's inter-frame compression.
    """
    images = [Image.fromarray(f) for f in frames]
    # Build the palette from frames spread across the whole animation so late
    # episodes are represented, not just the opening pose.
    sample_idx = np.linspace(0, len(images) - 1, min(12, len(images))).astype(int)
    montage = Image.new("RGB", (images[0].width, images[0].height * len(sample_idx)))
    for row, idx in enumerate(sample_idx):
        montage.paste(images[idx], (0, row * images[0].height))
    palette = montage.quantize(colors=colors, method=Image.Quantize.MEDIANCUT)

    mode = Image.Dither.FLOYDSTEINBERG if dither else Image.Dither.NONE
    return [im.quantize(palette=palette, dither=mode) for im in images]


def main(argv=None) -> int:
    args = parse_args(argv)
    meta, model = load_run(args.run)

    task = meta.get("task", "reach")
    template = meta.get("template", "reach_arm")
    env = envs.make(task, template, meta.get("params"))

    # The sim runs at 1/env.dt Hz; play it back at real time by keeping every
    # nth frame. Rounded to at least 1 so a fast --fps can't ask for fractions.
    stride = max(1, round((1.0 / env.dt) / args.fps))
    frame_ms = int(round(1000 * env.dt * stride))

    renderer = mujoco.Renderer(env.model, args.height, args.width)
    cam = make_camera(args)
    right, up = screen_basis(renderer, env.data, cam)

    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",")]
    else:
        seeds = pick_episodes(env, model, right, up, args.episodes, args.seed_pool)
    print(f"run={args.run} task={task}/{template} seeds={seeds} "
          f"stride={stride} ({1000 / frame_ms:.1f} fps)")

    backdrop = None
    if args.backdrop.lower() != "none":
        backdrop = np.array([int(v) for v in args.backdrop.split(",")], dtype=np.uint8)

    tail_steps = None if args.tail <= 0 else max(1, round(args.tail / env.dt))
    frames: list[np.ndarray] = []
    durations: list[int] = []
    for seed in seeds:
        ep_frames, stats = run_episode(env, model, seed, renderer, cam, stride,
                                       tail_steps)
        print(f"  seed {seed:>3}  success={stats['success']}  "
              f"final_distance={stats['distance']:.3f}m  "
              f"frames={len(ep_frames)} ({stats['steps'] * env.dt:.1f}s)")
        ep_durations = [frame_ms] * len(ep_frames)
        # Linger on the last frame so the eye registers the hit before the cut.
        ep_durations[-1] += int(args.hold * 1000)
        frames.extend(fill_backdrop(f, backdrop) for f in ep_frames)
        durations.extend(ep_durations)

    renderer.close()

    images = quantize(frames, args.colors, args.dither)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        args.out,
        save_all=True,
        append_images=images[1:],
        duration=durations,
        loop=0,          # 0 means forever
        optimize=True,
        disposal=1,      # leave each frame in place; the camera never moves
    )

    size = args.out.stat().st_size
    total_s = sum(durations) / 1000
    print(f"wrote {args.out}  {size / 1e6:.2f} MB  "
          f"{args.width}x{args.height}  {len(images)} frames  {total_s:.1f}s")
    if size > 4_000_000:
        print("  warning: over 4 MB for a git repo — try --colors 32 or a "
              "smaller --width/--height")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
