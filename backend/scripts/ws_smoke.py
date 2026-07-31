#!/usr/bin/env python
"""End-to-end websocket check: stream a real training run, then cancel one.

    python scripts/ws_smoke.py [ws://127.0.0.1:8000/ws/train]
"""

from __future__ import annotations

import asyncio
import json
import sys
from collections import Counter

import websockets

URL = sys.argv[1] if len(sys.argv) > 1 else "ws://127.0.0.1:8000/ws/train"


async def run(request: dict, stop_after: int | None = None) -> Counter:
    kinds: Counter = Counter()
    sizes: Counter = Counter()
    async with websockets.connect(URL, max_size=32 * 1024 * 1024) as ws:
        await ws.send(json.dumps(request))
        progress = 0
        while True:
            try:
                raw = await ws.recv()
            except websockets.exceptions.ConnectionClosed:
                # The server closes straight after a fatal error, with no
                # "closed" event. That's a legitimate end of stream.
                break
            ev = json.loads(raw)
            kind = ev["type"]
            kinds[kind] += 1
            sizes[kind] += len(raw)

            if kind == "error":
                print("  ERROR:", ev["message"][:400])
            elif kind == "accepted":
                print(f"  accepted run_id={ev['run_id']}")
            elif kind == "start":
                print(f"  start obs={ev['obs_dim']} act={ev['action_dim']} "
                      f"bodies={len(ev['scene']['bodies'])}")
            elif kind == "progress":
                progress += 1
                if progress % 10 == 0:
                    print(f"    it{ev['iteration']:>3} step={ev['step']:>7,} "
                          f"reward={ev.get('mean_reward', 0):>8.2f} fps={ev['fps']:,}")
                if stop_after and progress == stop_after:
                    print("  -> sending stop")
                    await ws.send(json.dumps({"action": "stop"}))
            elif kind == "rollout":
                print(f"    rollout it{ev['iteration']} frames={ev['n_frames']} "
                      f"({len(raw) / 1024:.0f} KB) stats={ev['stats']}")
            elif kind == "done":
                print(f"  done steps={ev['steps_completed']:,} wall={ev['wall_time']}s "
                      f"stopped_early={ev['stopped_early']}")
            elif kind == "eval":
                print(f"  eval {({k: v for k, v in ev.items() if k not in ('type', 'run_id')})}")
            elif kind == "saved":
                print(f"  saved {ev['run_id']}")
            elif kind == "closed":
                break

    total_kb = sum(sizes.values()) / 1024
    print(f"  events: {dict(kinds)}  |  {total_kb:.0f} KB over the wire")
    return kinds


async def main() -> int:
    print("[1] full run (small budget)")
    a = await run({"task": "reach", "template": "reach_arm", "total_steps": 40000,
                   "rollout_every": 5, "eval_episodes": 10, "save": True})

    print("\n[2] cancelled run")
    b = await run({"task": "locomotion", "template": "wheeled_bot",
                   "total_steps": 800000, "rollout_every": 0,
                   "eval_episodes": 3, "save": False}, stop_after=5)

    print("\n[3] invalid request (task/template mismatch)")
    c = await run({"task": "reach", "template": "quadruped", "total_steps": 10000})

    ok = (
        a["done"] == 1 and a["eval"] == 1 and a["saved"] == 1 and a["rollout"] >= 2
        and b["done"] == 1 and c["error"] == 1
    )
    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
