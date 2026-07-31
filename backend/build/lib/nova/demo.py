"""The whole NOVA story in one command.

Trains a robot, exports the policy as a portable artifact, scores it on episodes
it has never seen, and hands it back to the running app. If the UI is up, the
whole thing streams into it live.
"""

from __future__ import annotations

import json
import pathlib
import time
import urllib.error
import urllib.request

BAR = "─" * 66


def _say(step: int, title: str) -> None:
    print(f"\n\033[1m{BAR}\n {step}. {title}\n{BAR}\033[0m")


def _ui_is_up(url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{url}/api/health", timeout=2) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError, TimeoutError):
        return False


def run_demo(
    task: str = "reach",
    template: str = "reach_arm",
    steps: int | None = None,
    out: str | pathlib.Path = "demo_policy.onnx",
    url: str = "http://127.0.0.1:8000",
) -> int:
    import nova
    from nova import envs
    from nova.training import TrainConfig, train

    out = pathlib.Path(out)
    live = _ui_is_up(url)
    total = steps or envs.get_task(task)["typical_steps"]

    print("\n\033[1mNOVA — Next-gen Open Virtual AI\033[0m")
    print(f"  {'streaming to ' + url if live else 'no UI running; console only'}")
    if live:
        print("  open http://localhost:5173 to watch")

    # 1 ----------------------------------------------------------------
    _say(1, "What's available")
    for t in nova.templates():
        print(f"   {t['key']:<13} {t['name']:<16} {len(t['params'])} tunable parameters")
    for t in nova.tasks():
        print(f"   task {t['key']:<9} {t['description']}")

    # 2 ----------------------------------------------------------------
    _say(2, f"Train {template} to {task} ({total:,} steps)")
    cfg = TrainConfig(task=task, template=template, total_steps=total, rollout_every=8)
    stream = nova.attach(task, template, algo="PPO (built-in)", total_steps=total,
                         url=url) if live else None

    def on_event(ev: dict) -> None:
        if stream is not None and ev["type"] in ("progress", "rollout"):
            stream.send(ev)
        if ev["type"] == "progress" and ev["iteration"] % 10 == 0:
            metric = ""
            if "is_success" in ev:
                metric = f"  success={ev['is_success']:>4.0%}"
            elif "distance" in ev:
                metric = f"  distance={ev['distance']:>5.2f}m"
            print(f"   {ev['step']:>8,}/{total:,}  reward={ev.get('mean_reward', 0):>8.2f}"
                  f"{metric}  {ev['fps']:>6,} steps/s")

    started = time.perf_counter()
    result = train(cfg, on_event=on_event)
    if stream is not None:
        stream.finish(wall_time=round(time.perf_counter() - started, 2))
    print(f"   done in {result['wall_time']}s")

    # 3 ----------------------------------------------------------------
    _say(3, "Export it as a portable policy")
    manifest = nova.export(result["model"], out, task=task, template=template,
                           algo="PPO (stable-baselines3)", notes="nova demo")
    size = out.stat().st_size
    print(f"   {out}  ({size:,} bytes)")
    print(f"   carries: {manifest.task}/{manifest.template}, "
          f"{manifest.obs_dim}->{manifest.act_dim}, layout {manifest.obs_layout}")
    print("   ONNX, not a pickle — this file is data, so opening it can't run code.")

    # 4 ----------------------------------------------------------------
    _say(4, "Score it on episodes it has never seen")
    scores = nova.evaluate(out, episodes=30)
    scores.pop("rollout", None)
    _table(scores)

    # 5 ----------------------------------------------------------------
    _say(5, "Hand it back to the app")
    if live:
        try:
            req = urllib.request.Request(
                f"{url}/api/policies", data=out.read_bytes(),
                headers={"content-type": "application/octet-stream"}, method="POST")
            with urllib.request.urlopen(req, timeout=120) as resp:
                uploaded = json.loads(resp.read())
            print(f"   uploaded as {uploaded['policy_id']}")
            print("   it's in 'Bring your own policy' now — click it to replay.")
        except urllib.error.HTTPError as exc:
            print(f"   upload rejected: {exc.read()[:200].decode(errors='replace')}")
        except (urllib.error.URLError, OSError) as exc:
            print(f"   upload failed: {exc}")
    else:
        print(f"   start the app with ./dev.sh, then drop {out} on it.")

    print(f"\n\033[1mThat's the loop.\033[0m Anyone can do step 2 with their own "
          f"algorithm:\n   python examples/train_with_cem.py\n")
    return 0


def _table(scores: dict) -> None:
    pub, priv = scores["public"], scores["held_out"]
    rows = [("return", "mean_return", "{:.2f}")]
    if "success_rate" in pub:
        rows.append(("success", "success_rate", "{:.0%}"))
    if "mean_final_distance" in pub:
        rows.append(("distance", "mean_final_distance", "{:.3f}m"))

    print(f"   {'':<10}{'published seeds':>18}{'held-out seeds':>18}")
    for label, key, fmt in rows:
        a = fmt.format(pub[key]) if key in pub else "—"
        b = fmt.format(priv[key]) if key in priv else "—"
        print(f"   {label:<10}{a:>18}{b:>18}")
    print("\n   Held-out seeds come from a private salt, so a submission can't be")
    print("   tuned against the episodes it will be graded on.")
