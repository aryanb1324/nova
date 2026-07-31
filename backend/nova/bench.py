"""Report results with the spread they actually have.

A single training run tells you almost nothing about an RL algorithm. Two runs
of identical code with different seeds routinely land far apart, so a headline
number from one run is closer to an anecdote than a measurement.

Two different variances get confused constantly, and the distinction is the
whole point of this module:

  WITHIN-RUN   `evaluate()` averages one trained policy over 30 episodes. That
               measures how consistent that policy is across starting states.

  ACROSS-RUN   This module trains N times from N seeds and takes the spread of
               the per-run scores. That measures how reliably the *algorithm*
               produces a working policy at all - which is what a comparison
               between two algorithms is actually claiming.

Anything published should quote the second.
"""

from __future__ import annotations

import json
import os
import pathlib
import statistics
from dataclasses import asdict, dataclass, field

#: Set by the benchmark harness. A script that calls `record()` writes its
#: result here; with the variable unset the call does nothing.
OUT_ENV = "NOVA_BENCH_OUT"
SEED_ENV = "NOVA_SEED"

DEFAULT_SEEDS = (0, 1, 2, 3, 4)


def seed_from_env(default: int = 0) -> int:
    """The seed this process should use, so one script can be run N times."""
    try:
        return int(os.environ.get(SEED_ENV, default))
    except ValueError:
        return default


def record(label: str, scores: dict, **extra) -> None:
    """Append one run's result to the benchmark log, if one is collecting.

    Safe to leave in a script permanently: with no log configured it returns
    immediately.
    """
    path = os.environ.get(OUT_ENV)
    if not path:
        return

    entry = {"label": label, "seed": seed_from_env(), **extra}
    for split in ("public", "held_out"):
        if isinstance(scores.get(split), dict):
            entry[split] = {k: v for k, v in scores[split].items() if k != "rollout"}
    if "public" not in entry:
        # A flat score dict, as returned by evaluate(..., seeds=...).
        entry["public"] = {k: v for k, v in scores.items() if k != "rollout"}

    with open(path, "a") as fh:
        fh.write(json.dumps(entry) + "\n")


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------

@dataclass
class Aggregate:
    """One metric, summarized across training seeds."""

    metric: str
    n: int
    mean: float
    std: float
    lo: float
    hi: float
    values: list[float] = field(default_factory=list)

    def format(self, percent: bool = False, places: int = 2) -> str:
        if percent:
            return f"{self.mean * 100:.0f}% ± {self.std * 100:.0f}"
        return f"{self.mean:.{places}f} ± {self.std:.{places}f}"


def aggregate(values: list[float], metric: str) -> Aggregate | None:
    if not values:
        return None
    return Aggregate(
        metric=metric,
        n=len(values),
        mean=float(statistics.fmean(values)),
        # Sample standard deviation: N seeds estimate the spread of a process,
        # they do not describe a complete population. Undefined for a single
        # run - which is exactly the situation this module exists to expose.
        std=float(statistics.stdev(values)) if len(values) > 1 else 0.0,
        lo=min(values),
        hi=max(values),
        values=[round(float(v), 4) for v in values],
    )


def summarize(entries: list[dict], split: str = "held_out") -> dict[str, dict]:
    """Group per-run results by label, and summarize each metric across seeds."""
    by_label: dict[str, list[dict]] = {}
    for entry in entries:
        scores = entry.get(split) or entry.get("public") or {}
        by_label.setdefault(entry["label"], []).append(scores)

    out: dict[str, dict] = {}
    for label, runs in by_label.items():
        metrics = {
            k for run in runs for k, v in run.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        }
        summary = {}
        for metric in sorted(metrics):
            agg = aggregate([r[metric] for r in runs if metric in r], metric)
            if agg:
                summary[metric] = asdict(agg)
        out[label] = {"seeds": len(runs), "metrics": summary}
    return out


#: The reference implementations, and roughly how long each takes per seed on a
#: 10-core M-series. Used only to warn before a long sweep.
STARTERS = {
    "reinforce": 75,
    "ppo_minimal": 115,
    "cem": 105,
    "sb3_custom": 35,
}


def run_sweep(
    labels: tuple[str, ...] = tuple(STARTERS),
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    out: str | pathlib.Path = "bench_results.jsonl",
    on_progress=None,
) -> pathlib.Path:
    """Run each starter once per seed, appending results to `out`.

    Sequential on purpose: every run already saturates the cores, so running
    them concurrently would only make each one slower and the timings
    meaningless.

    Resumable - a (label, seed) already present in the log is skipped, so an
    interrupted sweep can be restarted without repeating work.
    """
    import subprocess
    import sys
    import time

    out = pathlib.Path(out)
    done = {(e["label"], e["seed"]) for e in load(out)}

    for seed in seeds:
        for label in labels:
            if (label, seed) in done:
                if on_progress:
                    on_progress(label, seed, "cached", 0.0)
                continue

            env = {**os.environ, SEED_ENV: str(seed), OUT_ENV: str(out.resolve()),
                   "OMP_NUM_THREADS": "1"}
            started = time.perf_counter()
            proc = subprocess.run(
                [sys.executable, "-m", f"nova.starters.{label}"],
                cwd=str(pathlib.Path(__file__).resolve().parents[1]),
                env=env, capture_output=True, text=True,
            )
            elapsed = time.perf_counter() - started
            if proc.returncode == 0:
                status = "ok"
            else:
                tail = (proc.stderr or proc.stdout).strip().splitlines()
                status = f"FAILED({proc.returncode}) {tail[-1][:80] if tail else ''}"
            if on_progress:
                on_progress(label, seed, status, elapsed)
    return out


def load(path: str | pathlib.Path) -> list[dict]:
    path = pathlib.Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def format_table(
    summary: dict[str, dict],
    metrics: tuple[str, ...] = (),
    percent_metrics: tuple[str, ...] = ("success_rate", "is_success"),
) -> str:
    """A markdown table of mean ± std, ready to paste into the README."""
    if not summary:
        return "(no results)"

    if not metrics:
        seen: list[str] = []
        for entry in summary.values():
            for m in entry["metrics"]:
                if m not in seen:
                    seen.append(m)
        preferred = [m for m in ("success_rate", "mean_return", "mean_final_distance")
                     if m in seen]
        metrics = tuple(preferred or seen[:3])

    rows = [
        "| run | seeds | " + " | ".join(metrics) + " |",
        "|---|---|" + "---|" * len(metrics),
    ]
    for label, entry in summary.items():
        cells = [label, str(entry["seeds"])]
        for metric in metrics:
            raw = entry["metrics"].get(metric)
            cells.append(
                Aggregate(**raw).format(percent=metric in percent_metrics)
                if raw else "—"
            )
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)
