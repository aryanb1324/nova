"""Does a policy survive a body it never trained on?

Train on one arm. Then, without any retraining, run that same policy on arms with
a longer forearm, weaker motors, heavier links. Nothing about the interface
changes - the observation layout belongs to the *task*, not the body, so the
policy still receives 18 well-formed numbers. They just describe a machine it has
never controlled.

This is the cheap, honest version of a question that matters well beyond this
project: how much of what a policy learned is about the task, and how much is
about the exact machine it was trained on? A policy that only works on its own
body has memorized a robot rather than learned to reach.

Two properties make the answer trustworthy:

  * Evaluation uses the held-out seed set, so transfer is measured on episodes
    the policy never saw either.
  * Every cell averages several training seeds, because a single run's transfer
    number is dominated by which run you happened to get.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field

from .. import robots
from ..bench import aggregate
from ..policies.evaluate import held_out_seeds, score


@dataclass(frozen=True)
class Body:
    """One robot, described as a multiplicative delta from the training body."""

    name: str
    delta: dict = field(default_factory=dict)

    def params_for(self, template: str) -> dict:
        """Concrete parameters. Multiplicative so a perturbation keeps its
        meaning across templates with different natural scales."""
        template_obj = robots.get(template)
        out = dict(template_obj.defaults())
        for key, factor in self.delta.items():
            if key in out:
                out[key] = out[key] * factor
        # resolve() clamps to each parameter's declared range, so a perturbation
        # can never ask for a body the simulator would refuse.
        return template_obj.resolve(out)

    def is_clamped(self, template: str) -> bool:
        """True if the declared range absorbed part of this perturbation.

        Worth surfacing: a request for a 60% longer link that silently clamps to
        40% is not the experiment you think you ran.
        """
        base = robots.get(template).defaults()
        want = {k: base[k] * f for k, f in self.delta.items() if k in base}
        got = self.params_for(template)
        return any(abs(got[k] - v) > 1e-9 for k, v in want.items())


def holding_torque(template: str, params: dict) -> tuple[float, float]:
    """(required, available) shoulder torque to hold a 2-link arm horizontal.

    Specific to `reach_arm`, and worth the specificity: without it a body can be
    physically incapable of the task and still be reported as a *transfer*
    failure. Scaling link radius by 1.6 multiplies mass by 2.56, which pushes
    required torque past what the motors deliver - the policy is then blameless
    and the number means nothing about generalization.
    """
    import mujoco

    model = mujoco.MjModel.from_xml_string(robots.build_mjcf(template, params))
    upper, fore = float(model.body_mass[2]), float(model.body_mass[3])
    l1, l2 = params["upper_length"], params["fore_length"]
    required = upper * 9.81 * (l1 / 2) + fore * 9.81 * (l1 + l2 / 2)
    return required, float(model.actuator_gear[1, 0])


def feasible(template: str, params: dict) -> bool:
    """Can this body hold itself against gravity at all?"""
    if template != "reach_arm":
        return True
    required, available = holding_torque(template, params)
    return available >= required


#: The perturbation ladder for the reaching arm. Chosen to separate two very
#: different kinds of change: geometry, which the policy can partly see through
#: its own observations, and actuation, which it cannot see at all.
#:
#: The last entry is deliberately under-actuated. It is kept as a control: a
#: study that reports transfer numbers should be able to show what a *physically
#: impossible* body looks like, so the reader can tell that apart from a policy
#: that merely failed to generalize.
REACH_BODIES: tuple[Body, ...] = (
    Body("baseline"),
    Body("forearm +20%", {"fore_length": 1.20}),
    Body("forearm +40%", {"fore_length": 1.40}),
    Body("upper arm +20%", {"upper_length": 1.20}),
    Body("both links +20%", {"upper_length": 1.20, "fore_length": 1.20}),
    Body("links -20%", {"upper_length": 0.80, "fore_length": 0.80}),
    Body("motors -30%", {"gear": 0.70}),
    Body("motors -50%", {"gear": 0.50}),
    Body("motors +50%", {"gear": 1.50}),
    Body("heavy links +30%", {"link_radius": 1.30}),
    Body("damping x3", {"joint_damping": 3.0}),
    Body("heavy links +60% (under-actuated)", {"link_radius": 1.60}),
)

BODIES = {"reach_arm": REACH_BODIES}


def bodies_for(template: str) -> tuple[Body, ...]:
    return BODIES.get(template, (Body("baseline"),))


def evaluate_across_bodies(policy, task: str, template: str,
                           bodies: tuple[Body, ...],
                           episodes: int = 30) -> list[dict]:
    """Score one trained policy on every body, using held-out episodes."""
    seeds = held_out_seeds(episodes)
    rows = []
    for body in bodies:
        params = body.params_for(template)
        result = score(policy, task, template, params, seeds)
        rows.append({
            "body": body.name,
            "delta": body.delta,
            "clamped": body.is_clamped(template),
            # Separates "the policy could not cope" from "no policy could".
            "feasible": feasible(template, params),
            **{k: v for k, v in result.items() if k != "episodes"},
        })
    return rows


def summarize(results: list[dict], metric: str = "success_rate") -> dict[str, dict]:
    """Collapse per-training-seed rows into one figure per body."""
    by_body: dict[str, list[float]] = {}
    feasibility: dict[str, bool] = {}
    for row in results:
        if metric in row:
            by_body.setdefault(row["body"], []).append(float(row[metric]))
            feasibility[row["body"]] = row.get("feasible", True)

    out = {}
    for body, values in by_body.items():
        agg = aggregate(values, metric)
        if agg:
            out[body] = {"mean": agg.mean, "std": agg.std, "n": agg.n,
                         "values": agg.values, "feasible": feasibility[body]}
    return out


def format_table(summary: dict[str, dict], baseline: str = "baseline",
                 percent: bool = True) -> str:
    """Markdown, with retention relative to the training body."""
    if not summary:
        return "(no results)"
    base = summary.get(baseline, {}).get("mean")

    rows = ["| body | success (held-out) | retained |", "|---|---|---|"]
    for name, entry in summary.items():
        value = (f"{entry['mean'] * 100:.0f}% ± {entry['std'] * 100:.0f}"
                 if percent else f"{entry['mean']:.2f} ± {entry['std']:.2f}")
        if name == baseline:
            retained = "—"
        elif not entry.get("feasible", True):
            # Reporting a percentage here would invite the wrong reading.
            retained = "n/a — body is under-actuated"
        else:
            retained = f"{entry['mean'] / base * 100:.0f}%" if base else "—"
        rows.append(f"| {name} | {value} | {retained} |")
    return "\n".join(rows)


def save(path: str | pathlib.Path, payload: dict) -> pathlib.Path:
    path = pathlib.Path(path)
    path.write_text(json.dumps(payload, indent=2))
    return path
