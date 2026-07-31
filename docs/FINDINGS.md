# Findings

Two measurements NOVA was built to make, and one mistake worth reading about.

Everything here reproduces from a clean checkout on a 10-core Apple M5, CPU only.

---

## 1. One run of an RL algorithm is an anecdote

`nova bench --seeds 5` — about 27 minutes.

Four reference algorithms, identical 400k-step budget on the same task and
comparable network sizes, scored on held-out episodes:

| algorithm | 1 seed (previously published) | 5 seeds |
|---|---|---|
| `reinforce` | 3% | **1% ± 1** |
| `cem` | 0% | **3% ± 4** |
| `ppo_minimal` | 87% | **70% ± 16** |
| `sb3_custom` | 100% | **87% ± 12** |

The single-seed numbers were flattering in both cases that mattered. That is the
ordinary outcome rather than bad luck: if you run once and report it, you report
the run you got, and nothing in that run tells you where it sat in the
distribution.

The spread is the more useful number. A ±16-point standard deviation on
`ppo_minimal` is wider than most differences a reader would want to draw out of
such a table — so a comparison between two algorithms within ~15 points of each
other, at this seed count, is not a result.

What does survive: `reinforce` and `ppo_minimal` are about the same length, see
the same environment and the same amount of data, and land 69 points apart. The
difference is a learned baseline, importance ratios, and reusing each batch for
several gradient steps.

**A distinction that gets conflated.** Averaging 30 episodes measures how
consistent *one policy* is across starting states. It says nothing about how
reliably the *algorithm* produces a working policy. Only re-training measures
that, and only the second is what an algorithm comparison claims.

---

## 2. Zero-shot transfer across bodies is more robust than expected — and mass is the expensive axis

`python -m examples.cross_body_study --seeds 3` — about 2 minutes.

Train on one arm; run the policy unchanged on arms it has never seen. The
observation layout belongs to the task rather than the body, so the policy still
receives 18 well-formed numbers — they simply describe a different machine.

Three training seeds, held-out episodes:

| body | success | retained |
|---|---|---|
| baseline | 96% ± 5 | — |
| forearm +20% | 92% ± 5 | 97% |
| forearm +40% | 86% ± 12 | 90% |
| upper arm +20% | 94% ± 7 | 99% |
| both links +20% | 92% ± 8 | 97% |
| links −20% | 98% ± 2 | 102% |
| motors −30% | 96% ± 5 | 100% |
| motors −50% | 96% ± 5 | 100% |
| motors +50% | 96% ± 5 | 100% |
| heavy links +30% | 89% ± 2 | 93% |
| damping ×3 | 97% ± 6 | 101% |
| heavy links +60% *(under-actuated)* | 30% ± 6 | n/a |

**Actuation is nearly free.** Halving motor torque, adding 50%, or tripling joint
damping each cost nothing measurable. This is the surprising half: motor strength
appears nowhere in the observation, so the policy cannot be adapting to it. The
likely explanation is slack in the task — two seconds to reach, dense shaping, and
a controller that already saturates — so a weaker arm still arrives in time. A
tighter time budget would probably expose it, which is the obvious next
experiment.

**Geometry degrades gracefully.** A 40% longer forearm still retains 90%. The
fingertip and the target are both in the observation, so part of a geometry
change is visible to the policy even though it never trained on one.

**Mass is what costs.** The only feasible perturbation that meaningfully hurts is
heavier limbs — 93% retained at +30%. It is the axis that changes what a given
torque *does*, rather than what the arm looks like.

### The mistake

The first version of this study reported `heavy links +60%` at 30% and concluded
that inertia breaks generalization. That conclusion was wrong.

Scaling link radius by 1.6 multiplies mass by 2.56, which raises the torque
needed to hold the arm horizontal to **9.46 N·m against 8.00 N·m available**. The
arm cannot hold itself up. No policy, however trained, can perform that task on
that body — so the number said nothing about transfer.

The study now computes required-versus-available holding torque for every body
and labels infeasible ones instead of scoring them as failures
(`nova/study/cross_body.py`). The under-actuated body is deliberately kept in the
table as a control, so a reader can see what physical impossibility looks like
next to genuine generalization loss.

This is the strongest argument for the discipline applied elsewhere in the
project: the wrong version of this result was more interesting than the right
one, and nothing but an explicit check distinguished them.

---

## What this is not

- **Not novel robotics research.** Small MuJoCo tasks, standard algorithms.
  Nothing here is state of the art and nothing is meant to be.
- **Not a sim-to-real claim.** No hardware, no domain randomization, no transfer
  to a physical robot.
- **Not a general result about RL.** One task family, one observation design, one
  machine. The transfer numbers describe *this* reach task; the methodology is
  the transferable part, not the percentages.
- **Not enough seeds to settle close calls.** Three to five seeds bounds gross
  differences. It does not adjudicate a 10-point gap.

## Reproducing

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r backend/requirements.txt
.venv/bin/pip install -e ./backend

cd backend
nova bench --seeds 5                            # section 1
python -m examples.cross_body_study --seeds 3   # section 2
```

Raw per-run data lands in `bench_results.jsonl` and `cross_body.json`, both
per-seed, so the aggregates can be recomputed or disputed.

## Open questions, all cheap to run

- Does the actuation-invariance survive a shorter episode budget?
- Does it hold for locomotion, where the body has to carry itself?
- Which observation additions — link lengths, mass, torque limits — would let a
  body-conditioned policy beat an unconditioned one?
- More seeds on the algorithm table, to see which gaps are real.

New robots and tasks register from your own package without forking; see the
"Use it as a library" section of the README.
