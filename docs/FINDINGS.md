# Findings

Five measurements NOVA was built to make. Three of them are corrections of
earlier results in this same document, which is the more useful half.

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

**Actuation appears nearly free.** Halving motor torque, adding 50%, or tripling
joint damping each cost nothing measurable. Motor strength appears nowhere in the
observation, so the policy cannot be adapting to it — which suggested slack in the
task rather than robustness in the policy.

> **Read section 3 before taking this at face value.** That suspicion was correct:
> under a tighter time budget the actuation perturbations collapse by 22–38 points.
> The invariance was headroom, not robustness.

**Geometry degrades gracefully.** A 40% longer forearm still retains 90%. The
fingertip and the target are both in the observation, so part of a geometry
change is visible to the policy even though it never trained on one.

**Mass is what costs** — at this budget. The only feasible perturbation that
meaningfully hurts here is heavier limbs, 93% retained at +30%. It is the axis that
changes what a given torque *does* rather than what the arm looks like, and section
3 shows it costs far more once the clock is tight (56% retained).

> Section 5 finds the opposite on a walking robot, where nearly doubling limb mass
> costs only 9%. "Mass is expensive" is a fact about this arm, not about robots.

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

## 3. That actuation result was an artifact of the time budget

`python -m examples.cross_body_horizon --mode study --horizon 40 --seeds 3`

Section 2 closed by guessing that actuation looked free because the task had
slack — two seconds to reach, dense shaping — and that a tighter budget would
expose it. It does.

First the slack itself, measured rather than assumed. `score()` reads only the
final step of an episode, so it cannot distinguish a policy that arrives at step
12 and parks from one that scrapes in at step 95. Recording the *first* step
inside the success radius separates them:

| body | first-arrival step (30 held-out episodes) |
|---|---|
| baseline | median 21, p75 29, p90 37, max 48 |
| motors −30% | median 27, p75 40, p90 52, max 67 |
| motors −50% | median 38, p75 52, p90 67, max 92 |
| motors +50% | median 16, p75 22, p90 26, max 35 |

The baseline never used more than half its budget. The −50% arm needed nearly all
of it. Both scored 96%, because both finished, and the clock hid the gap.

So: a 40-step variant (`reach_short`, 0.8 s — a `ReachEnv` subclass with a smaller
`max_steps` and an untouched observation, registered as a separate task so the
built-in `reach` keeps its 100-step horizon). 40 sits just above the baseline's
worst case and well below the weak arm's. Baseline trains to 98% ± 2 there, so
the horizon is tight rather than impossible. Both columns come from the same code
path, three seeds, held-out episodes:

| body | success @100 | success @40 | retained @100 | retained @40 |
|---|---|---|---|---|
| baseline | 96% ± 5 | 98% ± 2 | — | — |
| forearm +20% | 92% ± 5 | 94% ± 2 | 97% | 97% |
| forearm +40% | 86% ± 12 | 84% ± 13 | 90% | 86% |
| upper arm +20% | 94% ± 7 | 93% ± 7 | 99% | 95% |
| both links +20% | 92% ± 8 | 93% ± 3 | 97% | 95% |
| links −20% | 98% ± 2 | 100% ± 0 | 102% | 102% |
| motors −30% | 96% ± 5 | 94% ± 2 | 100% | 97% |
| **motors −50%** | 96% ± 5 | **77% ± 6** | 100% | **78%** |
| motors +50% | 96% ± 5 | 93% ± 3 | 100% | 95% |
| **heavy links +30%** | 89% ± 2 | **54% ± 2** | 93% | **56%** |
| **damping ×3** | 97% ± 6 | **62% ± 5** | 101% | **64%** |
| heavy links +60% *(under-actuated)* | 30% ± 6 | 28% ± 5 | n/a | n/a |

**Geometry is untouched; actuation collapses.** Every link-length perturbation
moves 0–3 points, inside the seed spread. The three perturbations that change what
a given torque *does* — weaker motors, heavier limbs, more damping — lose 22 to 38
points of retention. The distinction the study was built to draw, between what the
policy can partly see in its observation and what it cannot see at all, was real
the whole time. It was invisible at 100 steps because both arms had time to finish.

The right reading of the original number is not "the policy is robust to
actuation." It is **"the task did not ask."** Invariance measured with slack in the
budget is a statement about the budget.

`feasible()` still draws the line in the right place. `damping ×3` and `heavy links
+30%` both lose ~37 points here and both are correctly called feasible — the check
compares holding torque against available torque, certifying that a body can hold
itself up, not that it can move quickly. Those are genuine transfer failures on
bodies that *could* do the task, which is exactly what the under-actuated control
exists to be contrasted against.

---

## 4. The quadruped's bad gait was a physics bug, not a reward problem

The quadruped was the project's weakest demo: a 2.8 m shuffle where the rover
managed 13 m. The obvious diagnosis was the reward — it had no gait structure at
all, nothing rewarding foot clearance or penalizing thrash.

That diagnosis was wrong, and the reward work that followed from it made things
worse. Every standard legged-locomotion term, implemented and measured:

| variant | seeds | distance (m) | fall rate |
|---|---|---|---|
| baseline | 5 | 3.34 ± 0.83 | 0.67 |
| air-time + action-rate + tilt | 3 | 2.80 ± 0.32 | 0.98 |
| same, lower weight + fall penalty | 3 | 2.87 ± 0.14 | 0.90 |
| velocity tracking @ 1 m/s | 1 | 3.87 | 0.43 |
| **spawn fix, no reward change** | **5** | **7.28 ± 1.17** | **0.18** |
| spawn fix + air-time | 3 | 5.63 ± 1.00 | 0.32 |

The actual bug: the quadruped spawned with its **feet 8.5 cm below the floor**.
`build()` set a start height with a comment claiming the legs spawn "partly
folded" — but nothing folded them. `qpos0` was zeros, so the legs were locked
straight at 0.34 m under a torso at 0.285 m. The contact solver resolved the
interpenetration by launching the robot: with zero torque applied it left the
ground at **+1.94 m/s and peaked at 0.50 m**, 75% above its own spawn height.
Every episode began with a somersault the policy had to survive before it could
try to walk. That was the "scramble".

The fix solves 2-link IK for a crouched stance that puts each foot exactly on the
floor, and emits the legs already folded. Zero-torque launch becomes a −0.14 m/s
settle. Nothing about the reward changed — `locomotion.py` is byte-for-byte
unmodified, and the rover that shares it is per-seed identical to four decimal
places.

Two things worth taking from this. **A comment asserted behaviour the code never
implemented**, and it read plausibly enough to send the investigation at the
reward for a long time. And **reward engineering on top of a broken simulation
made the numbers worse while looking like progress** — the air-time term bought
leaping and a 98% fall rate on a robot that was already being thrown into the air.

A gap this exposes, still open: obs-layout versioning protects against changed
*observations*, but a change to a robot's *geometry* is not covered. A policy
exported against the old straight-legged quadruped is still accepted and simply
behaves differently.

---

---

## 5. On a robot that carries its own weight, mass is cheap and actuation is everything

`python -m examples.cross_body_locomotion --seeds 3` — about 6 minutes.

Section 2 concluded that mass was the expensive axis. That was measured on a
bolted-down arm, which only has to swing its limbs. A quadruped has to hold its
whole body up on them, so it should be the harsher test. It is not — it inverts
the result.

Three training seeds on the standard quadruped, scored zero-shot on bodies it
never saw:

| body | distance (m) | retained | native control |
|---|---|---|---|
| baseline | 7.28 ± 0.67 | — | — |
| legs +20% | 7.13 ± 1.54 | 98% | — |
| legs −20% | 5.96 ± 0.60 | 82% | — |
| torso longer +30% | 7.37 ± 1.25 | 101% | — |
| torso wider +30% | 7.24 ± 1.03 | 99% | — |
| **heavy legs +40%** | 6.60 ± 1.15 | **91%** | — |
| motors −30% | 6.11 ± 0.43 | 84% | — |
| **motors −50%** | 2.00 ± 0.92 | **27%** | 4.85 m |
| motors +50% | 2.66 ± 0.19 | 37% | **3.00 m** |
| **damping ×3** | 3.30 ± 0.43 | **45%** | 7.93 m |

**Mass barely registers.** Legs 40% thicker — nearly double the limb mass, on a
robot that must lift them every step — costs 9%. On the arm the same family of
change cost 44% once the clock was tight. The walking policy has proprioception
and a gravity vector in its observation and a dense velocity reward, so it can
feel a heavier leg and push harder; the arm's open-loop torque profile could not.

**Actuation still dominates.** Halving torque drops it to 27% of baseline.
Tripling damping drops it to 45%. Same split as the arm — what the policy cannot
observe is what breaks it — but here the split is between actuation and
*everything else*, geometry and mass alike.

### The control caught a third false conclusion

`feasible()` is arm-specific: it compares holding torque to available torque for
a two-link arm and returns `True` for everything else, so a quadruped gets no
static check at all. Rather than trust the transfer numbers, every body retaining
under 55% was given a policy trained natively *on it*.

That immediately paid for itself. **`motors +50%` transferred at 37% — but a
policy trained natively on it only reaches 3.00 m against the baseline's 7.28 m.**
A quadruped with 50% stronger motors is simply worse at this task; it has enough
torque to launch itself off the ground and the reward's alive bonus does not
discourage that. Reported without the control, it would have looked like a 63%
transfer collapse. It is not a transfer result at all.

The other two are real: `motors −50%` transfers at 2.00 m but trains to 4.85 m,
and `damping ×3` transfers at 3.30 m but trains to **7.93 m — better than
baseline**. High damping is a fine robot to be if you train for it, and a
terrible one to be handed.

This is the third time in this document that a dramatic number turned out to be
about the body rather than the policy. The first two were caught by a static
physics check; this one needed an empirical control, because no static check
covers a legged robot. The general lesson is not "add a torque check" — it is
that **a zero-shot transfer number is meaningless without knowing what a policy
trained on that body can do.**

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
  differences — a 22-point collapse is safe, a 3-point move is not. Nothing
  here adjudicates a 10-point gap.

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
