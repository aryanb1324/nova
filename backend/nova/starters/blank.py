"""Blank skeleton — the loop, with the algorithm left out.

Everything NOVA gives you is in the first 15 lines. The rest is yours.
Replace the marked line and you have your own agent.
"""

import numpy as np

import nova

TASK = "reach"          # "reach" or "locomotion"
TEMPLATE = "reach_arm"  # "reach_arm", "wheeled_bot", "quadruped"

env = nova.make(TASK, TEMPLATE)
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]

print(f"observations: {obs_dim}   actions: {act_dim}   "
      f"episode: {env.max_steps} steps at {1 / env.dt:.0f} Hz")
print("actions are continuous in [-1, 1]; observations are already scaled")

ITERATIONS = 40
EPISODES_PER_ITERATION = 10
TOTAL = ITERATIONS * EPISODES_PER_ITERATION * env.max_steps

steps = 0
with nova.attach(TASK, TEMPLATE, algo="my algorithm", total_steps=TOTAL) as run:
    for iteration in range(ITERATIONS):
        returns = []

        for _ in range(EPISODES_PER_ITERATION):
            obs, _ = env.reset()
            total = 0.0
            while True:
                # ------------------------------------------------------------
                # YOUR POLICY GOES HERE. Right now it flails at random.
                action = env.action_space.sample()
                # ------------------------------------------------------------

                obs, reward, terminated, truncated, info = env.step(action)
                total += reward
                steps += 1
                if terminated or truncated:
                    break
            returns.append(total)

        # ----------------------------------------------------------------
        # AND YOUR UPDATE GOES HERE.
        # ----------------------------------------------------------------

        mean_return = float(np.mean(returns))
        print(f"iteration {iteration + 1:>3}  return {mean_return:>8.2f}  steps {steps:,}")

        # Draws the point on the reward chart in the browser.
        run.log(step=steps, mean_reward=mean_return)

# When you have something worth keeping:
#
#   nova.export(policy, "mine.onnx", task=TASK, template=TEMPLATE)
#   print(nova.evaluate("mine.onnx"))
#
# `policy` needs to be a torch module mapping (batch, obs_dim) -> (batch, act_dim).
