#!/usr/bin/env python3
"""
Demo for Nek MARL HydroGym wrapper.
"""

import argparse
import numpy as np

from hydrogym.nek import NekMARLGymWrapper


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--config", type=str, required=True)
  parser.add_argument("--run-root", type=str, default="runs")
  parser.add_argument("--run-name", type=str, default=None)
  parser.add_argument(
      "--mode",
      type=str,
      default="blowing",
      choices=["blowing", "opposition"],
  )
  parser.add_argument("--steps", type=int, default=100)
  parser.add_argument("--blowing-amp", type=float, default=0.2)
  parser.add_argument("--opp-gain", type=float, default=0.5)
  return parser.parse_args()


def main():
  args = parse_args()

  env = NekMARLGymWrapper(
      config_path=args.config,
      run_root=args.run_root,
      run_name=args.run_name,
      reward_agg="mean",
  )

  obs = env.reset()

  for step in range(args.steps):
    # Compute per-agent action vector.
    if args.mode == "blowing":
      # Constant blowing for all agents.
      action = np.ones(env.action_space.shape, dtype=np.float32) * args.blowing_amp
    else:
      # Opposition control: negative feedback on the first state component.
      action = np.zeros(env.action_space.shape, dtype=np.float32)
      for i in range(env.n_agents):
        obs_start = i * env.per_agent_obs_size
        obs_val = float(obs[obs_start])
        act_start = i * env.per_agent_act_size
        action[act_start] = -args.opp_gain * obs_val

    obs, reward, done, info = env.step(action)
    if done:
      break

  env.close()


if __name__ == "__main__":
  main()
