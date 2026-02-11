#!/usr/bin/env python3
"""
Demo for Nek integration utilities similar to firedrake run-transient.py
"""

import argparse
import psutil
import numpy as np

import hydrogym.nek as hgym


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--config", type=str, required=True)
  parser.add_argument("--run-root", type=str, default="runs")
  parser.add_argument("--run-name", type=str, default=None)
  parser.add_argument("--dt", type=float, default=None, help="Time step (uses config default if not provided)")
  return parser.parse_args()


def log_postprocess(env):
  """Extract values to log from the environment"""
  mem_usage = psutil.virtual_memory().percent
  
  # Get reward from last step if available
  if hasattr(env, 'last_reward'):
    reward = env.last_reward
  else:
    reward = 0.0
  
  return reward, mem_usage


def controller(t, obs, env):
  """Simple controller function"""
  # Return zero action (no control)
  # User can modify this to implement their own control strategy
  # Action shape must match env.action_space.shape, not observation shape
  if hasattr(env, 'action_space'):
    return np.zeros(env.action_space.shape, dtype=np.float32)
  return None


def main():
  args = parse_args()

  # Create environment
  env = hgym.NekMARLGymWrapper(
      config_path=args.config,
      run_root=args.run_root,
      run_name=args.run_name,
      reward_agg="mean",
  )

  # Time step
  dt = -1*env.conf.simulation.dt if args.dt is None else args.dt
  max_steps = env.conf.runner.nb_interactions *\
            env.conf.runner.nb_warmup_episodes *\
            env.conf.runner.nb_episodes * env.conf.simulation.ndrl
  T_final = max_steps * dt

  # Set up the callback
  print_fmt = "t: {0:.2f},\t\t Reward: {1:.3f},\t\t Mem: {2:.1f}"
  log = hgym.io.LogCallback(
      postprocess=log_postprocess,
      nvals=2,
      interval=1,
      print_fmt=print_fmt,
      filename=f"{env.conf.simulation.CASENAME}_log.dat",
  )

  callbacks = [
      log,
      # hgym.io.CheckpointCallback(interval=10, filename="checkpoint"),
  ]

  hgym.print("Beginning integration")
  hgym.integrate(
      env,
      t_span=(0, T_final),
      dt=dt,
      callbacks=callbacks,
      max_steps=max_steps,
      controller=controller,
  )

  env.close()


if __name__ == "__main__":
  main()

