#!/usr/bin/env python3
"""
Run a transient simulation of a Nek environment, basically a wrapper around the hydrogym.integrate function.
The control here is zero actuation.
Demo for Nek integration utilities similar to firedrake run-transient.py.
"""

import argparse
import psutil
import numpy as np

import hydrogym.nek as hgym
from hydrogym.nek import make_afc_controller
from omegaconf import OmegaConf
from typing import List
from hydrogym.nek.configs import Config

def parse_omegaconf(conf_file: str, overrides: List[str]):
  conf = OmegaConf.merge(
      OmegaConf.structured(Config()),
      OmegaConf.load(conf_file),
      OmegaConf.from_dotlist(overrides),
  )
  return conf

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--config", type=str, required=True)
  parser.add_argument("--overrides", type=str, nargs="*", help="Config overrides, e.g. `other.gpus=4`")
  parser.add_argument("--steps", type=float, default=None)
  parser.add_argument("--ctrl_type", type=str, default="OC")
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


def main():
  # -- Parse the arguments --
  args = parse_args()
  # -- Parse the configuration --
  conf = parse_omegaconf(str(args.config), args.overrides)

  # -- Create the environment --
  env = hgym.NekMARLGymWrapper(
      conf=conf,
      reward_agg="mean",
  )

  # -- Create controller --
  controller = make_afc_controller(env, ctrl_type=args.ctrl_type)
  if controller is None:
    raise ValueError(f"Controller type {args.ctrl_type} not supported")
  else:
    print(f"Controller type {args.ctrl_type} created successfully")


  # -- Time step and max steps --
  dt = np.abs(env.conf.simulation.dt)
  ideal_max_transient_steps = env.conf.runner.nb_interactions *\
            env.conf.runner.nb_episodes *\
            env.conf.simulation.ndrl
  T_final = ideal_max_transient_steps * dt
  # if steps is provided, use it to calculate the final time
  max_steps = int(T_final / dt) if args.steps is None else args.steps
  T_final = max_steps * dt

  # -- Set up the callback --
  print_fmt = "t: {0:.2f},\t\t Reward: {1:.3f},\t\t Mem: {2:.1f}"
  log = hgym.io.LogCallback(
      postprocess=log_postprocess,
      nvals=2,
      interval=1,
      print_fmt=print_fmt,
      filename=f"log-files/callback_{env.conf.simulation.CASENAME}_log.dat",
  )

  callbacks = [
      log,
  ]


  # -- Integrate the environment --
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

