"""
Minimal initialization for Nek runs.
Prepares the run folder, copies case files, and writes RUN_PATH_*.txt.
"""

import argparse
import os
from pathlib import Path
from typing import List

from omegaconf import OmegaConf

from hydrogym.nek import Config
from hydrogym.nek.nek_lib.nek_utils import NEK_INIT, show_end


def parse_omegaconf(conf_file: str, overrides: List[str]):
  conf = OmegaConf.merge(
      OmegaConf.structured(Config()),
      OmegaConf.load(conf_file),
      OmegaConf.from_dotlist(overrides),
  )
  return conf


def main():

  # -- Load environment --
  parser = argparse.ArgumentParser()
  parser.add_argument("--config", type=Path, help="YAML configuration")
  parser.add_argument(
      "overrides",
      type=str,
      nargs="*",
      help="Config overrides, e.g. `other.gpus=4`",
  )
  args = parser.parse_args()

  conf = parse_omegaconf(str(args.config), args.overrides)

  # Set run name if restarting from a known agent.
  if conf.runner.agent_run_name != 0:
    conf.logging.run_name = conf.runner.agent_run_name
    conf.runner.load_agent = True
  else:
    conf.runner.rewrite_input_files = True

  # -- Create run root dir --
  if not os.path.exists(conf.logging.save_dir):
    os.makedirs(conf.logging.save_dir, exist_ok=True)
  print(f"[NEK] RUN ROOT DIR: {conf.logging.save_dir}", flush=True)

  # -- Create case run folder --
  run_folder = f"{conf.logging.save_dir}/{conf.logging.run_name}"
  if not os.path.exists(run_folder):
    os.makedirs(run_folder, exist_ok=True)
  print(f"[NEK] CASE RUN FOLDER: {run_folder}", flush=True)

  # -- Create rank run folder --
  if not conf.runner.evaluation:
    rank_folder = f"{run_folder}/train"
  else:
    rank_folder = f"{run_folder}/env_{conf.runner.rank:03d}"
  if not os.path.exists(rank_folder):
    os.makedirs(rank_folder, exist_ok=True)
  print(f"[NEK] RANK RUN FOLDER: {rank_folder}", flush=True)

  # -- Initialize Nek --
  initializer = NEK_INIT(
      nek=conf.simulation,
      drl=conf.runner,
      rank_folder=rank_folder,
  )
  initializer.main()

  # Write run path file for downstream launch scripts.
  dir_files_path = "dir-files"
  if not os.path.exists(dir_files_path):
    os.makedirs(dir_files_path, exist_ok=True)
  dir_files_path = f"{dir_files_path}/RUN_PATH_{conf.runner.agent_run_name}.txt"
  with open(dir_files_path, "w") as f:
    f.write(rank_folder + "\n")

  print(f"[NEK] RUN PATH: {dir_files_path}")
  show_end()


if __name__ == "__main__":
  main()
