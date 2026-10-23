"""Read the same shell settings used by preparation and rank wrappers."""
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
STAGES = {"smoke": "0_setup", "train": "1_tcf_train", "evaluate": "1_tcf_train",
          "deploy": "2_naca0012_deploy", "ucx": "2_naca0012_deploy",
          "collectives": "2_naca0012_deploy"}


def load_settings(stage):
    settings = ROOT / STAGES[stage] / "00_settings.sh"
    result = subprocess.run(["bash", "-c", 'set -e; source "$1"; env -0',
                             "settings", str(settings)], check=True, capture_output=True)
    return dict(item.decode().split("=", 1) for item in result.stdout.split(b"\0") if item)


def container_bindings(stage):
    result = subprocess.run(["bash", "-c", 'set -e; source "$1"; zs_bind_paths',
                             "settings", str(ROOT / STAGES[stage] / "00_settings.sh")],
                            check=True, capture_output=True, text=True)
    return result.stdout.strip()
