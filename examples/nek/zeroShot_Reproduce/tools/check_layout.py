#!/usr/bin/env python3
"""Validate source syntax and resolved paths without preparation or submission."""
import argparse
import ast
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.config import load_settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", action="store_true", help="Also require prepared TCF and wing inputs")
    args = parser.parse_args()
    python_count = shell_count = 0
    for directory in ("common", "sites", "tools", "tests", "0_setup", "1_tcf_train", "2_naca0012_deploy"):
        for path in (ROOT / directory).rglob("*"):
            if any(part in ("data", "run", "evaluate", "w", "r", "r40", "results", "results40", "__pycache__") for part in path.relative_to(ROOT).parts):
                continue
            if path.suffix == ".py":
                ast.parse(path.read_text(), filename=str(path))
                python_count += 1
            elif path.suffix in (".sh", ".slurm"):
                subprocess.run(["bash", "-n", str(path)], check=True)
                shell_count += 1
    states = {}
    missing = []
    for stage, prefix in (("smoke", "HG"), ("train", "TCF"), ("deploy", "WING")):
        env = load_settings(stage)
        keys = ["ZS_ROOT", "ZS_CACHE_ROOT", "ZS_DATA_ROOT", "ZS_SITE", prefix + "_IMAGE", prefix + "_CASE_DIR"]
        keys += {"smoke": ["HG_WORK_REL"], "train": ["TCF_CONFIG", "TCF_RESTART_SOURCE", "TCF_WORK_DIR", "TCF_RUN_DIR"],
                 "deploy": ["WING_DATA_ROOT", "WING_WORK_DIR", "WING_RUNTIME_CONFIG", "WING_RUNS_DIR", "WING_SOLVER_DIR", "WING_RESULTS_DIR", "WING_TCF_CHECKPOINT_DIR", "WING_HF_REVISION"]}[stage]
        states[stage] = {key: env[key] for key in keys}
        required = []
        if stage == "train":
            if not Path(env["TCF_CONFIG"]).is_file():
                raise RuntimeError("Missing reproduction-owned training YAML")
            if args.prepared:
                required = [Path(env["TCF_WORK_DIR"]) / "nek5000", Path(env["TCF_RESTART_SOURCE"]) / "init_6/rs6tcf0.f00006"]
        elif stage == "deploy" and args.prepared:
            probe = str(Path(env["WING_SOLVER_DIR"]) / "naca_wing.rea")
            if len(probe) > 131:
                raise RuntimeError(f"Nek solver path is too long ({len(probe)} > 131): {probe}")
            required = [Path(env["WING_WORK_DIR"]) / "PREPARED.json", Path(env["WING_RUNTIME_CONFIG"])]
        else:
            required = []
        if args.prepared:
            required.append(Path(env[prefix + "_IMAGE"]))
            missing += [str(path) for path in required if not path.is_file()]
    print(json.dumps({"syntax": {"python": python_count, "shell": shell_count}, "paths": states,
                      "prepared_inputs_checked": args.prepared, "missing": missing}, indent=2))
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
