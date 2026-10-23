#!/usr/bin/env python3
"""Print a complete Slurm submission plan. Submit only with --submit."""
import argparse
from pathlib import Path
import shlex
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.config import load_settings

SCRIPTS = {
    "smoke": ("HG", "03_hydrogym_nek_single_node.slurm", "hg-nek-test", "01:00:00"),
    "train": ("TCF", "03_tcf_train.slurm", "hg-tcf-train", "24:00:00"),
    "evaluate": ("TCF", "09_tcf_evaluate.slurm", "hg-tcf-eval", "12:00:00"),
    "deploy": ("WING", "05_naca0012_deploy.slurm", "hg-naca-zs", "24:00:00"),
    "ucx": ("WING", "diagnostics/08_ucx_test.slurm", "hg-naca-ucx-test", "01:00:00"),
    "collectives": ("WING", "diagnostics/09_collective_test.slurm", "hg-collective-test", "01:00:00"),
}


def command_for(stage, settings, begin=None, dependency=None, mv_data="true"):
    prefix, script, name, walltime = SCRIPTS[stage]
    case = Path(settings[prefix + "_CASE_DIR"]).resolve()
    if any(char.isspace() for char in str(case)) or any(c in str(case) for c in ",:"):
        raise ValueError("Use a workspace path without whitespace, commas or colons")
    ranks = int(settings[prefix + "_TOTAL_RANKS"])
    if stage == "smoke":
        layout = ["--nodes=1", "--ntasks=1", f"--cpus-per-task={ranks}"]
        if settings["ZS_SITE"] == "juwels":
            layout.append("--gres=mem96")
    else:
        nodes = int(settings[prefix + "_NODES"])
        per_node = int(settings[prefix + "_TASKS_PER_NODE"])
        if min(nodes, per_node, ranks) <= 0 or nodes * per_node < ranks:
            raise ValueError("Configured allocation cannot hold the required MPI ranks")
        layout = [f"--nodes={nodes}", f"--ntasks={ranks}",
                  f"--ntasks-per-node={per_node}", "--cpus-per-task=1",
                  "--hint=nomultithread", "--exclusive"]
    log_prefix = "eval" if stage == "evaluate" else "collective" if stage == "collectives" else "slurm"
    command = ["sbatch", "--parsable", f"--account={settings['ZS_ACCOUNT']}",
               f"--partition={settings['ZS_PARTITION']}", *layout,
               f"--time={walltime}", f"--job-name={name}", f"--chdir={case}",
               f"--output={case / 'log-files' / (log_prefix + '-%j.out')}", "--export=ALL"]
    if begin:
        command.append("--begin=" + begin)
    if dependency:
        command.append("--dependency=" + dependency)
    command.append(str(case / script))
    if stage == "deploy":
        command.append("--mv-data=" + mv_data)
    return command, case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=SCRIPTS)
    parser.add_argument("--submit", action="store_true", help="Actually submit; default only prints")
    parser.add_argument("--begin")
    parser.add_argument("--dependency")
    parser.add_argument("--chain", type=int, default=1, help="Number of serialized training allocations")
    parser.add_argument("--mv-data", choices=("true", "false"), default="true")
    args = parser.parse_args()
    if args.chain < 1 or (args.chain != 1 and args.stage != "train"):
        parser.error("--chain must be positive and is only supported for training")
    settings = load_settings(args.stage)
    previous = None
    for index in range(args.chain):
        dependency = "afterok:" + previous if previous else args.dependency
        command, case = command_for(args.stage, settings,
                                    args.begin if index == 0 else None, dependency, args.mv_data)
        if not args.submit:
            prefix = SCRIPTS[args.stage][0]
            exports = {key: settings[key] for key in ("ZS_ROOT", "ZS_SITE", prefix + "_CASE_DIR")}
            print("# PLAN ONLY; settings are loaded and exported by this wrapper")
            print(" ".join(f"{key}={shlex.quote(value)}" for key, value in exports.items()))
            print(shlex.join(command))
            previous = f"<job-{index + 1}>"
            continue
        (case / "log-files").mkdir(parents=True, exist_ok=True)
        result = subprocess.run(command, env=settings, check=True, text=True, capture_output=True)
        previous = result.stdout.strip().split(";", 1)[0]
        if not previous.isdigit():
            raise RuntimeError("Unrecognized sbatch response: " + result.stdout)
        print(previous, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
