#!/usr/bin/env python3
"""Plot the one-episode sliding-window mean reward during TCF training.

The smoothing follows marl_energy_cost/post_processing/exploration.ipynb: the
step-wise reward is averaged with a uniform window whose width is one complete
episode, and the horizontal coordinate is expressed in (possibly fractional)
episodes.  Complete episodes from archived allocations and the live allocation
are concatenated in chronological order; partial episodes are ignored.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import re
import shutil
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.config import load_settings, container_bindings

SETTINGS = load_settings("train")
os.environ.update(SETTINGS)
CASE_DIR = Path(SETTINGS["TCF_CASE_DIR"])
DEFAULT_RUN_DIR = Path(SETTINGS["TCF_RUN_DIR"])
DEFAULT_IMAGE = Path(SETTINGS["TCF_IMAGE"])
DEFAULT_MODULE = "Nek5000/1.0-gompi-2024a-SystemCUDA-LargeChannel"
DEFAULT_VENV = Path("/home/easybuild/venvs/hydrogym_cpu")


def ensure_plot_environment() -> None:
    """Re-execute inside the HydroGym container when plotting packages are absent."""
    missing = [name for name in ("numpy", "matplotlib") if importlib.util.find_spec(name) is None]
    if not missing:
        return

    if os.environ.get("HYDROGYM_PLOT_IN_CONTAINER") == "1":
        raise SystemExit(f"Missing Python packages inside the HydroGym environment: {', '.join(missing)}")

    apptainer = shutil.which("apptainer")
    image = Path(os.environ.get("TCF_IMAGE", DEFAULT_IMAGE))
    module = os.environ.get("TCF_NEK_MODULE", DEFAULT_MODULE)
    venv = Path(os.environ.get("TCF_PYTHON_VENV", DEFAULT_VENV))
    if apptainer is None or not image.is_file():
        raise SystemExit(
            f"Missing Python packages: {', '.join(missing)}. Install them or run this script "
            f"inside the HydroGym container ({image})."
        )

    bindings = container_bindings("train")

    command = [
        apptainer,
        "exec",
        "--bind",
        bindings,
        "--env",
        "HYDROGYM_PLOT_IN_CONTAINER=1",
        str(image),
        "bash",
        "--noprofile",
        "--norc",
        "-c",
        (
            "source /usr/share/lmod/lmod/init/bash; "
            "module --force purge; "
            "module load \"$1\"; "
            "source \"$2/bin/activate\"; "
            "shift 2; "
            "exec python \"$@\""
        ),
        "bash",
        module,
        str(venv),
        str(Path(__file__).resolve()),
        *sys.argv[1:],
    ]
    os.execv(apptainer, command)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the one-episode sliding-window averaged HydroGym reward."
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument(
        "--window-steps",
        type=int,
        help="Smoothing width in DRL steps (default: runner.nb_interactions from config-used.yml).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output PNG (default: RUN_DIR/plots/reward_sliding_window.png).",
    )
    parser.add_argument("--label", default="TCF large")
    parser.add_argument("--title", default="Exploration")
    parser.add_argument("--dpi", type=int, default=299)
    parser.add_argument("--ylim", nargs=2, type=float, metavar=("MIN", "MAX"))
    parser.add_argument("--show", action="store_true", help="Also open an interactive plot window.")
    return parser.parse_args()


def config_window(run_dir: Path) -> int:
    config = run_dir / "config-used.yml"
    if not config.is_file():
        raise SystemExit(f"Cannot determine the episode length: {config} is missing; use --window-steps.")
    pattern = re.compile(r"^\s*nb_interactions\s*:\s*([0-9]+)\b")
    for line in config.read_text().splitlines():
        match = pattern.match(line)
        if match:
            return int(match.group(1))
    raise SystemExit(f"runner.nb_interactions was not found in {config}; use --window-steps.")


def archive_sort_key(path: Path) -> tuple[int, int, str]:
    match = re.fullmatch(r"job-(\d+)(?:-(\d+))?", path.name)
    if match:
        return int(match.group(1)), int(match.group(2) or 0), path.name
    return sys.maxsize, 0, path.name


def history_segments(run_dir: Path) -> list[tuple[str, Path]]:
    segments: list[tuple[str, Path]] = []
    archive_root = run_dir / "history"
    if archive_root.is_dir():
        for job_dir in sorted(archive_root.glob("job-*"), key=archive_sort_key):
            solver_history = job_dir / "solver_history"
            if solver_history.is_dir():
                segments.append((job_dir.name, solver_history))

    live_history = run_dir / "current_solver" / "history"
    if live_history.is_dir():
        segments.append(("current", live_history))
    return segments


def load_complete_episodes(
    run_dir: Path, window_steps: int
) -> tuple[list[float], int, int, list[tuple[str, Path]]]:
    rewards: list[float] = []
    complete = 0
    incomplete = 0
    sources: list[tuple[str, Path]] = []

    for segment_name, segment in history_segments(run_dir):
        for csv_path in sorted(segment.glob("rewards_aggregated_*.csv")):
            sources.append((segment_name, csv_path))
            groups: dict[int, dict[int, float]] = {}
            with csv_path.open(newline="") as stream:
                reader = csv.DictReader(stream)
                required = {"episode", "step", "mean_reward"}
                if not required.issubset(reader.fieldnames or ()):
                    print(f"[WARN] skipping {csv_path}: missing {sorted(required)}", file=sys.stderr)
                    continue
                for row in reader:
                    try:
                        episode = int(row["episode"])
                        step = int(row["step"])
                        reward = float(row["mean_reward"])
                    except (KeyError, TypeError, ValueError):
                        # A live file can be observed while its final row is being written.
                        continue
                    groups.setdefault(episode, {})[step] = reward

            expected_steps = set(range(window_steps))
            for episode in sorted(groups):
                step_rewards = groups[episode]
                if len(step_rewards) != window_steps or set(step_rewards) != expected_steps:
                    incomplete += 1
                    continue
                rewards.extend(step_rewards[step] for step in range(window_steps))
                complete += 1

    return rewards, complete, incomplete, sources


def main() -> int:
    ensure_plot_environment()
    args = parse_args()

    import numpy as np
    import matplotlib

    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_dir = args.run_dir.expanduser().resolve()
    window_steps = args.window_steps or config_window(run_dir)
    if window_steps <= 0:
        raise SystemExit("--window-steps must be positive")

    raw_rewards, n_complete, n_incomplete, sources = load_complete_episodes(run_dir, window_steps)
    if not raw_rewards:
        raise SystemExit(f"No complete {window_steps}-step episodes found below {run_dir}")

    reward_pct = np.asarray(raw_rewards, dtype=np.float64) * 100.0
    # Equivalent to np.convolve(..., ones(window)/window, mode="valid"), but
    # linear-time for long training histories.
    cumulative = np.concatenate(([0.0], np.cumsum(reward_pct)))
    smooth = (cumulative[window_steps:] - cumulative[:-window_steps]) / window_steps
    episode = np.arange(window_steps, reward_pct.size + 1, dtype=np.float64) / window_steps

    max_index = int(np.argmax(smooth))
    max_reward = float(smooth[max_index])
    max_episode = int(np.ceil(episode[max_index]))

    fig, axis = plt.subplots(1, 1, figsize=(8, 6))
    legend = f"{args.label}: Max R={max_reward:.1f}%; Eps={max_episode}"
    axis.plot(episode, smooth, linewidth=2.5, color="black", label=legend)
    axis.set_xlabel("Episode")
    axis.set_ylabel("Reward [%]")
    axis.set_title(args.title)
    axis.set_xlim(1.0, max(1.01, float(episode[-1]) * 1.01))
    if args.ylim:
        axis.set_ylim(*args.ylim)
    axis.grid(True, alpha=0.35)
    axis.legend(loc="lower right", fontsize=11)
    fig.tight_layout()

    output = args.output or (run_dir / "plots" / "reward_sliding_window.png")
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=args.dpi)
    if args.show:
        plt.show()
    plt.close(fig)

    print(f"Sources: {len(sources)} CSV file(s) across {len(set(name for name, _ in sources))} segment(s)")
    print(f"Episodes: {n_complete} complete, {n_incomplete} incomplete/skipped")
    print(f"Window: {window_steps} DRL steps (one episode)")
    print(f"Maximum: {max_reward:.3f}% at episode {episode[max_index]:.3f} (reported episode {max_episode})")
    print(f"Saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
