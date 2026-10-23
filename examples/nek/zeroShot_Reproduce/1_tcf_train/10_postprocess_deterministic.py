#!/usr/bin/env python3
"""Post-process deterministic HydroGym large-channel evaluations.

The analysis follows the deterministic workflow in marl_energy_cost: reward
evolution in viscous time, representative actuation histories, and the joint
PDF of the two observations.  It also exports per-IC/ensemble statistics and a
conditional policy map.  Large trajectory arrays are opened read-only with
NumPy memory mapping.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.config import load_settings, container_bindings

SETTINGS = load_settings("train")
os.environ.update(SETTINGS)
CASE_DIR = Path(SETTINGS["TCF_CASE_DIR"])
DEFAULT_EVALUATION_ROOT = CASE_DIR / "evaluate"
DEFAULT_IMAGE = Path(SETTINGS["TCF_IMAGE"])
DEFAULT_MODULE = "Nek5000/1.0-gompi-2024a-SystemCUDA-LargeChannel"
DEFAULT_VENV = Path("/home/easybuild/venvs/hydrogym_cpu")


def ensure_analysis_environment() -> None:
    """Re-execute inside the HydroGym container if analysis packages are absent."""
    required = ("numpy", "matplotlib", "omegaconf")
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    if not missing:
        return
    if os.environ.get("HYDROGYM_POST_IN_CONTAINER") == "1":
        raise SystemExit(f"Missing packages inside HydroGym environment: {', '.join(missing)}")

    apptainer = shutil.which("apptainer")
    image = Path(os.environ.get("TCF_IMAGE", DEFAULT_IMAGE))
    module = os.environ.get("TCF_NEK_MODULE", DEFAULT_MODULE)
    venv = Path(os.environ.get("TCF_PYTHON_VENV", DEFAULT_VENV))
    if apptainer is None or not image.is_file():
        raise SystemExit(
            f"Missing Python packages: {', '.join(missing)}; cannot find usable container {image}"
        )

    bindings = container_bindings("train")

    command = [
        apptainer,
        "exec",
        "--bind",
        bindings,
        "--env",
        "HYDROGYM_POST_IN_CONTAINER=1",
        str(image),
        "bash",
        "--noprofile",
        "--norc",
        "-c",
        (
            "source /usr/share/lmod/lmod/init/bash; "
            "module --force purge; module load \"$1\"; "
            "source \"$2/bin/activate\"; shift 2; exec python \"$@\""
        ),
        "bash",
        module,
        str(venv),
        str(Path(__file__).resolve()),
        *sys.argv[1:],
    ]
    os.execv(apptainer, command)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyse a deterministic TCF evaluation.")
    parser.add_argument(
        "--evaluation",
        type=Path,
        help="Evaluation directory (default: newest complete evaluate/e-* run).",
    )
    parser.add_argument("--evaluation-root", type=Path, default=DEFAULT_EVALUATION_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory (default: EVALUATION/postprocess).",
    )
    parser.add_argument("--transition-tplus", type=float, default=500.0)
    parser.add_argument("--evaluation-end-tplus", type=float, default=1500.0)
    parser.add_argument("--agents", default="0,100,200", help="Representative actuator indices.")
    parser.add_argument("--bins", type=int, default=100)
    parser.add_argument("--min-bin-count", type=int, default=20)
    parser.add_argument("--max-samples", type=int, default=2_000_000)
    parser.add_argument("--seed", type=int, default=20010603)
    parser.add_argument("--saturation-threshold", type=float, default=0.99)
    parser.add_argument("--clip-percentiles", nargs=2, type=float, default=(0.5, 99.5))
    parser.add_argument("--dpi", type=int, default=240)
    parser.add_argument("--show", action="store_true")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Use completed_steps from progress.json when an IC lacks COMPLETE.",
    )
    return parser.parse_args()


def numeric_job_key(path: Path) -> tuple[int, float]:
    try:
        job = int(path.name.removeprefix("e-"))
    except ValueError:
        job = -1
    return job, path.stat().st_mtime


def resolve_evaluation(args: argparse.Namespace) -> Path:
    if args.evaluation is not None:
        evaluation = args.evaluation.expanduser().resolve()
    else:
        root = args.evaluation_root.expanduser().resolve()
        candidates = [path for path in root.glob("e-*") if (path / "COMPLETE").is_file()]
        if not candidates:
            raise SystemExit(f"No complete evaluation found below {root}")
        evaluation = max(candidates, key=numeric_job_key)
    if not evaluation.is_dir():
        raise SystemExit(f"Evaluation directory does not exist: {evaluation}")
    return evaluation


def completed_steps(ic_dir: Path, allow_partial: bool) -> int:
    if (ic_dir / "COMPLETE").is_file():
        summary = json.loads((ic_dir / "summary.json").read_text())
        return int(summary["completed_steps"])
    if not allow_partial:
        raise RuntimeError(f"Incomplete initial condition: {ic_dir}")
    progress = json.loads((ic_dir / "progress.json").read_text())
    count = int(progress.get("completed_steps", 0))
    if count < 1:
        raise RuntimeError(f"No complete records in {ic_dir}")
    return count


def parse_indices(value: str) -> list[int]:
    try:
        indices = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise SystemExit(f"Invalid --agents value: {value!r}") from error
    if not indices or any(index < 0 for index in indices):
        raise SystemExit("--agents must contain non-negative integer indices")
    return indices


def interp_rows(x_target, x_rows, values):
    import numpy as np

    return np.vstack(
        [np.interp(x_target, x_rows[index], values[index]) for index in range(len(values))]
    )


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    ensure_analysis_environment()
    args = parse_args()

    import matplotlib
    import numpy as np
    from omegaconf import OmegaConf

    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, TwoSlopeNorm

    if args.bins < 10 or args.max_samples < 1 or args.min_bin_count < 1:
        raise SystemExit("--bins must be >=10; --max-samples and --min-bin-count must be positive")
    if not 0 <= args.saturation_threshold <= 1:
        raise SystemExit("--saturation-threshold must be between 0 and 1")
    p_low, p_high = args.clip_percentiles
    if not 0 <= p_low < p_high <= 100:
        raise SystemExit("--clip-percentiles must satisfy 0 <= LOW < HIGH <= 100")
    if args.evaluation_end_tplus <= args.transition_tplus:
        raise SystemExit("--evaluation-end-tplus must exceed --transition-tplus")

    evaluation = resolve_evaluation(args)
    metadata_path = evaluation / "metadata.json"
    if not metadata_path.is_file():
        raise SystemExit(f"Missing metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("status") != "complete" and not args.allow_partial:
        raise SystemExit(f"Evaluation status is {metadata.get('status')!r}; use --allow-partial if intended")

    config_path = evaluation / "config-used.yml"
    if not config_path.is_file():
        config_path = Path(metadata["config"])
    cfg = OmegaConf.load(config_path)
    reynolds = abs(float(cfg.simulation.viscosity))
    utau = float(cfg.runner.u_tau)
    t_star = (1.0 / reynolds) / (utau**2)
    normalize_input = str(cfg.runner.normalize_input)
    checkpoint_step = int(metadata["checkpoint_step"])

    initial_conditions = [int(value) for value in metadata["initial_conditions"]]
    ic_dirs = [evaluation / f"ic_{ic:02d}" for ic in initial_conditions]
    counts = [completed_steps(folder, args.allow_partial) for folder in ic_dirs]
    representative_indices = parse_indices(args.agents)
    n_agents = int(metadata["n_agents"])
    obs_dim = int(metadata["observation_dimension"])
    if any(index >= n_agents for index in representative_indices):
        raise SystemExit(f"Representative agent index must be below {n_agents}")

    output = (args.output_dir or (evaluation / "postprocess")).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    samples_per_ic = max(1, args.max_samples // len(ic_dirs))

    time_rows: list[np.ndarray] = []
    reward_rows: list[np.ndarray] = []
    reward_spatial_std_rows: list[np.ndarray] = []
    act_mean_rows: list[np.ndarray] = []
    act_rms_rows: list[np.ndarray] = []
    saturation_rows: list[np.ndarray] = []
    obs_mean_rows: list[np.ndarray] = []
    obs_std_rows: list[np.ndarray] = []
    representative_rows: list[np.ndarray] = []
    sampled_obs: list[np.ndarray] = []
    sampled_act: list[np.ndarray] = []
    sampled_reward: list[np.ndarray] = []
    ic_summaries: list[dict[str, Any]] = []

    print(f"[POST] Evaluation: {evaluation}")
    print(f"[POST] Checkpoint: {checkpoint_step}; ICs: {initial_conditions}")
    for ic, folder, count in zip(initial_conditions, ic_dirs, counts):
        flow_time = np.load(folder / "flow_time.npy", mmap_mode="r")[:count]
        mean_reward = np.load(folder / "mean_reward.npy", mmap_mode="r")[:count]
        rewards = np.load(folder / "rewards.npy", mmap_mode="r")[:count]
        actuations = np.load(folder / "actuations.npy", mmap_mode="r")[:count]
        policy_actions = np.load(folder / "policy_actions.npy", mmap_mode="r")[:count]
        observations = np.load(folder / "observations.npy", mmap_mode="r")[:count]
        if observations.shape[1:] != (n_agents, obs_dim):
            raise RuntimeError(f"Unexpected observation shape in {folder}: {observations.shape}")

        # Align every IC at its first recorded post-action time. The missing
        # initial instant differs by only one control interval and is not saved.
        tplus = (np.asarray(flow_time, dtype=np.float64) - float(flow_time[0])) / t_star
        reward_ts = np.asarray(mean_reward, dtype=np.float64)
        reward_spatial_std = np.asarray(np.std(rewards, axis=1), dtype=np.float64)
        act = np.asarray(actuations[:, :, 0], dtype=np.float64)
        act_mean = np.mean(act, axis=1)
        act_rms = np.sqrt(np.mean(np.square(act), axis=1))
        saturation = np.mean(
            np.abs(np.asarray(policy_actions[:, :, 0])) >= args.saturation_threshold, axis=1
        )
        obs_mean = np.asarray(np.mean(observations, axis=1), dtype=np.float64)
        obs_std = np.asarray(np.std(observations, axis=1), dtype=np.float64)
        representative = np.asarray(actuations[:, representative_indices, 0], dtype=np.float64)

        evaluation_mask = (tplus >= args.transition_tplus) & (
            tplus <= args.evaluation_end_tplus
        )
        if not np.any(evaluation_mask):
            raise RuntimeError(
                f"IC {ic} has no samples in t+=[{args.transition_tplus}, "
                f"{args.evaluation_end_tplus}] (available 0..{tplus[-1]:.3f})"
            )
        eligible = np.flatnonzero(evaluation_mask)
        total_points = eligible.size * n_agents
        sample_count = min(samples_per_ic, total_points)
        flat = rng.integers(0, total_points, size=sample_count)
        sample_t = eligible[flat // n_agents]
        sample_agent = flat % n_agents
        sampled_obs.append(np.asarray(observations[sample_t, sample_agent, :], dtype=np.float32))
        sampled_act.append(np.asarray(actuations[sample_t, sample_agent, 0], dtype=np.float32))
        sampled_reward.append(np.asarray(rewards[sample_t, sample_agent], dtype=np.float32))

        selected_reward = reward_ts[evaluation_mask]
        selected_act_mean = act_mean[evaluation_mask]
        selected_act_rms = act_rms[evaluation_mask]
        selected_obs_mean = obs_mean[evaluation_mask]
        selected_obs_std = obs_std[evaluation_mask]
        row: dict[str, Any] = {
            "initial_condition": ic,
            "records": count,
            "tplus_final": float(tplus[-1]),
            "evaluation_records": int(np.count_nonzero(evaluation_mask)),
            "reward_mean": float(np.mean(selected_reward)),
            "reward_mean_percent": float(100.0 * np.mean(selected_reward)),
            "reward_temporal_std_percent": float(100.0 * np.std(selected_reward)),
            "reward_spatial_std_percent": float(100.0 * np.mean(reward_spatial_std[evaluation_mask])),
            "actuation_mean": float(np.mean(selected_act_mean)),
            "actuation_abs_mean": float(np.mean(np.abs(act[evaluation_mask]))),
            "actuation_rms": float(np.sqrt(np.mean(np.square(selected_act_rms)))),
            "actuation_rms_plus": float(np.sqrt(np.mean(np.square(selected_act_rms))) / utau),
            "policy_saturation_percent": float(100.0 * np.mean(saturation[evaluation_mask])),
        }
        for component in range(obs_dim):
            component_mean = selected_obs_mean[:, component]
            component_std = selected_obs_std[:, component]
            row[f"observation_{component}_mean"] = float(np.mean(component_mean))
            row[f"observation_{component}_std"] = float(
                np.sqrt(np.mean(np.square(component_std) + np.square(component_mean - np.mean(component_mean))))
            )

        ic_summaries.append(row)
        time_rows.append(tplus)
        reward_rows.append(reward_ts)
        reward_spatial_std_rows.append(reward_spatial_std)
        act_mean_rows.append(act_mean)
        act_rms_rows.append(act_rms)
        saturation_rows.append(saturation)
        obs_mean_rows.append(obs_mean)
        obs_std_rows.append(obs_std)
        representative_rows.append(representative)
        print(
            f"[POST] IC {ic}: R={row['reward_mean_percent']:.3f}% "
            f"a_rms+={row['actuation_rms_plus']:.4f} "
            f"saturation={row['policy_saturation_percent']:.3f}%"
        )

    # Common time grid over the interval available in every IC.
    final_common = min(float(times[-1]) for times in time_rows)
    common_tplus = time_rows[0][time_rows[0] <= final_common]
    reward_matrix = interp_rows(common_tplus, time_rows, reward_rows)
    reward_spatial_std_matrix = interp_rows(common_tplus, time_rows, reward_spatial_std_rows)
    act_mean_matrix = interp_rows(common_tplus, time_rows, act_mean_rows)
    act_rms_matrix = interp_rows(common_tplus, time_rows, act_rms_rows)
    saturation_matrix = interp_rows(common_tplus, time_rows, saturation_rows)
    obs_mean_matrix = np.stack(
        [interp_rows(common_tplus, time_rows, [row[:, k] for row in obs_mean_rows]) for k in range(obs_dim)],
        axis=2,
    )
    obs_std_matrix = np.stack(
        [interp_rows(common_tplus, time_rows, [row[:, k] for row in obs_std_rows]) for k in range(obs_dim)],
        axis=2,
    )
    common_eval = (common_tplus >= args.transition_tplus) & (
        common_tplus <= args.evaluation_end_tplus
    )
    ic_reward_means = np.array([row["reward_mean_percent"] for row in ic_summaries])
    ensemble_summary = {
        "evaluation": str(evaluation),
        "checkpoint_step": checkpoint_step,
        "initial_conditions": initial_conditions,
        "reward_definition": "100 * mean(1 - local_dUdy / baseline_dUdy)",
        "reward_is_net_power_saving": False,
        "transition_tplus": args.transition_tplus,
        "evaluation_end_tplus": args.evaluation_end_tplus,
        "viscous_time_scale": t_star,
        "friction_velocity": utau,
        "reynolds_number": reynolds,
        "time_origin": "first recorded post-action flow time in each IC",
        "reward_mean_percent": float(np.mean(ic_reward_means)),
        "reward_between_ic_std_percent": float(np.std(ic_reward_means)),
        "reward_between_ic_min_percent": float(np.min(ic_reward_means)),
        "reward_between_ic_max_percent": float(np.max(ic_reward_means)),
        "actuation_rms_plus_mean": float(np.mean([row["actuation_rms_plus"] for row in ic_summaries])),
        "policy_saturation_percent_mean": float(
            np.mean([row["policy_saturation_percent"] for row in ic_summaries])
        ),
        "notes": [
            "Actuations are Python-side commands before Nek5000's internal ZNMF correction.",
            "Observations are already in plus units when normalize_input=utau.",
            "Net power saving cannot be computed because pressure-work/actuation-power terms were not recorded.",
        ],
        "per_initial_condition": ic_summaries,
    }
    (output / "summary.json").write_text(json.dumps(ensemble_summary, indent=2) + "\n")

    summary_fields = list(ic_summaries[0].keys())
    write_csv(output / "ic_summary.csv", summary_fields, ic_summaries)

    timeseries_rows = []
    for index, tplus in enumerate(common_tplus):
        row: dict[str, Any] = {
            "tplus": float(tplus),
            "reward_mean_percent": float(100.0 * np.mean(reward_matrix[:, index])),
            "reward_between_ic_std_percent": float(100.0 * np.std(reward_matrix[:, index])),
            "reward_spatial_std_percent": float(100.0 * np.mean(reward_spatial_std_matrix[:, index])),
            "actuation_mean_plus": float(np.mean(act_mean_matrix[:, index]) / utau),
            "actuation_rms_plus": float(np.mean(act_rms_matrix[:, index]) / utau),
            "policy_saturation_percent": float(100.0 * np.mean(saturation_matrix[:, index])),
        }
        for component in range(obs_dim):
            row[f"observation_{component}_mean"] = float(np.mean(obs_mean_matrix[:, index, component]))
            row[f"observation_{component}_spatial_std"] = float(
                np.mean(obs_std_matrix[:, index, component])
            )
        timeseries_rows.append(row)
    write_csv(output / "ensemble_timeseries.csv", list(timeseries_rows[0]), timeseries_rows)

    # Reward, actuation and observation overview.
    colors = plt.cm.tab10(np.linspace(0, 1, len(initial_conditions)))
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    ax = axes[0, 0]
    for ic, tplus, reward, color in zip(initial_conditions, time_rows, reward_rows, colors):
        ax.plot(tplus, 100.0 * reward, color=color, alpha=0.35, lw=0.8, label=f"IC {ic}")
    reward_mean = 100.0 * np.mean(reward_matrix, axis=0)
    reward_std = 100.0 * np.std(reward_matrix, axis=0)
    ax.fill_between(common_tplus, reward_mean - reward_std, reward_mean + reward_std, color="black", alpha=0.18)
    ax.plot(common_tplus, reward_mean, color="black", lw=2.0, label="six-IC mean")
    ax.set_ylabel("Wall-shear reduction [%]")
    ax.set_title(
        f"Reward: {ensemble_summary['reward_mean_percent']:.2f} "
        f"± {ensemble_summary['reward_between_ic_std_percent']:.2f}% across ICs"
    )
    ax.legend(ncol=2, fontsize=8)

    ax = axes[0, 1]
    for ic_index, (ic, tplus, rms, color) in enumerate(
        zip(initial_conditions, time_rows, act_rms_rows, colors)
    ):
        ax.plot(tplus, rms / utau, color=color, alpha=0.35, lw=0.8, label=f"IC {ic}")
    ax.plot(common_tplus, np.mean(act_rms_matrix, axis=0) / utau, color="black", lw=2.0)
    ax.set_ylabel(r"Command RMS $a_{rms}/u_\tau$")
    ax.set_title("Actuation command magnitude")

    ax = axes[1, 0]
    observation_labels = [r"$u_t^+$", r"$v_n^+$"] + [f"obs {i}" for i in range(2, obs_dim)]
    for component in range(obs_dim):
        mean = np.mean(obs_mean_matrix[:, :, component], axis=0)
        between_ic = np.std(obs_mean_matrix[:, :, component], axis=0)
        ax.fill_between(common_tplus, mean - between_ic, mean + between_ic, alpha=0.18)
        ax.plot(common_tplus, mean, lw=1.6, label=observation_labels[component])
    ax.set_xlabel(r"Elapsed viscous time $t^+$")
    ax.set_ylabel("Spatial mean observation")
    ax.set_title("Observation means")
    ax.legend()

    ax = axes[1, 1]
    ax.plot(
        common_tplus,
        100.0 * np.mean(saturation_matrix, axis=0),
        color="tab:red",
        lw=1.6,
        label=f"|policy action| ≥ {args.saturation_threshold:g}",
    )
    ax.set_xlabel(r"Elapsed viscous time $t^+$")
    ax.set_ylabel("Saturated actuators [%]")
    ax.set_title("Policy saturation")
    ax.legend()
    for axis in axes.flat:
        axis.axvspan(0, args.transition_tplus, color="0.7", alpha=0.18)
        axis.axvline(args.evaluation_end_tplus, color="0.45", ls="--", lw=1.0)
        axis.set_xlim(0, final_common)
        axis.grid(alpha=0.25)
    fig.suptitle(f"TCF deterministic evaluation — checkpoint {checkpoint_step}")
    fig.tight_layout()
    fig.savefig(output / "deterministic_overview.png", dpi=args.dpi)

    # Representative actuator histories, matching the old deterministic plot.
    fig, axes = plt.subplots(len(representative_indices), 1, figsize=(11, 2.7 * len(representative_indices)), sharex=True)
    axes = np.atleast_1d(axes)
    for panel, agent_index in enumerate(representative_indices):
        for ic, tplus, representative, color in zip(
            initial_conditions, time_rows, representative_rows, colors
        ):
            axes[panel].plot(
                tplus,
                representative[:, panel] / utau,
                color=color,
                alpha=0.8,
                lw=0.8,
                label=f"IC {ic}",
            )
        axes[panel].axvspan(0, args.transition_tplus, color="0.7", alpha=0.18)
        axes[panel].axvline(args.evaluation_end_tplus, color="0.45", ls="--", lw=1.0)
        axes[panel].set_ylabel(f"Agent {agent_index}\n" + r"$a/u_\tau$")
        axes[panel].grid(alpha=0.25)
    axes[0].legend(ncol=len(initial_conditions), fontsize=8)
    axes[-1].set_xlabel(r"Elapsed viscous time $t^+$")
    fig.suptitle("Representative deterministic actuation commands")
    fig.tight_layout()
    fig.savefig(output / "representative_actuators.png", dpi=args.dpi)

    # Distribution and conditional-action products from a bounded random sample.
    obs_sample = np.concatenate(sampled_obs, axis=0)
    act_sample = np.concatenate(sampled_act, axis=0)
    reward_sample = np.concatenate(sampled_reward, axis=0)
    x_range = np.percentile(obs_sample[:, 0], [p_low, p_high])
    y_range = np.percentile(obs_sample[:, 1], [p_low, p_high])
    x_edges = np.linspace(*x_range, args.bins + 1)
    y_edges = np.linspace(*y_range, args.bins + 1)
    counts, _, _ = np.histogram2d(obs_sample[:, 0], obs_sample[:, 1], bins=(x_edges, y_edges))
    weighted, _, _ = np.histogram2d(
        obs_sample[:, 0], obs_sample[:, 1], bins=(x_edges, y_edges), weights=act_sample / utau
    )
    conditional_action = np.full_like(weighted, np.nan, dtype=np.float64)
    valid = counts >= args.min_bin_count
    conditional_action[valid] = weighted[valid] / counts[valid]
    probability = counts / np.sum(counts)
    positive = probability[probability > 0]
    norm = LogNorm(vmin=max(float(np.percentile(positive, 5)), 1e-8), vmax=float(np.max(positive)))
    action_limit = float(np.nanpercentile(np.abs(conditional_action), 99))
    action_limit = max(action_limit, 1e-6)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    mesh = axes[0, 0].pcolormesh(x_edges, y_edges, probability.T, shading="auto", cmap="viridis", norm=norm)
    fig.colorbar(mesh, ax=axes[0, 0], label="Probability per bin")
    axes[0, 0].set_title(f"Joint observation PDF ({len(obs_sample):,} samples)")
    axes[0, 0].set_xlabel(observation_labels[0])
    axes[0, 0].set_ylabel(observation_labels[1])

    mesh = axes[0, 1].pcolormesh(
        x_edges,
        y_edges,
        conditional_action.T,
        shading="auto",
        cmap="coolwarm",
        norm=TwoSlopeNorm(vmin=-action_limit, vcenter=0.0, vmax=action_limit),
    )
    fig.colorbar(mesh, ax=axes[0, 1], label=r"Mean command $a/u_\tau$")
    axes[0, 1].set_title(f"Conditional action (minimum {args.min_bin_count} samples/bin)")
    axes[0, 1].set_xlabel(observation_labels[0])
    axes[0, 1].set_ylabel(observation_labels[1])

    axes[1, 0].hist(100.0 * reward_sample, bins=args.bins, density=True, color="tab:blue", alpha=0.8)
    axes[1, 0].axvline(100.0 * np.mean(reward_sample), color="black", lw=1.5, label="sample mean")
    axes[1, 0].set_xlabel("Local wall-shear reduction [%]")
    axes[1, 0].set_ylabel("Probability density")
    axes[1, 0].set_title("Per-actuator reward distribution")
    axes[1, 0].legend()

    axes[1, 1].hist(act_sample / utau, bins=args.bins, density=True, color="tab:orange", alpha=0.8)
    axes[1, 1].set_xlabel(r"Command $a/u_\tau$")
    axes[1, 1].set_ylabel("Probability density")
    axes[1, 1].set_title("Actuation-command distribution")
    for axis in axes.flat:
        axis.grid(alpha=0.2)
    fig.suptitle(
        rf"Evaluation window: {args.transition_tplus:g} ≤ $t^+$ ≤ {args.evaluation_end_tplus:g}"
    )
    fig.tight_layout()
    fig.savefig(output / "state_action_distributions.png", dpi=args.dpi)

    np.savez_compressed(
        output / "state_action_histograms.npz",
        observation_0_edges=x_edges,
        observation_1_edges=y_edges,
        probability=probability,
        conditional_action_plus=conditional_action,
        sample_count=np.array(len(obs_sample)),
    )

    report = f"""# Deterministic channel evaluation

- Evaluation: `{evaluation}`
- Checkpoint: `{checkpoint_step}`
- Initial conditions: `{', '.join(map(str, initial_conditions))}`
- Evaluation interval: `{args.transition_tplus:g} <= t+ <= {args.evaluation_end_tplus:g}`
- Mean wall-shear reduction: `{ensemble_summary['reward_mean_percent']:.4f}%`
- Between-IC standard deviation: `{ensemble_summary['reward_between_ic_std_percent']:.4f}%`
- Mean actuation-command RMS: `{ensemble_summary['actuation_rms_plus_mean']:.5f} u_tau`
- Mean policy saturation: `{ensemble_summary['policy_saturation_percent_mean']:.4f}%`

The reward is HydroGym's normalized wall-shear reduction, `1 - dUdy/dUdy_ref`.
It is not net power saving: pressure-work and actuation-power terms were not
recorded. `actuations.npy` contains the Python-side command before Nek5000's
internal zero-net-mass-flux correction (`znmf_avg={metadata.get('znmf_avg')}`).
"""
    (output / "report.md").write_text(report)

    if args.show:
        plt.show()
    plt.close("all")
    print(f"[POST] Mean reward: {ensemble_summary['reward_mean_percent']:.4f}%")
    print(f"[POST] Between-IC std: {ensemble_summary['reward_between_ic_std_percent']:.4f}%")
    print(f"[POST] Saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
