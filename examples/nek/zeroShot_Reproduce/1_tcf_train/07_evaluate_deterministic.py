#!/usr/bin/env python3
"""Evaluate one saved policy on fixed large-channel initial conditions.

This driver deliberately lives outside the HydroGym package.  It uses the
public NekEnv interface, selects init_<IC> explicitly, and records transition-
aligned trajectories without changing either HydroGym or the training run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf

from hydrogym.nek import NekEnv, make_pettingzoo_env


MODEL_RE = re.compile(r"model_(\d+)_steps\.zip$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deterministically evaluate a HydroGym policy on fixed ICs."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--env", required=True)
    parser.add_argument("--local-dir", type=Path, required=True)
    parser.add_argument("--nproc", type=int, required=True)
    parser.add_argument("--solver-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-step",
        type=int,
        default=576000000,
        help="Exact SB3 timestep to evaluate (default: 576000000, the episode-40 policy).",
    )
    parser.add_argument(
        "--ics",
        default="1,2,3,4,5,6",
        help="Comma-separated deterministic init_<N> indices.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Interactions per IC (default: runner.nb_interactions).",
    )
    parser.add_argument("--flush-every", type=int, default=100)
    return parser.parse_args()


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def atomic_json(path: Path, payload: Any) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_ics(value: str) -> list[int]:
    try:
        result = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise ValueError(f"Invalid --ics value: {value!r}") from error
    if not result or len(result) != len(set(result)) or any(item < 1 for item in result):
        raise ValueError("--ics must contain unique positive integers")
    return result


def resolve_checkpoint(checkpoint_dir: Path, requested: int | None) -> tuple[int, Path, Path, dict]:
    manifest_path = checkpoint_dir / "latest.json"
    manifest: dict[str, Any] = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())

    if requested is not None:
        step = requested
    elif "step" in manifest:
        step = int(manifest["step"])
    else:
        candidates = []
        for path in checkpoint_dir.glob("model_*_steps.zip"):
            match = MODEL_RE.fullmatch(path.name)
            if match:
                candidates.append(int(match.group(1)))
        if not candidates:
            raise FileNotFoundError(f"No policy checkpoint found in {checkpoint_dir}")
        step = max(candidates)

    model = checkpoint_dir / f"model_{step}_steps.zip"
    vecnormalize = checkpoint_dir / f"vecnormalize_{step}_steps.pkl"
    for path in (model, vecnormalize):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Incomplete checkpoint at step {step}: missing {path}")
    return step, model, vecnormalize, manifest


def open_trajectory_files(folder: Path, steps: int, n_agents: int, obs_dim: int, act_dim: int):
    return {
        "observations": np.lib.format.open_memmap(
            folder / "observations.npy", mode="w+", dtype=np.float32,
            shape=(steps, n_agents, obs_dim),
        ),
        "policy_actions": np.lib.format.open_memmap(
            folder / "policy_actions.npy", mode="w+", dtype=np.float32,
            shape=(steps, n_agents, act_dim),
        ),
        "actuations": np.lib.format.open_memmap(
            folder / "actuations.npy", mode="w+", dtype=np.float32,
            shape=(steps, n_agents, act_dim),
        ),
        "rewards": np.lib.format.open_memmap(
            folder / "rewards.npy", mode="w+", dtype=np.float32,
            shape=(steps, n_agents),
        ),
        "mean_reward": np.lib.format.open_memmap(
            folder / "mean_reward.npy", mode="w+", dtype=np.float32,
            shape=(steps,),
        ),
        "flow_time": np.lib.format.open_memmap(
            folder / "flow_time.npy", mode="w+", dtype=np.float64,
            shape=(steps,),
        ),
    }


def flush_arrays(arrays: dict[str, np.memmap]) -> None:
    for array in arrays.values():
        array.flush()


def physical_actuation(base_env: NekEnv, policy_action: np.ndarray) -> np.ndarray:
    """Mirror the Python-side rescaling performed by NekEnv.step()."""
    sent = np.asarray(policy_action, dtype=np.float32).reshape(-1).copy()
    if base_env.rescale_actions:
        negative = sent < 0
        positive = sent > 0
        sent[negative] *= float(base_env.rescale_factors[0][0])
        sent[positive] *= float(base_env.rescale_factors[0][1])
    # For this case znmf_avg=1, so the remaining ZNMF correction is performed
    # inside Nek5000.  "actuations.npy" is therefore the command sent to Nek.
    if base_env.znmf_avg == -1:
        sent -= np.mean(sent)
    elif base_env.znmf_avg == -2:
        weights = np.empty(base_env.n_actuators, dtype=np.float64)
        for index in range(base_env.n_actuators):
            ix = int(base_env.actuator_info["ix"][index])
            iz = int(base_env.actuator_info["iz"][index])
            weights[index] = base_env.gll_weight[ix - 1] * base_env.gll_weight[iz - 1]
        sent -= np.average(sent, weights=weights)
    return sent.reshape(-1, 1)


def main() -> int:
    args = parse_args()
    args.config = args.config.resolve()
    args.local_dir = args.local_dir.resolve()
    args.solver_dir = args.solver_dir.resolve()
    args.checkpoint_dir = args.checkpoint_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    ics = parse_ics(args.ics)
    cfg = OmegaConf.load(args.config)
    steps = int(args.steps if args.steps is not None else cfg.runner.nb_interactions)
    if steps < 1 or args.flush_every < 1:
        raise ValueError("--steps and --flush-every must be positive")

    required_solver = ("nek5000", "tcf.par", "tcf.re2", "tcf.ma2", "int_pos")
    missing = [name for name in required_solver if not (args.solver_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"Unstaged solver directory {args.solver_dir}: missing {', '.join(missing)}")
    for ic in ics:
        restart = args.local_dir / args.env / "restart_files" / f"init_{ic}"
        if not restart.is_dir():
            raise FileNotFoundError(f"Missing deterministic restart directory: {restart}")

    step, source_model, source_vec, source_manifest = resolve_checkpoint(
        args.checkpoint_dir, args.checkpoint_step
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    policy_dir = args.output_dir / "policy"
    policy_dir.mkdir(exist_ok=False)
    model_path = policy_dir / source_model.name
    vec_path = policy_dir / source_vec.name
    shutil.copy2(source_model, model_path)
    shutil.copy2(source_vec, vec_path)
    if (args.checkpoint_dir / "latest.json").is_file():
        shutil.copy2(args.checkpoint_dir / "latest.json", policy_dir / "latest-at-start.json")

    # VecNormalize is a trusted local training artefact.  Evaluation does not
    # update its statistics; only its frozen observation transformation is used.
    with vec_path.open("rb") as stream:
        vectorizer = pickle.load(stream)
    vectorizer.training = False
    norm_obs = bool(vectorizer.__dict__.get("norm_obs", False))
    norm_reward = bool(vectorizer.__dict__.get("norm_reward", False))

    from stable_baselines3 import PPO, SAC, TD3

    algorithms = {"PPO": PPO, "SAC": SAC, "TD3": TD3}
    algo_name = str(cfg.runner.RL_algorithm).upper()
    if algo_name not in algorithms:
        raise ValueError(f"Unsupported runner.RL_algorithm: {algo_name}")
    model = algorithms[algo_name].load(str(model_path), device="cpu", print_system_info=False)

    metadata = {
        "status": "initializing",
        "created_at": now(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "interactive"),
        "algorithm": algo_name,
        "deterministic": True,
        "checkpoint_step": step,
        "checkpoint_manifest_at_start": source_manifest,
        "model_source": str(source_model),
        "model_snapshot": str(model_path),
        "model_sha256": sha256(model_path),
        "vecnormalize_source": str(source_vec),
        "vecnormalize_snapshot": str(vec_path),
        "vecnormalize_sha256": sha256(vec_path),
        "vecnormalize_norm_obs": norm_obs,
        "vecnormalize_norm_reward": norm_reward,
        "initial_conditions": ics,
        "steps_per_initial_condition": steps,
        "config": str(args.config),
        "solver_dir": str(args.solver_dir),
    }
    atomic_json(args.output_dir / "metadata.json", metadata)

    # HydroGym currently prefixes absolute run_root paths with cwd when forming
    # MPI's wdir.  As in training, use a path relative to the common work dir.
    solver_root = os.path.relpath(args.solver_dir, start=Path.cwd())
    env_config = {
        "environment_name": args.env,
        "nproc": args.nproc,
        "use_clean_cache": True,
        "local_fallback_dir": str(args.local_dir),
        "configuration_file": str(args.config),
        "rescale_actions": bool(cfg.runner.rescale_actions),
        "normalize_input": str(cfg.runner.normalize_input),
        "ctrl_min_amp": float(cfg.runner.ctrl_min_amp),
        "ctrl_max_amp": float(cfg.runner.ctrl_max_amp),
    }

    base_env: NekEnv | None = None
    summaries: list[dict[str, Any]] = []
    try:
        base_env = NekEnv(env_config=env_config, run_root=solver_root)
        # Old-format runner YAMLs do not accept this field through HydroGym's
        # new-schema override map.  Keep termination aligned with --steps,
        # including short smoke tests.
        base_env.nb_interactions = steps
        # Construct only to obtain HydroGym's stable agent ordering.  Stepping
        # the array-based base env avoids thousands of dict operations per step.
        pettingzoo_env = make_pettingzoo_env(base_env)
        agent_names = list(pettingzoo_env.possible_agents)
        n_agents = int(base_env.n_actuators)
        obs_dim = int(base_env.obs_per_actuator)
        act_dim = 1
        if len(agent_names) != n_agents:
            raise RuntimeError("HydroGym agent ordering does not match actuator count")
        (args.output_dir / "agent_names.txt").write_text("\n".join(agent_names) + "\n")
        actuator_info = args.solver_dir / "history" / "ACTUATOR_INFO.csv"
        if actuator_info.is_file():
            shutil.copy2(actuator_info, args.output_dir / "ACTUATOR_INFO.csv")

        metadata.update(
            status="running",
            n_agents=n_agents,
            observation_dimension=obs_dim,
            action_dimension=act_dim,
            hydrogym_normalize_input=str(base_env.normalize_input),
            rescale_actions=bool(base_env.rescale_actions),
            ctrl_command_scale=(
                float(base_env.rescale_factors[0][0]) if base_env.rescale_actions else 1.0
            ),
            znmf_avg=int(base_env.znmf_avg),
            array_semantics={
                "observations.npy": "pre-action input passed to model.predict",
                "policy_actions.npy": "deterministic SB3 action in policy action space",
                "actuations.npy": "rescaled command passed from Python to Nek5000",
                "rewards.npy": "per-actuator reward returned by the same transition",
                "mean_reward.npy": "mean of rewards.npy over actuators",
                "flow_time.npy": "Nek flow time after the same transition",
                "final_observation.npy": "policy input after the final transition",
            },
        )
        atomic_json(args.output_dir / "metadata.json", metadata)

        for ic in ics:
            ic_folder = args.output_dir / f"ic_{ic:02d}"
            ic_folder.mkdir(exist_ok=False)
            arrays = open_trajectory_files(ic_folder, steps, n_agents, obs_dim, act_dim)
            ic_state: dict[str, Any] = {
                "status": "running",
                "initial_condition": ic,
                "restart_directory": str(
                    args.local_dir / args.env / "restart_files" / f"init_{ic}"
                ),
                "started_at": now(),
                "completed_steps": 0,
                "requested_steps": steps,
            }
            atomic_json(ic_folder / "progress.json", ic_state)
            print(f"[EVAL] IC={ic} reset from init_{ic}", flush=True)

            # The old-format YAML cannot apply this override through env_config,
            # so set the two public runtime fields before every reset.
            base_env.random_init = -1
            base_env.rank = ic
            observation, _ = base_env.reset(seed=int(cfg.runner.seed) + ic)
            observation = np.asarray(observation, dtype=np.float32).reshape(n_agents, obs_dim)
            episode_start = np.ones(n_agents, dtype=bool)
            policy_state = None
            reward_total = 0.0

            try:
                for index in range(steps):
                    policy_observation = (
                        np.asarray(vectorizer.normalize_obs(observation), dtype=np.float32)
                        if norm_obs
                        else observation
                    )
                    action, policy_state = model.predict(
                        policy_observation,
                        state=policy_state,
                        episode_start=episode_start,
                        deterministic=True,
                    )
                    action = np.asarray(action, dtype=np.float32).reshape(n_agents, act_dim)
                    sent_action = physical_actuation(base_env, action)

                    next_observation, _, terminated, truncated, info = base_env.step(action[:, 0])
                    rewards = np.asarray(info["reward_per_actuator"], dtype=np.float32)
                    mean_reward = float(np.mean(rewards))

                    arrays["observations"][index] = policy_observation
                    arrays["policy_actions"][index] = action
                    arrays["actuations"][index] = sent_action
                    arrays["rewards"][index] = rewards
                    arrays["mean_reward"][index] = mean_reward
                    arrays["flow_time"][index] = float(info["time"])
                    reward_total += mean_reward
                    observation = np.asarray(next_observation, dtype=np.float32).reshape(n_agents, obs_dim)
                    episode_start.fill(False)

                    completed = index + 1
                    if completed % args.flush_every == 0 or completed == steps:
                        flush_arrays(arrays)
                        ic_state.update(
                            completed_steps=completed,
                            last_flow_time=float(info["time"]),
                            mean_reward_so_far=reward_total / completed,
                            updated_at=now(),
                        )
                        atomic_json(ic_folder / "progress.json", ic_state)
                        print(
                            f"[EVAL] IC={ic} step={completed}/{steps} "
                            f"time={float(info['time']):.6g} reward={mean_reward:.6g}",
                            flush=True,
                        )
                    if (terminated or truncated) and completed < steps:
                        raise RuntimeError(
                            f"IC {ic} terminated at step {completed}, before requested {steps}"
                        )

                final_policy_observation = (
                    np.asarray(vectorizer.normalize_obs(observation), dtype=np.float32)
                    if norm_obs
                    else observation
                )
                np.save(ic_folder / "final_observation.npy", final_policy_observation)
                with (ic_folder / "timeseries.csv").open("w", newline="") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(("step", "flow_time", "mean_reward"))
                    writer.writerows(
                        (i + 1, float(arrays["flow_time"][i]), float(arrays["mean_reward"][i]))
                        for i in range(steps)
                    )
                summary = {
                    "status": "complete",
                    "initial_condition": ic,
                    "completed_steps": steps,
                    "mean_reward": reward_total / steps,
                    "final_flow_time": float(arrays["flow_time"][steps - 1]),
                    "completed_at": now(),
                }
                atomic_json(ic_folder / "summary.json", summary)
                (ic_folder / "COMPLETE").write_text(f"completed_at={summary['completed_at']}\n")
                summaries.append(summary)
                atomic_json(args.output_dir / "summary.json", {"evaluations": summaries})
            except BaseException as error:
                flush_arrays(arrays)
                ic_state.update(status="failed", error=repr(error), failed_at=now())
                atomic_json(ic_folder / "progress.json", ic_state)
                raise
            finally:
                del arrays

        metadata.update(status="complete", completed_at=now())
        atomic_json(args.output_dir / "metadata.json", metadata)
    except BaseException as error:
        metadata.update(status="failed", failed_at=now(), error=repr(error))
        atomic_json(args.output_dir / "metadata.json", metadata)
        raise
    finally:
        if base_env is not None:
            base_env.close()

    (args.output_dir / "COMPLETE").write_text(
        f"completed_at={now()}\ncheckpoint_step={step}\ninitial_conditions={','.join(map(str, ics))}\n"
    )
    print(f"[EVAL] all initial conditions complete: {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("[EVAL] interrupted", file=sys.stderr, flush=True)
        raise
