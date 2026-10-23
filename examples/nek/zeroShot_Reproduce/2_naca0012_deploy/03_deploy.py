#!/usr/bin/env python3
"""Deterministically deploy a selected large-channel TD3 policy on the wing.

This controller is external to HydroGym.  It uses one shared policy for every
wing actuator and records transition-aligned observations, commands, and
rewards without changing package code.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import pickle
import re
import shutil
import signal
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf

from hydrogym.nek import NekEnv, make_pettingzoo_env


MODEL_RE = re.compile(r"model_(\d+)_steps\.zip$")
STOP_REQUESTED = False


def request_stop(signum, _frame) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print(f"[DEPLOY] received signal {signum}; stopping after this transition", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--policy-template", type=Path, required=True)
    parser.add_argument("--solver-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--nproc", type=int, required=True)
    parser.add_argument("--checkpoint-step", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--record-every", type=int, default=None)
    parser.add_argument("--flush-every", type=int, default=100)
    parser.add_argument("--max-seconds", type=int, default=82800)
    return parser.parse_args()


def load_policy_template(path: Path):
    spec = importlib.util.spec_from_file_location("wing_meta_policy", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load policy template: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_policy_template(template, cfg, args: argparse.Namespace) -> None:
    required = (
        "ENV_NAME", "NPROC", "NUM_STEPS", "RECORD_EVERY",
        "CHECKPOINT_DIR", "CHECKPOINT_STEP", "RUNNER_CONFIG",
        "SIMULATION_CONFIG", "POLICY_SPECS",
    )
    missing = [name for name in required if not hasattr(template, name)]
    if missing:
        raise ValueError(f"Policy template is missing: {', '.join(missing)}")
    if int(template.NPROC) != args.nproc:
        raise ValueError(f"Policy template requests {template.NPROC} Nek ranks, got {args.nproc}")
    if Path(template.CHECKPOINT_DIR).resolve() != args.checkpoint_dir:
        raise ValueError(
            "Policy template CHECKPOINT_DIR differs from --checkpoint-dir: "
            f"{template.CHECKPOINT_DIR} != {args.checkpoint_dir}"
        )
    runner = template.RUNNER_CONFIG
    simulation = template.SIMULATION_CONFIG
    if int(runner.get("random_init", 0)) != -2:
        raise ValueError(
            "RUNNER_CONFIG.random_init must be -2 to preserve the staged flow checkpoint"
        )
    if int(runner.get("rank", 0)) != 1:
        raise ValueError("RUNNER_CONFIG.rank must be 1 because only HF init_1 is staged")
    checks = []
    for section_name, mirrored, redirected in (
        ("runner", runner, set()),
        ("simulation", simulation, {"compile_path", "restart_folder"}),
    ):
        runtime_section = getattr(cfg, section_name)
        for key, expected in mirrored.items():
            if key in redirected:
                continue
            if not hasattr(runtime_section, key):
                checks.append((f"{section_name}.{key}", expected, "<missing>"))
                continue
            actual = getattr(runtime_section, key)
            checks.append((f"{section_name}.{key}", expected, actual))
    differences = [
        f"{key}: template={expected!r}, runtime={actual!r}"
        for key, expected, actual in checks
        if expected != actual
    ]
    if differences:
        raise ValueError("Policy template/runtime mismatch: " + "; ".join(differences))
    if not template.POLICY_SPECS:
        raise ValueError("POLICY_SPECS cannot be empty")
    for item in template.POLICY_SPECS:
        if str(item.get("algorithm", "")).upper() != "TD3":
            raise ValueError(f"Only TD3 policy specs are valid here: {item!r}")
        if not bool(item.get("shared_policy", False)):
            raise ValueError(f"Each wing group must select the shared policy: {item!r}")
        if Path(item.get("checkpoint_dir", "")).resolve() != args.checkpoint_dir:
            raise ValueError(f"Policy checkpoint directory is inconsistent: {item!r}")
        if item.get("checkpoint_step") != template.CHECKPOINT_STEP:
            raise ValueError(f"Policy checkpoint step is inconsistent: {item!r}")
        if list(map(float, item.get("action_bounds", []))) != [
            float(cfg.runner.ctrl_min_amp), float(cfg.runner.ctrl_max_amp)
        ]:
            raise ValueError(f"Policy action bounds differ from runtime config: {item!r}")


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


def resolve_checkpoint(
    checkpoint_dir: Path, requested: int | None
) -> tuple[int, Path, Path, dict[str, Any]]:
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
            raise FileNotFoundError(f"No model checkpoint found in {checkpoint_dir}")
        step = max(candidates)
    model = checkpoint_dir / f"model_{step}_steps.zip"
    vec = checkpoint_dir / f"vecnormalize_{step}_steps.pkl"
    for path in (model, vec):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Checkpoint {step} is incomplete: {path}")
    return step, model, vec, manifest


def open_arrays(folder: Path, records: int, steps: int, agents: int, obs_dim: int):
    return {
        "record_steps": np.lib.format.open_memmap(
            folder / "record_steps.npy", mode="w+", dtype=np.int64, shape=(records,)
        ),
        "observations": np.lib.format.open_memmap(
            folder / "observations.npy", mode="w+", dtype=np.float32,
            shape=(records, agents, obs_dim),
        ),
        "policy_actions": np.lib.format.open_memmap(
            folder / "policy_actions.npy", mode="w+", dtype=np.float32,
            shape=(records, agents, 1),
        ),
        "actuations": np.lib.format.open_memmap(
            folder / "actuations.npy", mode="w+", dtype=np.float32,
            shape=(records, agents, 1),
        ),
        "rewards": np.lib.format.open_memmap(
            folder / "rewards.npy", mode="w+", dtype=np.float32,
            shape=(records, agents),
        ),
        "record_flow_time": np.lib.format.open_memmap(
            folder / "record_flow_time.npy", mode="w+", dtype=np.float64,
            shape=(records,),
        ),
        "mean_reward": np.lib.format.open_memmap(
            folder / "mean_reward.npy", mode="w+", dtype=np.float32, shape=(steps,)
        ),
        "flow_time": np.lib.format.open_memmap(
            folder / "flow_time.npy", mode="w+", dtype=np.float64, shape=(steps,)
        ),
    }


def flush_arrays(arrays: dict[str, np.memmap]) -> None:
    for array in arrays.values():
        array.flush()


def command_sent_by_python(base_env: NekEnv, policy_action: np.ndarray) -> np.ndarray:
    sent = np.asarray(policy_action, dtype=np.float32).reshape(-1).copy()
    if base_env.rescale_actions:
        sent[sent < 0] *= float(base_env.rescale_factors[0][0])
        sent[sent > 0] *= float(base_env.rescale_factors[0][1])
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
    for attr in ("config", "policy_template", "solver_dir", "checkpoint_dir", "output_dir"):
        setattr(args, attr, getattr(args, attr).resolve())
    cfg = OmegaConf.load(args.config)
    template = load_policy_template(args.policy_template)
    validate_policy_template(template, cfg, args)
    steps = int(args.steps if args.steps is not None else template.NUM_STEPS)
    record_every = int(
        args.record_every
        if args.record_every is not None
        else template.RECORD_EVERY
    )
    if min(steps, record_every, args.flush_every, args.max_seconds) < 1:
        raise ValueError("steps, record/flush intervals, and max-seconds must be positive")
    if int(cfg.simulation.nproc) != args.nproc:
        raise ValueError(f"Config requests {cfg.simulation.nproc} Nek ranks, got {args.nproc}")

    required_solver = (
        "nek5000", "naca_wing.rea", "naca_wing.re2", "naca_wing.map",
        "naca_wing.wall", "naca_wing.restart", "SESSION.NAME",
    )
    missing = [name for name in required_solver if not (args.solver_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"Incomplete solver directory: {', '.join(missing)}")
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    requested_checkpoint = (
        args.checkpoint_step
        if args.checkpoint_step is not None
        else template.CHECKPOINT_STEP
    )
    checkpoint_step, source_model, source_vec, source_manifest = resolve_checkpoint(
        args.checkpoint_dir, requested_checkpoint
    )
    policy_dir = args.output_dir / "policy"
    policy_dir.mkdir()
    model_path = policy_dir / source_model.name
    vec_path = policy_dir / source_vec.name
    shutil.copy2(source_model, model_path)
    shutil.copy2(source_vec, vec_path)
    latest = args.checkpoint_dir / "latest.json"
    if latest.is_file():
        shutil.copy2(latest, policy_dir / "latest-at-start.json")

    with vec_path.open("rb") as stream:
        vectorizer = pickle.load(stream)
    vectorizer.training = False
    norm_obs = bool(vectorizer.__dict__.get("norm_obs", False))
    norm_reward = bool(vectorizer.__dict__.get("norm_reward", False))

    from stable_baselines3 import TD3

    if str(cfg.runner.RL_algorithm).upper() != "TD3":
        raise ValueError("The authoritative configuration does not select TD3")
    model = TD3.load(str(model_path), device="cpu", print_system_info=False)

    metadata: dict[str, Any] = {
        "status": "initializing",
        "created_at": now(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "interactive"),
        "deployment": "zero-shot large-channel TD3 policy on all NACA0012 actuators",
        "deterministic": True,
        "checkpoint_step": checkpoint_step,
        "checkpoint_manifest_at_start": source_manifest,
        "model_source": str(source_model),
        "model_snapshot": str(model_path),
        "model_sha256": sha256(model_path),
        "vecnormalize_source": str(source_vec),
        "vecnormalize_snapshot": str(vec_path),
        "vecnormalize_sha256": sha256(vec_path),
        "vecnormalize_norm_obs": norm_obs,
        "vecnormalize_norm_reward": norm_reward,
        "config": str(args.config),
        "config_sha256": sha256(args.config),
        "policy_template": str(args.policy_template),
        "policy_template_sha256": sha256(args.policy_template),
        "policy_groups": template.POLICY_SPECS,
        "environment_name": str(template.ENV_NAME),
        "solver_directory": str(args.solver_dir),
        "requested_steps": steps,
        "record_every": record_every,
        "max_seconds": args.max_seconds,
        "initial_condition": json.loads(
            (args.solver_dir / "restart-input.json").read_text()
        ),
    }
    atomic_json(args.output_dir / "metadata.json", metadata)

    base_env: NekEnv | None = None
    arrays: dict[str, np.memmap] | None = None
    started = time.monotonic()
    completed = 0
    completed_records = 0
    reward_sum = 0.0
    stop_reason = "requested_steps_complete"
    try:
        # HydroGym currently builds MPI's wdir as cwd/run_root, so use a path
        # relative to the common container working directory.
        solver_root = os.path.relpath(args.solver_dir, start=Path.cwd())
        base_env = NekEnv(conf=cfg, run_root=solver_root)
        base_env.nb_interactions = steps
        base_env.random_init = int(cfg.runner.random_init)
        base_env.rank = int(cfg.runner.rank)
        pettingzoo_env = make_pettingzoo_env(base_env)
        agent_names = list(pettingzoo_env.possible_agents)
        n_agents = int(base_env.n_actuators)
        obs_dim = int(base_env.obs_per_actuator)
        if len(agent_names) != n_agents:
            raise RuntimeError("PettingZoo and base-environment actuator order differ")
        (args.output_dir / "agent_names.txt").write_text("\n".join(agent_names) + "\n")
        actuator_info = args.solver_dir / "history" / "ACTUATOR_INFO.csv"
        if actuator_info.is_file():
            shutil.copy2(actuator_info, args.output_dir / "ACTUATOR_INFO.csv")

        n_records = (steps + record_every - 1) // record_every
        estimated_bytes = (
            n_records * n_agents * (obs_dim + 1 + 1 + 1) * 4
            + n_records * 16
            + steps * 12
        )
        free = shutil.disk_usage(args.output_dir).free
        if free < int(estimated_bytes * 1.1):
            raise RuntimeError(
                f"Trajectory needs about {estimated_bytes / 2**30:.2f} GiB; "
                f"only {free / 2**30:.2f} GiB is free"
            )
        arrays = open_arrays(args.output_dir, n_records, steps, n_agents, obs_dim)
        metadata.update(
            status="running",
            n_agents=n_agents,
            observation_dimension=obs_dim,
            action_dimension=1,
            planned_records=n_records,
            estimated_trajectory_bytes=estimated_bytes,
            hydrogym_normalize_input=str(base_env.normalize_input),
            hydrogym_rescale_actions=bool(base_env.rescale_actions),
            hydrogym_ctrl_scale=(
                float(base_env.rescale_factors[0][0]) if base_env.rescale_actions else 1.0
            ),
            znmf_avg=int(base_env.znmf_avg),
            array_semantics={
                "record_steps.npy": "zero-based transition indices stored in sparse arrays",
                "observations.npy": "pre-action input actually passed to model.predict",
                "policy_actions.npy": "deterministic TD3 output",
                "actuations.npy": "command passed by Python to Nek before internal v17 UTAU/ZNMF handling",
                "rewards.npy": "per-actuator reward from the same transition",
                "record_flow_time.npy": "post-transition Nek time for sparse records",
                "mean_reward.npy": "per-transition spatial mean reward (dense)",
                "flow_time.npy": "post-transition Nek time (dense)",
            },
        )
        atomic_json(args.output_dir / "metadata.json", metadata)

        observation, _ = base_env.reset()
        observation = np.asarray(observation, dtype=np.float32).reshape(n_agents, obs_dim)
        episode_start = np.ones(n_agents, dtype=bool)
        policy_state = None

        for index in range(steps):
            if STOP_REQUESTED:
                stop_reason = "signal"
                break
            if time.monotonic() - started >= args.max_seconds:
                stop_reason = "wallclock_guard"
                break

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
            action = np.asarray(action, dtype=np.float32).reshape(n_agents, 1)
            sent_action = command_sent_by_python(base_env, action)
            next_observation, _, terminated, truncated, info = base_env.step(action[:, 0])
            rewards = np.asarray(info["reward_per_actuator"], dtype=np.float32).reshape(n_agents)
            mean_reward = float(np.mean(rewards))

            arrays["mean_reward"][index] = mean_reward
            arrays["flow_time"][index] = float(info["time"])
            if index % record_every == 0:
                record = completed_records
                arrays["record_steps"][record] = index
                arrays["observations"][record] = policy_observation
                arrays["policy_actions"][record] = action
                arrays["actuations"][record] = sent_action
                arrays["rewards"][record] = rewards
                arrays["record_flow_time"][record] = float(info["time"])
                completed_records += 1

            completed = index + 1
            reward_sum += mean_reward
            observation = np.asarray(next_observation, dtype=np.float32).reshape(n_agents, obs_dim)
            episode_start.fill(False)

            if completed % args.flush_every == 0 or completed == steps:
                flush_arrays(arrays)
                metadata.update(
                    completed_steps=completed,
                    completed_records=completed_records,
                    last_flow_time=float(info["time"]),
                    mean_reward_so_far=reward_sum / completed,
                    updated_at=now(),
                )
                atomic_json(args.output_dir / "metadata.json", metadata)
                print(
                    f"[DEPLOY] step={completed}/{steps} time={float(info['time']):.6g} "
                    f"mean_reward={mean_reward:.6g}",
                    flush=True,
                )
            if (terminated or truncated) and completed < steps:
                stop_reason = "environment_terminated"
                break

        flush_arrays(arrays)
        final_policy_observation = (
            np.asarray(vectorizer.normalize_obs(observation), dtype=np.float32)
            if norm_obs
            else observation
        )
        np.save(args.output_dir / "final_observation.npy", final_policy_observation)
        with (args.output_dir / "timeseries.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("step", "flow_time", "mean_reward"))
            writer.writerows(
                (i + 1, float(arrays["flow_time"][i]), float(arrays["mean_reward"][i]))
                for i in range(completed)
            )

        complete = completed == steps
        metadata.update(
            status="complete" if complete else "partial",
            stop_reason=stop_reason,
            completed_steps=completed,
            completed_records=completed_records,
            mean_reward=(reward_sum / completed if completed else None),
            finished_at=now(),
        )
        atomic_json(args.output_dir / "metadata.json", metadata)
        marker = "COMPLETE" if complete else "PARTIAL"
        (args.output_dir / marker).write_text(
            f"finished_at={now()}\ncheckpoint_step={checkpoint_step}\n"
            f"completed_steps={completed}\nrequested_steps={steps}\nstop_reason={stop_reason}\n"
        )
    except BaseException as error:
        if arrays is not None:
            flush_arrays(arrays)
        metadata.update(
            status="failed",
            error=repr(error),
            completed_steps=completed,
            completed_records=completed_records,
            failed_at=now(),
        )
        atomic_json(args.output_dir / "metadata.json", metadata)
        raise
    finally:
        if base_env is not None:
            base_env.close()

    print(
        f"[DEPLOY] {metadata['status']}: {completed}/{steps} steps, "
        f"output={args.output_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("[DEPLOY] interrupted", file=sys.stderr, flush=True)
        raise
