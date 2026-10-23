#!/usr/bin/env python3
"""External, resumable SB3 driver for HydroGym's Nek PettingZoo environment.

This intentionally lives outside the HydroGym package.  It uses HydroGym's
public NekEnv/PettingZoo interface, while adding the continuation behavior that
the bundled getting-started trainer currently lacks: paired model/replay-buffer
checkpoints, a stable run directory, remaining-step accounting, and a clean
wall-clock stop before Slurm terminates the allocation.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from hydrogym.nek import NekEnv, make_pettingzoo_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--env", required=True)
    parser.add_argument("--local-dir", type=Path, required=True)
    parser.add_argument("--nproc", type=int, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, default=82_800)
    parser.add_argument("--keep-replay", type=int, default=2)
    return parser.parse_args()


def build_vec_env(args: argparse.Namespace, cfg):
    from stable_baselines3.common.vec_env import VecEnvWrapper, VecNormalize
    from supersuit import pettingzoo_env_to_vec_env_v1

    solver_dir = args.run_dir / "current_solver"
    required = ("tcf.par", "tcf.re2", "tcf.ma2", "int_pos")
    missing = [name for name in required if not (solver_dir / name).is_file()]
    if missing:
        raise RuntimeError(
            f"Nek solver directory was not staged: {solver_dir}; "
            f"missing {', '.join(missing)}"
        )

    # HydroGym currently constructs mpi_info['wdir'] as cwd/run_root, so an
    # absolute run_root would be prefixed a second time. The MPMD roles both
    # start in TCF_WORK_DIR; pass a path relative to that shared cwd.
    solver_root = os.path.relpath(solver_dir, start=Path.cwd())
    env_config = {
        "environment_name": args.env,
        "nproc": args.nproc,
        # True makes HFDataManager prefer a valid local_fallback_dir before it
        # attempts the network. False means "HF cache directly" and probes HF
        # first, which is undesirable on compute nodes.
        "use_clean_cache": True,
        "local_fallback_dir": str(args.local_dir),
        "configuration_file": str(args.config),
        "rescale_actions": bool(cfg.runner.rescale_actions),
        "normalize_input": str(cfg.runner.normalize_input),
        "ctrl_min_amp": float(cfg.runner.ctrl_min_amp),
        "ctrl_max_amp": float(cfg.runner.ctrl_max_amp),
    }
    base_env = NekEnv(env_config=env_config, run_root=solver_root)
    env = make_pettingzoo_env(base_env)
    # Nek exposes identically shaped observation/action spaces for every
    # actuator and all agents terminate together.  Padding and black-death
    # wrappers are therefore unnecessary; with thousands of actuators they
    # also route each parallel step through PettingZoo's sequential AEC path.
    env = pettingzoo_env_to_vec_env_v1(env)

    class VecEnvCompatWrapper(VecEnvWrapper):
        def __init__(self, venv):
            super().__init__(venv)
            self._pending_seed: int | None = None

        def seed(self, seed: int | None = None):
            """Bridge SB3's legacy seed() call to Gymnasium reset(seed=...)."""
            if seed is None:
                seed = int(np.random.randint(0, np.iinfo(np.uint32).max, dtype=np.uint32))
            self._pending_seed = int(seed)
            print(f"[SEED] queued environment seed={self._pending_seed}", flush=True)
            return [self._pending_seed + idx for idx in range(self.num_envs)]

        def reset(self, **kwargs):
            if self._pending_seed is not None:
                kwargs.setdefault("seed", self._pending_seed)
                print(f"[SEED] applying environment seed={kwargs['seed']}", flush=True)
                self._pending_seed = None
            result = self.venv.reset(**kwargs)
            return result[0] if isinstance(result, tuple) else result

        def step_wait(self):
            result = self.venv.step_wait()
            if len(result) == 5:
                obs, reward, terminated, truncated, info = result
                return obs, reward, np.logical_or(terminated, truncated), info
            return result

    env = VecEnvCompatWrapper(env)
    return env, VecNormalize


class CheckpointStore:
    _MODEL_RE = re.compile(r"model_(\d+)_steps\.zip$")

    def __init__(self, path: Path, keep_replay: int):
        self.path = path
        self.keep_replay = max(2, keep_replay)
        self.manifest = path / "latest.json"
        path.mkdir(parents=True, exist_ok=True)

    def paths(self, step: int) -> dict[str, Path]:
        return {
            "model": self.path / f"model_{step}_steps.zip",
            "replay": self.path / f"replay_buffer_{step}_steps.pkl",
            "vecnormalize": self.path / f"vecnormalize_{step}_steps.pkl",
        }

    def _complete(self, step: int) -> bool:
        return all(p.is_file() and p.stat().st_size > 0 for p in self.paths(step).values())

    def candidates(self) -> list[int]:
        steps = []
        for path in self.path.glob("model_*_steps.zip"):
            match = self._MODEL_RE.fullmatch(path.name)
            if match and self._complete(int(match.group(1))):
                steps.append(int(match.group(1)))
        return sorted(set(steps), reverse=True)

    def latest(self) -> tuple[int, dict[str, Path]] | None:
        if self.manifest.is_file():
            try:
                step = int(json.loads(self.manifest.read_text())["step"])
                if self._complete(step):
                    return step, self.paths(step)
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                pass
        candidates = self.candidates()
        return (candidates[0], self.paths(candidates[0])) if candidates else None

    def save(self, model, vec_env, reason: str) -> int:
        step = int(model.num_timesteps)
        final = self.paths(step)
        tmp = self.path / f".tmp-{step}-{os.getpid()}"
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir()
        staged = {
            "model": tmp / "model.zip",
            "replay": tmp / "replay.pkl",
            "vecnormalize": tmp / "vecnormalize.pkl",
        }
        try:
            model.save(staged["model"])
            model.save_replay_buffer(staged["replay"])
            vec_env.save(str(staged["vecnormalize"]))
            for key in ("model", "replay", "vecnormalize"):
                os.replace(staged[key], final[key])
            payload = {
                "step": step,
                "reason": reason,
                "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID", "interactive"),
            }
            manifest_tmp = self.path / f".latest-{os.getpid()}.json"
            manifest_tmp.write_text(json.dumps(payload, indent=2) + "\n")
            os.replace(manifest_tmp, self.manifest)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        self.prune_replay()
        print(f"[CHECKPOINT] step={step} reason={reason}", flush=True)
        return step

    def prune_replay(self) -> None:
        complete = self.candidates()
        keep = set(complete[: self.keep_replay])
        pattern = re.compile(r"replay_buffer_(\d+)_steps\.pkl$")
        for path in self.path.glob("replay_buffer_*_steps.pkl"):
            match = pattern.fullmatch(path.name)
            if match and int(match.group(1)) not in keep:
                path.unlink()
                print(f"[CHECKPOINT] removed old replay buffer {path.name}", flush=True)


def main() -> int:
    args = parse_args()
    args.config = args.config.resolve()
    args.local_dir = args.local_dir.resolve()
    args.run_dir = args.run_dir.resolve()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    cfg = OmegaConf.load(args.config)

    from stable_baselines3 import PPO, SAC, TD3
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.noise import NormalActionNoise

    algorithms = {"PPO": PPO, "SAC": SAC, "TD3": TD3}
    algo_name = str(cfg.runner.RL_algorithm).upper()
    if algo_name not in algorithms:
        raise ValueError(f"Unsupported runner.RL_algorithm: {algo_name}")
    Algorithm = algorithms[algo_name]

    store = CheckpointStore(args.run_dir / "checkpoints", args.keep_replay)
    resume = store.latest()
    raw_env, VecNormalize = build_vec_env(args, cfg)

    if resume:
        resume_step, paths = resume
        print(f"[RESUME] loading complete checkpoint at step {resume_step}", flush=True)
        env = VecNormalize.load(str(paths["vecnormalize"]), raw_env)
        env.training = True
        model = Algorithm.load(paths["model"], env=env, device="cpu", print_system_info=False)
        model.load_replay_buffer(paths["replay"], truncate_last_traj=True)
        print(f"[RESUME] replay buffer loaded: {paths['replay']}", flush=True)
    else:
        env = VecNormalize(raw_env, norm_obs=False, norm_reward=False, clip_obs=0.0)
        model_kwargs = {
            "policy": "MlpPolicy",
            "env": env,
            "learning_rate": float(cfg.runner.learning_rate),
            "policy_kwargs": {"net_arch": {"qf": [16, 16, 64], "pi": [8]}},
            "gamma": float(getattr(cfg.runner, "gamma", 0.99)),
            "verbose": 1,
            "tensorboard_log": str(args.run_dir / "tensorboard"),
            "seed": int(cfg.runner.seed),
            "device": "cpu",
        }
        if algo_name == "PPO":
            model_kwargs.update(batch_size=int(cfg.runner.batch_size))
        else:
            n_actions = env.action_space.shape[0]
            model_kwargs.update(
                batch_size=int(cfg.runner.batch_size),
                buffer_size=int(cfg.runner.buffer_size),
                learning_starts=int(cfg.runner.learning_starts),
                train_freq=(int(cfg.runner.train_steps), "step"),
                gradient_steps=int(cfg.runner.gradient_steps),
                tau=float(cfg.runner.tau),
                action_noise=NormalActionNoise(
                    mean=np.zeros(n_actions),
                    sigma=float(cfg.runner.action_noise) * np.ones(n_actions),
                ),
            )
            if algo_name == "TD3":
                model_kwargs.update(
                    policy_delay=int(cfg.runner.policy_delay),
                    target_policy_noise=float(cfg.runner.target_policy_noise),
                    target_noise_clip=float(cfg.runner.target_noise_clip),
                )
        model = Algorithm(**model_kwargs)
        print("[RESUME] no checkpoint found; training from scratch", flush=True)

    target = int(cfg.runner.nb_episodes) * int(cfg.runner.nb_interactions) * int(env.num_envs)
    remaining = max(0, target - int(model.num_timesteps))
    save_calls = int(cfg.runner.nb_interactions) * int(cfg.runner.ckpt_int)
    deadline = time.monotonic() + args.max_seconds

    print(
        f"[PLAN] algorithm={algo_name} vec_envs={env.num_envs} "
        f"current={model.num_timesteps} target={target} remaining={remaining} "
        f"checkpoint_every={save_calls} vector steps max_seconds={args.max_seconds}",
        flush=True,
    )

    class ManagedCheckpointCallback(BaseCallback):
        def __init__(self):
            super().__init__(verbose=0)
            self.last_saved_step = int(model.num_timesteps)

        def _save(self, reason: str) -> None:
            self.last_saved_step = store.save(self.model, env, reason)

        def _on_step(self) -> bool:
            if self.n_calls % save_calls == 0:
                self._save("periodic")
            if time.monotonic() >= deadline:
                if self.last_saved_step != int(self.model.num_timesteps):
                    self._save("wallclock")
                print("[TIME] clean wall-clock stop requested", flush=True)
                return False
            return True

    callback = ManagedCheckpointCallback()
    complete_file = args.run_dir / "COMPLETE"
    try:
        if remaining:
            model.learn(
                total_timesteps=remaining,
                callback=callback,
                log_interval=max(1, int(env.num_envs)),
                reset_num_timesteps=False,
                tb_log_name=f"{algo_name}_tcf_large",
            )
        if callback.last_saved_step != int(model.num_timesteps):
            store.save(model, env, "final" if model.num_timesteps >= target else "stopped")
        if model.num_timesteps >= target:
            complete_file.write_text(
                f"completed_at={time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n"
                f"timesteps={model.num_timesteps}\n"
                f"target={target}\n"
            )
            print(f"[DONE] training target reached; wrote {complete_file}", flush=True)
    except BaseException:
        # Preserve progress on ordinary Python errors when MPI remains healthy.
        try:
            if callback.last_saved_step != int(model.num_timesteps):
                store.save(model, env, "exception")
        except BaseException as save_error:
            print(f"[CHECKPOINT] emergency save failed: {save_error}", file=sys.stderr, flush=True)
        raise
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

