#!/usr/bin/env python3
"""Download the official wing data once and build an external run workspace.

The default mode is a side-effect-free plan.  A real ~174.53 GB download only
starts when both --download and --accept-large-download are supplied.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


EXPECTED_FILES = {
    ".NEK5000_v17": None,
    "environment_config.yaml": 802,
    "forparam.i": 53,
    "mask_naca_wing0.f00002": 3_628_126_792,
    "naca_wing.map": 24_037_637,
    "naca_wing.re2": 47_061_476,
    "naca_wing.rea": 7_817,
    "naca_wing.restart": 13,
    "naca_wing.wall": 900,
    "stat_pts.in": 108_363,
    **{
        f"restart_files/init_1/rs8naca_wing0.f{i:05d}": 21_353_285_320
        for i in range(1, 9)
    },
}
KNOWN_BYTES = sum(size for size in EXPECTED_FILES.values() if size is not None)
RESERVE_BYTES = 10 * 2**30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--environment", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--policy-template", type=Path, required=True)
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--module-case-dir", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--max-workers", type=int, default=4)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--download", action="store_true")
    mode.add_argument("--reuse-local", type=Path, help="Reuse immutable files from a reference environment")
    parser.add_argument("--source-prepared", type=Path, help="Reference PREPARED.json for --reuse-local")
    parser.add_argument("--accept-large-download", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def missing_bytes(env_dir: Path) -> int:
    total = 0
    for relative, expected in EXPECTED_FILES.items():
        if expected is None:
            continue
        path = env_dir / relative
        present = path.stat().st_size if path.is_file() else 0
        total += max(expected - present, 0)
    return total


def validate_environment(env_dir: Path) -> dict[str, int]:
    missing = []
    too_small = []
    actual = {}
    for relative, expected in EXPECTED_FILES.items():
        path = env_dir / relative
        if not path.is_file():
            missing.append(relative)
            continue
        actual[relative] = path.stat().st_size
        if expected is not None and path.stat().st_size < expected:
            too_small.append(f"{relative}: {path.stat().st_size} < {expected}")
    if missing or too_small:
        details = []
        if missing:
            details.append("missing=" + ", ".join(missing))
        if too_small:
            details.append("truncated=" + ", ".join(too_small))
        raise RuntimeError("Incomplete HF environment: " + "; ".join(details))
    return actual


def atomic_json(path: Path, payload: Any) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def load_policy_template(path: Path):
    spec = importlib.util.spec_from_file_location("wing_meta_policy", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load standalone policy configuration: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    args = parse_args()
    if args.max_workers < 1:
        raise ValueError("--max-workers must be positive")

    data_root = args.data_root.resolve()
    env_dir = data_root / args.environment
    workspace = args.workspace.resolve()
    config_dir = args.config_dir.resolve()
    policy_template = args.policy_template.resolve()
    module_case = args.module_case_dir.resolve()

    print(f"Repository:       {args.repo}@{args.revision}")
    print(f"Environment:      {args.environment}")
    print(f"Destination:      {env_dir}")
    print(f"Known data size:  {KNOWN_BYTES / 1e9:.2f} GB ({KNOWN_BYTES / 2**30:.2f} GiB)")
    print(f"Workspace:        {workspace}")
    print(f"Policy config:    {policy_template}")

    if not args.download and args.reuse_local is None:
        print("PLAN ONLY: no files were downloaded or changed.")
        print("Re-run with: --download --accept-large-download")
        return 0
    if args.download and not args.accept_large_download:
        raise RuntimeError(
            "Refusing the ~174.53 GB transfer without --accept-large-download"
        )
    if not policy_template.is_file():
        raise FileNotFoundError(policy_template)
    template = load_policy_template(policy_template)
    for attribute in ("ENV_NAME", "RUNNER_CONFIG", "SIMULATION_CONFIG", "POLICY_SPECS"):
        if not hasattr(template, attribute):
            raise ValueError(f"Policy configuration is missing {attribute}")
    if str(template.ENV_NAME) != args.environment:
        raise ValueError(
            f"Policy ENV_NAME={template.ENV_NAME!r} does not match {args.environment!r}"
        )
    for name in ("nek5000", "SIZE"):
        if not (module_case / name).is_file():
            raise FileNotFoundError(f"Nek module case file is missing: {module_case / name}")

    if args.download:
        # Reused inputs are symlinks into an immutable reference dataset.
        # Never let a downloader replace or follow those inputs during refresh.
        candidates = [env_dir, *[env_dir / relative for relative in EXPECTED_FILES]]
        for candidate in candidates:
            ancestors = [candidate, *candidate.parents]
            if any(path.is_symlink() for path in ancestors if path == env_dir or env_dir in path.parents):
                raise RuntimeError("Refusing download into reused input symlinks; choose a fresh data directory")

    data_root.mkdir(parents=True, exist_ok=True)
    workspace.mkdir(parents=True, exist_ok=True)
    # PREPARED.json is the job's commit marker.  Remove it before a requested
    # refresh so an interrupted transfer cannot be mistaken for a valid setup.
    (workspace / "PREPARED.json").unlink(missing_ok=True)
    need = missing_bytes(env_dir) if args.download else 0
    free = shutil.disk_usage(data_root).free
    if free < need + RESERVE_BYTES:
        raise RuntimeError(
            f"Insufficient free space: need about {(need + RESERVE_BYTES) / 2**30:.2f} GiB "
            f"including reserve, have {free / 2**30:.2f} GiB"
        )

    if args.download:
        try:
            from huggingface_hub import HfApi, snapshot_download
        except ImportError as error:
            raise RuntimeError("huggingface_hub is not installed in the selected environment") from error
    
        revision_sha = HfApi().dataset_info(args.repo, revision=args.revision).sha
        snapshot_download(
            repo_id=args.repo,
            repo_type="dataset",
            revision=revision_sha,
            allow_patterns=[f"{args.environment}/*", f"{args.environment}/**"],
            local_dir=str(data_root),
            max_workers=args.max_workers,
            force_download=args.force_download,
            cache_dir=str(args.cache_dir.resolve()),
            local_dir_use_symlinks=False,
            resume_download=True,
        )
    else:
        local_source = args.reuse_local.resolve()
        if args.source_prepared is None:
            raise ValueError("--reuse-local requires --source-prepared")
        provenance = json.loads(args.source_prepared.read_text())
        if (provenance.get("hf_resolved_revision") != args.revision
                or provenance.get("hf_repository") != args.repo
                or provenance.get("environment") != args.environment
                or Path(provenance["environment_directory"]).resolve() != local_source):
            raise ValueError("Local source provenance does not match the pinned dataset")
        if env_dir.resolve() == local_source:
            raise ValueError("Local source and sandbox destination must be different")
        validate_environment(local_source)
        env_dir.mkdir(parents=True, exist_ok=True)
        for relative in EXPECTED_FILES:
            source = local_source / relative
            target = env_dir / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() or target.is_symlink():
                if not target.is_symlink() or target.resolve() != source.resolve():
                    raise FileExistsError(f"Refusing to replace existing input: {target}")
            else:
                target.symlink_to(source.resolve())
        revision_sha = args.revision
    actual_sizes = validate_environment(env_dir)

    config_dir.mkdir(parents=True, exist_ok=True)
    (workspace / "log-files").mkdir(exist_ok=True)
    for name in ("nek5000", "SIZE"):
        shutil.copy2(module_case / name, workspace / name)
    (workspace / "nek5000").chmod((workspace / "nek5000").stat().st_mode | 0o111)

    standalone_config = OmegaConf.create(
        {
            "runner": dict(template.RUNNER_CONFIG),
            "simulation": dict(template.SIMULATION_CONFIG),
        }
    )

    # HF supplies packaged mesh/layout fields.  The standalone Python config
    # wins all overlapping deployment values.
    hf_config = OmegaConf.load(env_dir / "environment_config.yaml")
    runtime = OmegaConf.merge(hf_config, standalone_config)
    runtime.simulation.compile_path = str(workspace)
    runtime.simulation.restart_folder = str(env_dir / "restart_files")
    runtime_path = config_dir / "runtime.yml"
    OmegaConf.save(runtime, runtime_path)

    marker = {
        "input_mode": "download" if args.download else "local_immutable_symlinks",
        "local_source": str(args.reuse_local.resolve()) if args.reuse_local else None,
        "prepared_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "hf_repository": args.repo,
        "hf_requested_revision": args.revision,
        "hf_resolved_revision": revision_sha,
        "environment": args.environment,
        "environment_directory": str(env_dir),
        "known_expected_bytes": KNOWN_BYTES,
        "actual_known_file_bytes": sum(actual_sizes.values()),
        "files": actual_sizes,
        "policy_config": str(policy_template),
        "policy_config_sha256": sha256(policy_template),
        "runtime_config": str(runtime_path),
        "runtime_config_sha256": sha256(runtime_path),
        "nek_module_case_directory": str(module_case),
        "nek_executable_sha256": sha256(workspace / "nek5000"),
        "merge_policy": (
            "HF config supplies packaged structural fields; meta_policy_naca_wing.py "
            "wins overlapping values; compile_path/restart_folder point to this workspace."
        ),
    }
    atomic_json(workspace / "PREPARED.json", marker)
    (Path(args.data_root).resolve().parent / "log-files").mkdir(exist_ok=True)

    print(f"Prepared:         {workspace / 'PREPARED.json'}")
    print(f"Runtime config:   {runtime_path}")
    print("The preparation is complete; no Slurm job was submitted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
