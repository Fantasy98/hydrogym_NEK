#!/usr/bin/env python3
"""Prepare a complete Nek5000 run directory before the MPMD launch."""

from __future__ import annotations

import argparse
from pathlib import Path

from omegaconf import OmegaConf

from hydrogym.nek.nek_lib.nek_utils import NEK_INIT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--restart-root", type=Path, required=True)
    parser.add_argument("--solver-dir", type=Path, required=True)
    return parser.parse_args()


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required Nek input is missing: {path}")


def main() -> int:
    args = parse_args()
    config = args.config.resolve()
    workspace = args.workspace.resolve()
    restart_root = args.restart_root.resolve()
    solver_dir = args.solver_dir.resolve()

    require_file(config)
    for name in ("nek5000", "tcf.re2", "tcf.ma2", "tcf.usr", "int_pos"):
        require_file(workspace / name)

    if not restart_root.is_dir():
        raise FileNotFoundError(f"Restart root is missing: {restart_root}")

    solver_dir.mkdir(parents=True, exist_ok=True)
    if any(solver_dir.iterdir()):
        raise RuntimeError(f"Refusing to stage into non-empty directory: {solver_dir}")

    cfg = OmegaConf.load(config)
    # NEK_INIT resolves these paths itself. Make them absolute so this helper is
    # independent of the directory from which it is invoked.
    cfg.simulation.compile_path = str(workspace)
    cfg.simulation.restart_folder = str(restart_root)

    initializer = NEK_INIT(
        nek=cfg.simulation,
        drl=cfg.runner,
        rank_folder=str(solver_dir),
    )
    if not initializer.main():
        raise RuntimeError("HydroGym NEK_INIT did not complete")

    casename = str(cfg.simulation.CASENAME)
    required = [
        solver_dir / "nek5000",
        solver_dir / f"{casename}.re2",
        solver_dir / f"{casename}.ma2",
        solver_dir / f"{casename}.usr",
        solver_dir / f"{casename}.par",
        solver_dir / "int_pos",
        solver_dir / "SESSION.NAME",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    restarts = list(solver_dir.glob("rs*"))
    if missing or not restarts:
        details = ", ".join(missing) if missing else "no rs* restart files"
        raise RuntimeError(f"Incomplete Nek solver directory: {details}")

    # Nek starts in workspace and reads SESSION.NAME there before changing to
    # solver_dir. Publish this only after every required input has been staged.
    (workspace / "SESSION.NAME").write_text(f"{casename}\n{solver_dir}\n")

    print(f"[STAGE] Nek solver directory ready: {solver_dir}", flush=True)
    print(f"[STAGE] Bootstrap restart files: {len(restarts)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
