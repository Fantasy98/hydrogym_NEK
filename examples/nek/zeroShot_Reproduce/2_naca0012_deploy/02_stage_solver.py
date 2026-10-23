#!/usr/bin/env python3
"""Initialize or reuse a persistent v17 NACA0012 solver directory."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from omegaconf import OmegaConf


V17_PARAMETERS = {
    "density": "p001",
    "viscosity": "p002",
    "numSteps": "p011",
    "dt": "p012",
    "writeInterval": "p015",
    "p_residualTol": "p021",
    "v_residualTol": "p022",
    "target_cfl": "p026",
    "writePTS": "p051",
    "CHKPFNUMBER": "p067",
    "writeLA2": "p070",
    "CHKPINTERVAL": "p075",
    "READCHKPT": "p076",
    "AVSTEP": "p087",
    "IOSTEP": "p088",
    "ndrl": "p089",
    "znmf_avg": "p090",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--environment-dir", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--solver-dir", type=Path, required=True)
    parser.add_argument("--restart-from", type=Path, help="Initial seed directory containing rs8 files and the .restart selector")
    return parser.parse_args()


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required input is missing: {path}")


def relative_symlink(source: Path, destination: Path) -> None:
    if destination.is_symlink():
        destination.unlink()
    elif destination.exists():
        raise RuntimeError(f"Expected an immutable input symlink: {destination}")
    destination.symlink_to(os.path.relpath(source, start=destination.parent))


def validate_restart(solver: Path, casename: str, expected_sizes: dict, *, writable: bool) -> dict:
    """Follow the v17 committed-set selector, never filesystem timestamps."""
    selector = solver / f"{casename}.restart"
    require_file(selector)
    selected = int(selector.read_text().strip())
    if selected not in (0, 1):
        raise ValueError(f"Invalid v17 checkpoint selector in {selector}: {selected}")
    if writable:
        for path in [selector, *solver.glob(f"rs8{casename}0.f?????")]:
            if path.is_symlink() or path.stat().st_nlink != 1:
                raise RuntimeError(f"Writable restart must be an independent regular file: {path}")
    members = []
    for index in range(4 * selected + 1, 4 * selected + 5):
        name = f"rs8{casename}0.f{index:05d}"
        path = solver / name
        require_file(path)
        expected = expected_sizes[f"restart_files/init_1/{name}"]
        if path.stat().st_size != expected:
            raise RuntimeError(f"Incomplete checkpoint: {path} (expected {expected} bytes)")
        with path.open("rb") as stream:
            header = stream.read(132).decode("ascii").split()
        if len(header) < 11 or header[0] != "#std" or int(header[1]) != 8:
            raise ValueError(f"Invalid rs8 checkpoint header: {path}")
        members.append({"file": name, "flow_time": float(header[7]), "nek_step": int(header[8])})
    for previous, current in zip(members, members[1:]):
        if current["nek_step"] != previous["nek_step"] + 1 or current["flow_time"] <= previous["flow_time"]:
            raise ValueError(f"Inconsistent checkpoint set selected by {selector}")
    return {"selector": selected, "members": members}


def rewrite_v17_rea(path: Path, simulation) -> None:
    seen = set()
    output = []
    for line in path.read_text().splitlines(keepends=True):
        parts = line.split()
        if len(parts) > 1 and parts[1] in V17_PARAMETERS.values():
            key = next(name for name, token in V17_PARAMETERS.items() if token == parts[1])
            if not hasattr(simulation, key):
                raise KeyError(f"Runtime config has no simulation.{key}")
            parts[0] = f"{float(getattr(simulation, key)):.6E}"
            line = "\t".join(parts) + "\n"
            seen.add(key)
        output.append(line)
    missing = sorted(set(V17_PARAMETERS) - seen)
    if missing:
        raise RuntimeError(f"The v17 .rea file lacks parameters for: {', '.join(missing)}")
    path.write_text("".join(output))


def main() -> int:
    args = parse_args()
    config = args.config.resolve()
    environment = args.environment_dir.resolve()
    workspace = args.workspace.resolve()
    solver = args.solver_dir.resolve()
    require_file(config)
    cfg = OmegaConf.load(config)
    casename = str(cfg.simulation.CASENAME)
    if str(cfg.simulation.solver_version).lower() != "v17":
        raise RuntimeError("This staging script is intentionally restricted to solver_version=v17")
    if int(cfg.runner.random_init) != -2:
        raise ValueError(
            "This deployment requires runner.random_init=-2 so HydroGym does not "
            "copy HF restart files over the persistent checkpoint during reset"
        )
    if int(cfg.runner.rank) != 1:
        raise ValueError(
            "This HF wing payload contains only restart_files/init_1; runner.rank must be 1"
        )

    required_environment = (
        f"{casename}.rea",
        f"{casename}.re2",
        f"{casename}.map",
        f"{casename}.wall",
        f"{casename}.restart",
        f"mask_{casename}0.f00002",
        "stat_pts.in",
        "forparam.i",
    )
    for name in required_environment:
        require_file(environment / name)
    for name in ("nek5000", "SIZE", "PREPARED.json"):
        require_file(workspace / name)

    probe = solver / f"{casename}.rea"
    if len(str(probe)) > 131:
        raise RuntimeError(f"Nek CHARACTER*132 path limit exceeded ({len(str(probe))}): {probe}")
    expected_sizes = json.loads((workspace / "PREPARED.json").read_text())["files"]
    if solver.exists():
        restart_input = validate_restart(solver, casename, expected_sizes, writable=True)
        restart_input.update(mode="reuse", source=str(solver))
    else:
        seed = args.restart_from.resolve() if args.restart_from else environment / "restart_files" / "init_1"
        selector = (seed if args.restart_from else environment) / f"{casename}.restart"
        require_file(selector)
        solver.parent.mkdir(parents=True, exist_ok=True)
        # Publish only a complete, validated seed. Interrupted copies cannot be
        # mistaken for an initialized solver directory on the next submission.
        staging = Path(tempfile.mkdtemp(prefix=f".{solver.name}-seed-", dir=solver.parent))
        try:
            shutil.copy2(selector, staging / selector.name)
            for index in range(1, 9):
                source = seed / f"rs8{casename}0.f{index:05d}"
                require_file(source)
                print(f"[STAGE] Copying writable restart: {source.name}", flush=True)
                subprocess.run(["cp", "--reflink=auto", "--preserve=timestamps", "--", str(source), str(staging / source.name)], check=True)
            restart_input = validate_restart(staging, casename, expected_sizes, writable=True)
            restart_input.update(mode="import" if args.restart_from else "initial", source=str(seed))
            staging.rename(solver)
        finally:
            if staging.exists():
                shutil.rmtree(staging)

    # Preserve writable checkpoint files and selector; refresh small launch
    # inputs and link only immutable mesh/mask payloads.
    for name in ("nek5000", "SIZE"):
        shutil.copy2(workspace / name, solver / name)
    for name in (f"{casename}.rea", "forparam.i"):
        shutil.copy2(environment / name, solver / name)
    for name in (
        f"{casename}.re2",
        f"{casename}.map",
        f"{casename}.wall",
        f"mask_{casename}0.f00002",
        "stat_pts.in",
    ):
        relative_symlink(environment / name, solver / name)

    rewrite_v17_rea(solver / f"{casename}.rea", cfg.simulation)
    (solver / "SESSION.NAME").write_text(f"{casename}\n{solver}\n")
    shutil.copy2(config, solver / "runtime-config.yml")
    (solver / "restart-input.json").write_text(json.dumps(restart_input, indent=2) + "\n")
    print(f"[STAGE] Solver directory: {solver}", flush=True)
    print(f"[STAGE] Nek path length:  {len(str(probe))}/131", flush=True)
    print(f"[STAGE] Restart: {restart_input['mode']}, committed set {restart_input['selector']}", flush=True)
    print(f"[STAGE] Saved flow times: {[member['flow_time'] for member in restart_input['members']]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
