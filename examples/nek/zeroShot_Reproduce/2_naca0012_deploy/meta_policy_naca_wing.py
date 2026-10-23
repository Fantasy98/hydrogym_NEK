#!/usr/bin/env python3
"""Standalone meta-policy configuration for NACA0012 zero-shot deployment.

The compiled suction-side actuators use the newly trained TCF TD3 policy. The
workspace preparer combines these deployment values with the packaged HF
environment's v17 mesh fields to generate HydroGym's runtime YAML.
"""

import os
from pathlib import Path


ENV_NAME = "NACA0012_3D_Re200000_AOA0"
CASENAME = "naca_wing"
NPROC = 4096
NUM_STEPS = 100_000 # Usually we can not reach it for a 24-hour run
RECORD_EVERY = 100
INITIAL_CONDITION = 1

# Keep the selected episode-40 policy when invoking the Python driver
# directly as well as through Slurm. WING_MODEL_STEP can explicitly override it.
CHECKPOINT_DIR = Path(os.environ.get(
    "WING_TCF_CHECKPOINT_DIR",
    str(Path(__file__).resolve().parents[1] / "1_tcf_train/run/checkpoints"),
)).resolve()
CHECKPOINT_STEP = 576000000
POLICY_ROOT = str(CHECKPOINT_DIR)


RUNNER_CONFIG = {
    "ctrl_min_amp": -1.0,
    "ctrl_max_amp": 1.0,
    "npl_state": 2,
    "vars_record_freq": RECORD_EVERY,
    "nb_interactions": NUM_STEPS,
    "RL_algorithm": "TD3",
    "rescale_actions": True,
    "evaluation": True,
    "rank": INITIAL_CONDITION,
    # -2 preserves the staged flow checkpoint across jobs. A positive value
    # would randomly select and copy an init_N on every environment reset.
    "random_init": -2,
    "normalize_input": "None",
    "u_tau": 0.048,
    "dUdy": 480.54,
}


SIMULATION_CONFIG = {
    "CASENAME": CASENAME,
    "solver_version": "v17",
    "exeName": "nek5000",
    "TOTCTRL": 5000,
    "nproc": NPROC,
    "ndrl": 40, # Ensure 0.6 t+
    "znmf_avg": 1,
    "target_cfl": 3.0,
    "numSteps": 10_000_000,
    "dt": -4.0e-5,  # syntax in Nek
    "writeInterval": 5000,
    "writeLA2": 5000,
    "p_residualTol": 1.0e-8,
    "v_residualTol": 1.0e-8,
    "density": 1.0,
    "viscosity": -2.0e5, # syntax in Nek
    "READCHKPT": 1,
    "CHKPFNUMBER": 6,
    "CHKPINTERVAL": 5000,
    "AVSTEP": 10,
    "IOSTEP": 5000,
}


# One shared model covers the compiled suction-side control region. Each
# HydroGym interaction already advances simulation.ndrl=40 Nek steps.
POLICY_SPECS = [
    {
        "name": "TCF-TD3-SS",
        # We make the limit slightly larger than the wing chord, 
        # to avoid clipping the control region.
        "x_range": [0.24, 0.88], 
        "side": "SS",
        "algorithm": "TD3",
        "shared_policy": True,
        "checkpoint_dir": str(CHECKPOINT_DIR),
        "checkpoint_step": CHECKPOINT_STEP,
        "drl_step": 1,
        # Observations are scaled inside the v17 executable.  Keep the
        # demo-style Python divisor at one to prevent double normalization.
        "u_tau": 1.0,
        "configured_u_tau": RUNNER_CONFIG["u_tau"],
        "baseline_dudy": RUNNER_CONFIG["dUdy"],
        "action_bounds": [
            RUNNER_CONFIG["ctrl_min_amp"],
            RUNNER_CONFIG["ctrl_max_amp"],
        ],
    },
]
