#!/usr/bin/env bash
# Shared settings for the resumable large-channel training workflow.

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/common/settings.sh"
TCF_PROJECT_ROOT="${ZS_ROOT}"
TCF_CASE_DIR="${TCF_CASE_DIR:-${ZS_ROOT}/1_tcf_train}"
TCF_CONFIG="${TCF_CONFIG:-${TCF_CASE_DIR}/config/pettingzoo_tcflarge_re207.yml}"

TCF_IMAGE="${TCF_IMAGE:-${ZS_IMAGE}}"
TCF_NEK_MODULE="${TCF_NEK_MODULE:-Nek5000/1.0-gompi-2024a-SystemCUDA-LargeChannel}"
TCF_PYTHON_VENV="${TCF_PYTHON_VENV:-${ZS_PYTHON_VENV}}"

TCF_ENV_NAME="${TCF_ENV_NAME:-TCFlarge_3D_Re207}"
TCF_NPROC_NEK="${TCF_NPROC_NEK:-512}"
TCF_TOTAL_RANKS=$((TCF_NPROC_NEK + 1))
TCF_NODES="${TCF_NODES:-$(((TCF_TOTAL_RANKS + ZS_CORES_PER_NODE - 1) / ZS_CORES_PER_NODE))}"
TCF_TASKS_PER_NODE="${TCF_TASKS_PER_NODE:-${ZS_CORES_PER_NODE}}"

# This is the known-compatible large-channel initial-condition set used by
# marl_energy_cost for the Re_tau=207, Re_b=3300 case.
TCF_RESTART_SOURCE="${TCF_RESTART_SOURCE:-${ZS_DATA_ROOT}/channel_restarts}"

TCF_DATA_ROOT="${TCF_DATA_ROOT:-${TCF_CASE_DIR}/data}"
TCF_WORK_DIR="${TCF_WORK_DIR:-${TCF_CASE_DIR}/workspace}"
TCF_RUN_DIR="${TCF_RUN_DIR:-${TCF_CASE_DIR}/run}"
TCF_CHECKPOINT_DIR="${TCF_CHECKPOINT_DIR:-${TCF_RUN_DIR}/checkpoints}"

# Stop the controller one hour before Slurm's 24-hour limit. Nek's own config
# has WALLTIME=23:45; this earlier controller deadline permits a clean replay-
# buffer checkpoint and MPI shutdown.
TCF_TRAIN_SECONDS="${TCF_TRAIN_SECONDS:-82800}"
TCF_KEEP_REPLAY="${TCF_KEEP_REPLAY:-2}"

export TCF_PROJECT_ROOT TCF_CASE_DIR TCF_CONFIG
export TCF_IMAGE TCF_NEK_MODULE TCF_PYTHON_VENV
export TCF_ENV_NAME TCF_NPROC_NEK TCF_TOTAL_RANKS TCF_NODES TCF_TASKS_PER_NODE
export TCF_RESTART_SOURCE TCF_DATA_ROOT TCF_WORK_DIR TCF_RUN_DIR TCF_CHECKPOINT_DIR
export TCF_TRAIN_SECONDS TCF_KEEP_REPLAY

