#!/bin/bash

# Unified run script for mini_channel demo.
# This mirrors the flow in drl_repo_TEST/execs/run-script:
#   1) initialize run folder
#   2) read nproc + run path
#   3) mpirun: rank0 python + ranks>0 nek5000

ORIG_ARGS=("$@")
set --
source ~/.bashrc.openmpi_ucx
source ~/.bashrc.miniforge
set -- "${ORIG_ARGS[@]}"

unset OMPI_MCA_pml OMPI_MCA_osc UCX_TLS UCX_NET_DEVICES
export HWLOC_HIDE_ERRORS=1
export UCX_TLS=sm,self,tcp,cma,sysv,posix
export OMPI_MCA_btl=self,vader,tcp

LOG_DIR="log-files"
mkdir -p ${LOG_DIR}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default values
CONFIG_NAME="MC16-TD3.yml"
MODE="blowing"
STEPS=100

while [[ $# -gt 1 ]]; do
    case "$1" in
        --config)
            CONFIG_NAME="$2"
            shift 2
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        --steps)
            STEPS="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

CONFIG_PATH="${SCRIPT_DIR}/conf/${CONFIG_NAME}"

mpirun -n 1 python "${SCRIPT_DIR}/nek_initial.py" "${CONFIG_PATH}" \
  > "${LOG_DIR}/log.initial.${CONFIG_NAME}" 2>&1

AGENT_RUN_NAME=$(grep -ri 'agent_run_name' "${CONFIG_PATH}" | awk -F':' '{gsub(/ /,"",$2); print $2}')
NTOT=$(grep -ri 'nproc' "${CONFIG_PATH}" | awk -F':' '{gsub(/ /,"",$2); print $2}')
RUN_PATH=$(head -n 1 "RUN_PATH_${AGENT_RUN_NAME}.txt")
RUN_ROOT=$(dirname "${RUN_PATH}")
RUN_NAME=$(basename "${RUN_PATH}")

echo "CONFIG: ${CONFIG_NAME}, NTOT: ${NTOT}, RUN_PATH: ${RUN_PATH}"

mpirun -n $((1 + ${NTOT})) bash -c "
if [ \$OMPI_COMM_WORLD_RANK -eq 0 ]; then
  python \"${SCRIPT_DIR}/nek_marl_wrapper_demo.py\" \
    --config \"${CONFIG_PATH}\" --mode ${MODE} --steps ${STEPS} \
    --run-root \"${RUN_ROOT}\" --run-name \"${RUN_NAME}\"
else
  cd ${RUN_PATH} && ./nek5000
fi
"  > "${LOG_DIR}/log.run.${CONFIG_NAME}" 2>&1
