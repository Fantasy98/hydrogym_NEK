#!/bin/bash

# Unified run script for mini_channel demo.
# This mirrors the flow in drl_repo_TEST/execs/run-script:
#   1) initialize run folder
#   2) read nproc + run path
#   3) mpirun: rank0 python + ranks>0 nek5000

ORIG_ARGS=("$@")
set --
# -- Load environment --
# # If the setup is in the part of the container this should not be needed.
# if [[ -f /home/firedrake/.bashrc.openmpi_ucx ]]; then
#     source /home/firedrake/.bashrc.openmpi_ucx
# else
#     echo "Warning: /home/firedrake/.bashrc.openmpi_ucx not found; continuing."
# fi
# if [[ -f /home/firedrake/bin/activate ]]; then
#     source /home/firedrake/bin/activate
# else
#     echo "Warning: /home/firedrake/bin/activate not found; continuing."
# fi

source /home/firedrake/.bashrc.openmpi_ucx
source /home/firedrake/firedrake/bin/activate

#source ~/.bashrc.miniforge
set -- "${ORIG_ARGS[@]}"

unset OMPI_MCA_pml OMPI_MCA_osc UCX_TLS UCX_NET_DEVICES
export HWLOC_HIDE_ERRORS=1
export UCX_TLS=sm,self,tcp,cma,sysv,posix
export OMPI_MCA_btl=self,vader,tcp
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

LOG_DIR="log-files"
mkdir -p ${LOG_DIR}

# SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_DIR=$(pwd)

# -- Default values --
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

# -- Initialize run folder --
CONFIG_PATH="${SCRIPT_DIR}/conf/${CONFIG_NAME}"

# # Go to the compile folder and compile the case.
# COMPILE_PATH=$(grep -ri 'compile_path' "${CONFIG_PATH}" | awk -F':' '{gsub(/ /,"",$2); print $2}')
# COMPILE_PATH=$(echo "${COMPILE_PATH}" | sed 's/"//g')
# if [[ "${COMPILE_PATH}" != /* ]]; then
#     COMPILE_PATH="${SCRIPT_DIR}/${COMPILE_PATH}"
# fi
# echo "COMPILE_PATH: ${COMPILE_PATH}"
# if [[ ! -d "${COMPILE_PATH}" ]]; then
#     echo "Error: compile_path does not exist: ${COMPILE_PATH}"
#     exit 1
# fi
# cd "${COMPILE_PATH}"
# bash compile_script --clean
# bash compile_script --all
# echo "Compiled the case."
# cd "${SCRIPT_DIR}" # Go back to the script directory.


# -- Initialize run folder --
mpirun -n 1 python3 "${SCRIPT_DIR}/nek_initial.py" --config "${CONFIG_PATH}" \
  > "${LOG_DIR}/log.initial.${CONFIG_NAME}" 2>&1

AGENT_RUN_NAME=$(grep -ri 'agent_run_name' "${CONFIG_PATH}" | awk -F':' '{gsub(/ /,"",$2); print $2}')
NTOT=$(grep -ri 'nproc' "${CONFIG_PATH}" | awk -F':' '{gsub(/ /,"",$2); print $2}')
RUN_PATH=$(head -n 1 "RUN_PATH_${AGENT_RUN_NAME}.txt")
RUN_ROOT=$(dirname "${RUN_PATH}")
RUN_NAME=$(basename "${RUN_PATH}")

echo "CONFIG: ${CONFIG_NAME}, NTOT: ${NTOT}, RUN_PATH: ${RUN_PATH}"

# Sanity checks before MPMD launch.
if [[ -z "${NTOT}" || ! "${NTOT}" =~ ^[0-9]+$ || "${NTOT}" -lt 1 ]]; then
    echo "Error: invalid nproc (NTOT) from config: '${NTOT}'"
    exit 1
fi
if [[ -z "${RUN_PATH}" || ! -d "${RUN_PATH}" ]]; then
    echo "Error: RUN_PATH not found: '${RUN_PATH}'"
    exit 1
fi

mpirun -n 1 python3 "${SCRIPT_DIR}/nek_marl_wrapper_demo.py" --config "${CONFIG_PATH}" :\
 -n "${NTOT}" bash -c "cd ${RUN_PATH} && ./nek5000" > "${LOG_DIR}/log.run.${CONFIG_NAME}" 2>&1

# mpirun -n 4 python3 mpi_split_test.py