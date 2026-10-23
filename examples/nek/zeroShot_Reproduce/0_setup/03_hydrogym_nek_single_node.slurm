#!/usr/bin/env bash
# Step 3: one Python controller rank plus ten Nek5000 ranks on one node.
# Submit with:
#   python3 tools/submit.py smoke --submit


set -euo pipefail

# Use an absolute location because Slurm executes a spooled copy of this file.
: "${HG_CASE_DIR:?Use tools/submit.py to supply the setup path}"
# shellcheck source=00_settings.sh
source "${HG_CASE_DIR}/00_settings.sh"

if [[ ! -s "${HG_IMAGE}" ]]; then
    echo "Error: container image not found: ${HG_IMAGE}" >&2
    exit 1
fi

work_dir="${HG_PROJECT_ROOT}/${HG_WORK_REL}"
if [[ ! -f "${work_dir}/SESSION.NAME" ]]; then
    echo "Error: prepared Nek5000 workspace not found: ${work_dir}" >&2
    echo "Run 0_setup/02_prepare_nek_workspace.sh on a login node first." >&2
    exit 1
fi

if [[ "${SLURM_CPUS_PER_TASK}" -lt "${HG_TOTAL_RANKS}" ]]; then
    echo "Error: need at least ${HG_TOTAL_RANKS} CPUs for ${HG_TOTAL_RANKS} MPI ranks." >&2
    exit 1
fi

echo "Job ID:            ${SLURM_JOB_ID}"
echo "Node:              ${SLURM_JOB_NODELIST}"
echo "Environment:       ${HG_ENV_NAME}"
echo "Controller ranks:  1"
echo "Nek5000 ranks:     ${HG_NPROC_NEK}"
echo "Container:         ${HG_IMAGE}"
echo "Work directory:    ${work_dir}"

export OMP_NUM_THREADS=1

srun \
    --ntasks=1 \
    --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
    --cpu-bind=none \
    apptainer exec \
        --cleanenv \
        --bind "${HG_PROJECT_ROOT}:/work" \
        --pwd "/work/${HG_WORK_REL}" \
        --env "XDG_CACHE_HOME=/work/${HG_RUNTIME_HOME_REL}/.cache" \
        --env "HF_HOME=/work/${HG_HF_CACHE_REL}" \
        --env "HF_HUB_OFFLINE=1" \
        --env "MPI4PY_RC_FINALIZE=0" \
        --env "PYTHONNOUSERSITE=1" \
        --env "OMP_NUM_THREADS=1" \
        "${HG_IMAGE}" \
        bash --noprofile --norc -c '
            set -eo pipefail

            source /usr/share/lmod/lmod/init/bash
            module purge
            module load "$1"
            source "$2/bin/activate"
            set -u

            echo "Container MPI: $(mpirun --version | sed -n 1p)"
            echo "Nek5000:      $(command -v nek5000)"
            echo "Python:       $(command -v python)"
            python -c "import hydrogym.nek; print('"'"'HydroGym:    '"'"' + hydrogym.nek.__file__)"

            exec mpirun \
                --host "localhost:$3" \
                --map-by core \
                --bind-to core \
                -np 1 python "/work/$4/test_nek_direct.py" \
                    --env "$5" \
                    --local-dir "/work/$6" \
                    --steps "$7" \
                    --nproc "$8" \
                : \
                -np "$8" nek5000
        ' bash \
        "${HG_NEK_MODULE}" \
        "${HG_PYTHON_VENV}" \
        "${HG_TOTAL_RANKS}" \
        "${HG_EXAMPLE_REL}" \
        "${HG_ENV_NAME}" \
        "${HG_DATA_CACHE_REL}" \
        "${HG_NUM_STEPS}" \
        "${HG_NPROC_NEK}"

echo "HydroGym/Nek5000 single-node test completed successfully."
