#!/usr/bin/env bash
# Compare explicit UCX settings with job 15710149, using the same rank count,
# checkpoint, and initial data. Two policy interactions advance ~80 Nek steps.
# Preview from the bundle root: python3 tools/submit.py ucx


set -euo pipefail

# The submission wrapper exports the original case directory for Slurm.
: "${WING_CASE_DIR:?Use tools/submit.py to supply the case path}"
source "${WING_CASE_DIR}/00_settings.sh"

export WING_NPROC_NEK=4096
export WING_TOTAL_RANKS=4097
export WING_NODES=$(((WING_TOTAL_RANKS + ZS_CORES_PER_NODE - 1) / ZS_CORES_PER_NODE))
export WING_TASKS_PER_NODE="${ZS_CORES_PER_NODE}"
export WING_MODEL_STEP=1037646720
export WING_STEPS=2
export WING_FLUSH_EVERY=1
export WING_DEPLOY_SECONDS=3000
export WING_MV_DATA=false
# A diagnostic must never advance or archive the persistent production flow.
export WING_SOLVER_DIR="${WING_RUNS_DIR}/j-${SLURM_JOB_ID}/s"
export WING_RESTART_FROM=""

# APPTAINERENV preserves the comma-separated transport list and overrides
# image defaults in every container launched by the production workflow.
export APPTAINERENV_OMPI_MCA_pml=ucx
export APPTAINERENV_UCX_TLS=rc,sm,ud,self
export APPTAINERENV_UCX_NET_DEVICES=all

if [[ "${SLURM_NTASKS:?Submit this script with sbatch}" != "4097" ]]; then
    echo "[ERR] This comparison requires 4097 total tasks." >&2
    exit 1
fi
for checkpoint in \
    "model_${WING_MODEL_STEP}_steps.zip" \
    "vecnormalize_${WING_MODEL_STEP}_steps.pkl"; do
    if [[ ! -s "${WING_TCF_CHECKPOINT_DIR}/${checkpoint}" ]]; then
        echo "[ERR] Missing baseline checkpoint: ${WING_TCF_CHECKPOINT_DIR}/${checkpoint}" >&2
        exit 1
    fi
done

echo "[UCX TEST] Baseline job: 15710149; checkpoint: ${WING_MODEL_STEP}"
echo "[UCX TEST] Two policy interactions (~80 Nek steps); compare Nek steps 10-55."
echo "[UCX TEST] Controller deadline: ${WING_DEPLOY_SECONDS}s; no prior-run archiving."

# Verify the overrides survive the same module initialization used by ranks.
# This checks configuration propagation, not the active MPI transport.
apptainer exec \
    --bind "$(zs_bind_paths)" \
    "${WING_IMAGE}" \
    bash --noprofile --norc -c '
        set -eo pipefail
        source /usr/share/lmod/lmod/init/bash
        module --force purge
        module load "$1"
        set -u
        printf "[UCX TEST] OMPI_MCA_pml=%s\n" "${OMPI_MCA_pml:-unset}"
        printf "[UCX TEST] UCX_TLS=%s\n" "${UCX_TLS:-unset}"
        printf "[UCX TEST] UCX_NET_DEVICES=%s\n" "${UCX_NET_DEVICES:-unset}"
        [[ "${OMPI_MCA_pml:-}" == ucx &&
           "${UCX_TLS:-}" == rc,sm,ud,self &&
           "${UCX_NET_DEVICES:-}" == all ]] || {
            echo "[ERR] UCX overrides did not survive module initialization." >&2
            exit 1
        }
        ompi_info --version
        ucx_info -v
    ' bash "${WING_NEK_MODULE}"

# Reuse production staging, rank wrappers, and per-job output directories.
# The central submission wrapper supplies the diagnostic allocation.
exec bash "${WING_CASE_DIR}/05_naca0012_deploy.slurm" --mv-data=false
