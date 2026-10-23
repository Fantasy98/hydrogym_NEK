#!/usr/bin/env bash
# Deterministic evaluation: 1 Python controller + 512 Nek5000 ranks.


set -euo pipefail

: "${TCF_CASE_DIR:?Use tools/submit.py to supply the case path}"
# shellcheck source=00_settings.sh
source "${TCF_CASE_DIR}/00_settings.sh"

# Keep the Nek working path short. Its Fortran session/file-name buffer is 132
# characters; the former run/evaluations/job-<id>/solver path was truncated.
TCF_EVAL_ROOT="${TCF_EVAL_ROOT:-${TCF_CASE_DIR}/evaluate/e-${SLURM_JOB_ID}}"
TCF_EVAL_SOLVER_DIR="${TCF_EVAL_ROOT}/s"
TCF_EVAL_ICS="${TCF_EVAL_ICS:-1,2,3,4,5,6}"
TCF_EVAL_FLUSH="${TCF_EVAL_FLUSH:-100}"
export TCF_EVAL_ROOT TCF_EVAL_ICS TCF_EVAL_FLUSH
# Evaluate the episode-40 policy; do not follow latest.json.
export TCF_EVAL_STEP="${TCF_EVAL_STEP:-576000000}"
export TCF_EVAL_STEPS="${TCF_EVAL_STEPS:-}"

for path in "${TCF_IMAGE}" "${TCF_CONFIG}" "${TCF_WORK_DIR}/nek5000" \
            "${TCF_DATA_ROOT}/${TCF_ENV_NAME}/environment_config.yaml" \
            "${TCF_CHECKPOINT_DIR}" \
            "${TCF_CHECKPOINT_DIR}/model_${TCF_EVAL_STEP}_steps.zip" \
            "${TCF_CHECKPOINT_DIR}/vecnormalize_${TCF_EVAL_STEP}_steps.pkl"; do
    if [[ ! -e "${path}" ]]; then
        echo "[ERR] Missing ${path}" >&2
        exit 1
    fi
done

if [[ -e "${TCF_EVAL_ROOT}" ]]; then
    echo "[ERR] Refusing to overwrite evaluation output: ${TCF_EVAL_ROOT}" >&2
    exit 1
fi

# Nek constructs <session-directory>/<casename>.rea in a CHARACTER*132
# buffer, even when the binary mesh is later read from .re2. Leave one byte of
# headroom so a custom TCF_EVAL_ROOT fails here instead of inside 512 ranks.
nek_probe="${TCF_EVAL_SOLVER_DIR}/tcf.rea"
if (( ${#nek_probe} > 131 )); then
    echo "[ERR] Nek evaluation path is too long (${#nek_probe} characters):" >&2
    echo "      ${nek_probe}" >&2
    echo "      Choose a shorter TCF_EVAL_ROOT (maximum probe length: 131)." >&2
    exit 1
fi
mkdir -p "${TCF_EVAL_SOLVER_DIR}"

# Build an evaluation-only solver directory.  Training's current_solver,
# checkpoints, and histories are never modified.
apptainer exec \
    --bind "$(zs_bind_paths)" \
    --pwd "${TCF_WORK_DIR}" \
    --env "PYTHONNOUSERSITE=1" \
    "${TCF_IMAGE}" \
    bash --noprofile --norc -c '
        set -eo pipefail
        source /usr/share/lmod/lmod/init/bash
        module --force purge
        module load "$1"
        source "$2/bin/activate"
        set -u
        exec python "$3/02_stage_solver.py" \
            --config "$4" \
            --workspace "$5" \
            --restart-root "$6" \
            --solver-dir "$7"
    ' bash \
    "${TCF_NEK_MODULE}" \
    "${TCF_PYTHON_VENV}" \
    "${TCF_CASE_DIR}" \
    "${TCF_CONFIG}" \
    "${TCF_WORK_DIR}" \
    "${TCF_RESTART_SOURCE}" \
    "${TCF_EVAL_SOLVER_DIR}"

cp "${TCF_CONFIG}" "${TCF_EVAL_ROOT}/config-used.yml"
sha256sum "${TCF_CONFIG}" "${TCF_IMAGE}" > "${TCF_EVAL_ROOT}/inputs.sha256"

mpmd_file="${TCF_EVAL_ROOT}/mpmd.conf"
printf '0 %s controller\n' "${TCF_CASE_DIR}/08_eval_rank_wrapper.sh" > "${mpmd_file}"
printf '1-%s %s solver\n' "${TCF_NPROC_NEK}" "${TCF_CASE_DIR}/08_eval_rank_wrapper.sh" >> "${mpmd_file}"

echo "Job:              ${SLURM_JOB_ID}"
echo "Nodes:            ${SLURM_JOB_NODELIST}"
echo "Nek ranks:        ${TCF_NPROC_NEK}"
echo "Initial states:   ${TCF_EVAL_ICS}"
echo "Checkpoint step:  ${TCF_EVAL_STEP}"
echo "Steps per IC:     ${TCF_EVAL_STEPS:-config default}"
echo "Evaluation root:  ${TCF_EVAL_ROOT}"
echo "Nek path length:  ${#nek_probe}/131 (${nek_probe})"
echo "Started:          $(date --iso-8601=seconds)"

srun \
    --mpi="${ZS_MPI}" \
    --ntasks="${TCF_TOTAL_RANKS}" \
    --cpu-bind=cores \
    --kill-on-bad-exit=1 \
    --multi-prog "${mpmd_file}"

echo "Finished: $(date --iso-8601=seconds)"
