#!/usr/bin/env bash
# 1 deterministic TD3 controller + 4096 Nek5000 ranks on JUWELS Cluster CPU nodes.

set -euo pipefail

: "${WING_CASE_DIR:?Use tools/submit.py to supply the case path}"
# shellcheck source=00_settings.sh
source "${WING_CASE_DIR}/00_settings.sh"

mv_data="${WING_MV_DATA}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --mv-data=*)
            mv_data="${1#--mv-data=}"
            shift
            ;;
        --mv-data)
            [[ $# -ge 2 ]] || { echo "[ERR] --mv-data needs true or false" >&2; exit 2; }
            mv_data="$2"
            shift 2
            ;;
        *)
            echo "[ERR] Unknown job argument: $1" >&2
            echo "Usage: sbatch $(basename "$0") [--mv-data=true|false]" >&2
            exit 2
            ;;
    esac
done

case "${mv_data,,}" in
    true|yes|1) mv_data="true" ;;
    false|no|0) mv_data="false" ;;
    *)
        echo "[ERR] --mv-data must be true or false, got: ${mv_data}" >&2
        exit 2
        ;;
esac

environment_dir="${WING_DATA_ROOT}/${WING_ENV_NAME}"
run_root="$(realpath -m -- "${WING_RUNS_DIR}/j-${SLURM_JOB_ID}")"
solver_dir="$(realpath -m -- "${WING_SOLVER_DIR}")"

for path in \
    "${WING_IMAGE}" \
    "${WING_RUNTIME_CONFIG}" \
    "${WING_POLICY_TEMPLATE}" \
    "${WING_WORK_DIR}/PREPARED.json" \
    "${WING_WORK_DIR}/nek5000" \
    "${environment_dir}/environment_config.yaml" \
    "${environment_dir}/restart_files/init_1/rs8naca_wing0.f00001" \
    "${WING_TCF_CHECKPOINT_DIR}" \
    "${WING_TCF_CHECKPOINT_DIR}/model_${WING_MODEL_STEP}_steps.zip" \
    "${WING_TCF_CHECKPOINT_DIR}/vecnormalize_${WING_MODEL_STEP}_steps.pkl"; do
    if [[ ! -e "${path}" ]]; then
        echo "[ERR] Missing ${path}" >&2
        echo "Run ${WING_CASE_DIR}/01_prepare_workspace.sh first." >&2
        exit 1
    fi
done

if [[ -e "${run_root}" ]]; then
    echo "[ERR] Refusing to overwrite ${run_root}" >&2
    exit 1
fi

# Hold the solver lock through archiving, staging and srun. Independent
# experiments must use different WING_SOLVER_DIR values.
mkdir -p "$(dirname "${solver_dir}")"
exec 7>"${solver_dir}.lock"
flock -n 7 || {
    echo "[ERR] Another job or archiver is using ${solver_dir}" >&2
    exit 1
}

# Only the last launched job owns the outputs in this persistent solver.
# Cancellation/time limits may prevent that job from writing RUN_FINISHED.
if [[ "${mv_data}" == "true" && -s "${solver_dir}.last-run" ]]; then
    IFS= read -r previous_root < "${solver_dir}.last-run"
    previous_exit=1
    if [[ -f "${previous_root}/RUN_FINISHED" ]]; then
        previous_exit="$(awk -F= '$1 == "srun_exit" { print $2; exit }' \
            "${previous_root}/RUN_FINISHED")"
    fi
    echo "[MV-DATA] Archiving previous run: ${previous_root}"
    bash "${WING_CASE_DIR}/06_archive_nek_data.sh" \
        --run-root "${previous_root}" \
        --results-root "${WING_RESULTS_DIR}" \
        --case-name naca_wing \
        --solver-exit-code "${previous_exit}" \
        --solver-lock-held
elif [[ "${mv_data}" == "false" ]]; then
    echo "[MV-DATA] Disabled; existing solver outputs may be overwritten by Nek."
fi

mkdir -p "${run_root}"
ln -s "${WING_CASE_DIR}/log-files/slurm-${SLURM_JOB_ID}.out" \
    "${run_root}/slurm-output.log"

# Fail before starting 4097 tasks if the Fortran v17 session path is too long.
nek_probe="${solver_dir}/naca_wing.rea"
if (( ${#nek_probe} > 131 )); then
    echo "[ERR] Nek path exceeds its 131-character safe limit (${#nek_probe}):" >&2
    echo "      ${nek_probe}" >&2
    exit 1
fi

apptainer exec \
    --bind "$(zs_bind_paths)" \
    --pwd "${WING_WORK_DIR}" \
    --env "PYTHONNOUSERSITE=1" \
    "${WING_IMAGE}" \
    bash --noprofile --norc -c '
        set -eo pipefail
        source /usr/share/lmod/lmod/init/bash
        module --force purge
        module load "$1"
        source "$2/bin/activate"
        set -u
        cmd=(python "$3/02_stage_solver.py"
            --config "$4"
            --environment-dir "$5"
            --workspace "$6"
            --solver-dir "$7")
        [[ -n "$8" ]] && cmd+=(--restart-from "$8")
        exec "${cmd[@]}"
    ' bash \
    "${WING_NEK_MODULE}" \
    "${WING_PYTHON_VENV}" \
    "${WING_CASE_DIR}" \
    "${WING_RUNTIME_CONFIG}" \
    "${environment_dir}" \
    "${WING_WORK_DIR}" \
    "${solver_dir}" \
    "${WING_RESTART_FROM}"

# Rank wrappers use this link; it points at live state, not an old snapshot.
# The UCX test instead owns a real job-specific solver at this same path.
if [[ "${solver_dir}" != "${run_root}/s" ]]; then
    ln -s "${solver_dir}" "${run_root}/s"
fi
cp "${solver_dir}/restart-input.json" "${run_root}/restart-input.json"

cp "${WING_RUNTIME_CONFIG}" "${run_root}/runtime-config-used.yml"
cp "${WING_POLICY_TEMPLATE}" "${run_root}/meta_policy_naca_wing.py"
sha256sum \
    "${WING_RUNTIME_CONFIG}" \
    "${WING_POLICY_TEMPLATE}" \
    "${WING_IMAGE}" \
    > "${run_root}/inputs.sha256"

mpmd_file="${run_root}/mpmd.conf"
printf '0 %s controller %s\n' "${WING_CASE_DIR}/04_rank_wrapper.sh" "${run_root}" > "${mpmd_file}"
printf '1-%s %s solver %s\n' \
    "${WING_NPROC_NEK}" "${WING_CASE_DIR}/04_rank_wrapper.sh" "${run_root}" \
    >> "${mpmd_file}"

echo "Job:              ${SLURM_JOB_ID}"
echo "Nodes:            ${SLURM_JOB_NODELIST}"
echo "Controller ranks: 1"
echo "Nek ranks:        ${WING_NPROC_NEK}"
echo "Checkpoint:       ${WING_MODEL_STEP}"
echo "Interactions:     ${WING_STEPS:-100000 from standalone config}"
echo "Record interval:  ${WING_RECORD_EVERY:-100 from standalone config}"
echo "Move Nek data:    ${mv_data}"
echo "Run root:         ${run_root}"
echo "Persistent solver: ${solver_dir}"
echo "Nek path length:  ${#nek_probe}/131"
echo "Started:          $(date --iso-8601=seconds)"

last_run_tmp="${solver_dir}.last-run.$$"
printf '%s\n' "${run_root}" > "${last_run_tmp}"
mv -f -- "${last_run_tmp}" "${solver_dir}.last-run"

set +e
srun \
    --mpi="${ZS_MPI}" \
    --ntasks="${WING_TOTAL_RANKS}" \
    --cpu-bind=cores \
    --kill-on-bad-exit=1 \
    --multi-prog "${mpmd_file}"
srun_exit=$?
set -e

finished="$(date --iso-8601=seconds)"
finish_tmp="${run_root}/.RUN_FINISHED.$$"
printf 'finished_at=%s\nsrun_exit=%s\n' "${finished}" "${srun_exit}" > "${finish_tmp}"
mv -f -- "${finish_tmp}" "${run_root}/RUN_FINISHED"

echo "Finished: ${finished}"
echo "srun exit: ${srun_exit}; current run awaits the next mv-data pass"

exit "${srun_exit}"
