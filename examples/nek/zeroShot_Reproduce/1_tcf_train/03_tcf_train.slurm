#!/usr/bin/env bash
# 1 Python controller + 512 Nek5000 ranks on 11 JUWELS Cluster CPU nodes.


set -euo pipefail

: "${TCF_CASE_DIR:?Use tools/submit.py to supply the case path}"
# shellcheck source=00_settings.sh
source "${TCF_CASE_DIR}/00_settings.sh"

if [[ -f "${TCF_RUN_DIR}/COMPLETE" ]]; then
    echo "Training is already complete: ${TCF_RUN_DIR}/COMPLETE"
    cat "${TCF_RUN_DIR}/COMPLETE"
    exit 0
fi

for path in "${TCF_IMAGE}" "${TCF_CONFIG}" "${TCF_WORK_DIR}/nek5000" \
            "${TCF_DATA_ROOT}/${TCF_ENV_NAME}/environment_config.yaml"; do
    if [[ ! -e "${path}" ]]; then
        echo "[ERR] Missing ${path}" >&2
        echo "Run ${TCF_CASE_DIR}/01_prepare_tcf.sh first." >&2
        exit 1
    fi
done

mkdir -p "${TCF_RUN_DIR}/history" "${TCF_CHECKPOINT_DIR}"

# Only one continuation may manipulate current_solver/checkpoints at a time.
exec 9>"${TCF_RUN_DIR}/training.lock"
if ! flock -n 9; then
    echo "[ERR] Another training allocation holds ${TCF_RUN_DIR}/training.lock" >&2
    exit 1
fi

# Preserve compact HydroGym/Nek logs from the previous allocation, then rebuild
# the disposable solver directory. RL state remains in run/checkpoints.
solver_dir="${TCF_RUN_DIR}/current_solver"
if [[ -d "${solver_dir}" ]]; then
    previous_job="unknown"
    [[ -f "${TCF_RUN_DIR}/current_job.txt" ]] && previous_job="$(sed -n '1p' "${TCF_RUN_DIR}/current_job.txt")"
    archive="${TCF_RUN_DIR}/history/job-${previous_job}"
    suffix=0
    while [[ -e "${archive}" ]]; do
        suffix=$((suffix + 1))
        archive="${TCF_RUN_DIR}/history/job-${previous_job}-${suffix}"
    done
    mkdir -p "${archive}"
    [[ -d "${solver_dir}/history" ]] && mv "${solver_dir}/history" "${archive}/solver_history"
    [[ -f "${solver_dir}/current_conf.yml" ]] && cp "${solver_dir}/current_conf.yml" "${archive}/"
    case "${solver_dir}" in
        "${TCF_RUN_DIR}"/current_solver) rm -rf -- "${solver_dir}" ;;
        *) echo "[ERR] Refusing unsafe cleanup target: ${solver_dir}" >&2; exit 1 ;;
    esac
fi
printf '%s\n' "${SLURM_JOB_ID}" > "${TCF_RUN_DIR}/current_job.txt"

# Nek workers start at the same time as the Python controller and need their
# case files before HydroGym's MPI handshake. Run HydroGym's own initializer
# now, while no Nek ranks are active, to build a complete solver directory and
# render tcf.par from the selected YAML configuration.
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
    "${solver_dir}"

cp "${TCF_CONFIG}" "${TCF_RUN_DIR}/config-used.yml"
sha256sum "${TCF_CONFIG}" "${TCF_IMAGE}" > "${TCF_RUN_DIR}/inputs.sha256"

mpmd_file="${TCF_RUN_DIR}/mpmd-${SLURM_JOB_ID}.conf"
printf '0 %s controller\n' "${TCF_CASE_DIR}/02_rank_wrapper.sh" > "${mpmd_file}"
printf '1-%s %s solver\n' "${TCF_NPROC_NEK}" "${TCF_CASE_DIR}/02_rank_wrapper.sh" >> "${mpmd_file}"

echo "Job:              ${SLURM_JOB_ID}"
echo "Nodes:            ${SLURM_JOB_NODELIST}"
echo "Controller ranks: 1"
echo "Nek ranks:        ${TCF_NPROC_NEK}"
echo "Run state:        ${TCF_RUN_DIR}"
echo "Started:          $(date --iso-8601=seconds)"

srun \
    --mpi="${ZS_MPI}" \
    --ntasks="${TCF_TOTAL_RANKS}" \
    --cpu-bind=cores \
    --kill-on-bad-exit=1 \
    --multi-prog "${mpmd_file}"

echo "Finished: $(date --iso-8601=seconds)"

