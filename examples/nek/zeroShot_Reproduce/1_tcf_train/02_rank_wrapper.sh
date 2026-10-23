#!/usr/bin/env bash
# One Slurm task enters the container and becomes either the Python controller
# or one Nek5000 rank. The roles still share a single 513-rank PMIx world.

set -euo pipefail

role="${1:?usage: 02_rank_wrapper.sh controller|solver}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=00_settings.sh
source "${SCRIPT_DIR}/00_settings.sh"

case "${role}" in
    controller|solver) ;;
    *) echo "[ERR] Unknown rank role: ${role}" >&2; exit 2 ;;
esac

exec apptainer exec \
    --bind "$(zs_bind_paths)" \
    --pwd "${TCF_WORK_DIR}" \
    --env "HF_HUB_OFFLINE=1" \
    --env "PYTHONNOUSERSITE=1" \
    --env "OMP_NUM_THREADS=1" \
    --env "MPI4PY_RC_FINALIZE=0" \
    --env "UCX_WARN_UNUSED_ENV_VARS=n" \
    --env "HWLOC_HIDE_ERRORS=1" \
    "${TCF_IMAGE}" \
    bash --noprofile --norc -c '
        set -eo pipefail
        source /usr/share/lmod/lmod/init/bash
        module --force purge
        module load "$1"
        source "$2/bin/activate"
        set -u
        role="$3"
        if [[ "$role" == controller ]]; then
            exec python "$4/02_train_resumable.py" \
                --config "$5" \
                --env "$6" \
                --local-dir "$7" \
                --nproc "$8" \
                --run-dir "$9" \
                --max-seconds "${10}" \
                --keep-replay "${11}"
        fi

        # Nek case files are located through SESSION.NAME, but the MSTEP
        # checkpoint reader opens rs6tcf0.f* relative to the process working
        # directory. HydroGym also copies new episode restarts here. Therefore
        # solver ranks, unlike the controller, must execute in current_solver.
        solver_dir="${9}/current_solver"
        if [[ ! -f "$solver_dir/SESSION.NAME" || ! -x "$solver_dir/nek5000" ]]; then
            echo "[ERR] Nek solver directory is not staged: $solver_dir" >&2
            exit 1
        fi
        cd "$solver_dir"
        exec ./nek5000
    ' bash \
    "${TCF_NEK_MODULE}" \
    "${TCF_PYTHON_VENV}" \
    "${role}" \
    "${TCF_CASE_DIR}" \
    "${TCF_CONFIG}" \
    "${TCF_ENV_NAME}" \
    "${TCF_DATA_ROOT}" \
    "${TCF_NPROC_NEK}" \
    "${TCF_RUN_DIR}" \
    "${TCF_TRAIN_SECONDS}" \
    "${TCF_KEEP_REPLAY}"

