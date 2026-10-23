#!/usr/bin/env bash
# Enter the container as either the deterministic evaluator or one Nek rank.

set -euo pipefail

role="${1:?usage: 08_eval_rank_wrapper.sh controller|solver}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=00_settings.sh
source "${SCRIPT_DIR}/00_settings.sh"

case "${role}" in
    controller|solver) ;;
    *) echo "[ERR] Unknown rank role: ${role}" >&2; exit 2 ;;
esac

: "${TCF_EVAL_ROOT:?TCF_EVAL_ROOT was not exported by the Slurm script}"
: "${TCF_EVAL_ICS:=1,2,3,4,5,6}"
: "${TCF_EVAL_FLUSH:=100}"

exec apptainer exec \
    --bind "$(zs_bind_paths)" \
    --pwd "${TCF_WORK_DIR}" \
    --env "HF_HUB_OFFLINE=1" \
    --env "PYTHONNOUSERSITE=1" \
    --env "OMP_NUM_THREADS=1" \
    --env "MPI4PY_RC_FINALIZE=0" \
    --env "UCX_WARN_UNUSED_ENV_VARS=n" \
    --env "HWLOC_HIDE_ERRORS=1" \
    --env "TCF_EVAL_ROOT=${TCF_EVAL_ROOT}" \
    --env "TCF_EVAL_ICS=${TCF_EVAL_ICS}" \
    --env "TCF_EVAL_STEP=${TCF_EVAL_STEP:-}" \
    --env "TCF_EVAL_STEPS=${TCF_EVAL_STEPS:-}" \
    --env "TCF_EVAL_FLUSH=${TCF_EVAL_FLUSH}" \
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
            options=()
            [[ -n "${TCF_EVAL_STEP}" ]] && options+=(--checkpoint-step "${TCF_EVAL_STEP}")
            [[ -n "${TCF_EVAL_STEPS}" ]] && options+=(--steps "${TCF_EVAL_STEPS}")
            exec python "$4/07_evaluate_deterministic.py" \
                --config "$5" \
                --env "$6" \
                --local-dir "$7" \
                --nproc "$8" \
                --solver-dir "${TCF_EVAL_ROOT}/s" \
                --checkpoint-dir "$9" \
                --output-dir "${TCF_EVAL_ROOT}" \
                --ics "${TCF_EVAL_ICS}" \
                --flush-every "${TCF_EVAL_FLUSH}" \
                "${options[@]}"
        fi

        solver_dir="${TCF_EVAL_ROOT}/s"
        if [[ ! -f "$solver_dir/SESSION.NAME" || ! -x "$solver_dir/nek5000" ]]; then
            echo "[ERR] Evaluation solver directory is not staged: $solver_dir" >&2
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
    "${TCF_CHECKPOINT_DIR}"
