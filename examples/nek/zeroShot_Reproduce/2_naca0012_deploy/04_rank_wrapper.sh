#!/usr/bin/env bash
# Enter the container as either the Python controller or one Nek5000 rank.

set -euo pipefail

role="${1:?usage: 04_rank_wrapper.sh controller|solver}"
run_root="${2:?missing run root}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=00_settings.sh
source "${SCRIPT_DIR}/00_settings.sh"

case "${role}" in
    controller|solver) ;;
    *) echo "[ERR] Unknown rank role: ${role}" >&2; exit 2 ;;
esac

# Use the UCX configuration validated by 08_ucx_test.slurm for every rank.
# APPTAINERENV passes the comma-separated transport list into the container.
export APPTAINERENV_OMPI_MCA_pml=ucx
export APPTAINERENV_UCX_TLS=rc,sm,ud,self
export APPTAINERENV_UCX_NET_DEVICES=all

exec apptainer exec \
    --bind "$(zs_bind_paths)" \
    --pwd "${WING_WORK_DIR}" \
    --env "HF_HUB_OFFLINE=1" \
    --env "PYTHONNOUSERSITE=1" \
    --env "OMP_NUM_THREADS=1" \
    --env "MPI4PY_RC_FINALIZE=0" \
    --env "UCX_WARN_UNUSED_ENV_VARS=n" \
    --env "HWLOC_HIDE_ERRORS=1" \
    "${WING_IMAGE}" \
    bash --noprofile --norc -c '
        set -eo pipefail
        source /usr/share/lmod/lmod/init/bash
        module --force purge
        module load "$1"
        source "$2/bin/activate"
        set -u
        role="$3"
        run_root="$4"
        if [[ "$role" == controller ]]; then
            printf "[MPI] OMPI_MCA_pml=%s UCX_TLS=%s UCX_NET_DEVICES=%s\n" \
                "$OMPI_MCA_pml" "$UCX_TLS" "$UCX_NET_DEVICES"
            cmd=(python "$5/03_deploy.py"
                --config "$6"
                --policy-template "$7"
                --solver-dir "${run_root}/s"
                --checkpoint-dir "$8"
                --output-dir "${run_root}/out"
                --nproc "$9"
                --flush-every "${10}"
                --max-seconds "${11}")
            [[ -n "${12}" ]] && cmd+=(--checkpoint-step "${12}")
            [[ -n "${13}" ]] && cmd+=(--steps "${13}")
            [[ -n "${14}" ]] && cmd+=(--record-every "${14}")
            exec "${cmd[@]}"
        fi

        solver_dir="${run_root}/s"
        if [[ ! -f "${solver_dir}/SESSION.NAME" || ! -x "${solver_dir}/nek5000" ]]; then
            echo "[ERR] Nek solver directory is not staged: ${solver_dir}" >&2
            exit 1
        fi
        cd "${solver_dir}"
        exec ./nek5000
    ' bash \
    "${WING_NEK_MODULE}" \
    "${WING_PYTHON_VENV}" \
    "${role}" \
    "${run_root}" \
    "${WING_CASE_DIR}" \
    "${WING_RUNTIME_CONFIG}" \
    "${WING_POLICY_TEMPLATE}" \
    "${WING_TCF_CHECKPOINT_DIR}" \
    "${WING_NPROC_NEK}" \
    "${WING_FLUSH_EVERY}" \
    "${WING_DEPLOY_SECONDS}" \
    "${WING_MODEL_STEP}" \
    "${WING_STEPS}" \
    "${WING_RECORD_EVERY}"
