#!/usr/bin/env bash
set -euo pipefail
profile="${1:?missing profile}"
pass="${2:?missing pass}"
phase="${3:?missing phase}"
output="${4:?missing diagnostic output directory}"
source "${WING_CASE_DIR:?}/00_settings.sh"
export APPTAINERENV_OMPI_MCA_pml=ucx
export APPTAINERENV_UCX_TLS=rc,sm,ud,self
export APPTAINERENV_UCX_NET_DEVICES=all
exec apptainer exec \
    --bind "$(zs_bind_paths)" \
    --pwd "${WING_WORK_DIR}" \
    --env OMP_NUM_THREADS=1 --env HWLOC_HIDE_ERRORS=1 \
    --env UCX_WARN_UNUSED_ENV_VARS=n \
    "${WING_IMAGE}" bash --noprofile --norc -c '
        set -eo pipefail
        source /usr/share/lmod/lmod/init/bash
        module --force purge
        module load "$1"
        set -u
        profile="$2"; pass="$3"; phase="$4"; output="$5"
        # Avoid inheriting an unrelated experiment through the submit environment.
        for name in ${!OMPI_MCA_coll*}; do unset "$name"; done
        case "$profile" in
            default) ;;
            tuned_fixed) export OMPI_MCA_coll=^han ;;
            small_rd|large_ring)
                export OMPI_MCA_coll=^han
                export OMPI_MCA_coll_tuned_use_dynamic_rules=1
                export OMPI_MCA_coll_tuned_dynamic_rules_filename="$output/$profile.rules"
                ;;
            *) echo "Unknown collective profile: $profile" >&2; exit 2 ;;
        esac
        # Only one solver rank emits selection diagnostics, in separate probe runs.
        # Keep verbosity off during every timed measurement.
        if [[ "$phase" == probe && "${SLURM_PROCID:-}" == 1 ]]; then
            export OMPI_MCA_coll_base_verbose=60
            export OMPI_MCA_coll_han_verbose=30
        fi
        exec "$output/collective_bench" "$profile" "$pass" "$phase" "$6"
    ' bash "${WING_NEK_MODULE}" "$profile" "$pass" "$phase" "$output" "${SLURM_NTASKS:?}"
