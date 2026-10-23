#!/usr/bin/env bash
# Run preparation/validation Python in the pinned environment, without MPI init.
# Usage: tools/container_python.sh train|deploy|smoke PYTHON_ARGUMENTS...
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
stage="${1:?usage: container_python.sh train|deploy|smoke PYTHON_ARGUMENTS...}"
shift
case "${stage}" in
    train) source "${ROOT}/1_tcf_train/00_settings.sh"; image="${TCF_IMAGE}"; module="${TCF_NEK_MODULE}"; venv="${TCF_PYTHON_VENV}" ;;
    deploy) source "${ROOT}/2_naca0012_deploy/00_settings.sh"; image="${WING_IMAGE}"; module="${WING_NEK_MODULE}"; venv="${WING_PYTHON_VENV}" ;;
    smoke) source "${ROOT}/0_setup/00_settings.sh"; image="${HG_IMAGE}"; module="${HG_NEK_MODULE}"; venv="${HG_PYTHON_VENV}" ;;
    *) echo "Unknown stage: ${stage}" >&2; exit 2 ;;
esac
[[ -f "${image}" ]] || { echo "Missing image: ${image}" >&2; exit 1; }
mkdir -p "${ZS_CACHE_ROOT}" "${HF_HOME}"
exec apptainer exec --bind "$(zs_bind_paths)" --pwd "${ZS_ROOT}" \
    --env PYTHONNOUSERSITE=1 --env PYTHONDONTWRITEBYTECODE=1 \
    --env MPI4PY_RC_INITIALIZE=0 --env MPI4PY_RC_FINALIZE=0 \
    "${image}" bash --noprofile --norc -c '
        set -eo pipefail
        source /usr/share/lmod/lmod/init/bash
        module --force purge
        module load "$1"
        source "$2/bin/activate"
        shift 2
        exec python -B "$@"
    ' bash "${module}" "${venv}" "$@"
