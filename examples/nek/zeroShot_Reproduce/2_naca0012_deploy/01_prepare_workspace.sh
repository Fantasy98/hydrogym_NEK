#!/usr/bin/env bash
# Standalone preparation entry point.  It enters the HydroGym container, loads
# the NACA0012 Nek module, and invokes the downloader/config builder.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=00_settings.sh
source "${SCRIPT_DIR}/00_settings.sh"

if [[ ! -f "${WING_IMAGE}" ]]; then
    echo "[ERR] Container image not found: ${WING_IMAGE}" >&2
    exit 1
fi
if [[ ! -f "${WING_POLICY_TEMPLATE}" ]]; then
    echo "[ERR] Standalone policy configuration not found: ${WING_POLICY_TEMPLATE}" >&2
    exit 1
fi

if [[ "${1:-}" == "--reuse-existing" ]]; then
    shift
    : "${WING_LOCAL_SOURCE:?Set WING_LOCAL_SOURCE for local dataset reuse}"
    : "${WING_SOURCE_PREPARED:?Set WING_SOURCE_PREPARED for local dataset reuse}"
    set -- --reuse-local "${WING_LOCAL_SOURCE}" --source-prepared "${WING_SOURCE_PREPARED}" "$@"
fi
exec apptainer exec \
    --bind "$(zs_bind_paths)" \
    --pwd "${WING_CASE_DIR}" \
    --env "PYTHONNOUSERSITE=1" \
    --env "HF_HUB_DOWNLOAD_TIMEOUT=3600" \
    "${WING_IMAGE}" \
    bash --noprofile --norc -c '
        set -eo pipefail
        source /usr/share/lmod/lmod/init/bash
        module --force purge
        module load "$1"
        source "$2/bin/activate"
        set -u
        : "${NEK_CASE_DIR:?NACA0012 module did not define NEK_CASE_DIR}"
        exec python "$3/01_prepare_workspace.py" \
            --repo "$4" \
            --environment "$5" \
            --data-root "$6" \
            --workspace "$7" \
            --policy-template "$8" \
            --config-dir "$9" \
            --revision "${10}" \
            --cache-dir "${11}" \
            --module-case-dir "${NEK_CASE_DIR}" \
            "${@:12}"
    ' bash \
    "${WING_NEK_MODULE}" \
    "${WING_PYTHON_VENV}" \
    "${WING_CASE_DIR}" \
    "${WING_HF_REPO}" \
    "${WING_ENV_NAME}" \
    "${WING_DATA_ROOT}" \
    "${WING_WORK_DIR}" \
    "${WING_POLICY_TEMPLATE}" \
    "${WING_CONFIG_DIR}" \
    "${WING_HF_REVISION}" \
    "${HF_HOME}/hub" \
    "$@"
