#!/usr/bin/env bash
# Sourcing settings is read-only: no directory creation, downloads or launches.
ZS_ROOT="${ZS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ZS_SITE="${ZS_SITE:-juwels}"
case "${ZS_SITE}" in juwels|jureca) ;; *) echo "Unknown ZS_SITE: ${ZS_SITE}" >&2; return 2 ;; esac
source "${ZS_ROOT}/sites/${ZS_SITE}.sh"
# Optional local overrides stay out of the distributable source bundle.
if [[ -f "${ZS_ROOT}/sites/local.sh" ]]; then source "${ZS_ROOT}/sites/local.sh"; fi
ZS_DATA_ROOT="${ZS_DATA_ROOT:-${ZS_ROOT}/data}"
ZS_CACHE_ROOT="${ZS_CACHE_ROOT:-${ZS_ROOT}/.cache}"
ZS_CONTAINER_ROOT="${ZS_CONTAINER_ROOT:-${ZS_ROOT}/containers}"
ZS_IMAGE="${ZS_IMAGE:-${ZS_CONTAINER_ROOT}/images/hydrogym-nvhpc-26.1.sif}"
ZS_IMAGE_URI="${ZS_IMAGE_URI:-docker://clagemann/hydrogym-nvhpc-26.1_cuda-12.9_turing_ampere@sha256:60a752cef00fd8dc6b505ca6faae6c1bf02abf6e19551e5e6e12250305ae9b53}"
ZS_PYTHON_VENV="${ZS_PYTHON_VENV:-/home/easybuild/venvs/hydrogym_cpu}"
# Host utilities use the system interpreter, independently of Conda/PATH.
ZS_HOST_PYTHON="${ZS_HOST_PYTHON:-/usr/bin/python3}"
ZS_HF_REPO="${ZS_HF_REPO:-dynamicslab/HydroGym-environments}"
ZS_HF_REVISION="${ZS_HF_REVISION:-cf5ef6748443ae72e9647f398487e98d421a7243}"
HF_HOME="${ZS_CACHE_ROOT}/huggingface"
XDG_CACHE_HOME="${ZS_CACHE_ROOT}/xdg"
APPTAINER_CACHEDIR="${ZS_CONTAINER_ROOT}/cache"
APPTAINER_TMPDIR="${ZS_CONTAINER_ROOT}/tmp"
export ZS_ROOT ZS_SITE ZS_DATA_ROOT ZS_CACHE_ROOT ZS_CONTAINER_ROOT ZS_IMAGE ZS_IMAGE_URI
export ZS_HOST_PYTHON ZS_PYTHON_VENV ZS_HF_REPO ZS_HF_REVISION HF_HOME XDG_CACHE_HOME
export APPTAINER_CACHEDIR APPTAINER_TMPDIR
export APPTAINERENV_HF_HOME="${HF_HOME}" APPTAINERENV_XDG_CACHE_HOME="${XDG_CACHE_HOME}"
export PYTHONDONTWRITEBYTECODE=1 APPTAINERENV_PYTHONDONTWRITEBYTECODE=1

# Bind configured paths at identical host/container locations. Include resolved
# targets of external immutable inputs; never rely on a personal scratch root.
zs_bind_paths() {
    local path resolved answer=""
    local -A seen=()
    for path in "${ZS_ROOT}" "${ZS_DATA_ROOT}" "${ZS_CACHE_ROOT}" \
        "${TCF_DATA_ROOT:-}" "${TCF_WORK_DIR:-}" "${TCF_RUN_DIR:-}" \
        "${TCF_RESTART_SOURCE:-}" "${TCF_CHECKPOINT_DIR:-}" "${TCF_EVAL_ROOT:-}" \
        "${WING_LOCAL_SOURCE:-}" "${WING_SOURCE_PREPARED:-}" "${WING_DATA_ROOT:-}" "${WING_WORK_DIR:-}" "${WING_CONFIG_DIR:-}" \
        "${WING_RUNS_DIR:-}" "${WING_RESULTS_DIR:-}" "${WING_TCF_CHECKPOINT_DIR:-}" \
        "${WING_RESTART_FROM:-}" "${WING_SOLVER_DIR:-}"; do
        [[ -n "${path}" ]] || continue
        resolved="$(realpath -m -- "${path}")"
        while [[ ! -d "${resolved}" && "${resolved}" != / ]]; do resolved="$(dirname "${resolved}")"; done
        [[ "${resolved}" != / ]] || { echo "Cannot bind missing path: ${path}" >&2; return 2; }
        [[ -z "${seen[${resolved}]:-}" ]] || continue
        seen["${resolved}"]=1
        if [[ "${path}" == "${WING_LOCAL_SOURCE:-}" || "${path}" == "${WING_SOURCE_PREPARED:-}" ]]; then
            answer+="${answer:+,}${resolved}:${resolved}:ro"
        else
            answer+="${answer:+,}${resolved}:${resolved}"
        fi
    done
    printf '%s\n' "${answer}${ZS_EXTRA_BINDS:+,${ZS_EXTRA_BINDS}}"
}
