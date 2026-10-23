#!/usr/bin/env bash
# Prepare the large channel; --download obtains its pinned HF package first.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=00_settings.sh
source "${SCRIPT_DIR}/00_settings.sh"

download=0
if [[ $# -eq 1 && "$1" == --download ]]; then
    download=1
elif [[ $# -ne 0 ]]; then
    echo "Usage: $0 [--download]" >&2
    echo "--download obtains the complete pinned HF channel package and verifies the original restart hashes." >&2
    [[ $# -eq 1 && "$1" == --help ]] && exit 0
    exit 2
fi

for path in "${TCF_IMAGE}" "${TCF_CONFIG}"; do
    if [[ ! -e "${path}" ]]; then
        echo "[ERR] Required input is missing: ${path}" >&2
        exit 1
    fi
done

if [[ "${download}" == 1 ]]; then
    "${ZS_ROOT}/tools/host_python.sh" "${ZS_ROOT}/tools/download_channel.py" --download
fi
"${ZS_ROOT}/tools/host_python.sh" "${ZS_ROOT}/tools/channel_restarts.py" --source "${TCF_RESTART_SOURCE}"
mkdir -p "${ZS_CACHE_ROOT}" "${TCF_CASE_DIR}/log-files"
env_dir="${TCF_DATA_ROOT}/${TCF_ENV_NAME}"
mkdir -p "${env_dir}" "${TCF_WORK_DIR}" "${TCF_RUN_DIR}" "${TCF_CHECKPOINT_DIR}"

# The source data are immutable inputs. Symlinks avoid duplicating six large
# restart sets and make their provenance explicit.
ln -sfn "${TCF_CONFIG}" "${env_dir}/environment_config.yaml"
ln -sfn "${TCF_RESTART_SOURCE}" "${env_dir}/restart_files"
: > "${env_dir}/.NEK5000"

# Copy the small case/runtime files out of the image. Absolute symlinks into a
# SIF would not be meaningful to host-side tools, so these are real files.
apptainer exec \
    --cleanenv \
    --bind "$(zs_bind_paths)" \
    "${TCF_IMAGE}" \
    bash --noprofile --norc -c '
        set -eo pipefail
        source /usr/share/lmod/lmod/init/bash
        module --force purge
        module load "$1"
        set -u
        env_dir="$2"
        work_dir="$3"
        for name in tcf.re2 tcf.ma2 tcf.par int_pos; do
            cp -f -- "$NEK_CASE_DIR/$name" "$env_dir/$name"
        done
        for name in nek5000 tcf.re2 tcf.ma2 tcf.par tcf.usr SIZE int_pos; do
            cp -f -- "$NEK_CASE_DIR/$name" "$work_dir/$name"
        done
        chmod u+x "$work_dir/nek5000"
    ' bash "${TCF_NEK_MODULE}" "${env_dir}" "${TCF_WORK_DIR}"

ln -sfn "${env_dir}/environment_config.yaml" "${TCF_WORK_DIR}/environment_config.yaml"
ln -sfn "${env_dir}/restart_files" "${TCF_WORK_DIR}/restart_files"

# This is an early bootstrap file. NekEnv rewrites it to point at the current
# solver directory before the communicator handshake.
printf 'tcf\n%s\n' "${TCF_WORK_DIR}" > "${TCF_WORK_DIR}/SESSION.NAME"

echo "Prepared large-channel inputs:"
echo "  HydroGym package : ${env_dir}"
echo "  Nek workspace    : ${TCF_WORK_DIR}"
echo "  restart source   : ${TCF_RESTART_SOURCE}"
echo "  config           : ${TCF_CONFIG}"
echo
echo "Submission preview: ${ZS_ROOT}/tools/host_python.sh ${ZS_ROOT}/tools/submit.py train"

