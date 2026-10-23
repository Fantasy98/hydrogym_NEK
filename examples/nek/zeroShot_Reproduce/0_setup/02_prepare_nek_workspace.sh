#!/usr/bin/env bash
# Step 2: download the Nek5000 environment data and prepare train_run.
# Run this on a login node, where Hugging Face is reachable.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=00_settings.sh
source "${script_dir}/00_settings.sh"

command -v apptainer >/dev/null 2>&1 || {
    echo "Error: apptainer is not available in PATH." >&2
    exit 1
}

if [[ ! -s "${HG_IMAGE}" ]]; then
    echo "Error: container image not found: ${HG_IMAGE}" >&2
    echo "Run ${script_dir}/01_pull_hydrogym_image.sh first." >&2
    exit 1
fi

if [[ ! -f "${HG_PROJECT_ROOT}/${HG_EXAMPLE_REL}/test_nek_direct.py" ]]; then
    echo "Error: HydroGym Nek5000 example was not found under ${HG_PROJECT_ROOT}." >&2
    exit 1
fi

mkdir -p \
    "${HG_PROJECT_ROOT}/${HG_DATA_CACHE_REL}" \
    "${HG_PROJECT_ROOT}/${HG_HF_CACHE_REL}" \
    "${HG_PROJECT_ROOT}/${HG_RUNTIME_HOME_REL}"

force_download="${HG_FORCE_DOWNLOAD:-0}"
if [[ "${force_download}" != "0" && "${force_download}" != "1" ]]; then
    echo "Error: HG_FORCE_DOWNLOAD must be 0 or 1." >&2
    exit 1
fi

echo "Preparing environment: ${HG_ENV_NAME}"
echo "Host work directory: ${HG_PROJECT_ROOT}/${HG_WORK_REL}"
echo "Data cache:          ${HG_PROJECT_ROOT}/${HG_DATA_CACHE_REL}"

apptainer exec \
    --cleanenv \
    --bind "${HG_PROJECT_ROOT}:/work" \
    --pwd "/work/${HG_EXAMPLE_REL}" \
    --env "XDG_CACHE_HOME=/work/${HG_RUNTIME_HOME_REL}/.cache" \
    --env "HF_HOME=/work/${HG_HF_CACHE_REL}" \
    --env "PYTHONNOUSERSITE=1" \
    "${HG_IMAGE}" \
    bash --noprofile --norc -c '
        set -eo pipefail

        source /usr/share/lmod/lmod/init/bash
        module purge
        module load "$1"
        source "$2/bin/activate"
        set -u

        prepare_args=(
            --env "$3"
            --repo "$8"
            --revision "$9"
            --work-dir "/work/$4"
            --cache-dir "/work/$5"
            --restart-index "$6"
        )
        if [[ "$7" == "1" ]]; then
            prepare_args+=(--force-download)
        fi

        python ./prepare_workspace.py "${prepare_args[@]}"
    ' bash \
    "${HG_NEK_MODULE}" \
    "${HG_PYTHON_VENV}" \
    "${HG_ENV_NAME}" \
    "${HG_WORK_REL}" \
    "${HG_DATA_CACHE_REL}" \
    "${HG_RESTART_INDEX}" \
    "${force_download}" \
    "${ZS_HF_REPO}" \
    "${ZS_HF_REVISION}"

work_dir="${HG_PROJECT_ROOT}/${HG_WORK_REL}"
if [[ ! -f "${work_dir}/SESSION.NAME" ]]; then
    echo "Error: workspace preparation did not create ${work_dir}/SESSION.NAME" >&2
    exit 1
fi
if ! compgen -G "${work_dir}/*.par" >/dev/null; then
    echo "Error: workspace preparation did not create/link a Nek5000 .par file." >&2
    exit 1
fi

echo
echo "Nek5000 workspace is ready: ${work_dir}"
echo "Submission preview: ${ZS_ROOT}/tools/host_python.sh ${ZS_ROOT}/tools/submit.py smoke"
