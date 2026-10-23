#!/usr/bin/env bash
# Step 1: convert the pinned HydroGym Docker image into an Apptainer SIF.
# This is a large download (the Docker registry reports about 37.2 GB).

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=00_settings.sh
source "${script_dir}/00_settings.sh"

command -v apptainer >/dev/null 2>&1 || {
    echo "Error: apptainer is not available in PATH." >&2
    exit 1
}

if ! id -nG | tr ' ' '\n' | grep -qx container; then
    echo "Error: this account is not in the JUWELS 'container' group." >&2
    echo "Request Container Runtime Engine access through JuDoor first." >&2
    exit 1
fi

mkdir -p \
    "${HG_CONTAINER_ROOT}/images" \
    "${HG_CONTAINER_ROOT}/cache" \
    "${HG_CONTAINER_ROOT}/tmp"

export APPTAINER_CACHEDIR="${HG_CONTAINER_ROOT}/cache"
export APPTAINER_TMPDIR="${HG_CONTAINER_ROOT}/tmp"

echo "Image destination: ${HG_IMAGE}"
echo "Pinned source:     ${HG_IMAGE_URI}"
echo "Scratch capacity:"
df -h "${HG_CONTAINER_ROOT}"

if [[ -s "${HG_IMAGE}" && "${HG_FORCE_PULL:-0}" != "1" ]]; then
    echo
    echo "Image already exists; leaving it unchanged."
    echo "Set HG_FORCE_PULL=1 to replace it."
else
    pull_options=()
    if [[ "${HG_FORCE_PULL:-0}" == "1" ]]; then
        pull_options+=(--force)
    fi

    apptainer pull "${pull_options[@]}" "${HG_IMAGE}" "${HG_IMAGE_URI}"
fi

echo
apptainer inspect "${HG_IMAGE}" >/dev/null
ls -lh "${HG_IMAGE}"
echo "HydroGym image is ready."

