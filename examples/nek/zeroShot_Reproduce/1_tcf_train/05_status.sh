#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=00_settings.sh
source "${SCRIPT_DIR}/00_settings.sh"

echo "Run directory: ${TCF_RUN_DIR}"
if [[ -f "${TCF_RUN_DIR}/COMPLETE" ]]; then
    echo "State: COMPLETE"
    cat "${TCF_RUN_DIR}/COMPLETE"
elif [[ -f "${TCF_CHECKPOINT_DIR}/latest.json" ]]; then
    echo "State: resumable checkpoint available"
    cat "${TCF_CHECKPOINT_DIR}/latest.json"
else
    echo "State: no complete RL checkpoint yet"
fi

echo
echo "Checkpoint storage:"
du -sh "${TCF_CHECKPOINT_DIR}" 2>/dev/null || true
find "${TCF_CHECKPOINT_DIR}" -maxdepth 1 -type f -printf '%TY-%Tm-%Td %TH:%TM  %12s  %f\n' 2>/dev/null | sort -r | head -20

if command -v squeue >/dev/null 2>&1; then
    echo
    echo "Queued/running jobs:"
    squeue --name=hg-tcf-train --user="${USER}" -o '%.18i %.9T %.10M %.10l %.6D %R'
fi

