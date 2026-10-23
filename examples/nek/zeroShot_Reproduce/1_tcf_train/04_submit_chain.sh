#!/usr/bin/env bash
# Print a chain by default; append --submit only for real submissions.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
count="${1:-1}"
if (( $# )); then shift; fi
exec "${SCRIPT_DIR}/../tools/host_python.sh" "${SCRIPT_DIR}/../tools/submit.py" train --chain "${count}" "$@"
