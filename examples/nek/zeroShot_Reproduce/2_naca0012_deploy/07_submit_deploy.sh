#!/usr/bin/env bash
# Print by default; --begin is optional, --submit explicitly launches a job.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/../tools/host_python.sh" "${SCRIPT_DIR}/../tools/submit.py" deploy "$@"
