#!/usr/bin/env bash
# Run standard-library host utilities without relying on Conda or PATH's Python.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT}/common/settings.sh"
if [[ "${ZS_HOST_PYTHON}" != /* || ! -x "${ZS_HOST_PYTHON}" ]]; then
    echo "Set ZS_HOST_PYTHON to an absolute executable path for Python 3.9+ (default: /usr/bin/python3)." >&2
    exit 2
fi
# Ignore host PYTHONHOME/PYTHONPATH and user-site packages. Explicit -B still
# disables bytecode when -E ignores PYTHONDONTWRITEBYTECODE.
"${ZS_HOST_PYTHON}" -E -s -B -c '
import sys
if sys.version_info < (3, 9):
    raise SystemExit("Host utilities require Python 3.9+; set ZS_HOST_PYTHON to a suitable system interpreter.")
'
exec "${ZS_HOST_PYTHON}" -E -s -B "$@"
