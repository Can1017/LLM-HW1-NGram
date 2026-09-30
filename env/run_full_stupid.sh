#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
exec "$HW1_VENV/bin/python" "$HW1_CODE/run_full_stupid.py" "$@"
