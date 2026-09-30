#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"
if [ ! -f "$HW1_CODE/third_party/kenlm/CMakeLists.txt" ]; then
  git -C "$HW1_CODE" submodule update --init third_party/kenlm
fi
if [ ! -x "$HW1_CODE/.tools/kenlm/bin/lmplz" ]; then
  bash "$SCRIPT_DIR/build_kenlm_minimal.sh"
fi
if ! "$HW1_VENV/bin/python" -c 'import kenlm' >/dev/null 2>&1; then
  bash "$SCRIPT_DIR/build_kenlm_python.sh"
fi
