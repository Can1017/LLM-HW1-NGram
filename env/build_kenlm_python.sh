#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
if [ ! -x "$HW1_VENV/bin/python" ]; then
  python3 -m venv "$HW1_VENV"
fi
export PATH="$HW1_CODE/.tools/cmake-root/usr/bin:$PATH"
export CMAKE_ROOT="$HW1_CODE/.tools/cmake-root/usr/share/cmake-3.22"
export LD_LIBRARY_PATH="$HW1_CODE/.tools/cmake-root/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
cmake -S "$HW1_CODE/third_party/kenlm" -B "$HW1_CODE/.tools/build-kenlm" \
  -DKENLM_MAX_ORDER=10 -DENABLE_PYTHON=ON -DPYTHON_EXECUTABLE="$HW1_VENV/bin/python" \
  -DPYTHON_INCLUDE_DIR=/usr/include/python3.10 \
  -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.10.so
cmake --build "$HW1_CODE/.tools/build-kenlm" -j4 --target kenlm_python
cp "$HW1_CODE/.tools/build-kenlm/lib/kenlm.so" "$HW1_VENV/lib/python3.10/site-packages/kenlm.so"
"$HW1_VENV/bin/python" -c 'import kenlm; print(kenlm.__file__)'
