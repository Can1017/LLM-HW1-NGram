#!/usr/bin/env bash
# 在 WSL(Ubuntu 22.04) 中【免 root】搭建 KenLM 环境，全部落在工作区(D:)：
#   1) 创建 Python 虚拟环境并安装 requirements.txt（含 pip 版 cmake）
#   2) 免 root 获取 Boost（apt-get download + dpkg -x，见 fetch_boost_debs.sh）
#   3) 从 third_party/kenlm 源码编译 lmplz / build_binary / query / count_ngrams
#   4) 从同一份源码编译并安装 kenlm 的 Python 模块
#   5) 冒烟测试
# 幂等：重复运行会跳过已完成的步骤。
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env.sh"
KENLM_SRC="$HW1_CODE/third_party/kenlm"
BUILD="$HW1_TOOLS/build-kenlm"
BIN="$HW1_TOOLS/kenlm/bin"
PREFIX="$HW1_TOOLS/debroot"
LOG() { echo "[$(date +%H:%M:%S)] $*"; }

# ---- 1) 虚拟环境 ----
if [ ! -x "$HW1_VENV/bin/python" ]; then
  LOG "创建虚拟环境 $HW1_VENV"
  python3 -m venv "$HW1_VENV"
fi
LOG "安装 Python 依赖"
"$HW1_VENV/bin/python" -m pip install --no-cache-dir --upgrade pip wheel setuptools
"$HW1_VENV/bin/python" -m pip install --no-cache-dir -r "$HW1_CODE/requirements.txt"

# ---- 2) Boost ----
if [ ! -f "$PREFIX/usr/include/boost/version.hpp" ]; then
  LOG "获取 Boost（免 root）"
  bash "$HERE/fetch_boost_debs.sh"
fi

# ---- 3) 编译 KenLM 命令行工具 ----
if [ ! -x "$BIN/lmplz" ] || [ ! -x "$BIN/build_binary" ] || [ ! -x "$BIN/query" ]; then
  [ -d "$KENLM_SRC" ] || { echo "缺少 $KENLM_SRC（请先 git clone https://github.com/kpu/kenlm）"; exit 1; }
  mkdir -p "$BUILD" "$BIN"
  LOG "cmake 配置"
  "$HW1_VENV/bin/cmake" -S "$KENLM_SRC" -B "$BUILD" \
      -DCMAKE_BUILD_TYPE=Release -DKENLM_MAX_ORDER=6 \
      -DBoost_NO_BOOST_CMAKE=ON -DBoost_NO_SYSTEM_PATHS=ON \
      -DBOOST_ROOT="$PREFIX/usr" -DBOOST_INCLUDEDIR="$PREFIX/usr/include" \
      -DBOOST_LIBRARYDIR="$PREFIX/usr/lib/x86_64-linux-gnu" \
      -DCMAKE_BUILD_RPATH="$PREFIX/usr/lib/x86_64-linux-gnu" \
      -DCMAKE_INSTALL_RPATH="$PREFIX/usr/lib/x86_64-linux-gnu"
  LOG "编译（-j$(nproc)）"
  "$HW1_VENV/bin/cmake" --build "$BUILD" -j"$(nproc)" --target lmplz build_binary query count_ngrams
  for t in lmplz build_binary query count_ngrams; do cp -f "$BUILD/bin/$t" "$BIN/$t"; done
fi
LOG "命令行工具就绪: $(ls "$BIN" | tr '\n' ' ')"

# ---- 4) Python 模块 ----
if ! "$HW1_VENV/bin/python" -c "import kenlm" 2>/dev/null; then
  LOG "编译安装 kenlm Python 模块"
  "$HW1_VENV/bin/python" -m pip install --no-cache-dir "$KENLM_SRC"
fi

# ---- 5) 冒烟测试 ----
LOG "冒烟测试"
"$BIN/lmplz" --help 2>&1 | head -3 || true
"$HW1_VENV/bin/python" - <<'EOF'
import kenlm, sys
print("python", sys.version.split()[0], "| kenlm module OK:", kenlm.__file__)
EOF
LOG "完成"
