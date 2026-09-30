#!/usr/bin/env bash
# 免 root 获取 Boost：apt-get download 只下载 .deb（不需要 root），再用 dpkg -x 解压到工作区私有前缀。
# 不修改系统目录，全部落在 D: 盘工作区内。
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE="$(cd "$HERE/.." && pwd)"
TOOLS="$CODE/.tools"
DEBS="$TOOLS/debs"
PREFIX="$TOOLS/debroot"
mkdir -p "$DEBS" "$PREFIX"
export TMPDIR="$CODE/.tmp"; mkdir -p "$TMPDIR"

# 需要的 Boost 组件（KenLM 的 lmplz/build_binary/query 使用 program_options、system、thread；测试用 unit_test_framework）
SEEDS="libboost-program-options-dev libboost-system-dev libboost-thread-dev libboost-test-dev"
PKGS=$(apt-cache depends --recurse --no-recommends --no-suggests --no-conflicts --no-breaks --no-replaces --no-enhances $SEEDS 2>/dev/null \
        | grep -E '^libboost' | sort -u | tr '\n' ' ')
echo "将下载的软件包: $PKGS"
cd "$DEBS"
# 只下载缺失的 .deb
need=""
for p in $PKGS; do
  ls "$DEBS/${p}"_*.deb >/dev/null 2>&1 || need="$need $p"
done
if [ -n "$need" ]; then apt-get download $need; fi
for d in "$DEBS"/*.deb; do dpkg -x "$d" "$PREFIX"; done
echo "解压完成 -> $PREFIX"
du -sh "$DEBS" "$PREFIX"
ls "$PREFIX/usr/lib/x86_64-linux-gnu" | grep -E 'boost_(program_options|system|thread)' | head -12
ls "$PREFIX/usr/include/boost/version.hpp" && grep -E 'define BOOST_LIB_VERSION' "$PREFIX/usr/include/boost/version.hpp"
