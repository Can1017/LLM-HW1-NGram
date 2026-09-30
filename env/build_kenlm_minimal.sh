#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
if [ ! -f "$HW1_CODE/.tools/debroot/usr/include/boost/version.hpp" ]; then
  bash "$(dirname "$0")/fetch_boost_debs.sh"
fi
cd "$HW1_CODE/.tmp"
for package in cmake cmake-data libarchive13 libjsoncpp25 librhash0; do
  if ! ls "${package}"_*.deb >/dev/null 2>&1; then
    apt-get download "$package"
  fi
done
mkdir -p "$HW1_CODE/.tools/cmake-root"
for deb in cmake_*_amd64.deb cmake-data_*_all.deb libarchive13_*_amd64.deb libjsoncpp25_*_amd64.deb librhash0_*_amd64.deb; do
  dpkg -x "$deb" "$HW1_CODE/.tools/cmake-root"
done
export PATH="$HW1_CODE/.tools/cmake-root/usr/bin:$PATH"
export CMAKE_ROOT="$HW1_CODE/.tools/cmake-root/usr/share/cmake-3.22"
export LD_LIBRARY_PATH="$HW1_CODE/.tools/cmake-root/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
cmake --version
cmake -S "$HW1_CODE/third_party/kenlm" -B "$HW1_CODE/.tools/build-kenlm" \
  -DCMAKE_BUILD_TYPE=Release -DKENLM_MAX_ORDER=10 \
  -DBoost_NO_BOOST_CMAKE=ON -DBoost_NO_SYSTEM_PATHS=ON \
  -DBOOST_ROOT="$HW1_CODE/.tools/debroot/usr" \
  -DBOOST_INCLUDEDIR="$HW1_CODE/.tools/debroot/usr/include" \
  -DBOOST_LIBRARYDIR="$HW1_CODE/.tools/debroot/usr/lib/x86_64-linux-gnu" \
  -DCMAKE_BUILD_RPATH="$HW1_CODE/.tools/debroot/usr/lib/x86_64-linux-gnu" \
  -DCMAKE_INSTALL_RPATH="$HW1_CODE/.tools/debroot/usr/lib/x86_64-linux-gnu"
cmake --build "$HW1_CODE/.tools/build-kenlm" -j4 --target lmplz build_binary query
mkdir -p "$HW1_CODE/.tools/kenlm/bin"
for binary in lmplz build_binary query; do
  cp "$HW1_CODE/.tools/build-kenlm/bin/$binary" "$HW1_CODE/.tools/kenlm/bin/$binary"
done
