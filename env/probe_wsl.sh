#!/usr/bin/env bash
# 探测 WSL 环境：编译 KenLM 所需的依赖是否齐全（只读，不修改系统）
echo "== user/os =="; id -un; . /etc/os-release && echo "$PRETTY_NAME"; uname -r
echo "== cpu/mem =="; nproc; grep -m1 'model name' /proc/cpuinfo; free -h | sed -n 1,2p
echo "== mounts (D: is DrvFs) =="; mount | grep -E ' /mnt/d | / ' | head -3
echo "== tools =="; for t in gcc g++ make cmake git python3 pip3 taskset curl wget; do printf '%-8s ' "$t"; command -v "$t" || echo "(missing)"; done
echo "== packages =="
for p in build-essential cmake libboost-program-options-dev libboost-system-dev libboost-thread-dev libboost-test-dev zlib1g-dev libbz2-dev liblzma-dev python3-dev python3-venv python3-pip libeigen3-dev; do
  if dpkg -s "$p" >/dev/null 2>&1; then printf '%-36s installed %s\n' "$p" "$(dpkg -s "$p" | awk -F': ' '/^Version/{print $2}')"; else printf '%-36s MISSING\n' "$p"; fi
done
echo "== apt sources / proxy =="; grep -rhE '^(deb|URIs)' /etc/apt/sources.list /etc/apt/sources.list.d 2>/dev/null | head -4
apt-config dump 2>/dev/null | grep -i proxy || echo "(no apt proxy configured)"
env | grep -i '^https\?_proxy' | head -2
echo "== /dev/shm =="; df -h /dev/shm | tail -1
echo "== python =="; python3 --version; python3 -c 'import venv, ensurepip; print("venv+ensurepip ok")'
