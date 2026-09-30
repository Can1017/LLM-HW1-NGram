# 被其它脚本 source：统一目录与环境变量。
# 约束：所有数据、临时文件、缓存都必须落在 D:\PostGraduate\LLM\hw1\code 之内（WSL 中即 /mnt/d/PostGraduate/LLM/hw1/code），
# 绝不写 C 盘。
_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export HW1_CODE="$(cd "$_ENV_DIR/.." && pwd)"
export HW1_TOOLS="$HW1_CODE/.tools"          # 编译产物、解压的 Boost 头文件/库
export HW1_VENV="$HW1_CODE/.venv"            # Python 虚拟环境
export TMPDIR="$HW1_CODE/.tmp"
export XDG_CACHE_HOME="$HW1_CODE/.cache"
export PIP_CACHE_DIR="$HW1_CODE/.cache/pip"
export MPLCONFIGDIR="$HW1_CODE/.cache/mpl"
export PYTHONPYCACHEPREFIX="$HW1_CODE/.cache/pycache"
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
export LC_ALL=C.UTF-8
export LANG=C.UTF-8
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$PIP_CACHE_DIR" "$MPLCONFIGDIR" "$PYTHONPYCACHEPREFIX" "$HW1_CODE/logs"
# KenLM 可执行文件（lmplz / build_binary / query）与虚拟环境优先
export PATH="$HW1_TOOLS/kenlm/bin:$HW1_VENV/bin:$PATH"
# 免 root 解压的 Boost 运行库（可执行文件已带 RPATH，这里再兜底一次）
export LD_LIBRARY_PATH="$HW1_TOOLS/debroot/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
