#!/usr/bin/env bash
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-python3}"
CXX_BIN="${CXX:-g++}"
AGENT_LIB_DIR="${PYNDS_AGENT_LIB_DIR:-$ROOT_DIR/vllm-uc/agent/release}"
AGENT_LIB_DIR_EXPLICIT=false
[[ -n "${PYNDS_AGENT_LIB_DIR:-}" ]] && AGENT_LIB_DIR_EXPLICIT=true
AGENT_MODE=auto

usage() {
    cat <<'EOF'
用法：
  bash install.sh [--agent-lib-dir DIR] [--python PYTHON] [--no-agent | --mock]

选项：
  --agent-lib-dir DIR  包含 libkv_agent.so 和 libceph-common.so.2 的目录。
                       默认：vllm-uc/agent/release；也可用 PYNDS_AGENT_LIB_DIR。
  --python PYTHON     目标 Python 解释器，默认 python3；也可用 PYTHON。
  --no-agent          强制非集群构建，不链接厂商 agent 库。
  --mock              使用 mock agent，仅用于构建验证，不用于生产。
  -h, --help          显示帮助。

默认自动检测：两个厂商库齐全则支持集群模式；都未提供则构建非集群版本。
非集群版本请在 nds_config.conf 中设置 sceneMode = 0。

系统依赖：Linux、g++、make、Python >= 3.9、libibverbs、OpenSSL、zlib 开发文件。
若系统启用 PEP 668，请先创建并激活 venv，再用 --python 指定解释器；不建议使用 --break-system-packages。
EOF
}

while (($#)); do
    case "$1" in
        --agent-lib-dir)
            [[ $# -ge 2 ]] || { echo "错误：--agent-lib-dir 缺少目录参数" >&2; exit 2; }
            AGENT_LIB_DIR="$2"
            AGENT_LIB_DIR_EXPLICIT=true
            shift 2
            ;;
        --python)
            [[ $# -ge 2 ]] || { echo "错误：--python 缺少解释器参数" >&2; exit 2; }
            PYTHON="$2"
            shift 2
            ;;
        --no-agent)
            AGENT_MODE=none
            shift
            ;;
        --mock)
            AGENT_MODE=mock
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "错误：未知参数 $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

trap 'echo "安装失败：请根据上方错误定位依赖或编译问题。" >&2' ERR

[[ "$(uname -s)" == "Linux" ]] || {
    echo "错误：pynds_kv 原生扩展仅支持 Linux，当前为 $(uname -s)。" >&2
    exit 1
}
for command_name in "$PYTHON" "$CXX_BIN" make; do
    command -v "$command_name" >/dev/null 2>&1 || {
        echo "错误：找不到命令 $command_name。" >&2
        exit 1
    }
done

"$PYTHON" -c 'import sys; assert sys.version_info >= (3, 9), "Python >= 3.9 is required"'

if [[ "$AGENT_LIB_DIR" != /* ]]; then
    AGENT_LIB_DIR="$ROOT_DIR/$AGENT_LIB_DIR"
fi
agent_library="$AGENT_LIB_DIR/libkv_agent.so"
ceph_library="$AGENT_LIB_DIR/libceph-common.so.2"
if [[ "$AGENT_MODE" == auto ]]; then
    if [[ -f "$agent_library" && -f "$ceph_library" ]]; then
        AGENT_MODE=release
    elif [[ -f "$agent_library" || -f "$ceph_library" ]]; then
        echo "错误：集群 agent 依赖不完整，需要同时提供：" >&2
        echo "  $agent_library" >&2
        echo "  $ceph_library" >&2
        exit 1
    elif [[ "$AGENT_LIB_DIR_EXPLICIT" == true ]]; then
        echo "错误：指定的厂商库目录中未找到 libkv_agent.so 和 libceph-common.so.2：$AGENT_LIB_DIR" >&2
        echo "非集群构建可省略 --agent-lib-dir，或显式使用 --no-agent。" >&2
        exit 1
    else
        AGENT_MODE=none
    fi
fi

missing_headers=()
for header in infiniband/verbs.h openssl/md5.h zlib.h; do
    if ! printf '#include <%s>\n' "$header" | "$CXX_BIN" -x c++ -E - >/dev/null 2>&1; then
        missing_headers+=("$header")
    fi
done
if ((${#missing_headers[@]})); then
    echo "错误：缺少开发头文件：${missing_headers[*]}" >&2
    echo "Debian/Ubuntu：sudo apt install build-essential libibverbs-dev libssl-dev zlib1g-dev" >&2
    echo "RHEL/CentOS：sudo dnf install gcc-c++ make rdma-core-devel openssl-devel zlib-devel" >&2
    exit 1
fi

case "$AGENT_MODE" in
    release)
        echo "agent 构建模式：release（支持集群模式）"
        ;;
    none)
        echo "agent 构建模式：none（仅非集群模式，nds_config.conf 需设置 sceneMode = 0）"
        ;;
    mock)
        echo "警告：正在使用 mock agent，构建产物不可用于生产。"
        ;;
esac

echo "[1/5] 清理旧构建产物"
make -C vllm clean
make -C vllm-uc clean
rm -rf build dist

echo "[2/5] 从源码构建 libndsclient.so"
make -C vllm-uc BUILD=release AGENT_MODE="$AGENT_MODE" \
    AGENT_LIB_DIR="$AGENT_LIB_DIR" CXX="$CXX_BIN"

echo "[3/5] 构建 libndskv.so 并整理运行时库"
mkdir -p vllm/NDS_lib vllm/NDS_bin
cp -a vllm-uc/NDS_bin/libndsclient.so vllm/NDS_lib/
make -C vllm BUILD=release CXX="$CXX_BIN"
cp -a vllm-uc/NDS_bin/libndsclient.so vllm/NDS_bin/
if [[ "$AGENT_MODE" == release ]]; then
    cp -a "$agent_library" "$ceph_library" vllm/NDS_bin/
fi

echo "[4/5] 构建并安装 wheel"
"$PYTHON" -m pip wheel --no-deps --no-build-isolation --wheel-dir dist .
wheels=(dist/pynds_kv-*.whl)
[[ ${#wheels[@]} -eq 1 && -f "${wheels[0]}" ]] || {
    echo "错误：未生成唯一 wheel 文件。" >&2
    exit 1
}
"$PYTHON" -m pip install --force-reinstall --no-deps "${wheels[0]}"

echo "[5/5] 验证导入和原生库解析"
"$PYTHON" - <<'PY'
from pathlib import Path
import _pynds_kv_native
import pynds_kv

native_dir = Path(_pynds_kv_native.__file__).resolve().parent
required = {"libndskv.so", "libndsclient.so"}
missing = sorted(name for name in required if not (native_dir / name).is_file())
if missing:
    raise RuntimeError(f"Installed native libraries are incomplete: {missing}")
print(f"pynds_kv {pynds_kv.__version__} 安装成功")
print(f"扩展：{Path(pynds_kv.__file__).resolve()}")
print(f"原生库：{native_dir}")
PY
