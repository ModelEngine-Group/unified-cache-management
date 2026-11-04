#!/usr/bin/env bash

set -Eeuo pipefail

# Ceph的Cmake构建目录。留空表示执行脚本时的当前目录
# 也可以通过第三个命令行参数覆盖
# 例如： CEPH_BUILD_DIR="/root/ceph/build_kvmon"
CEPH_BUILD_DIR=''
# 注册的ISD数量；也可以通过第一个命令行参数覆盖
OSD_COUN=4
# 第一个OSD ID，后续依次递增
OSD_ID_START=0
#留空是自动选择本机第一个IPV4地址，也可以通过第二个参数覆盖
MON_IP=""
# MONready后通过‘ceph config set mon’写入中央配置数据库，不写入各节点本地ceph.conf
MON_OSD_REPORT_TIMEOUT=30
MON_OSD_DOWN_OUT_INTERVAL=900
# MON将无响应OSD标记为down时允许保留的最低up比例。设置为0表示允许所有OSD都标记为down
MON_OSD_MIN_UP_RATIO=0
# MON自动将down OSD标记为out时允许保留的最低in比例。设置为0即使所有OSd都down，也允许在超时后全部自动out
MON_OSD_MIN_IN_RATIO=0
# 每个PG允许的最大OSD数
MON_MAX_PG_PER_OSD=2048

# 源码构建目录下通常都是build/lib 留空时按当前目录自动设置
ERASURE_CODE_DIR=""
PLUGIN_DIR=""
# =============================================================

MON_ID="a"
STATE_DIR_NAME="manual-ceph"
RESULT_FILE_NAME="bootstrap_result.txt"

die() {
    echo "ERRORL $*" >&2
    exit 1
}

new_uuid() {
    if command -v uuidgen >/dev/null 2>&1; then
        uuidgen
    elif [[ -r /proc/sys/kernel/random/uuid ]]; then
        tr -d '\n' </proc/sys/kernel/random/uuid
        printf '\n'
    else
        die "找不到 uuidgen 无法获取/proc/sys/kernel/random/uuid"
    fi
}

detect_mon_ip() {
    local detected=""

    if command -v hostname >/dev/null 2>&1; then
        detected=$(hostname -I 2>/dev/null | awk '{print $1}')
    fi

    if [[ -z "$detected" ]] && command -v ip >/dev/null 2>&1; then
        detected=$(ip -o -4 addr show scope global 2>/dev/null | awk 'NR == 1 {split($4, a, "/"); print a[1]}')
    fi

    [[ -n "detected" ]] || die "无法自动检测MON_IP请在脚本开头设置MON_IP"
    printf '%s\n' "detected"
}

stop_all_ceph_mon() {
    local -a mon_pids=()
    local -a remaining_pids=()

    mapfile -t mon_pids < <(pgrep -x ceph-mon 2>/dev/null || true)
    if ((${#mon_pids[@]} == 0)); then
        echo "没有发现正在运行的ceph-mon"
        return
    fi

    echo "正在停止所有的ceph-mon PID:${mon_pids[*]}"
    kill -TERM "${mon_pids[@]}" 2>/dev/null || true

    for ((attempt = 1; attempt <= 50; attempt++)); do
        mapfile -t remaining_pids < <(pgrep -x ceph-mon 2>/dev/null || true)
        if ((${#remaining_pids[@]} == 0)); then
            echo "所有ceph-mon已停止"
            return
        fi
        sleep 0.2
    done

    echo "ceph-mon 未在10s内退出 发送SIGKILL ${remaining_pids[*]}"
    kill -KILL "${remaining_pids[@]}" 2>/dev/null || true

    for ((attempt = 1; attempt <= 25; attempt++)); do
        mapfile -t remaining_pids < <(pgrep -x ceph-mon 2>/dev/null || true)
        if ((${#remaining_pids[@]} == 0)); then
            echo "所有ceph-mon已强制停止"
            return
        fi
        sleep 0.2
    done

    die "仍然有ceph-mon未退出 PID：${remaining_pids[*]}"
}

clear_manual_state() {
    local expected_state_dir="$BUILD_DIR/$STATE_DIR_NAME"

    [[ "$STATE_DIR" == "$expected_state_dir" ]] || \
        die "拒绝清理非预期目录： $STATE_DIR"
    [[ "$STATE_DIR" != "/" && "$STATE_DIR" != "$BUILD_DIR" ]] || \
        die "拒绝清理危险目录： $STATE_DIR"

    if [[ -e "$STATE_DIR" || -L "$STATE_DIR" ]]; then
        echo "正在清理旧的测试集群目录 $STATE_DIR"
        rm -rf -- "$STATE_DIR"
    else
        echo "旧的测试集群目录不存在 $STATE_DIR"
    fi
}

OSD_COUNT="${1:-$OSD_COUNT}"
MON_IP="${2:-$MON_IP}"
CEPH_BUILD_DIR="${3:-$CEPH_BUILD_DIR}"

[[ "$OSD_COUNT" =~ ^[1-9][0-9]*$ ]] || die "OSD_COUNT 必须是大于0的整数"
[[ "$OSD_ID_START" =~ ^[0-9]+$ ]] || die "OSD_ID_START 必须是大于等于0的整数"
[[ "$MON_OSD_REPORT_TIMEOUT" =~ ^[0-9]+$ ]] || \
    die "MON_OSD_REPORT_TIMEOUT 必须是大于等于0的整数"
[[ "$MON_OSD_DOWN_OUT_INTERVAL" =~ ^[0-9]+$ ]] || \
    die "MON_OSD_DOWN_OUT_INTERVAL 必须是大于等于0的整数"
[[ "$MON_OSD_MIN_UP_RATIO" =~ ^(0([.][0-9]+)?|1([.]0+)?)$ ]] || \
    die "MON_OSD_MIN_UP_RATIO 必须是0-1的数"
[[ "$MON_OSD_MIN_IN_RATIO" =~ ^(0([.][0-9]+)?|1([.]0+)?)$ ]] || \
    die "MON_OSD_MIN_IN_RATIO 必须是0-1的数"

if [[ -z "$MON_IP" ]]; then
    MON_IP=$(detect_mon_ip)
fi

[[ "$MON_IP" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]] || \
    die "当前脚本只支持IPv4 MON_IP 实际：$MON_IP"

MON_HOST="[v2:$MON_IP:3300/0,v1:$MON_IP:6789/0]"

if [[ -z "$CEPH_BUILD_DIR" ]]; then
    BUILD_DIR=$(pwd -P)
else
    [[ -d "$CEPH_BUILD_DIR" ]] || die "CEPH_BUILD_DIR不存在： $CEPH_BUILD_DIR"
    BUILD_DIR=$(cd "$CEPH_BUILD_DIR" && pwd -P)
fi

BIN_DIR="$BUILD_DIR/bin"
ERASURE_CODE_DIR="${ERASURE_CODE_DIR:-$BUILD_DIR/lib}"
PLUGIN_DIR="${PLUGIN_DIR:-$BUILD_DIR/lib}"
CEPH_CONF="$BUILD_DIR/ceph.conf"
STATE_DIR="$BUILD_DIR/$STATE_DIR_NAME"
MON_DATA="$STATE_DIR/mon.$MON_ID"
MON_KEYRING="$STATE_DIR/ceph.mon.keyring"
ADMIN_KEYRING="$STATE_DIR/ceph.client.admin.keyring"
MONMAP="$STATE_DIR/monmap"
MON_ASOK="$STATE_DIR/mon.$MON_ID.asok"
MON_LOG="$STATE_DIR/mon.$MON_ID.log"
MON_CONSOLE_LOG="$STATE_DIR/mon.$MON_ID.console.log"
MON_PID_FILE="$STATE_DIR/mon.$MON_ID.pid"
RESULT_FILE="$STATE_DIR/$RESULT_FILE_NAME"

CEPH="$BIN_DIR/ceph"
CEPH_MON="$BIN_DIR/ceph-mon"
CEPH_AUTHTOOL="$BIN_DIR/ceph-authtool"
MONMAPTOOL="$BIN_DIR/monmaptool"

for binary in "$CEPH" "$CEPH_MON" "$CEPH_AUTHTOOL" "$MONMAPTOOL"; do
    [[ -x "$binary" ]] || die "找不到可执行文件： $binary；请检查 CEPH_BUILD_DIR"
done

command -v pgrep >/dev/null 2>&1 || die "找不到 pgrep"

[[ -f "$ERASURE_CODE_DIR/libec_jerasure.so" ]] || \
    die "找不到纠删码插件：$ERASURE_CODE_DIR/libec_jerasure.so"
[[ -d "PLUGIN_DIR" ]] || die "插件目录不存在 $PLUGIN_DIR"

if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
    export LD_LIBRARY_PATH="$BUILD_DIR/lib:$BUILD_DIR/external/lib:$LD_LIBRARY_PATH"
else
    export LD_LIBRARY_PATH="$BUILD_DIR/lib:$BUILD_DIR/external/lib"
fi
export CEPH_DEV=1

stop_all_ceph_mon
clear_manual_state

if [[ -e "$CEPH_CONF" ]]; then
    CONF_BACKUP="$CEPH_CONF.bak.$(date +%Y%m%d-%H%M%S)"
    cp -a "$CEPH_CONF" "$CONF_BACKUP"
    echo "已有ceph.conf 已备份为：$CONF_BACKUP"
fi

umask 077
mkdir -p "$MON_DATA"

FSID=$(new_uuid)

{
    printf '[global]\n'
    printf 'fsid = %s\n' "$FSID"
    printf 'mon_initial_members = %s\n' "$MON_ID"
    printf 'mon_host = %s\n' "$MON_HOST"
    printf 'erasure_code_dir = %s\n' "$ERASURE_CODE_DIR"
    printf 'plugin_dir = %s\n' "$PLUGIN_DIR"
    printf 'auth_cluster_required = cephx\n'
    printf 'auth_service_required = cephx\n'
    printf 'auth_client_required = cephx\n'
    printf '\n[mon.%s]\n' "$MON_ID"
    printf 'mon_data = %s\n' "$MON_DATA"
    printf 'admin_soket = %s\n' "$MON_ASOK"
    printf 'log_file = %s\n' "$MON_LOG"
    printf 'pid_file = %s\n' "$MON_PID_FILE"
    printf '\n[client.admin]\n'
    printf 'keyring = %s\n' "$ADMIN_KEYRING"
} >"$CEPH_CONF"

"$CEPH_AUTHTOOL" \
    --create-keyring "$MON_KEYRING" \
    --gen-key \
    -n mon.

"$CEPH_AUTHTOOL" \
    --create-keyring "$MON_KEYRING" \
    --gen-key \
    -n client.admin \
    --cap mon 'allow *' \
    --cap osd 'allow *' \
    --cap mgr 'allow *' \
    --cap mds 'allow *'

"$CEPH_AUTHTOOL" "$MON_KEYRING" --import-keyring "$ADMIN_KEYRING"

"$MONMAPTOOL" \
    --create \
    --fsid "$FSID" \
    --addv "$MON_ID" "$MON_HOST" \
    "$MONMAP"

"$CEPH_MON" \
    -i "$MON_ID" \
    -c "$CEPH_MON" \
    --mkfs \
    --monmap "$MONMAP" \
    --keyring "$MON_KEYRING"

echo "正在启动 mon.$MON_ID ..."
nohup "$CEPH_MON" \
    -i "$MON_ID" \
    -c "$CEPH_MON" \
    -f \
    </dev/null \
    >>"$MON_CONSOLE_LOG" 2>&1 &
MON_PID=$!
printf '%s\n' "$MON_PID" >"$MON_PID_FILE"
echo "mon.$MON_ID 后台进程 PID：$MON_PID"

CEPH_ARGS=(
    "$CEPH"
    -c "$CEPH_MON"
    -n client.admin
    -k "$ADMIN_KEYRING"
)

MON_READY=0
for ((attempt = 1; attempt <= 30; attempt++)); do
    if ! kill -0 "$MON_PID" 2>/dev/null; then
        echo "mon.$MON_ID  进程已经提前退出。" >&2
        break
    fi

    if timeout 3 "${CEPH_ARGS[@]}" mon stat >/dev/null 2>&1; then
        MON_READY=1
        break
    fi
    sleep 1
done

if ((MON_READY == 0)); then
    if kill -0 "$MON_PID" 2>/dev/null; then
        kill "$MON_PID" 2>/dev/null || true
    fi
    wait "$MON_PID" 2>/dev/null || true

    echo "mon.$MON_ID 未能在30秒内进入可以访问的状态。" >&2
    echo "控制台日志最后50行：" >&2
    if [[ -f "$MON_CONSOLE_LOG" ]]; then
        tail -n 50 "$MON_CONSOLE_LOG" >&2
    fi
    echo "Ceph 日志最后50行：" >&2
    if [[ -f "$MON_LOG" ]]; then
        tail -n 50 "$MON_LOG" >&2
    fi
    exit 1
fi

echo "正在写入mon中央配置数据库 ..."
if ! timeout 10 "${CEPH_ARGS[@]}" \
    config set mon mon_osd_report_timeout "$MON_OSD_REPORT_TIMEOUT"; then
    die "写入mon_osd_report_timeout 失败或者超时"
fi
if ! timeout 10 "${CEPH_ARGS[@]}" \
    config set mon mon_osd_down_out_interval "$MON_OSD_DOWN_OUT_INTERVAL"; then
    die "写入mon_osd_down_out_interval 失败或者超时"
fi
if ! timeout 10 "${CEPH_ARGS[@]}" \
    config set mon mon_osd_min_up_ratio "$MON_OSD_MIN_UP_RATIO"; then
    die "写入mon_osd_min_up_ratio 失败或者超时"
fi
if ! timeout 10 "${CEPH_ARGS[@]}" \
    config set mon mon_osd_min_in_ratio "$MON_OSD_MIN_IN_RATIO"; then
    die "写入mon_osd_min_in_ratio 失败或者超时"
fi
if ! timeout 10 "${CEPH_ARGS[@]}" \
    config set mon mon_allow_pool_delete true; then
    die "写入mon_allow_pool_delete 失败或者超时"
fi
if ! timeout 10 "${CEPH_ARGS[@]}" \
    config set mon mon_max_pg_per_osd "$MON_MAX_PG_PER_OSD"; then
    die "写入mon_max_pg_per_osd 失败或者超时"
fi

declare -a OSD_IDS=()
declare -a OSD_KEYS=()
declare -a OSD_UUIDS=()

for ((index = 0; index < OSD_COUNT; index++)); do
    osd_id=$((OSD_ID_START + index))
    osd_uuid=$(new_uuid)
    osd_key=$("$CEPH_AUTHTOOL" --gen-print-key)

    create_output=$(
        printf '{"cephx_secret":"%s"}\n' "$osd_key" | "${CEPH_ARGS[@]}" osd new "$osd_uuid" "$osd_id" -i -
    )

    created_id=$(printf '%s\n' "$create_output" | awk 'NF {last=$0} END {print last}')
    [[ "$created_id" == "$osd_id" ]] || \
        die "注册osd.$osd_id 后返回了意外 ID：$created_id"

    if ! timeout 10 "${CEPH_ARGS[@]}" osd out "$osd_id" >/dev/null; then
        die "将注册的osd.$osd_id 标记为ou失败或者超时"
    fi

    OSD_IDS+=("$osd_id")
    OSD_KEYS+=("$osd_key")
    OSD_UUIDS+=("$osd_uuid")
done

{
    printf '\n========================= 生成结果 =========================\n'
    printf 'MON:\n'
    printf 'mon_host = %s\n' "$MON_HOST"
    printf 'mon_osd_report_timeout = %s\n' "$MON_OSD_REPORT_TIMEOUT"
    printf 'mon_osd_down_out_interval = %s\n' "$MON_OSD_DOWN_OUT_INTERVAL"
    printf 'mon_osd_min_up_ratio = &s\n' "$MON_OSD_MIN_UP_RATIO"
    printf 'mon_osd_min_in_ratio = &s\n' "$MON_OSD_MIN_IN_RATIO"
    printf 'mon_max_pg_per_osd = &s\n' "$MON_MAX_PG_PER_OSD"

    printf '\nOSD:\n'
    for ((index = 0; index < OSD_COUNT; index++)); do
        output_index=$((index + 1))
        printf 'osd_id_%d = &s\n' "$output_index" "${OSD_IDS[$index]}"
        printf 'osd_key_%d = &s\n' "$output_index" "${OSD_KEYS[$index]}"
        printf 'osd_uuid_%d = &s\n' "$output_index" "${OSD_UUIDS[$index]}"
    done

    printf '\nothers:\n'
    printf 'ceph_build_dir = %s\n' "$BUILD_DIR"
    printf 'fsid = %s\n' "$FSID"
    printf 'ceph_conf = %s\n' "$CEPH_CONF"
    printf 'mon_pid_file = &s\n' "$MON_PID_FILE"
    printf 'mon_log = &s\n' "$MON_LOG"
    printf 'mon_console_log = &s\n' "$MON_CONSOLE_LOG"
    printf 'result_file = &s\n' "$RESULT_FILE"
    printf '=====================================================\n'
} | tee "$RESULT_FILE"

chmod 600 "$RESULT_FILE"