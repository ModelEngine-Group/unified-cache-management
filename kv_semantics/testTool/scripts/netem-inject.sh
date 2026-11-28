#!/usr/bin/env bash
# ============================================================================
# netem-inject.sh —— 网络延迟 / 丢包故障注入（tc netem）
#
# 用途：在承载 testTool 流量的网卡发送侧（egress）注入延迟 / 抖动 / 丢包 /
#       带宽限制，模拟网络劣化，配合 sop/6 的 P99 指标与 Prometheus/Grafana
#       看板观察 testTool 的时延分位、错误率与恢复行为
#
# 用法（目标 Linux 机器执行；需 root 或 CAP_NET_ADMIN）：
#   bash scripts/netem-inject.sh show [dev]           查看注入状态（缺省全部网卡）
#   bash scripts/netem-inject.sh add <dev> [参数...]  注入（replace 语义，重复执行覆盖）
#   bash scripts/netem-inject.sh del <dev>            清除注入恢复链路
#   参数：
#     --delay  5ms     单向延迟（--jitter 存在时作为均值）
#     --jitter 1ms     延迟抖动幅度（须与 --delay 同用）
#     --dist   normal  抖动分布：normal | pareto | paretonormal
#     --loss   0.1     丢包率 %（RoCE 链路建议 0.01 ~ 1 起步）
#     --corr   25      延迟 / 丢包时间相关性 %（模拟突发式劣化，0 ~ 99）
#     --rate   100mbit 可选：同步限制发送带宽
#     --limit  100000  可选：netem 队列长度（包数；默认放大到 10 万——内核默认
#                      仅 1000，高 IOPS 注入延迟时队列打满会产生计划外丢包）
#
# 生效范围（重要，选错设备注入无效）：
#   1. TCP / UDP 链路：完全生效，注入在实际通信网卡上
#   2. SoftRoCE（rxe）：数据面走内核 UDP 栈，注入在 rxe 绑定的底层网卡上生效
#      （rdma link show 可查 rxe 与 netdev 的绑定关系，脚本 add 时自动提示）
#   3. 硬件 RoCE / InfiniBand（mlx5、Hi1822 等）：verbs 数据面 kernel-bypass，
#      不经内核 qdisc，netem 不生效——脚本检测到会警告；此类链路的延迟 /
#      丢包需在交换机或对端机器注入
#   4. FAKE 链路：进程内模拟不经网络栈，netem 无效
#
# 注意：
#   - 注入在本机发送侧，只影响出向流量；回程方向的劣化需在对端机器执行
#   - RoCE RC 传输的丢包恢复是 go-back-N 整段重传，丢包语义比 TCP 剧烈得多：
#     P99 先于均值恶化，建议从 0.01% 起步观察，勿直接上 1% 以上
#   - 与 conf 键 ratelimit_mb 的区别：本脚本限线路（网卡发送），ratelimit_mb
#     限应用层提交速率，两者叠加可观察排队放大效应
# ============================================================================
set -euo pipefail

# ---------- 工具函数 ----------
die() { echo "错误：$*" >&2; exit 1; }
info() { echo "[netem] $*"; }
warn() { echo "[netem][警告] $*" >&2; }

# 校验时间参数格式（us / ms / s）
valid_time() { printf '%s' "$1" | grep -Eq '^[0-9]+(\.[0-9]+)?(us|ms|s)$'; }
# 校验非负数值
valid_num() { printf '%s' "$1" | grep -Eq '^[0-9]+(\.[0-9]+)?$'; }
# 校验 0 <= 值 <= 上限（支持小数）
num_le() { awk -v v="$1" -v m="$2" 'BEGIN { exit !(v >= 0 && v <= m) }'; }

usage() {
    cat <<'EOF'
用法：bash scripts/netem-inject.sh <show|add|del> [参数...]
  show [dev]            查看注入状态（缺省列出全部网卡与 RDMA 设备概览）
  add  <dev> [参数...]   注入延迟 / 丢包（replace 语义，重复执行覆盖旧值）
  del  <dev>             清除注入恢复链路
参数：
  --delay  5ms     单向延迟（netem 仅作用于发送侧 egress）
  --jitter 1ms     延迟抖动（须配 --delay）
  --dist   normal  抖动分布 normal | pareto | paretonormal
  --loss   0.1     丢包率 %（0 ~ 100；RoCE 链路建议 0.01 ~ 1 起步）
  --corr   25      延迟 / 丢包时间相关性 %（0 ~ 99，模拟突发式劣化）
  --rate   100mbit 可选：同步限制发送带宽
  --limit  100000  可选：netem 队列长度（包数）
示例：
  bash scripts/netem-inject.sh add eth0 --delay 5ms --jitter 1ms --loss 0.1 --corr 25
  bash scripts/netem-inject.sh add eth0 --delay 500ms
  bash scripts/netem-inject.sh show eth0
  bash scripts/netem-inject.sh del eth0
EOF
}

# ---------- RDMA 设备辅助（提示注入点是否生效）----------

# 列出硬件 RDMA 设备关联的网卡名（RoCE 端口 gid 的 ndevs 属性；IB 链路无此属性返回空）
list_hw_rdma_netdevs() {
    local ib f
    for ib in /sys/class/infiniband/*; do
        if [ -d "$ib" ]; then
            case "$(basename "$ib")" in
                rxe*|siw*) continue ;;
            esac
            for f in "$ib"/ports/*/gid_attrs/ndevs/*; do
                if [ -f "$f" ]; then
                    cat "$f" 2>/dev/null || true
                fi
            done
        fi
    done | sort -u
}

# 打印 dev 与 RDMA 的关系提示（add / show 指定 dev 时调用）
check_rdma_hint() {
    local dev="$1" hits="" ndev
    if [ ! -d /sys/class/infiniband ] || [ -z "$(ls -A /sys/class/infiniband 2>/dev/null)" ]; then
        return 0
    fi
    if command -v rdma >/dev/null 2>&1; then
        hits=$(rdma link show 2>/dev/null | grep -E "[[:space:]]netdev[[:space:]]+${dev}([[:space:]]|$)" || true)
    fi
    if [ -n "$hits" ]; then
        info "$dev 是 SoftRoCE(rxe) 底层网卡：RDMA 数据走内核 UDP 栈，netem 生效"
        return 0
    fi
    for ndev in $(list_hw_rdma_netdevs); do
        if [ "$ndev" = "$dev" ]; then
            warn "$dev 是硬件 RoCE/IB 网卡：verbs 数据面 kernel-bypass，"
            warn "netem 只影响其上的 TCP/IP 流量，不影响 RDMA 流量；硬件链路请在交换机或对端注入"
            return 0
        fi
    done
}

# RDMA 设备概览（show 无设备参数时调用）
print_rdma_overview() {
    local d name kind
    if [ ! -d /sys/class/infiniband ] || [ -z "$(ls -A /sys/class/infiniband 2>/dev/null)" ]; then
        echo "  本机无 RDMA 设备（按通信网卡注入即可）"
        return 0
    fi
    for d in /sys/class/infiniband/*; do
        if [ -d "$d" ]; then
            name=$(basename "$d")
            kind="硬件 RoCE/IB（kernel-bypass，netem 对 verbs 流量无效）"
            case "$name" in
                rxe*|siw*) kind="软件模拟 SoftRoCE/SIW（走内核，netem 有效）" ;;
            esac
            echo "  $name：$kind"
        fi
    done
    if command -v rdma >/dev/null 2>&1; then
        echo "  rxe 绑定关系："
        rdma link show 2>/dev/null || true
    fi
}

# ---------- 子命令 ----------

cmd_show() {
    local dev="${1:-}"
    if [ -n "$dev" ]; then
        ip link show dev "$dev" >/dev/null 2>&1 || die "网卡 $dev 不存在"
        tc -s qdisc show dev "$dev"
        check_rdma_hint "$dev"
    else
        echo "== 网卡 qdisc（有 netem 行即注入中）=="
        tc -s qdisc show
        echo "== RDMA 设备概览 =="
        print_rdma_overview
    fi
}

cmd_add() {
    local dev="${1:-}"
    if [ -z "$dev" ]; then
        usage >&2
        die "add 需要设备名"
    fi
    shift
    ip link show dev "$dev" >/dev/null 2>&1 || die "网卡 $dev 不存在"

    # ---- 参数解析 ----
    local delay="" jitter="" dist="normal" loss="" corr="0" rate="" limit=100000 val
    while [ $# -gt 0 ]; do
        case "$1" in
            --delay)
                [ $# -ge 2 ] || die "--delay 缺参数"
                val="$2"; shift 2
                valid_time "$val" || die "--delay 格式应为 <数值>us|ms|s（如 5ms）"
                delay="$val" ;;
            --jitter)
                [ $# -ge 2 ] || die "--jitter 缺参数"
                val="$2"; shift 2
                valid_time "$val" || die "--jitter 格式应为 <数值>us|ms|s（如 1ms）"
                jitter="$val" ;;
            --dist)
                [ $# -ge 2 ] || die "--dist 缺参数"
                val="$2"; shift 2
                case "$val" in
                    normal|pareto|paretonormal) dist="$val" ;;
                    *) die "--dist 仅支持 normal|pareto|paretonormal" ;;
                esac ;;
            --loss)
                [ $# -ge 2 ] || die "--loss 缺参数"
                val="$2"; shift 2
                valid_num "$val" || die "--loss 应为非负数值"
                num_le "$val" 100 || die "--loss 范围 0 ~ 100"
                loss="$val" ;;
            --corr)
                [ $# -ge 2 ] || die "--corr 缺参数"
                val="$2"; shift 2
                valid_num "$val" || die "--corr 应为非负数值"
                num_le "$val" 99 || die "--corr 范围 0 ~ 99（100 会导致 netem 行为异常）"
                corr="$val" ;;
            --rate)
                [ $# -ge 2 ] || die "--rate 缺参数"
                val="$2"; shift 2
                printf '%s' "$val" | grep -Eq '^[0-9]+(bit|kbit|mbit|gbit)$' || die "--rate 格式如 100mbit"
                rate="$val" ;;
            --limit)
                [ $# -ge 2 ] || die "--limit 缺参数"
                val="$2"; shift 2
                printf '%s' "$val" | grep -Eq '^[0-9]+$' || die "--limit 应为正整数"
                limit="$val" ;;
            *)
                usage >&2
                die "未知参数：$1" ;;
        esac
    done

    if [ -z "$delay" ] && [ -z "$loss" ]; then
        die "至少指定 --delay 或 --loss 之一"
    fi
    if [ -n "$jitter" ] && [ -z "$delay" ]; then
        die "--jitter 须与 --delay 同用"
    fi
    if [ "$corr" != "0" ] && [ -z "$jitter" ] && [ -z "$loss" ]; then
        warn "--corr 仅在 --jitter 或 --loss 存在时生效，本次忽略"
    fi

    # ---- 组装 netem 参数（语法：delay TIME [JITTER [CORR%]] [distribution TYPE]；
    #      loss PCT% [CORR%]；rate / limit 独立关键字）----
    local args=("limit" "$limit")
    if [ -n "$delay" ]; then
        args+=("delay" "$delay")
        if [ -n "$jitter" ]; then
            args+=("$jitter")
            if [ "$corr" != "0" ]; then
                args+=("${corr}%")
            fi
            args+=("distribution" "$dist")
        fi
    fi
    if [ -n "$loss" ]; then
        args+=("loss" "${loss}%")
        if [ "$corr" != "0" ]; then
            args+=("${corr}%")
        fi
    fi
    if [ -n "$rate" ]; then
        args+=("rate" "$rate")
    fi

    check_rdma_hint "$dev"
    local before
    before=$(tc qdisc show dev "$dev" 2>/dev/null | head -n 1 || true)
    if ! tc qdisc replace dev "$dev" root netem "${args[@]}"; then
        die "tc 注入失败（权限不足时需 root 或 CAP_NET_ADMIN；容器需以 --cap-add NET_ADMIN 启动）"
    fi
    info "已注入（$dev）：${args[*]}"
    if printf '%s' "$before" | grep -q netem; then
        info "覆盖了原有注入：$before"
    else
        info "注入前 qdisc：$before"
    fi
    tc -s qdisc show dev "$dev"
}

cmd_del() {
    local dev="${1:-}"
    if [ -z "$dev" ]; then
        usage >&2
        die "del 需要设备名"
    fi
    ip link show dev "$dev" >/dev/null 2>&1 || die "网卡 $dev 不存在"
    if tc qdisc show dev "$dev" 2>/dev/null | grep -q netem; then
        tc qdisc del dev "$dev" root || die "tc 删除失败（权限不足时需 root 或 CAP_NET_ADMIN）"
        info "已清除 $dev 上的注入，链路恢复"
    else
        info "$dev 当前无 netem 注入，未改动现有 qdisc"
    fi
    tc -s qdisc show dev "$dev"
}

# ---------- 命令分发 ----------
if [ $# -eq 0 ]; then
    usage
    exit 0
fi
cmd="$1"
shift
case "$cmd" in
    show|add|del) "cmd_$cmd" "$@" ;;
    help|-h|--help) usage ;;
    *)
        usage >&2
        die "未知命令：$cmd" ;;
esac
