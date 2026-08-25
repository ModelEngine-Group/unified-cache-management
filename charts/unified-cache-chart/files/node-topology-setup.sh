#!/bin/bash
# 运行时拓扑初始化：
# 1. 计算 master/worker 地址与 rank
# 2. 加载当前节点对应的 nodeTopologyConfig
# 3. 打印最终生效的拓扑环境变量

load_node_topology_env() {
    NODE_TOPOLOGY_CONFIG_FILE="/etc/node-topology-config/${NODE_NAME}"
    echo ""
    echo "已加载的环境变量："

    if [[ ! -f "$NODE_TOPOLOGY_CONFIG_FILE" ]]; then
        echo "ℹ 未挂载节点配置文件，跳过 nodeTopologyConfig 注入"
        return
    fi

    echo "✓ 加载节点 ${NODE_NAME} 的配置文件: $NODE_TOPOLOGY_CONFIG_FILE"
    set -a
    source "$NODE_TOPOLOGY_CONFIG_FILE"
    set +a

    while IFS='=' read -r raw_key _; do
        key="$(echo "$raw_key" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
        if [[ -z "$key" || "$key" == \#* ]]; then
            continue
        fi
        if [[ -n "${!key}" ]]; then
            echo "  ${key}=${!key}"
        fi
    done < "$NODE_TOPOLOGY_CONFIG_FILE"
}

print_topology_summary() {
    echo "  MASTER_IP=$MASTER_IP"
    echo "  MASTER_PORT=$MASTER_PORT"
    echo "  NODE_RANK=$NODE_RANK"
    echo "  LOCAL_RANK=$LOCAL_RANK"
    echo "========================================"
}

# ===== 互联网卡探测与扇出（详见 plan/iface-auto-detect-2026-06-03.md） =====

# 探测“互联网卡”，回显 "<iface> <iface_ip>"；逐级降级，绝不阻断启动。
# 依赖镜像内完整 iproute2（vLLM/CANN 镜像通常具备）。
detect_interconnect_iface() {
    local iface="" ipaddr=""
    # 信号①：到 master 的出口网卡（多机最准，精确回答“走哪块网卡到达对端”）
    if [[ -n "${MASTER_IP:-}" && "${MASTER_IP:-}" != "${POD_IP:-}" ]]; then
        iface="$(ip -o route get "$MASTER_IP" 2>/dev/null \
                 | awk '{for(i=1;i<=NF;i++) if($i=="dev"){print $(i+1); exit}}')"
    fi
    # 信号②：node IP(HOST_IP) 精确归属的网卡（rank0/单机，或①失败）
    if [[ -z "$iface" ]]; then
        iface="$(ip -o -4 addr show 2>/dev/null \
                 | awk -v ip="${HOST_IP:-${POD_IP:-}}" '{split($4,a,"/"); if(a[1]==ip){print $2; exit}}')"
    fi
    # 信号③：默认路由兜底（最不可靠）
    if [[ -z "$iface" ]]; then
        iface="$(ip -o route show default 2>/dev/null \
                 | awk '{for(i=1;i<=NF;i++) if($i=="dev"){print $(i+1); exit}}')"
        [[ -n "$iface" ]] && echo "[WARN][iface] 退回默认路由网卡 '$iface'，可能不是互联网卡" >&2
    fi
    # 过滤虚拟网卡：命中即作废，避免把 docker0/veth 等扇出
    case "$iface" in
        lo|docker*|cni*|flannel*|cali*|veth*|tunl*|kube*|virbr*)
            echo "[WARN][iface] 探测到 '$iface' 形似虚拟接口，忽略；建议用 nodeTopologyConfig 显式指定" >&2
            iface="" ;;
    esac
    if [[ -n "$iface" ]]; then
        ipaddr="$(ip -o -4 addr show dev "$iface" 2>/dev/null | awk '{split($4,a,"/"); print a[1]; exit}')"
    fi
    echo "$iface $ipaddr"
}

# 校验显式配置的网卡确实存在且有 IPv4。高速/NPU 网卡不一定承载 K8s HOST_IP，
# 因此默认不把 HOST_IP 不匹配当成告警；需要严格校验时设置 UC_VERIFY_IFACE_HOST_IP=true。
verify_explicit_iface() {
    local name="${VLLM_NETWORK_INTERFACE:-${VLLM_USE_NETIF:-${HCCL_SOCKET_IFNAME:-${HCCL_IF_NAME:-${GLOO_SOCKET_IFNAME:-${TP_SOCKET_IFNAME:-${NCCL_SOCKET_IFNAME:-}}}}}}}"
    local ips=""
    [[ -z "$name" ]] && return 0
    name="${name%%,*}"
    name="${name#^}"
    if ! ip link show "$name" >/dev/null 2>&1; then
        echo "[WARN][iface] 配置的网卡 '$name' 在本节点不存在" >&2
        return 0
    fi
    ips="$(ip -o -4 addr show dev "$name" 2>/dev/null | awk '{split($4,a,"/"); print a[1]}' | paste -sd, -)"
    if [[ -z "$ips" ]]; then
        echo "[WARN][iface] 配置网卡 '$name' 没有 IPv4 地址，无法派生 HCCL_IF_IP" >&2
        return 0
    fi
    echo "[INFO][iface] 配置网卡 '$name' IPv4=${ips}（HOST_IP=${HOST_IP:-未知}）"
    if [[ "${UC_VERIFY_IFACE_HOST_IP:-false}" == "true" && -n "${HOST_IP:-}" ]]; then
        ip -o -4 addr show dev "$name" 2>/dev/null | grep -qw "$HOST_IP" \
            || echo "[WARN][iface] 配置网卡 '$name' 未承载 HOST_IP=$HOST_IP，可能配错节点映射" >&2
    fi
    return 0
}

# 探测失败时打印诊断信息（不吞错），帮助定位：缺 ip / 非 hostNetwork / 无路由。
dump_iface_diagnostics() {
    echo "[iface][diag] HOST_IP=[${HOST_IP:-}] POD_IP=[${POD_IP:-}] MASTER_IP=[${MASTER_IP:-}]" >&2
    if command -v ip >/dev/null 2>&1; then
        echo "[iface][diag] ip -o -4 addr show:" >&2
        ip -o -4 addr show 2>&1 | sed 's/^/[iface][diag]   /' >&2
        echo "[iface][diag] default route: $(ip route show default 2>&1)" >&2
    else
        echo "[iface][diag] 'ip' 命令不存在（镜像可能缺 iproute2）" >&2
    fi
    echo "[iface][diag] /proc/net/route 默认网卡: $(awk 'NR>1 && $2=="00000000"{print $1; exit}' /proc/net/route 2>/dev/null)" >&2
}

# 探测/覆盖并扇出互联网卡环境变量。WARN 不阻断（临时关闭 errexit）。
# 优先级：nodeTopologyConfig 显式值 > UC_FORCE_IFACE > 自动探测 > 不设置。
apply_iface_env() {
    local _restore_e=0
    case $- in *e*) _restore_e=1; set +e ;; esac

    local IFACE="" IFACE_IP="" iface_source="" v can_detect=1
    local vars="${UC_IFACE_ENV_VARS:-GLOO_SOCKET_IFNAME TP_SOCKET_IFNAME NCCL_SOCKET_IFNAME HCCL_SOCKET_IFNAME HCCL_IF_NAME VLLM_NETWORK_INTERFACE VLLM_USE_NETIF}"

    # hostNetwork 守卫：非 hostNetwork 下 Pod 内看不到宿主机网卡，跳过探测以免选错
    if [[ -n "${POD_IP:-}" && -n "${HOST_IP:-}" && "${POD_IP}" != "${HOST_IP}" ]]; then
        can_detect=0
        echo "[WARN][iface] 疑似非 hostNetwork（POD_IP=$POD_IP ≠ HOST_IP=$HOST_IP），跳过自动探测；如需请开启 hostNetwork 或显式 nodeTopologyConfig" >&2
    fi

    local explicit_iface="${VLLM_NETWORK_INTERFACE:-${VLLM_USE_NETIF:-${HCCL_SOCKET_IFNAME:-${HCCL_IF_NAME:-${GLOO_SOCKET_IFNAME:-${TP_SOCKET_IFNAME:-${NCCL_SOCKET_IFNAME:-}}}}}}}"
    if [[ -n "$explicit_iface" ]]; then
        IFACE="${explicit_iface%%,*}"
        IFACE="${IFACE#^}"
        iface_source="explicit"
        echo "[INFO][iface] 使用显式网卡配置=$IFACE 派生本机通信 IP"
    elif [[ -n "${UC_FORCE_IFACE:-}" ]]; then
        IFACE="$UC_FORCE_IFACE"
        iface_source="force"
        echo "[INFO][iface] 使用 forceInterface=$IFACE"
    elif [[ "${UC_AUTO_DETECT_IFACE:-true}" == "true" && "$can_detect" == "1" ]]; then
        read -r IFACE IFACE_IP < <(detect_interconnect_iface)
        [[ -n "$IFACE" ]] && iface_source="auto"
        [[ -n "$IFACE" ]] && echo "[INFO][iface] 探测到互联网卡: $IFACE (${IFACE_IP:-未知IP})"
    fi

    if [[ -n "$IFACE" && -z "$IFACE_IP" ]]; then
        IFACE_IP="$(ip -o -4 addr show dev "$IFACE" 2>/dev/null | awk '{split($4,a,"/"); print a[1]; exit}')"
    fi

    if [[ -n "$IFACE" ]]; then
        for v in $vars; do
            if [[ -z "${!v:-}" ]]; then
                export "$v=$IFACE"
            elif [[ "${!v}" != "$IFACE" ]]; then
                echo "[INFO][iface] 保留显式 $v=${!v}（未用探测值 $IFACE 覆盖）"
            fi
        done
        [[ -n "$IFACE_IP" && -z "${HCCL_IF_IP:-}"   ]] && export HCCL_IF_IP="$IFACE_IP"
        # 已固定网卡/IP，关闭 vLLM 多 IP 探测（仅未显式设置时）
        [[ -z "${VLLM_DETECT_MULTI_IP:-}" ]] && export VLLM_DETECT_MULTI_IP="0"
        # 打印最终注入的网卡相关环境变量
        echo "[INFO][iface] 注入的环境变量："
        for v in $vars HCCL_IF_IP VLLM_HOST_IP VLLM_DETECT_MULTI_IP; do
            [[ -n "${!v:-}" ]] && echo "  ${v}=${!v}"
        done
        if [[ -n "${VLLM_HOST_IP:-}" ]]; then
            echo "[INFO][iface] vLLM get_ip() 将使用 VLLM_HOST_IP=${VLLM_HOST_IP}"
        else
            echo "[WARN][iface] 未注入 VLLM_HOST_IP；如需固定 Mooncake/vLLM 注册地址，请在 nodeTopologyConfig 中按节点显式设置" >&2
        fi
    else
        echo "[WARN][iface] 未能确定互联网卡，未注入 *_IFNAME（交由镜像/库默认）" >&2
        if [[ -n "${VLLM_HOST_IP:-}" ]]; then
            echo "[INFO][iface] vLLM get_ip() 将使用 VLLM_HOST_IP=${VLLM_HOST_IP}"
        fi
        dump_iface_diagnostics
        echo "[WARN][iface] 建议：显式配置顶层 nodeTopologyConfig（按节点名指定 GLOO_/HCCL_/NCCL_SOCKET_IFNAME 等），" >&2
        echo "[WARN][iface]   或设置 forceInterface 强制指定网卡。多机务必配置，否则 HCCL/NCCL 可能选错网卡导致建链失败。" >&2
    fi

    verify_explicit_iface

    [[ "$_restore_e" == "1" ]] && set -e
    return 0
}

# 主流程（测试可用 UC_TOPO_SKIP_MAIN=1 跳过，仅加载函数定义）
if [[ -z "${UC_TOPO_SKIP_MAIN:-}" ]]; then
    # 获取当前节点名（优先使用环境变量，否则使用 hostname）
    NODE_IP=${NODE_IP:-$(hostname)}

    echo "========================================"
    echo "  Rank Table 环境变量加载"
    echo "========================================"
    echo "当前节点：$NODE_IP"
    echo ""

    # kthena-only：REPLICA_COUNT 由 pod 注入(=1+workerReplicas)；rank/master 读 kthena 注入的
    # WORKER_INDEX(worker)/ENTRY_ADDRESS(worker)。entry(rank0) 用自身 POD_IP，worker 解析 ENTRY_ADDRESS→host IP。
    export NODE_RANK="${WORKER_INDEX:-0}"
    export LOCAL_RANK="${NODE_RANK}"
    if [[ "${NODE_RANK}" == "0" || -z "${ENTRY_ADDRESS:-}" ]]; then
        export MASTER_IP="${POD_IP}"
    else
        while true; do
            MASTER_IP="$(getent hosts "${ENTRY_ADDRESS}" | awk 'NR==1{print $1}')"
            [[ -n "$MASTER_IP" ]] && break
            echo "等待 entry pod (${ENTRY_ADDRESS}) 就绪..."
            sleep 2
        done
        export MASTER_IP
    fi
    export MASTER_PORT="${MASTER_PORT:-${POD_PORT:-8000}}"

    echo "Master: $MASTER_IP:$MASTER_PORT, Local: $POD_IP, Rank Index: $NODE_RANK Total: $REPLICA_COUNT"
    load_node_topology_env
    apply_iface_env
    print_topology_summary
fi
