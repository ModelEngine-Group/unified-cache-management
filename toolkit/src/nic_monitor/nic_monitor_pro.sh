#!/bin/bash

# ================= 配置项 =================
LOG_DIR="$(pwd)/net_log"
DEFAULT_DURATION_HOURS=12
DEFAULT_FOREGROUND_INTERVAL=2
DEFAULT_BACKGROUND_INTERVAL=10
STAT_CYCLE_SECONDS=3600     # 后台日志每隔多少秒进行一次阶段统计 (默认3600秒=1小时)
# ==========================================

# 颜色定义
RED='\033[1;31m'
GREEN='\033[1;32m'
YELLOW='\033[1;33m'
BLUE='\033[1;34m'
CYAN='\033[1;36m'
NC='\033[0m' # No Color

# -------- 权限检查 --------
if [ "$EUID" -ne 0 ]; then
  echo -e "${RED}请使用 root 用户或 sudo 执行此脚本，以避免 ethtool 权限阻塞！${NC}"
  exit 1
fi

# -------- 核心功能函数 --------

get_phy_eths() {
    for eth in $(ls /sys/class/net/); do
        if [[ -d "/sys/class/net/$eth/device" ]]; then echo "$eth"; fi
    done | sort -V
}

get_eth_speed() {
    local eth=$1
    local spd=$(ethtool "$eth" 2>/dev/null | awk '/Speed/ {gsub(/[^0-9]/,""); print $0}')
    echo "${spd:----}"
}

get_eth_driver() {
    local eth=$1
    ethtool -i "$eth" 2>/dev/null | awk '/^driver:/ {print $2}'
}

format_flow() {
    local bps=$1
    awk -v val="$bps" 'BEGIN {
        if (val >= 1024*1024) printf "%.2f MB/s", val/1024/1024;
        else if (val >= 1024) printf "%.2f KB/s", val/1024;
        else if (val >= 0) printf "%.1f B/s", val;
        else printf "0 B/s";
    }'
}

format_speed() {
    local spd=$1
    if [[ "$spd" == "---" ]]; then echo "---";
    elif (( spd >= 1000 )); then echo "$((spd/1000))Gb/s";
    else echo "${spd}Mb/s"; fi
}

format_total_flow() {
    local bytes=$1
    awk -v b="$bytes" 'BEGIN {
        if (b < 0) b = 0;
        if (b >= 1024*1024*1024) printf "%.2f GB", b/1024/1024/1024;
        else if (b >= 1024*1024) printf "%.2f MB", b/1024/1024;
        else if (b >= 1024) printf "%.2f KB", b/1024;
        else printf "%.0f B", b;
    }'
}

calc_util() {
    local rx_bps=$1 tx_bps=$2 spd=$3
    if [[ "$spd" == "---" || "$spd" == "0" ]]; then echo -e "${YELLOW}N/A${NC}"; return; fi
    # 恢复与老脚本一致的 1024 进制算法： bps * 8 / 1024 / 1024 / spd_Mbps * 100
    awk -v rx="$rx_bps" -v tx="$tx_bps" -v spd="$spd" 'BEGIN {
        current = (rx > tx) ? rx : tx;
        util = current * 8 / 1024 / 1024 / spd * 100;
        if (util >= 80) color = "\033[1;31m"; else if (util >= 30) color = "\033[1;33m"; else color = "\033[1;32m";
        printf "%s%3.0f%%\033[0m", color, util;
    }'
}

read_counters() {
    local eth=$1 drv=${DRV_MAP[$eth]}
    local rx="" tx="" stats
    # 一次性获取 ethtool 输出，减少进程调用开销
    stats=$(ethtool -S "$eth" 2>/dev/null)
    
    if [[ "$drv" == "mlx5_core"* ]]; then
        # 修复：使用 grep -m1 兼容前导空格，且只取第一行防止重复字段
        rx=$(echo "$stats" | grep -m1 "rx_bytes_phy" | awk '{print $2}')
        tx=$(echo "$stats" | grep -m1 "tx_bytes_phy" | awk '{print $2}')
    else
        rx=$(echo "$stats" | grep -m1 "mac_rx_total_oct_num" | awk '{print $2}')
        tx=$(echo "$stats" | grep -m1 "mac_tx_total_oct_num" | awk '{print $2}')
    fi
    
    # 如果未获取到特定计数器，降级使用 /proc/net/dev
    if [[ -z "$rx" || -z "$tx" ]]; then
        local proc_data=$(grep -E "^\s*${eth}:" /proc/net/dev)
        rx=$(echo "$proc_data" | awk -F: '{print $2}' | awk '{print $1}')
        tx=$(echo "$proc_data" | awk -F: '{print $2}' | awk '{print $9}')
    fi
    echo "$rx $tx"
}

# -------- 初始化系统信息 --------
init_system() {
    ETHS=($(get_phy_eths))
    if [ ${#ETHS[@]} -eq 0 ]; then echo -e "${RED}未检测到物理网卡，退出。${NC}"; exit 1; fi
    declare -gA DRV_MAP SPD_MAP RX0_MAP TX0_MAP T0_MAP
    declare -gA INIT_RX_MAP INIT_TX_MAP STAT_RX_MAP STAT_TX_MAP
    for eth in "${ETHS[@]}"; do
        DRV_MAP[$eth]=$(get_eth_driver "$eth")
        SPD_MAP[$eth]=$(get_eth_speed "$eth")
    done
}

# -------- 前台动态展示模式 --------
run_foreground() {
    local interval=$1
    trap 'echo -en "\033[?25h"; echo -e "\n${YELLOW}监控已停止。${NC}"; exit 0' SIGINT SIGTERM
    for eth in "${ETHS[@]}"; do
        local counters=$(read_counters "$eth")
        RX0_MAP[$eth]=$(echo $counters | awk '{print $1}'); TX0_MAP[$eth]=$(echo $counters | awk '{print $2}')
        T0_MAP[$eth]=$(date +%s.%N)
    done
    clear; echo -en "\033[?25l"
    while true; do
        sleep "$interval"; echo -en "\033[H"
        echo -e "${CYAN}=================== 物理网卡实时性能监控 ===================${NC}"
        printf "${BLUE}%-14s %-12s %-10s %-15s %-15s %-10s %-6s${NC}\n" "时间" "网卡" "驱动" "接收速率" "发送速率" "端口速率" "利用率"
        echo "-------------------------------------------------------------------------"
        local ts=$(date +%H:%M:%S)
        for eth in "${ETHS[@]}"; do
            local counters=$(read_counters "$eth")
            local rx1=$(echo $counters | awk '{print $1}'); local tx1=$(echo $counters | awk '{print $2}'); local t1=$(date +%s.%N)
            local dt=$(awk -v t1="$t1" -v t0="${T0_MAP[$eth]}" 'BEGIN { printf "%.6f", t1 - t0 }')
            dt=$(awk -v dt="$dt" -v interval="$interval" 'BEGIN { if (dt <= 0) print interval; else print dt }')
            local rx_bps=$(awk -v r1="$rx1" -v r0="${RX0_MAP[$eth]}" -v dt="$dt" 'BEGIN { printf "%.6f", (r1 - r0) / dt }')
            local tx_bps=$(awk -v t1_v="$tx1" -v t0="${TX0_MAP[$eth]}" -v dt="$dt" 'BEGIN { printf "%.6f", (t1_v - t0) / dt }')
            printf "%-14s %-12s %-10s %-15s %-15s %-10s " "$ts" "$eth" "${DRV_MAP[$eth]}" "$(format_flow "$rx_bps")" "$(format_flow "$tx_bps")" "$(format_speed "${SPD_MAP[$eth]}")"
            echo -e "$(calc_util "$rx_bps" "$tx_bps" "${SPD_MAP[$eth]}")"
            RX0_MAP[$eth]=$rx1; TX0_MAP[$eth]=$tx1; T0_MAP[$eth]=$t1
        done
        echo "============================================================"; echo -e "按 Ctrl+C 退出 | 采样间隔: ${interval}s | 更新于: $ts"; echo -en "\033[J"
    done
}

# -------- 后台落盘与统计模式 --------
LOG_FILE=""; CSV_FILE=""; PID_FILE=""; STOP_REQUESTED=0

bg_cleanup() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 收到停止信号，准备生成最终统计..." 
    STOP_REQUESTED=1
}

run_background() {
    local duration_hrs=$1 interval=$2

    if [[ "$6" != "--daemon" ]]; then
        # --- 启动入口逻辑 ---
        for pidf in "${LOG_DIR}"/*.pid; do
            if [ -f "$pidf" ]; then
                old_pid=$(cat "$pidf")
                if kill -0 "$old_pid" 2>/dev/null; then
                    echo -e "${RED}错误: 已有后台监控正在运行 (PID: $old_pid)！${NC}" >&2; exit 1
                else rm -f "$pidf"; fi
            fi
        done

        local timestamp=$(date +%Y%m%d_%H%M%S)
        local base_name="${LOG_DIR}/Eth_Perf_Monitor_${timestamp}"
        LOG_FILE="${base_name}.log"; CSV_FILE="${base_name}.csv"; PID_FILE="${base_name}.pid"
        
        mkdir -p "$LOG_DIR" || { echo -e "${RED}无法创建目录 ${LOG_DIR}${NC}" >&2; exit 1; }
        touch "$LOG_FILE" "$CSV_FILE" || { echo -e "${RED}无写入权限 ${LOG_DIR}${NC}" >&2; exit 1; }
        
        echo -e "${GREEN}监控文件准备完毕：${NC}" >&2
        echo -e "  PID文件: ${CYAN}${PID_FILE}${NC}" >&2
        echo -e "  日志文件: ${CYAN}${LOG_FILE}${NC}" >&2
        echo -e "  数据文件: ${CYAN}${CSV_FILE}${NC}" >&2
        echo -e "${YELLOW}正在转入后台执行...${NC}" >&2
        
        local script_path=$(readlink -f "$0")
        nohup "$script_path" --stat-cycle-seconds "$STAT_CYCLE_SECONDS" bg "$duration_hrs" "$interval" "$LOG_FILE" "$CSV_FILE" "$PID_FILE" --daemon >> "$LOG_FILE" 2>&1 &
        local child_pid=$!; echo $child_pid > "$PID_FILE"
        
        sleep 0.5
        if kill -0 "$child_pid" 2>/dev/null; then exit 0
        else echo -e "${RED}后台进程启动失败！请查看 ${LOG_FILE}${NC}" >&2; exit 1; fi
    fi

    # --- 守护进程执行逻辑 ---
    LOG_FILE=$3; CSV_FILE=$4; PID_FILE=$5
    local duration_sec=$((duration_hrs * 3600))
    local end_time=$(( $(date +%s) + duration_sec ))
    local last_stat_epoch=$(date +%s)
    
    for eth in "${ETHS[@]}"; do
        local counters=$(read_counters "$eth")
        INIT_RX_MAP[$eth]=$(echo $counters | awk '{print $1}'); INIT_TX_MAP[$eth]=$(echo $counters | awk '{print $2}')
        RX0_MAP[$eth]=${INIT_RX_MAP[$eth]}; TX0_MAP[$eth]=${INIT_TX_MAP[$eth]}
        STAT_RX_MAP[$eth]=${INIT_RX_MAP[$eth]}; STAT_TX_MAP[$eth]=${INIT_TX_MAP[$eth]}
        T0_MAP[$eth]=$(date +%s.%N)
    done

    # 写入 CSV 表头
    echo "时间,网卡,驱动,接收速率,发送速率,端口速率,利用率" > "$CSV_FILE"
    
    # 写入 Log 启动信息
    echo "======================= 监控启动 ======================="
    echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "预计结束: $(date -d @${end_time} '+%Y-%m-%d %H:%M:%S')"
    echo "采样间隔: ${interval}秒 | 阶段统计周期: $((STAT_CYCLE_SECONDS/60))分钟"
    echo "========================================================"

    trap bg_cleanup SIGINT SIGTERM

    while [ $(date +%s) -lt $end_time ] && [ "$STOP_REQUESTED" -eq 0 ]; do
        sleep "$interval"
        local current_epoch=$(date +%s)
        local current_ts=$(date "+%Y-%m-%d %H:%M:%S")
        local data_block=""

        for eth in "${ETHS[@]}"; do
            local counters=$(read_counters "$eth")
            local rx1=$(echo $counters | awk '{print $1}'); local tx1=$(echo $counters | awk '{print $2}'); local t1=$(date +%s.%N)
            local dt=$(awk -v t1="$t1" -v t0="${T0_MAP[$eth]}" 'BEGIN { printf "%.6f", t1 - t0 }')
            dt=$(awk -v dt="$dt" -v interval="$interval" 'BEGIN { if (dt <= 0) print interval; else print dt }')
            local rx_bps=$(awk -v r1="$rx1" -v r0="${RX0_MAP[$eth]}" -v dt="$dt" 'BEGIN { printf "%.6f", (r1 - r0) / dt }')
            local tx_bps=$(awk -v t1_v="$tx1" -v t0="${TX0_MAP[$eth]}" -v dt="$dt" 'BEGIN { printf "%.6f", (t1_v - t0) / dt }')
            
            local util_str
            if [[ "${SPD_MAP[$eth]}" == "---" || "${SPD_MAP[$eth]}" == "0" ]]; then util_str="N/A"
            else util_str=$(awk -v rx="$rx_bps" -v tx="$tx_bps" -v spd="${SPD_MAP[$eth]}" 'BEGIN { current=(rx>tx)?rx:tx; printf "%.0f%%", current*8/1024/1024/spd*100; }')
            fi

            data_block+="${current_ts},${eth},${DRV_MAP[$eth]},$(format_flow "$rx_bps"),$(format_flow "$tx_bps"),$(format_speed "${SPD_MAP[$eth]}"),${util_str}\n"
            RX0_MAP[$eth]=$rx1; TX0_MAP[$eth]=$tx1; T0_MAP[$eth]=$t1
        done
        
        printf "%b" "$data_block" >> "$CSV_FILE"

        # 检查是否需要阶段统计写入 Log
        if (( current_epoch - last_stat_epoch >= STAT_CYCLE_SECONDS )); then
            echo "" >> "$LOG_FILE"
            echo "[$current_ts] --- 阶段统计 (过去 $((STAT_CYCLE_SECONDS/60)) 分钟) ---" >> "$LOG_FILE"
            for eth in "${ETHS[@]}"; do
                local stat_rx_bytes=$(awk -v f="${RX0_MAP[$eth]}" -v i="${STAT_RX_MAP[$eth]}" 'BEGIN { printf "%.0f", f - i }')
                local stat_tx_bytes=$(awk -v f="${TX0_MAP[$eth]}" -v i="${STAT_TX_MAP[$eth]}" 'BEGIN { printf "%.0f", f - i }')
                echo "[$current_ts] $eth: 接收 $(format_total_flow "$stat_rx_bytes"), 发送 $(format_total_flow "$stat_tx_bytes")" >> "$LOG_FILE"
                STAT_RX_MAP[$eth]=${RX0_MAP[$eth]}; STAT_TX_MAP[$eth]=${TX0_MAP[$eth]}
            done
            last_stat_epoch=$current_epoch
        fi
    done

    # ---- 生成最终统计 ----
    echo "" >> "$LOG_FILE"
    echo "======================= 最终统计汇总 =======================" >> "$LOG_FILE"
    echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')" >> "$LOG_FILE"
    for eth in "${ETHS[@]}"; do
        local total_rx_bytes=$(awk -v f="${RX0_MAP[$eth]}" -v i="${INIT_RX_MAP[$eth]}" 'BEGIN { printf "%.0f", f - i }')
        local total_tx_bytes=$(awk -v f="${TX0_MAP[$eth]}" -v i="${INIT_TX_MAP[$eth]}" 'BEGIN { printf "%.0f", f - i }')
        echo "$eth ($(format_speed ${SPD_MAP[$eth]})): 总接收 $(format_total_flow "$total_rx_bytes"), 总发送 $(format_total_flow "$total_tx_bytes")" >> "$LOG_FILE"
    done
    echo "============================================================" >> "$LOG_FILE"
    
    rm -f "$PID_FILE"
}

# -------- 主入口 --------
POSITIONAL_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --log-dir)
            if [[ -z "$2" ]]; then echo "错误: --log-dir 需要参数"; exit 1; fi
            LOG_DIR="$2"; shift 2 ;;
        --stat-cycle-seconds)
            if [[ -z "$2" ]]; then echo "错误: --stat-cycle-seconds 需要参数"; exit 1; fi
            STAT_CYCLE_SECONDS="$2"; shift 2 ;;
        --help|-h)
            POSITIONAL_ARGS+=("$1"); shift ;;
        *)
            POSITIONAL_ARGS+=("$1"); shift ;;
    esac
done
set -- "${POSITIONAL_ARGS[@]}"

case "$1" in
    fg) init_system; run_foreground ${2:-$DEFAULT_FOREGROUND_INTERVAL} ;;
    bg) init_system; shift; run_background "${1:-$DEFAULT_DURATION_HOURS}" "${2:-$DEFAULT_BACKGROUND_INTERVAL}" "${@:3}" ;;
    *)  echo "用法: $0 { fg [间隔秒] | bg [时长小时] [间隔秒] }"
        echo "      [--log-dir 路径] [--stat-cycle-seconds 秒数]"
        echo ""
        echo "  fg [间隔]       前台动态美观展示 (默认2秒刷新, Ctrl+C退出)"
        echo "  bg [时长] [间隔] 后台守护进程监控 (默认12小时, 10秒采样)"
        echo "  --log-dir       后台日志输出目录 (默认: 当前目录/net_log)"
        echo "  --stat-cycle-seconds 后台阶段统计周期 (默认: 3600秒)"
        echo ""
        echo "示例:"
        echo "  $0 fg           # 前台实时监控，2秒刷新"
        echo "  $0 fg 5         # 前台实时监控，5秒刷新"
        echo "  $0 bg           # 后台监控12小时，数据存入 $LOG_DIR"
        echo "  $0 bg 24 5      # 后台监控24小时，5秒采样一次"
        echo "  $0 bg 24 5 --log-dir /mnt/test/net_log --stat-cycle-seconds 600"
        exit 1 ;;
esac
