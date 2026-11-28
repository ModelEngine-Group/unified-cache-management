#!/usr/bin/env bash
# ============================================================================
# stop-instances.sh — 批量停止 testTool 实例
#
# 用法（在 testTool 目录下执行；从任意目录发起亦可，路径以脚本自身位置推算）：
#   bash scripts/stop-instances.sh [配置前缀]
#     配置前缀：缺省停止全部实例（扫描 logs/*.pid）；指定前缀则仅停止
#     logs/<前缀>_N.pid 对应的实例（与 start-instances.sh 的前缀约定一致）
#
# 停止方式：SIGTERM 优雅退出（异步日志刷盘后退出），最多等待 3 秒，
# 仍存活则 SIGKILL 兜底；pid 对应进程已不存在时仅清理 pid 文件。
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

# 进程存活判定：kill -0 对僵尸进程同样返回 0（父进程已退出、容器 PID 1 非
# init 不回收时进程停留僵尸态），僵尸视为已退出
is_alive() {
    kill -0 "$1" 2>/dev/null || return 1
    [ "$(ps -p "$1" -o stat= 2>/dev/null | cut -c1)" != "Z" ]
}

if [ ! -d logs ]; then
    echo "无运行实例（logs/ 目录不存在）"
    exit 0
fi

if [ $# -ge 1 ]; then
    pidFiles=$(ls "logs/${1}_"*.pid 2>/dev/null || true)
else
    pidFiles=$(ls logs/*.pid 2>/dev/null || true)
fi
if [ -z "$pidFiles" ]; then
    echo "无运行实例"
    exit 0
fi

for pidFile in $pidFiles; do
    pid=$(cat "$pidFile")
    if is_alive "$pid"; then
        kill "$pid"
        # 优雅退出等待，最多 3 秒
        for _ in $(seq 1 30); do
            is_alive "$pid" || break
            sleep 0.1
        done
        if is_alive "$pid"; then
            kill -9 "$pid" 2>/dev/null || true
            echo "$(basename "$pidFile"): pid $pid 未优雅退出，已 SIGKILL"
        else
            echo "$(basename "$pidFile"): pid $pid 已停止"
        fi
    else
        echo "$(basename "$pidFile"): 进程不存在，清理 pid 文件"
    fi
    rm -f "$pidFile"
done
