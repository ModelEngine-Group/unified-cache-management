#!/usr/bin/env bash
# ============================================================================
# start-instances.sh — 批量启动 testTool 实例
#
# 用法（在 testTool 目录下执行；从任意目录发起亦可，路径以脚本自身位置推算）：
#   bash scripts/start-instances.sh <实例个数> [配置前缀]
#     实例个数：必填正整数，启动 1..N 号实例
#     配置前缀：缺省 test_config；实例 N 的主配置为 conf/<前缀>_N.conf
#
# 每实例文件约定（N 为实例号）：
#   主配置    conf/<前缀>_N.conf     必须存在——启动前统一校验，任一缺失整体报错退出
#   kv client conf/kv_client_N.conf  存在则用实例专属，否则共用 conf/kv_client.conf
#   input     input/<前缀>_N.txt    存在则用实例专属，否则共用 input/input.txt
#
# 进程管理：nohup 后台启动；pid 写 logs/<前缀>_N.pid，控制台输出重定向到
# logs/<前缀>_N.out；实例已在运行（pid 存活）时跳过不重复启动。
# 注意：各实例主配置的 log.dir 必须指向不同目录，否则日志互相覆盖；
# 多实例并发建议在各配置中错开 startcoreid 绑核范围。
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

# 进程存活判定：kill -0 对僵尸进程同样返回 0（父进程已退出、容器 PID 1 非
# init 不回收时进程停留僵尸态），僵尸视为已退出
is_alive() {
    kill -0 "$1" 2>/dev/null || return 1
    [ "$(ps -p "$1" -o stat= 2>/dev/null | cut -c1)" != "Z" ]
}

if [ $# -lt 1 ] || ! [[ "$1" =~ ^[1-9][0-9]*$ ]]; then
    echo "用法: bash scripts/start-instances.sh <实例个数> [配置前缀]" >&2
    exit 1
fi
count=$1
prefix=${2:-test_config}

# 启动前统一校验主配置齐全（避免起一半才报缺）
missing=0
for i in $(seq 1 "$count"); do
    if [ ! -f "conf/${prefix}_${i}.conf" ]; then
        echo "错误: 缺少实例 $i 主配置 conf/${prefix}_${i}.conf" >&2
        missing=1
    fi
done
if [ "$missing" -ne 0 ]; then exit 1; fi

mkdir -p logs
for i in $(seq 1 "$count"); do
    mainConf="conf/${prefix}_${i}.conf"
    kvConf="conf/kv_client.conf"
    if [ -f "conf/kv_client_${i}.conf" ]; then kvConf="conf/kv_client_${i}.conf"; fi
    inputFile="input/input.txt"
    if [ -f "input/${prefix}_${i}.txt" ]; then inputFile="input/${prefix}_${i}.txt"; fi

    pidFile="logs/${prefix}_${i}.pid"
    if [ -f "$pidFile" ] && is_alive "$(cat "$pidFile")"; then
        echo "实例 $i 已在运行 (pid $(cat "$pidFile"))，跳过"
        continue
    fi

    nohup ./build/testTool "$mainConf" "$kvConf" "$inputFile" > "logs/${prefix}_${i}.out" 2>&1 &
    echo $! > "$pidFile"
    sleep 0.5
    if is_alive "$(cat "$pidFile")"; then
        echo "实例 $i 已启动: pid=$(cat "$pidFile") main=$mainConf kv=$kvConf input=$inputFile"
    else
        echo "实例 $i 启动失败（进程已退出，详见 logs/${prefix}_${i}.out）"
    fi
done
