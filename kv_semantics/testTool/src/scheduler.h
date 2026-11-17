/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "request/request.h"
#include "request/request_orc.h"
#include "request/request_manager.h"
#include "utils/config_util.h"
#include "worker/device_allocator.h"
#include "worker/executor.h"
#include "worker/kv_client_util.h"

/**
 * 调度器：统一管理 kv client 与全部执行器的生命周期——
 * 初始化（kv client 建链 + 值缓冲池一次性分配注册，尺寸与内存类型来自 conf，与 input 无关；
 * 参考 kv_test 用法：region 注册仅发生在建链阶段，执行期只复用）、
 * 按行序逐任务驱动执行器执行流量模型（sleep 暂停）、
 * 秒级周期汇总各执行器埋点指标并输出、停止时注销请求内存并关闭 client。
 */
class Scheduler {
public:
    Scheduler();
    /** 兜底析构：未显式 Shutdown 时停止指标线程并释放全部资源 */
    ~Scheduler();

    /**
     * 初始化：kv client 建链、值缓冲池分配（buffercount 块 × buffersize 字节，内存类型按
     * buffertype：host 主机内存 / device 设备内存）并完成内存注册、创建执行器
     * @param config 测试配置（取 buffersize / buffercount / buffertype / deviceid）
     */
    bool initializeAndRegister(const std::string &confPath, const ConfigUtil &config,
                              std::string &err);

    /**
     * 按行序执行 input 任务流（sleep 暂停、逐任务组装执行；组装时任务第 i 条 entry 绑定第 i 块
     * 复用缓冲）；重复执行语义，false 仅当工具链失败（err 说明首次失败）
     * @param requestManager 已解析的任务流
     * @param testmode 值生成模式（1 系统生成，2 行尾 value 字段）
     * @param runtime 运行时长秒；>0 限时模式，跑满即停（时间到不开新圈，当前圈跑完）；
     *               =0 圈数模式（以 loop 为准）
     * @param loop 循环执行圈数（整个任务流重复 loop 遍），仅 runtime=0 时生效（Check 已保证 >0）
     */
    bool Run(const RequestManager &requestManager, int testmode, int runtime, int loop,
             std::string &err);

    /** 停止指标线程、注销请求内存、释放值缓冲池、关闭 kv client 并释放执行器；可重复调用 */
    void Shutdown();

private:
    /**
     * 组装单个执行任务的数据包：任务第 i 条 entry 绑定第 i 块复用缓冲（容量与条数不足时报错）、
     * 值块生成经分配器写入缓冲（device 语义为 H2D）、KVBuffer 组装（region.size 为本次 IO 长度）、
     * 基准表更新与查询
     */
    bool buildTaskRequest(const OrcTask &task, int testmode, Request &request, std::string &err);

    /** input 侧操作到执行侧操作的映射（GET→LOAD、PUT→STORE、EXISTS→QUERY、DELETE→DELETE） */
    static OpType MapOp(OrcOp op);

    /** 指标汇总线程主循环：每秒采样全部执行器累计指标，输出窗口速率 */
    void MetricsLoop();

    /** 采集全部执行器当前指标快照 */
    std::vector<ExecutorMetrics> SnapshotExecutors() const;

    /** 输出一个汇总窗口的指标（窗口增量速率：IOPS / 流量 / TTFT / 错误率 / 未命中 / 比对不一致） */
    void OutputMetricsWindow(const std::vector<ExecutorMetrics> &last,
                             const std::vector<ExecutorMetrics> &current) const;

    /** 输出运行结束时的指标总计（全部执行器累计值汇总） */
    void OutputMetricsTotal() const;

    /** kv client 工具封装（生命周期归本类） */
    std::unique_ptr<KvClientUtil> client_;
    /** 全部执行器（当前单个，多线程并发时按 numjobs 创建多个） */
    std::vector<std::unique_ptr<Executor>> executors_;
    /** 值缓冲内存分配器（host 主机内存 / device 设备内存；生命周期归本类） */
    std::unique_ptr<DeviceAllocator> allocator_;
    /** 值缓冲池各块地址（建链时一次性分配注册；任务组装时按 entry 序号复用，地址注册后不得再变） */
    std::vector<std::uint64_t> bufferAddrs_;
    /** 值缓冲单块容量字节 */
    std::size_t bufferSize_{0};
    /** 已注册内存区域凭证（与值缓冲块逐项对应，注销时使用） */
    std::vector<kv::RegisteredMemory> registered_;
    /** put 基准表：keyLow64 → 最近一次写入块内容（get 回读比对基准） */
    std::unordered_map<std::uint64_t, std::vector<std::uint8_t>> keyBaseline_;
    /** 指标汇总线程 */
    std::thread metricsThread_;
    /** 指标线程运行标志（false 时线程退出） */
    std::atomic<bool> metricsRunning_{false};
};
