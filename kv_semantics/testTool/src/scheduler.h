/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */

#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include "request/request.h"
#include "request/request_manager.h"
#include "request/request_orc.h"
#include "utils/config_util.h"
#include "worker/device_allocator.h"
#include "worker/executor.h"
#include "worker/kv_client_util.h"

/**
 * 任务流的一个执行段：两次 sleep 步骤之间的任务区间，段末全员停顿（脉冲）后进下一段
 */
struct FlowSegment {
    /** 段起始步骤下标（含） */
    std::size_t start{0};
    /** 段结束步骤下标（不含；指向段边界 sleep 步骤或任务流末尾） */
    std::size_t end{0};
    /** 段末停顿秒数（段边界 sleep 步骤的秒数；末段无边界 sleep 时为 0） */
    std::uint32_t pauseSec{0};
};

/**
 * 调度器：统一管理 kv client 与全部执行器的生命周期——
 * 初始化（kv client 建链 + 值缓冲池一次性分配注册，尺寸与内存类型来自 conf，与 input 无关；
 * 参考 kv_test 用法：region 注册仅发生在建链阶段，执行期只复用）、
 * 多 worker 并发执行任务流（numjobs 个线程共享任务流按步骤下标原子领取，每任务仅执行一次；
 * sleep 步骤为段边界，全员到齐后统一停顿——脉冲语义）、
 * 秒级周期汇总各执行器埋点指标并输出、停止时注销请求内存并关闭 client。
 */
class Scheduler {
public:
    Scheduler();
    /** 兜底析构：未显式 Shutdown 时停止指标线程并释放全部资源 */
    ~Scheduler();

    /**
     * 初始化：kv client 建链、值缓冲池分配（buffercount 块 × buffersize 字节，内存类型按
     * buffertype：cpu 主机内存 / gpu CUDA 显存 / npu ACL NPU 内存）并完成内存注册、创建执行器
     * @param config 测试配置（取 buffersize / buffercount / buffertype / deviceid）
     */
    bool initializeAndRegister(const std::string &confPath, const ConfigUtil &config,
                               std::string &err);

    /**
     * 执行 input 任务流：numjobs 个 worker 线程并发执行——任务流按 sleep 步骤切段，段内任务经
     * 原子游标领取（每任务仅执行一次），段末全员到齐后统一停顿（脉冲）再放下一段；任务组装时
     * 从共享缓冲池领用独占块；重复执行语义，false 仅当工具链失败（err 说明首次失败）
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
     * 组装单个执行任务的数据包：任务第 i 条 entry 绑定领用块 owned[i]（条数超池的检查在领用前
     * 由调用方完成）、值块生成经分配器写入缓冲（device 语义为 H2D）、KVBuffer 组装（region.size
     * 为本次 IO 长度）、基准表更新与查询（加锁，多 worker 并发安全）
     */
    bool buildTaskRequest(const OrcTask &task, int testmode, const std::vector<std::size_t> &owned,
                          Request &request, std::string &err);

    /**
     * worker 线程主循环：绑核并命名后进入领任务循环——锁内查退出标志与当前段末，段内经原子
     * 游标领取任务（每任务仅执行一次）并发执行，本段领完进 barrier 全员同步；持续执行语义，
     * 单任务工具链失败记录后继续
     */
    void WorkerLoop(int workerId, int testmode);

    /**
     * 段末全员 barrier：全员到齐（本段任务全部完成）后由最后到者统一停顿（脉冲，秒数取段边界
     * sleep 步骤）并推进段、判定圈末退出（限时跑满不开新圈、圈数跑满退出），再开闸放下一段
     */
    void EnterBarrier();

    /** 领取 count 块独占值缓冲（owned 输出块下标；空闲不足时阻塞等待其他任务归还，调用方已保证
     * count 不超池总量） */
    void AcquireBuffers(std::size_t count, std::vector<std::size_t> &owned);

    /** 归还值缓冲块到空闲池并唤醒等待领用的 worker */
    void ReleaseBuffers(const std::vector<std::size_t> &owned);

    /** 绑定当前线程到指定 CPU 核（Linux 平台直调；失败仅告警不绑，不影响执行） */
    void BindThreadToCore(int core);

    /** 记录工具链失败并打错误日志（保留首次失败描述，线程安全） */
    void RecordFail(const std::string &message);

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
    /** put 基准表：keyLow64 → 最近一次写入块内容（get 回读比对基准；多 worker 并发访问加锁） */
    std::unordered_map<std::uint64_t, std::vector<std::uint8_t>> keyBaseline_;

    // ---- 多 worker 并发执行状态（Run 期间有效，worker 全部 join 后失效） ----
    /** worker 数与起始绑核编号（conf：numjobs / startcoreid；worker i 绑 startcoreid+i） */
    int numJobs_{1};
    int startCoreId_{0};
    /** 执行期任务流与段表（Run 启动时构建，执行期只读） */
    const RequestManager *flow_{nullptr};
    std::vector<FlowSegment> segments_;
    /** 段内领任务游标（原子；段推进时在 barrierMutex_ 内重置为新段起点） */
    std::atomic<std::size_t> cursor_{0};
    /** 段末全员 barrier：segIndex_ / barrierArrived_ / barrierGen_ / exiting_ / rounds_ 由本锁保护
     */
    std::mutex barrierMutex_;
    std::condition_variable barrierCv_;
    std::size_t segIndex_{0};
    int barrierArrived_{0};
    int barrierGen_{0};
    bool exiting_{false};
    /** 已完成圈数（圈末 barrier 时累加，Run 结束时输出） */
    int rounds_{0};
    /** 限时/圈数模式参数与起始时刻（Run 启动时写入，圈末 barrier 判定读取） */
    int runtimeSecs_{0};
    int loopLimit_{0};
    std::chrono::steady_clock::time_point startTime_{};
    /** 共享值缓冲池空闲块下标（在飞任务独占领用，不足阻塞等待；bufferMutex_ 保护） */
    std::vector<std::size_t> freeBuffers_;
    std::mutex bufferMutex_;
    std::condition_variable bufferCv_;
    /** put 基准表并发保护（PUT 写 / GET 读两触点加锁） */
    std::mutex baselineMutex_;
    /** 工具链失败聚合（anyFail_ / firstErr_ 由 failMutex_ 保护；保留首次失败描述） */
    std::mutex failMutex_;
    bool anyFail_{false};
    std::string firstErr_;
    /** 指标汇总线程 */
    std::thread metricsThread_;
    /** 指标线程运行标志（false 时线程退出） */
    std::atomic<bool> metricsRunning_{false};
};
