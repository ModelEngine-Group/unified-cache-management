/**
 * MIT License
 *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE. */

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <thread>
#include <vector>

/** 指标分桶的操作接口数（与 worker/executor.h 的 OpType 对齐：exist / put / get / delete） */
constexpr std::size_t kMetricsOpCount = 4;

/**
 * 升序排序后取分位时延（nearest-rank 法：rank = ceil(quantile * n)，取第 rank 个样本）。
 * 样本容器会被原地排序（复用调用方缓冲，避免大样本二次拷贝）
 * @param samples 时延样本集合（微秒；调用后内容有序）
 * @param quantile 分位点（0 < quantile <= 1）
 * @return 分位时延（微秒）；空样本返回 0
 */
std::uint64_t LatencyQuantileUs(std::vector<std::uint32_t> &samples, double quantile);

/** 聚合指标快照（调度器启动以来的累计口径，Prometheus 文本渲染输入） */
struct MetricsSnapshot {
    /** 分接口操作数（entry 级全量口径） */
    std::uint64_t opsByOp[kMetricsOpCount] = {0, 0, 0, 0};
    /** 分接口流量字节（成功 entry 口径） */
    std::uint64_t bytesByOp[kMetricsOpCount] = {0, 0, 0, 0};
    /** 分接口任务时延累计（纳秒；任务提交到完成的往返） */
    std::uint64_t latencyNsByOp[kMetricsOpCount] = {0, 0, 0, 0};
    /** 分接口完成任务数（时延样本数） */
    std::uint64_t tasksByOp[kMetricsOpCount] = {0, 0, 0, 0};
    /** 分接口时延样本聚合（微秒；滚动最近样本，P50/P90/P99 由渲染侧计算） */
    std::vector<std::uint32_t> latencySamplesUs[kMetricsOpCount];
    /** 全局失败操作数（提交/等待失败与 entry 级错误，不含 NOT_FOUND 观察） */
    std::uint64_t errorsTotal = 0;
    /** 全局 NOT_FOUND 观察计数 */
    std::uint64_t notFoundTotal = 0;
    /** 运行时长（秒，调度器启动起） */
    double uptimeSecs = 0.0;
};

/**
 * 渲染 Prometheus 文本协议（exposition format）：分接口 summary（P50/P90/P99 分位 +
 * sum/count）与 counter/gauge 族；指标名前缀 testtool_
 * @param snapshot 聚合快照（只读；样本副本内部排序）
 * @return 文本正文（HTTP 响应体）
 */
std::string RenderPrometheusText(const MetricsSnapshot &snapshot);

/**
 * 极简 Prometheus 抓取端点（HTTP/1.1 GET /metrics，pull 模式）：单线程 accept 循环、
 * 每连接读一次请求即回响应并关闭；正文经 render 回调现取现渲染（须线程安全）。
 * 非 POSIX 平台（_WIN32）编译为空回退（Start 返回失败并说明不支持）
 */
class MetricsExporter {
public:
    /**
     * 启动抓取端点监听
     * @param render 抓取时的正文渲染回调（每次抓取调用一次；抛异常时按 500 响应）
     * @param port 监听端口（绑定 INADDR_ANY）
     * @param err 失败时的错误描述
     * @return false 端口占用 / 创建失败 / 平台不支持
     */
    bool Start(std::function<std::string()> render, int port, std::string &err);

    /** 停止监听并回收线程（幂等，可重复调用） */
    void Stop();

    MetricsExporter() = default;
    ~MetricsExporter();
    MetricsExporter(const MetricsExporter &) = delete;
    MetricsExporter &operator=(const MetricsExporter &) = delete;

private:
    /** accept 循环线程主体：逐连接接收请求（不解析路径，一律返回指标文本）→ 响应 → 关闭 */
    void AcceptLoop();

    std::function<std::string()> render_;
    int listenFd_{-1};
    std::thread acceptThread_;
    std::atomic<bool> running_{false};
};
