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
 * SOFTWARE.
 * */
#include "asu_transport_impl.h"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include "asu_transport.h"
#include "trans_provider.h"
#include "connection_manager.h"
#include "logger.h"
#include "transport_config_parser.h"

namespace UC::ASU {

namespace {

std::chrono::steady_clock::time_point TaskDeadline(std::uint64_t timeoutMs)
{
    if (timeoutMs == 0) { return std::chrono::steady_clock::time_point::max(); }
    return std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);
}

}  // namespace

AsuTransportImpl::AsuTransportImpl() = default;

AsuTransportImpl::~AsuTransportImpl() { Shutdown(); }

Status AsuTransportImpl::Init(const std::string& configPath,
                              std::shared_ptr<TransProvider> transProvider)
{
    TransportConfig config;
    auto status = LoadTransportConfig(configPath, config);
    if (!status.ok()) { return status; }
    return Init(config, std::move(transProvider));
}

Status AsuTransportImpl::Init(const TransportConfig& config,
                              std::shared_ptr<TransProvider> transProvider)
{
    UC_DEBUG("AsuTransportImpl::Init start");
    if (worker_.joinable()) {
        UC_DEBUG("AsuTransportImpl::Init already initialized");
        return Status::OK();
    }
    config_ = config;
    transProvider_ = std::move(transProvider);
    if (config_.maxErrorCount == 0) {
        return Status::Error(StatusCode::INVALID_ARGUMENT, "maxErrorCount must be greater than 0");
    }
    if (config_.completionPollSpinLimit == 0) {
        return Status::Error(StatusCode::INVALID_ARGUMENT,
                             "completionPollSpinLimit must be greater than 0");
    }

    if (!transProvider_) {
        UC_ERROR("AsuTransportImpl::Init: TransProvider is null");
        return Status::Error(StatusCode::NOT_INITIALIZED, "TransProvider is null");
    }

    std::string localIp;
    auto it = config_.attrs.find("localIp");
    if (it != config_.attrs.end()) { localIp = it->second; }

    std::uint32_t timeout = 5000;
    auto tit = config_.attrs.find("timeout");
    if (tit != config_.attrs.end()) {
        timeout = static_cast<std::uint32_t>(std::stoul(tit->second));
    }

    connManager_.reset();
    connManager_ = std::make_unique<ConnectionManager>(*transProvider_, localIp, timeout,
                                                       config_.maxErrorCount);

    const std::uint32_t qp_num = config_.qpNum;
    UC_DEBUG("AsuTransportImpl::Init endpoints={} qp_num={}", config_.endpoints.size(), qp_num);
    for (const auto& ep : config_.endpoints) {
        auto s = connManager_->AddGroup(ep, qp_num);
        if (!s.ok()) {
            UC_DEBUG("AsuTransportImpl::Init AddGroup FAILED: {}", s.message);
            const auto shutdownStatus = Shutdown();
            if (!shutdownStatus.ok()) {
                UC_WARN(
                    "AsuTransportImpl::Init cleanup failed after AddGroup failure: code={} "
                    "message={}",
                    static_cast<int>(shutdownStatus.code), shutdownStatus.message);
            }
            return s;
        }
    }

    const auto serverCapabilities = connManager_->GetServerCapabilities();
    for (std::size_t index = 0; index < serverCapabilities.size(); ++index) {
        const auto& capabilities = serverCapabilities[index];
        UC_INFO(
            "AsuTransportImpl::Init ServerKvCapabilities group={} queueNum={} ioQueueDepth={} "
            "ioQueueKeyConcurrency={} connectionKeyConcurrency={} singleValueMaxBytes={} "
            "batchValueMaxBytes={} batchStoreKeys={} batchLoadKeys={} deleteKeys={} "
            "queryKeys={} keyLength={} kvCapabilities={}",
            index, capabilities.queueNum, capabilities.ioQueueDepth,
            capabilities.ioQueueKeyConcurrency, capabilities.connectionKeyConcurrency,
            capabilities.singleValueMaxBytes, capabilities.batchValueMaxBytes,
            capabilities.batchStoreKeys, capabilities.batchLoadKeys, capabilities.deleteKeys,
            capabilities.queryKeys, capabilities.keyLength, capabilities.kvCapabilities);
        auto applyCapLimit = [](auto& cfgField, auto capValue) {
            if (capValue != 0) {
                cfgField = std::min(cfgField, static_cast<std::size_t>(capValue));
            }
        };
        applyCapLimit(config_.asuBatchStoreIoNum, capabilities.batchStoreKeys);
        applyCapLimit(config_.asuBatchLoadIoNum, capabilities.batchLoadKeys);
        applyCapLimit(config_.asuDeleteIoNum, capabilities.deleteKeys);
        applyCapLimit(config_.asuQueryIoNum, capabilities.queryKeys);
    }
    UC_INFO("AsuTransportImpl::Init effective batch limits: store={} load={} delete={} query={}",
            config_.asuBatchStoreIoNum, config_.asuBatchLoadIoNum, config_.asuDeleteIoNum,
            config_.asuQueryIoNum);

    connManager_->StartRecoverLoop();

    taskExecutor_ = std::make_unique<TransportTaskExecutor>(config_, transProvider_, connManager_);
    auto status = taskExecutor_->Init();
    if (!status.ok()) {
        const auto shutdownStatus = Shutdown();
        if (!shutdownStatus.ok()) {
            UC_WARN(
                "AsuTransportImpl::Init cleanup failed after task executor initialization "
                "failure: code={} message={}",
                static_cast<int>(shutdownStatus.code), shutdownStatus.message);
        }
        return status;
    }
    auto queueDepth = std::max<std::size_t>(2, static_cast<std::size_t>(config_.maxInflightTasks));
    executeQueue_.Setup(queueDepth + 1);
    stopWorker_.store(false, std::memory_order_release);
    stopCompletionWorker_.store(false, std::memory_order_release);
    worker_ = std::thread(&AsuTransportImpl::WorkerLoop, this);
    completionWorker_ = std::thread(&AsuTransportImpl::CompletionLoop, this);
    UC_DEBUG("AsuTransportImpl::Init OK: queueDepth={}", queueDepth);
    return Status::OK();
}

Status AsuTransportImpl::Shutdown()
{
    Status finalStatus = Status::OK();
    {
        std::lock_guard<std::mutex> lock(producerMu_);
        stopWorker_.store(true, std::memory_order_release);
    }

    if (worker_.joinable()) { worker_.join(); }

    if (completionWorker_.joinable() && config_.timeoutMs != 0) {
        bool hasInflightTask = false;
        for (const auto& ctx : taskManager_.GetAll()) {
            if (ctx != nullptr &&
                ctx->state.load(std::memory_order_acquire) == TransportTaskState::INFLIGHT) {
                hasInflightTask = true;
                break;
            }
        }
        if (hasInflightTask) {
            std::this_thread::sleep_for(std::chrono::milliseconds(config_.timeoutMs));
        }
    }

    stopCompletionWorker_.store(true, std::memory_order_release);
    if (completionWorker_.joinable()) { completionWorker_.join(); }

    for (const auto& ctx : taskManager_.GetAll()) {
        if (ctx == nullptr) { continue; }
        if (!taskExecutor_->Cancel(ctx)) { continue; }
        taskManager_.NotifyCompletion(ctx);
    }
    for (const auto& ctx : taskManager_.GetAll()) {
        if (ctx != nullptr) { (void)taskManager_.Remove(ctx->taskId); }
    }
    if (taskExecutor_) {
        const auto status = taskExecutor_->Shutdown();
        if (!status.ok()) {
            if (finalStatus.ok()) { finalStatus = status; }
        } else {
            taskExecutor_.reset();
        }
    }

    if (connManager_) {
        auto status = connManager_->Shutdown();
        if (!status.ok() && finalStatus.ok()) { finalStatus = status; }
        connManager_.reset();
    }

    UC_DEBUG("AsuTransportImpl::Shutdown OK");
    return finalStatus;
}

Status AsuTransportImpl::CheckHealth()
{
    if (!worker_.joinable() || !completionWorker_.joinable()) {
        return Status::Error(StatusCode::NOT_INITIALIZED, "transport worker is not running");
    }
    return Status::OK();
}

Status AsuTransportImpl::Submit(const TransportTaskPtr& task)
{
    if (!task) { return Status::Error(StatusCode::INVALID_ARGUMENT, "transport task is null"); }
    const auto entryCount = IsEntryBatchOp(task->opType) ? task->entries.size() : task->keys.size();
    task->entryStatus.assign(entryCount, Status::OK());
    task->deadline = TaskDeadline(config_.timeoutMs);
    return SubmitTask(task);
}

Status AsuTransportImpl::Cancel(TaskId taskId)
{
    auto ctx = taskManager_.Get(taskId);
    if (!ctx) { return Status::Error(StatusCode::TASK_NOT_FOUND, "transport task not found"); }

    if (!taskExecutor_->Cancel(ctx)) { return Status::OK(); }
    taskManager_.NotifyCompletion(ctx);
    return Status::OK();
}

Status AsuTransportImpl::SubmitTask(const TransportTaskPtr& task)
{
    std::lock_guard<std::mutex> lock(producerMu_);
    if (stopWorker_.load(std::memory_order_acquire) || !worker_.joinable()) {
        task->taskId = kInvalidTaskId;
        return Status::Error(StatusCode::NOT_INITIALIZED, "transport is not accepting tasks");
    }

    TaskId taskId = kInvalidTaskId;
    auto status = taskManager_.Submit(task, taskId);
    if (!status.ok()) { return status; }

    auto submittedTask = taskManager_.Get(taskId);
    if (!submittedTask) {
        task->taskId = kInvalidTaskId;
        return Status::Error(StatusCode::INTERNAL_ERROR, "transport task disappeared after submit");
    }

    if (!executeQueue_.TryPush(std::move(submittedTask))) {
        taskManager_.Remove(taskId);
        task->taskId = kInvalidTaskId;
        return Status::Error(StatusCode::RESOURCE_BUSY, "transport task queue is full");
    }
    return Status::OK();
}

void AsuTransportImpl::WorkerLoop()
{
    executeQueue_.ConsumerLoop(stopWorker_, [this](TransportTaskPtr task) {
        if (!task) { return; }
        if (taskExecutor_->Execute(task)) { taskManager_.NotifyCompletion(task); }
    });
}

void AsuTransportImpl::CompletionLoop()
{
    std::size_t noCompletionRounds = 0;

    while (!stopCompletionWorker_.load(std::memory_order_acquire)) {
        const auto tasks = taskManager_.GetAll();
        if (tasks.empty()) {
            noCompletionRounds = 0;
            std::this_thread::sleep_for(std::chrono::microseconds(50));
            continue;
        }

        bool completedTask = false;
        for (const auto& ctx : tasks) {
            if (taskExecutor_->Poll(ctx)) {
                taskManager_.NotifyCompletion(ctx);
                completedTask = true;
            }
        }

        if (completedTask) {
            noCompletionRounds = 0;
        } else if (++noCompletionRounds >= config_.completionPollSpinLimit) {
            noCompletionRounds = 0;
            std::this_thread::yield();
        }
    }
}

void AsuTransportImpl::SetTransProvider(std::shared_ptr<TransProvider> provider)
{
    transProvider_ = std::move(provider);
}

std::unique_ptr<AsuTransport> CreateAsuTransport() { return std::make_unique<AsuTransportImpl>(); }

extern "C" std::unique_ptr<AsuTransport> UcmAsuCreateAsuTransport() { return CreateAsuTransport(); }

}  // namespace UC::ASU
