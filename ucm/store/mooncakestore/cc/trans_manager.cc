/**
 * MIT License
 *
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
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
#include "trans_manager.h"
#include <acl/acl_rt.h>
#include <fmt/format.h>
#include <span>
#include <utility>
#include "logger/logger.h"
#include "replica.h"

namespace UC::MooncakeStore {

TransManager::TransManager() = default;

TransManager::~TransManager() { Close(); }

Status TransManager::Setup(const Config& config)
{
    config_ = config;
    backend_ = config.storeBackend;

    auto s = SetupRealClient(config);
    if (s.Failure()) { return s; }

    if (backend_) {
        s = SetupBackendPipeline(config);
        if (s.Failure()) { return s; }
    }

    UC_INFO("TransManager setup ok, backend={}, localBufSize={}", backend_ ? "yes" : "none",
            config.localBufferSize);
    return Status::OK();
}

Status TransManager::SetupRealClient(const Config& config)
{
    realClient_ = mooncake::RealClient::create();
    if (!realClient_) {
        UC_ERROR("RealClient::create failed");
        return Status::Error("RealClient::create failed");
    }

    int rc = realClient_->setup_real(
        config.localHostname, config.metadataServer, config.globalSegmentSize,
        config.localBufferSize, config.protocol, config.deviceName.empty() ? "" : config.deviceName,
        config.masterServerAddress);
    if (rc != 0) {
        UC_ERROR("RealClient::setup_real failed, rc={}", rc);
        realClient_.reset();
        return Status::Error("RealClient::setup_real failed");
    }

    return Status::OK();
}

Status TransManager::SetupBackendPipeline(const Config& config)
{
    stopFlag_ = false;

    dumpGetBufferPool_ = std::make_unique<ThreadPool<DumpCtx>>();
    dumpGetBufferPool_->SetNWorker(config.dumpGetBufferNum)
        .SetWorkerFn([this](DumpCtx& ctx, auto& unused) { OnDumpGetBuffer(ctx, unused); })
        .Run();

    dumpSubmitPool_ = std::make_unique<ThreadPool<DumpSubmitCtx>>();
    dumpSubmitPool_->SetNWorker(config.dumpSubmitNum)
        .SetWorkerFn([this](DumpSubmitCtx& ctx, auto& unused) { OnDumpSubmit(ctx, unused); })
        .Run();

    dumpWaitPool_ = std::make_unique<ThreadPool<DumpWaitCtx>>();
    dumpWaitPool_->SetNWorker(config.dumpWaitNum)
        .SetWorkerFn([this](DumpWaitCtx& ctx, auto& unused) { OnDumpWait(ctx, unused); })
        .Run();

    loadWaitPool_ = std::make_unique<ThreadPool<LoadWaitCtx>>();
    loadWaitPool_->SetNWorker(config.loadWaitNum)
        .SetWorkerFn([this](LoadWaitCtx& ctx, auto& unused) { OnLoadWait(ctx, unused); })
        .Run();

    loadPutPool_ = std::make_unique<ThreadPool<LoadPutCtx>>();
    loadPutPool_->SetNWorker(config.loadPutNum)
        .SetWorkerFn([this](LoadPutCtx& ctx, auto& unused) { OnLoadPut(ctx, unused); })
        .Run();

    loadGetPool_ = std::make_unique<ThreadPool<LoadGetCtx>>();
    loadGetPool_->SetNWorker(config.loadGetNum)
        .SetWorkerFn([this](LoadGetCtx& ctx, auto& unused) { OnLoadGet(ctx, unused); })
        .Run();

    UC_DEBUG("Backend pipeline started (dump: {} + {} + {} workers, load: {} + {} + {} workers)",
             config.dumpGetBufferNum, config.dumpSubmitNum, config.dumpWaitNum, config.loadWaitNum,
             config.loadPutNum, config.loadGetNum);

    return Status::OK();
}

void TransManager::Close()
{
    if (closed_.exchange(true, std::memory_order_acq_rel)) { return; }

    stopFlag_ = true;

    dumpGetBufferPool_.reset();
    dumpSubmitPool_.reset();
    dumpWaitPool_.reset();

    loadWaitPool_.reset();
    loadPutPool_.reset();
    loadGetPool_.reset();

    if (realClient_) { realClient_.reset(); }
    backend_ = nullptr;
}

std::shared_ptr<mooncake::RealClient> TransManager::GetRealClient() const { return realClient_; }

void TransManager::ProcessTask(Detail::TaskHandle handle, TransTask& task,
                               std::shared_ptr<TaskState> state, HostBufferPool& bufPool)
{
    if (!state) { return; }
    state->status.store(TaskStatus::RUNNING, std::memory_order_release);

    if (task.shards.empty()) {
        state->Complete(TaskStatus::SUCCESS);
        return;
    }

    if (!realClient_) {
        state->Complete(TaskStatus::FAILED, "realClient not initialized");
        return;
    }

    if (task.prerequisiteHandle != 0) {
        auto event = reinterpret_cast<aclrtEvent>(task.prerequisiteHandle);
        auto ret = aclrtSynchronizeEvent(event);
        if (ret != 0) {
            UC_WARN("aclrtSynchronizeEvent failed, ret={}, falling back to device sync", ret);
            aclrtSynchronizeDevice();
        }
    }

    if (task.type == TaskType::DUMP) {
        std::string err;
        ProcessDump(task, err);
        state->Complete(err.empty() ? TaskStatus::SUCCESS : TaskStatus::FAILED, std::move(err));
    } else {
        ProcessLoad(task, state, bufPool);
    }
}

void TransManager::BuildBatchFromShards(const std::vector<TransShard>& shards,
                                        std::vector<std::string>& keys,
                                        std::vector<std::vector<void*>>& buffers,
                                        std::vector<std::vector<size_t>>& sizes)
{
    keys.reserve(shards.size());
    buffers.reserve(shards.size());
    sizes.reserve(shards.size());

    for (auto& s : shards) {
        keys.push_back(s.key);
        buffers.push_back(s.addrs);
        sizes.push_back(s.sizes);
    }
}

bool TransManager::CheckMooncakeHit(const std::vector<int>& results)
{
    for (size_t i = 0; i < results.size(); ++i) {
        if (results[i] < 0) { return false; }
    }
    return true;
}

Status TransManager::BuildMissShardBatch(const std::vector<TransShard>& shards,
                                         const std::vector<int>& results, HostBufferPool& bufPool,
                                         LoadWaitCtx& ctx, Detail::TaskDesc& desc)
{
    for (size_t i = 0; i < shards.size(); ++i) {
        if (results[i] >= 0) { continue; }

        auto& shard = shards[i];
        size_t totalSize = 0;
        for (auto sz : shard.sizes) { totalSize += sz; }
        if (totalSize == 0) { continue; }

        void* buf = bufPool.AcquireWithTimeout(std::chrono::milliseconds(30000));
        if (!buf) {
            UC_ERROR("BuildMissShardBatch: host buffer pool exhausted for key={}", shard.key);
            for (auto* b : ctx.hostBufs) { bufPool.Release(b); }
            return Status::Error("host buffer pool exhausted");
        }
        if (totalSize > bufPool.UnitSize()) {
            UC_ERROR("BuildMissShardBatch: shard too large: key={}, need={}, have={}", shard.key,
                     totalSize, bufPool.UnitSize());
            bufPool.Release(buf);
            for (auto* b : ctx.hostBufs) { bufPool.Release(b); }
            return Status::Error("shard too large for buffer pool");
        }

        ctx.keys.push_back(shard.key);
        ctx.shards.push_back(shard);
        ctx.hostBufs.push_back(buf);
        ctx.hostBufSizes.push_back(totalSize);

        desc.push_back(Detail::Shard{shard.owner, shard.index, {buf}});
    }

    return Status::OK();
}

void TransManager::SubmitPosixLoad(LoadWaitCtx& ctx, const Detail::TaskDesc& desc,
                                   std::shared_ptr<TaskState> state, HostBufferPool& bufPool)
{
    auto loadRes = backend_->Load(std::move(const_cast<Detail::TaskDesc&>(desc)));
    if (!loadRes) {
        UC_ERROR("SubmitPosixLoad: Posix batch Load submit failed: {}", loadRes.Error());
        for (auto* b : ctx.hostBufs) { bufPool.Release(b); }
        state->Complete(TaskStatus::FAILED, "Posix batch Load submit failed");
        return;
    }
    ctx.posixHandle = loadRes.Value();

    UC_DEBUG("SubmitPosixLoad: submitted {} miss shards to Posix", ctx.keys.size());
    EnqueueLoadTransfer(std::move(ctx));
}

void TransManager::BuildPutBatch(const LoadPutCtx& ctx,
                                 std::vector<std::span<const char>>& putValues)
{
    putValues.reserve(ctx.keys.size());
    for (size_t i = 0; i < ctx.keys.size(); ++i) {
        putValues.emplace_back(static_cast<const char*>(ctx.hostBufs[i]), ctx.hostBufSizes[i]);
    }
}

void TransManager::DoPutBatch(const LoadPutCtx& ctx,
                              const std::vector<std::span<const char>>& putValues, LoadGetCtx& gctx)
{
    mooncake::ReplicateConfig cfg;
    cfg.replica_num = config_.replicaNum;
    cfg.with_soft_pin = config_.withSoftPin;

    int putRc = realClient_->put_batch(ctx.keys, putValues, cfg);

    if (putRc < 0) {
        UC_ERROR("DoPutBatch: put_batch failed, rc={}", putRc);
        if (gctx.state && !gctx.state->IsTerminal()) {
            gctx.state->Complete(TaskStatus::FAILED, fmt::format("put_batch failed, rc={}", putRc));
        }
        for (auto* b : gctx.hostBufs) { gctx.bufPool->Release(b); }
        return;
    }

    UC_DEBUG("DoPutBatch: put_batch done, {} keys", ctx.keys.size());

    loadGetPool_->Push(std::move(gctx));
}

void TransManager::BuildGetBatch(const LoadGetCtx& ctx, std::vector<std::vector<void*>>& getBuffers,
                                 std::vector<std::vector<size_t>>& getSizes)
{
    getBuffers.reserve(ctx.keys.size());
    getSizes.reserve(ctx.keys.size());

    for (auto& shard : ctx.shards) {
        getBuffers.push_back(shard.addrs);
        getSizes.push_back(shard.sizes);
    }
}

void TransManager::DoGetInto(LoadGetCtx& ctx, const std::vector<std::vector<void*>>& getBuffers,
                             const std::vector<std::vector<size_t>>& getSizes)
{
    auto getResults = realClient_->batch_get_into_multi_buffers(ctx.keys, getBuffers, getSizes,
                                                                /*prefer_same_node=*/true);

    std::string err;
    for (size_t i = 0; i < getResults.size(); ++i) {
        if (getResults[i] < 0) {
            err = fmt::format("get_into failed for key={}, rc={}", ctx.keys[i], getResults[i]);
            UC_ERROR("DoGetInto: {}", err);
            break;
        }
    }

    if (err.empty()) {
        UC_DEBUG("DoGetInto: batch_get_into done, {} shards to NPU", ctx.keys.size());
    }

    if (ctx.state && !ctx.state->IsTerminal()) {
        ctx.state->Complete(err.empty() ? TaskStatus::SUCCESS : TaskStatus::FAILED, std::move(err));
    }
}

void TransManager::ProcessDump(TransTask& task, std::string& err)
{
    std::vector<std::string> keys;
    std::vector<std::vector<void*>> allBuffers;
    std::vector<std::vector<size_t>> allSizes;

    BuildBatchFromShards(task.shards, keys, allBuffers, allSizes);

    mooncake::ReplicateConfig cfg;
    cfg.replica_num = config_.replicaNum;
    cfg.with_soft_pin = config_.withSoftPin;

    auto results = realClient_->batch_put_from_multi_buffers(keys, allBuffers, allSizes, cfg);
    for (size_t i = 0; i < results.size(); ++i) {
        if (results[i] < 0) {
            err = fmt::format("batch_put key={} err={}", keys[i], results[i]);
            UC_ERROR("ProcessDump: batch_put_from_multi_buffers failed: key={}, rc={}", keys[i],
                     results[i]);
            return;
        }
    }

    if (backend_) {
        DumpCtx ctx;
        ctx.keys = std::move(keys);
        ctx.shards = task.shards;
        EnqueueBackendDump(std::move(ctx));
    }
}

void TransManager::EnqueueBackendDump(DumpCtx ctx) { dumpGetBufferPool_->Push(std::move(ctx)); }

void TransManager::OnDumpGetBuffer(DumpCtx& ctx, auto&)
{
    auto handles = realClient_->batch_get_buffer(ctx.keys);

    std::vector<void*> bufferPtrs;
    bufferPtrs.reserve(handles.size());
    bool ok = true;
    for (size_t i = 0; i < handles.size(); ++i) {
        if (!handles[i]) {
            UC_ERROR("OnDumpGetBuffer: get_buffer returned null for key={}", ctx.keys[i]);
            ok = false;
            break;
        }
        bufferPtrs.push_back(handles[i]->ptr());
    }

    if (!ok) { return; }

    dumpSubmitPool_->Push(DumpSubmitCtx{ctx.keys, ctx.shards, std::move(bufferPtrs)});
}

void TransManager::OnDumpSubmit(DumpSubmitCtx& ctx, auto&)
{
    UC_DEBUG("OnDumpSubmit: submitting {} keys to Posix", ctx.keys.size());

    Detail::TaskDesc desc;
    desc.brief = "Mooncake2Posix";
    for (size_t i = 0; i < ctx.shards.size(); ++i) {
        desc.push_back(
            Detail::Shard{ctx.shards[i].owner, ctx.shards[i].index, {ctx.bufferPtrs[i]}});
    }

    auto res = backend_->Dump(std::move(desc));
    if (!res) {
        UC_ERROR("OnDumpSubmit: Dump submit failed: {}", res.Error());
        return;
    }

    dumpWaitPool_->Push(DumpWaitCtx{res.Value(), ctx.keys.size()});
}

void TransManager::OnDumpWait(DumpWaitCtx& ctx, auto&)
{
    auto s = backend_->Wait(ctx.posixHandle);
    if (s.Failure()) {
        UC_ERROR("OnDumpWait: Wait failed: {}", s);
    } else {
        UC_DEBUG("OnDumpWait: persisted {} shards ok", ctx.shardCount);
    }
}

void TransManager::ProcessLoad(TransTask& task, std::shared_ptr<TaskState> state,
                               HostBufferPool& bufPool)
{
    std::vector<std::string> keys;
    std::vector<std::vector<void*>> allBuffers;
    std::vector<std::vector<size_t>> allSizes;

    BuildBatchFromShards(task.shards, keys, allBuffers, allSizes);

    auto results = realClient_->batch_get_into_multi_buffers(keys, allBuffers, allSizes, false);

    if (CheckMooncakeHit(results)) {
        UC_DEBUG("ProcessLoad: all {} shards hit in Mooncake pool", keys.size());
        state->Complete(TaskStatus::SUCCESS);
        return;
    }

    size_t missCount = 0;
    for (auto r : results) {
        if (r < 0) { ++missCount; }
    }
    UC_DEBUG("ProcessLoad: {}/{} shards miss, falling back to Posix", missCount, keys.size());

    if (!backend_) {
        state->Complete(TaskStatus::FAILED, "Mooncake pool miss with no backend");
        return;
    }

    LoadWaitCtx wctx;
    wctx.state = state;
    wctx.bufPool = &bufPool;

    Detail::TaskDesc loadDesc;
    loadDesc.brief = "Posix2Mooncake";

    auto s = BuildMissShardBatch(task.shards, results, bufPool, wctx, loadDesc);
    if (s.Failure()) {
        state->Complete(TaskStatus::FAILED, s.ToString());
        return;
    }

    if (wctx.keys.empty()) {
        state->Complete(TaskStatus::SUCCESS);
        return;
    }

    SubmitPosixLoad(wctx, loadDesc, state, bufPool);
}

void TransManager::EnqueueLoadTransfer(LoadWaitCtx ctx) { loadWaitPool_->Push(std::move(ctx)); }

void TransManager::OnLoadWait(LoadWaitCtx& ctx, auto&)
{
    auto waitStatus = backend_->Wait(ctx.posixHandle);
    if (waitStatus.Failure()) {
        UC_ERROR("OnLoadWait: Posix Wait failed: {}", waitStatus);
        for (auto* b : ctx.hostBufs) { ctx.bufPool->Release(b); }
        if (ctx.state && !ctx.state->IsTerminal()) {
            ctx.state->Complete(TaskStatus::FAILED, "Posix Wait failed");
        }
        return;
    }

    UC_DEBUG("OnLoadWait: Posix batch Load done, {} shards", ctx.keys.size());

    LoadPutCtx pctx;
    pctx.state = ctx.state;
    pctx.keys = std::move(ctx.keys);
    pctx.shards = std::move(ctx.shards);
    pctx.hostBufs = std::move(ctx.hostBufs);
    pctx.hostBufSizes = std::move(ctx.hostBufSizes);
    pctx.bufPool = ctx.bufPool;

    loadPutPool_->Push(std::move(pctx));
}

void TransManager::OnLoadPut(LoadPutCtx& ctx, auto&)
{
    std::vector<std::span<const char>> putValues;
    BuildPutBatch(ctx, putValues);

    LoadGetCtx gctx;
    gctx.state = ctx.state;
    gctx.keys = ctx.keys;
    gctx.shards = ctx.shards;
    gctx.hostBufs = std::move(ctx.hostBufs);
    gctx.bufPool = ctx.bufPool;

    DoPutBatch(ctx, putValues, gctx);
}

void TransManager::OnLoadGet(LoadGetCtx& ctx, auto&)
{
    std::vector<std::vector<void*>> getBuffers;
    std::vector<std::vector<size_t>> getSizes;

    BuildGetBatch(ctx, getBuffers, getSizes);
    DoGetInto(ctx, getBuffers, getSizes);

    for (auto* b : ctx.hostBufs) { ctx.bufPool->Release(b); }
}

}  // namespace UC::MooncakeStore
