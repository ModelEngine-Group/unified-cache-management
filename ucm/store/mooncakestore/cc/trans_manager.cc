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
#include <numeric>
#include "logger/logger.h"
#include "metrics_api.h"

namespace UC::MooncakeStore {

namespace {

size_t TaskBytes(const TransTask& task)
{
    size_t bytes = 0;
    for (const auto& shard : task.shards) {
        bytes += std::accumulate(shard.sizes.begin(), shard.sizes.end(), size_t{0});
    }
    return bytes;
}

}  // namespace

Status TransManager::Setup(const Config& config)
{
    config_ = config;
    timeoutMs_ = config.timeoutMs;
    localRankSize_ = config.localRankSize;

    auto s = SetupRealClient(config);
    if (s.Failure()) { return s; }

    size_t hostBufUnitSize =
        std::accumulate(config.tensorSizeList.begin(), config.tensorSizeList.end(), uint64_t{0});
    if (hostBufUnitSize > 0 && config.hostBufPoolSize > 0) {
        s = bufPool_.Setup(config.deviceId, config.hostBufPoolSize, hostBufUnitSize,
                           config.ioDirect);
        if (s.Failure()) { return s; }
    }

    if (localRankSize_ > 1 && config.storeBackend) {
        s = shareLoadQ_.Setup(config, &failureSet_, config.storeBackend);
        if (s.Failure()) { return s; }
    }

    ShareLoadQueue* shareLoadQPtr =
        (localRankSize_ > 1 && config.storeBackend) ? &shareLoadQ_ : nullptr;
    s = loadQ_.Setup(config, &failureSet_, realClient_, config.storeBackend, &bufPool_,
                     shareLoadQPtr);
    if (s.Failure()) { return s; }
    s = dumpQ_.Setup(config, &failureSet_, realClient_, config.storeBackend, &bufPool_);
    if (s.Failure()) { return s; }

    UC_INFO("TransManager setup ok, backend={}, localBufSize={}, localRankSize={}",
            config.storeBackend ? "yes" : "none", config.localBufferSize, localRankSize_);
    return Status::OK();
}

void TransManager::Close()
{
    loadQ_.Close();
    dumpQ_.Close();
    shareLoadQ_.Close();
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

void TransManager::Dispatch(TaskPtr t, WaiterPtr w)
{
    const auto id = t->id;
    const auto brief = t->brief;
    const auto num = t->shards.size();
    const auto size = TaskBytes(*t);
    const auto tp = w->startTp;
    const auto isLoad = t->type == TaskType::LOAD;
    w->SetEpilog([id, brief, num, size, tp, isLoad] {
        auto cost = NowTime::Now() - tp;
        auto costMs = cost * 1e3;
        auto bwGbps = cost > 0 ? static_cast<double>(size) / cost / 1e9 : 0.0;
        UC_DEBUG("Mooncake task({},{},{},{}) finished, cost {:.3f}ms.", id, brief, num, size,
                 costMs);
        if (isLoad) {
            UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("mooncake_load_duration_ms"), costMs);
            UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("mooncake_load_bandwidth_gbps"), bwGbps);
            UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("mooncake_load_blocks_total"),
                                     static_cast<double>(num));
            UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("mooncake_load_bytes_total"),
                                     static_cast<double>(size));
        } else {
            UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("mooncake_dump_duration_ms"), costMs);
            UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("mooncake_dump_bandwidth_gbps"), bwGbps);
            UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("mooncake_dump_blocks_total"),
                                     static_cast<double>(num));
            UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("mooncake_dump_bytes_total"),
                                     static_cast<double>(size));
        }
    });
    if (t->type == TaskType::LOAD) {
        loadQ_.Submit(t, w);
    } else {
        dumpQ_.Submit(t, w);
    }
}

}  // namespace UC::MooncakeStore
