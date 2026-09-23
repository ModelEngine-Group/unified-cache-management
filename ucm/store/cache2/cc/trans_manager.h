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
#pragma once

#include "cache_buffer.h"
#include "dump_queue.h"
#include "global_config.h"
#include "load_queue.h"
#include "metrics_api.h"
#include "template/task_wrapper.h"
#include "time/now_time.h"
#include "trans_task.h"

namespace UC::Cache2 {

template <typename LoadQT = LoadQ<>, typename DumpQT = DumpQ<>>
class TransManager : public Detail::TaskWrapper<Task, Detail::TaskHandle> {
    size_t shardSize_{0};
    LoadQT loadQ_;
    DumpQT dumpQ_;

public:
    Status Setup(const Config& config, Buffer* buffer)
    {
        timeoutMs_ = config.timeoutMs;
        shardSize_ = config.shardSize;
        auto s = loadQ_.Setup(config, &failureSet_, buffer);
        if (s.Failure()) [[unlikely]] { return s; }
        return dumpQ_.Setup(config, &failureSet_, buffer);
    }

protected:
    void Dispatch(TaskPtr t, WaiterPtr w) override
    {
        const auto isLoad = t->type == Task::Type::LOAD;
        const auto shards = t->desc.size();
        const auto bytes = shardSize_ * shards;
        w->SetEpilog([isLoad, shards, bytes, tp = w->startTp] {
            auto cost = NowTime::Now() - tp;
            auto costMs = cost * 1e3;
            auto bwGbps = cost > 0 ? static_cast<double>(bytes) / cost / 1e9 : 0.0;
            if (isLoad) {
                Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_load_duration_ms"), costMs);
                Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_load_bandwidth_gbps"), bwGbps);
                Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_load_bytes_total"),
                                     static_cast<double>(bytes));
            } else {
                Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_dump_duration_ms"), costMs);
                Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_dump_bandwidth_gbps"), bwGbps);
                Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_dump_shards_total"),
                                     static_cast<double>(shards));
                Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_dump_bytes_total"),
                                     static_cast<double>(bytes));
            }
        });
        if (isLoad) {
            loadQ_.Submit(t, w);
        } else {
            dumpQ_.Submit(t, w);
        }
    }
};

}  // namespace UC::Cache2
