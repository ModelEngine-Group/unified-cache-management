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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "trans_provider.h"
#include "types.h"

namespace UC::ASU {

struct TransportTask;

struct AsuEndpoint {
    std::string ip;
    std::uint16_t port{0};
    Protocol protocol{Protocol::ROCE};
    std::int32_t numaNode{-1};
    std::string hcaName;
    std::uint8_t hcaPort{1};
    std::unordered_map<std::string, std::string> attrs;
};

struct TransportConfig {
    // TODO: 拆分Config，按逻辑模块细化
    std::string asuName;
    AsuId asuId{0};
    // Local logical device ID used by the transport provider.
    std::int32_t deviceId{-1};
    std::vector<AsuEndpoint> endpoints;

    TransProviderType providerType{TransProviderType::AICPU};

    std::uint32_t qpNum{1};

    std::uint32_t maxInflightTasks{1024};
    std::uint64_t maxInflightBytes{1ULL << 30};

    std::uint64_t timeoutMs{100};
    std::uint32_t maxErrorCount{2};
    std::size_t completionPollSpinLimit{16};

    bool enableDeviceDirect{true};
    bool enableHostFallback{false};
    bool preconnect{true};
    bool bindCqPoller{true};

    // Slot sizes are caller-visible capacities; BufferManager computes the
    // aligned physical stride used for allocation and memory registration.
    std::size_t sendBufferSlotSize{4160};
    std::size_t sendBufferSlotNum{128};
    // Maximum memory required by a batch store/retrieve response flag buffer.
    std::size_t flagBufferSlotSize{71};
    std::size_t flagBufferSlotNum{4096};
    std::size_t asuBatchLoadIoNum{110};
    std::size_t asuBatchStoreIoNum{110};
    std::size_t asuDeleteIoNum{254};
    std::size_t asuQueryIoNum{256};

    // Transport attrs loaded from config, including SQE request attrs
    // (kv_ns_id, dtype, dspec, lr, sc) and send attrs (kernel_count, quiet_count).
    std::unordered_map<std::string, std::string> attrs;
};

template <typename T>
struct BatchView {
    const T* data{nullptr};
    std::size_t size{0};

    const T& operator[](std::size_t i) const noexcept { return data[i]; }
    bool empty() const noexcept { return size == 0; }
};

class AsuTransport {
public:
    virtual ~AsuTransport() = default;

    virtual Status Init(const TransportConfig& config,
                        std::shared_ptr<TransProvider> transProvider) = 0;
    virtual Status Init(const std::string& configPath,
                        std::shared_ptr<TransProvider> transProvider) = 0;
    virtual Status Shutdown() = 0;
    virtual Status CheckHealth() = 0;

    virtual Status Submit(const std::shared_ptr<TransportTask>& task) = 0;

    // Best-effort cancellation, does not interrupt underlying UB/RoCE IO
    virtual Status Cancel(TaskId taskId) = 0;
};

std::unique_ptr<AsuTransport> CreateAsuTransport();

}  // namespace UC::ASU
