/**
 * MIT License
 *
 * Copyright (c) 2026 relat-ivity
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
#ifndef RDMA_CHANNEL_GDR_H
#define RDMA_CHANNEL_GDR_H

#include <infiniband/verbs.h>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

struct RDMAChannelConfig {
    int cqDepth = 1024;
    int qpSendWr = 1024;
    int qpRecvWr = 1;
};

class RDMAChannel {
public:
    RDMAChannel(int32_t deviceId, std::string nicName, const RDMAChannelConfig& config);
    ~RDMAChannel();

    RDMAChannel(const RDMAChannel&) = delete;
    RDMAChannel& operator=(const RDMAChannel&) = delete;
    RDMAChannel(RDMAChannel&&) = delete;
    RDMAChannel& operator=(RDMAChannel&&) = delete;

    ibv_mr* RegisterHostMemory(void* buffer, size_t bytes);
    ibv_mr* RegisterGpuMemory(void* buffer, size_t bytes);
    uint64_t SubmitWrite(uint64_t localAddress, uint32_t localLKey, uint64_t remoteAddress,
                         uint32_t remoteRKey, size_t bytes);
    void Wait(uint64_t targetWorkRequestId);

    int32_t DeviceId() const noexcept { return deviceId_; }
    const std::string& NicName() const noexcept { return nicName_; }

private:
    void PollOneCompletion();

    ibv_context* context_ = nullptr;
    ibv_pd* protectionDomain_ = nullptr;
    ibv_cq* completionQueue_ = nullptr;
    ibv_qp* queuePair_ = nullptr;
    int32_t deviceId_ = 0;
    std::string nicName_;
    uint64_t nextWorkRequestId_ = 1;
    uint64_t completedWorkRequestId_ = 0;
    uint32_t outstandingWorkRequests_ = 0;
    uint32_t maxOutstandingWorkRequests_ = 0;
};

class ChannelManager {
public:
    static ChannelManager& Instance();

    void Initialize(int32_t deviceNumber, const std::vector<std::string>& nicNames,
                    const RDMAChannelConfig& config);
    RDMAChannel& Get(int32_t deviceId);
    const std::vector<int32_t>& DeviceIds() const noexcept { return deviceIds_; }
    bool IsInitialized() const noexcept;
    void Shutdown() noexcept;

private:
    ChannelManager() = default;

    mutable std::mutex mutex_;
    std::unordered_map<int32_t, std::unique_ptr<RDMAChannel>> channels_;
    std::vector<int32_t> deviceIds_;
    bool initialized_ = false;
};

#endif  // RDMA_CHANNEL_GDR_H
