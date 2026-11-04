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
#include "rdma_channel_gdr.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <thread>

#include "error_handle_gdr.h"

namespace {

constexpr int kIbvPort = 1;

struct QPEndpoint {
    uint32_t qpn = 0;
    uint16_t lid = 0;
    uint8_t gid[16] = {};
};

QPEndpoint QueryEndpoint(ibv_qp* queuePair, ibv_context* context)
{
    QPEndpoint endpoint;
    endpoint.qpn = queuePair->qp_num;

    ibv_port_attr portAttr = {};
    GDR_IBV_ASSERT(ibv_query_port(context, kIbvPort, &portAttr));
    endpoint.lid = portAttr.lid;

    union ibv_gid gid = {};
    GDR_IBV_ASSERT(ibv_query_gid(context, kIbvPort, 0, &gid));
    std::memcpy(endpoint.gid, &gid, sizeof(endpoint.gid));
    return endpoint;
}

void ConnectQueuePair(ibv_qp* queuePair, const QPEndpoint& remoteEndpoint, bool isRoce,
                      ibv_mtu pathMtu)
{
    {
        ibv_qp_attr attr = {};
        attr.qp_state = IBV_QPS_INIT;
        attr.pkey_index = 0;
        attr.port_num = kIbvPort;
        attr.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE;
        GDR_IBV_ASSERT(ibv_modify_qp(
            queuePair, &attr,
            IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS));
    }

    {
        ibv_qp_attr attr = {};
        attr.qp_state = IBV_QPS_RTR;
        attr.path_mtu = pathMtu;
        attr.dest_qp_num = remoteEndpoint.qpn;
        attr.rq_psn = 0;
        attr.max_dest_rd_atomic = 1;
        attr.min_rnr_timer = 12;
        attr.ah_attr.port_num = kIbvPort;
        attr.ah_attr.sl = 0;
        attr.ah_attr.src_path_bits = 0;
        if (isRoce) {
            attr.ah_attr.is_global = 1;
            attr.ah_attr.grh.hop_limit = 64;
            std::memcpy(&attr.ah_attr.grh.dgid, remoteEndpoint.gid, sizeof(remoteEndpoint.gid));
            attr.ah_attr.grh.sgid_index = 0;
            attr.ah_attr.dlid = 0;
        } else {
            attr.ah_attr.is_global = 0;
            attr.ah_attr.dlid = remoteEndpoint.lid;
        }
        GDR_IBV_ASSERT(ibv_modify_qp(
            queuePair, &attr,
            IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER));
    }

    {
        ibv_qp_attr attr = {};
        attr.qp_state = IBV_QPS_RTS;
        attr.timeout = 14;
        attr.retry_cnt = 7;
        attr.rnr_retry = 7;
        attr.sq_psn = 0;
        attr.max_rd_atomic = 1;
        GDR_IBV_ASSERT(ibv_modify_qp(
            queuePair, &attr,
            IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
                IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC));
    }
}

ibv_context* OpenNicContext(const std::string& nicName)
{
    int deviceCount = 0;
    ibv_device** deviceList = ibv_get_device_list(&deviceCount);
    ASSERT(deviceList != nullptr);
    ASSERT(deviceCount > 0);

    ibv_context* context = nullptr;
    for (int index = 0; index < deviceCount; ++index) {
        if (nicName == ibv_get_device_name(deviceList[index])) {
            context = ibv_open_device(deviceList[index]);
            break;
        }
    }

    if (context == nullptr) {
        ibv_free_device_list(deviceList);
        std::fprintf(stderr, "RDMA device not found: %s\n", nicName.c_str());
        std::exit(EXIT_FAILURE);
    }

    ibv_free_device_list(deviceList);
    return context;
}

}  // namespace

RDMAChannel::RDMAChannel(int32_t deviceId, std::string nicName, const RDMAChannelConfig& config)
    : deviceId_(deviceId), nicName_(std::move(nicName))
{
    CUDA_ASSERT(cudaSetDevice(deviceId_));
    ASSERT(config.cqDepth > 0);
    ASSERT(config.qpSendWr > 0);
    ASSERT(config.qpRecvWr > 0);

    context_ = OpenNicContext(nicName_);
    ASSERT(context_ != nullptr);

    protectionDomain_ = ibv_alloc_pd(context_);
    ASSERT(protectionDomain_ != nullptr);

    ibv_port_attr portAttr = {};
    GDR_IBV_ASSERT(ibv_query_port(context_, kIbvPort, &portAttr));
    const bool isRoce = (portAttr.lid == 0);

    ibv_device_attr deviceAttr = {};
    GDR_IBV_ASSERT(ibv_query_device(context_, &deviceAttr));

    ASSERT(config.cqDepth <= static_cast<int>(deviceAttr.max_cqe));
    ASSERT(config.qpSendWr <= static_cast<int>(deviceAttr.max_qp_wr));
    ASSERT(config.qpRecvWr <= static_cast<int>(deviceAttr.max_qp_wr));
    ASSERT(config.qpSendWr + config.qpRecvWr <= static_cast<int>(deviceAttr.max_qp_wr));

    const int cqDepth = config.cqDepth;
    completionQueue_ = ibv_create_cq(context_, cqDepth, nullptr, nullptr, 0);
    ASSERT(completionQueue_ != nullptr);

    ibv_qp_init_attr qpInitAttr = {};
    qpInitAttr.send_cq = completionQueue_;
    qpInitAttr.recv_cq = completionQueue_;
    qpInitAttr.cap.max_send_wr = config.qpSendWr;
    qpInitAttr.cap.max_recv_wr = config.qpRecvWr;
    qpInitAttr.cap.max_send_sge = 1;
    qpInitAttr.cap.max_recv_sge = 1;
    qpInitAttr.qp_type = IBV_QPT_RC;
    qpInitAttr.sq_sig_all = 0;

    queuePair_ = ibv_create_qp(protectionDomain_, &qpInitAttr);
    ASSERT(queuePair_ != nullptr);

    const auto cqWindow = static_cast<uint32_t>(std::max(1, cqDepth - 1));
    maxOutstandingWorkRequests_ = std::max<uint32_t>(
        1u, std::min<uint32_t>(qpInitAttr.cap.max_send_wr, cqWindow));

    const QPEndpoint endpoint = QueryEndpoint(queuePair_, context_);
    ConnectQueuePair(queuePair_, endpoint, isRoce, portAttr.active_mtu);
}

RDMAChannel::~RDMAChannel()
{
    if (queuePair_ != nullptr) { ibv_destroy_qp(queuePair_); }
    if (completionQueue_ != nullptr) { ibv_destroy_cq(completionQueue_); }
    if (protectionDomain_ != nullptr) { ibv_dealloc_pd(protectionDomain_); }
    if (context_ != nullptr) { ibv_close_device(context_); }
}

ibv_mr* RDMAChannel::RegisterHostMemory(void* buffer, size_t bytes)
{
    ASSERT(buffer != nullptr);
    ASSERT(bytes > 0);
    ibv_mr* memoryRegion =
        ibv_reg_mr(protectionDomain_, buffer, bytes, IBV_ACCESS_LOCAL_WRITE);
    ASSERT(memoryRegion != nullptr);
    return memoryRegion;
}

ibv_mr* RDMAChannel::RegisterGpuMemory(void* buffer, size_t bytes)
{
    ASSERT(buffer != nullptr);
    ASSERT(bytes > 0);
    CUDA_ASSERT(cudaSetDevice(deviceId_));
    ibv_mr* memoryRegion = ibv_reg_mr(
        protectionDomain_, buffer, bytes, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    ASSERT(memoryRegion != nullptr);
    return memoryRegion;
}

uint64_t RDMAChannel::SubmitWrite(uint64_t localAddress, uint32_t localLKey, uint64_t remoteAddress,
                                  uint32_t remoteRKey, size_t bytes)
{
    ASSERT(bytes > 0);
    ASSERT(bytes <= std::numeric_limits<uint32_t>::max());

    while (outstandingWorkRequests_ >= maxOutstandingWorkRequests_) { PollOneCompletion(); }

    ibv_sge scatterGather = {};
    scatterGather.addr = localAddress;
    scatterGather.length = static_cast<uint32_t>(bytes);
    scatterGather.lkey = localLKey;

    ibv_send_wr workRequest = {};
    workRequest.wr_id = nextWorkRequestId_;
    workRequest.opcode = IBV_WR_RDMA_WRITE;
    workRequest.sg_list = &scatterGather;
    workRequest.num_sge = 1;
    workRequest.send_flags = IBV_SEND_SIGNALED;
    workRequest.wr.rdma.remote_addr = remoteAddress;
    workRequest.wr.rdma.rkey = remoteRKey;

    ibv_send_wr* badWorkRequest = nullptr;
    GDR_IBV_ASSERT(ibv_post_send(queuePair_, &workRequest, &badWorkRequest));

    ++outstandingWorkRequests_;
    return nextWorkRequestId_++;
}

void RDMAChannel::Wait(uint64_t targetWorkRequestId)
{
    while (completedWorkRequestId_ < targetWorkRequestId) { PollOneCompletion(); }
}

void RDMAChannel::PollOneCompletion()
{
    ibv_wc workCompletion = {};
    while (true) {
        const int pollResult = ibv_poll_cq(completionQueue_, 1, &workCompletion);
        ASSERT(pollResult >= 0);
        if (pollResult == 0) {
            std::this_thread::yield();
            continue;
        }
        break;
    }

    GDR_WC_ASSERT(workCompletion.status, workCompletion.wr_id);
    completedWorkRequestId_ = std::max(completedWorkRequestId_, workCompletion.wr_id);
    if (outstandingWorkRequests_ > 0) { --outstandingWorkRequests_; }
}

ChannelManager& ChannelManager::Instance()
{
    static ChannelManager manager;
    return manager;
}

void ChannelManager::Initialize(int32_t deviceNumber, const std::vector<std::string>& nicNames,
                                const RDMAChannelConfig& config)
{
    std::lock_guard<std::mutex> lock(mutex_);
    ASSERT(!initialized_);
    ASSERT(deviceNumber > 0);
    ASSERT(nicNames.size() == static_cast<size_t>(deviceNumber));

    channels_.clear();
    deviceIds_.clear();
    deviceIds_.reserve(static_cast<size_t>(deviceNumber));
    for (int32_t deviceId = 0; deviceId < deviceNumber; ++deviceId) {
        channels_.emplace(deviceId,
                          std::make_unique<RDMAChannel>(deviceId, nicNames[deviceId], config));
        deviceIds_.push_back(deviceId);
    }
    initialized_ = true;
}

RDMAChannel& ChannelManager::Get(int32_t deviceId)
{
    std::lock_guard<std::mutex> lock(mutex_);
    ASSERT(initialized_);
    const auto it = channels_.find(deviceId);
    ASSERT(it != channels_.end());
    return *it->second;
}

bool ChannelManager::IsInitialized() const noexcept
{
    std::lock_guard<std::mutex> lock(mutex_);
    return initialized_;
}

void ChannelManager::Shutdown() noexcept
{
    std::lock_guard<std::mutex> lock(mutex_);
    channels_.clear();
    deviceIds_.clear();
    initialized_ = false;
}
