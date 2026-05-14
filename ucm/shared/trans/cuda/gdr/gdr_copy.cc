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
#include "gdr_copy.h"
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstring>
#include <cuda_runtime.h>
#include <infiniband/verbs.h>
#include <limits>
#include <mutex>
#include <poll.h>
#include <stdexcept>
#include <sys/eventfd.h>
#include <unistd.h>
#include <utility>
#include <vector>
#include "gdr_mr_buffer.h"
#include "logger/logger.h"

namespace {

constexpr int kIbvPort = 1;
constexpr int kTargetCqDepth = 4096;
constexpr int kTargetSendWr = 4096;
constexpr size_t kInflightRingCapacity = 8192;
constexpr size_t kInflightRingMask = kInflightRingCapacity - 1;
static_assert((kInflightRingCapacity & kInflightRingMask) == 0,
              "inflight ring capacity must be a power of two");

struct RegisteredMrRange {
    uint64_t addr;
    size_t len;
    struct ibv_mr* mr;
};

struct MrRef {
    struct ibv_mr* mr{nullptr};
    bool owned{false};
};

struct Endpoint {
    uint32_t qpn{};
    uint16_t lid{};
    uint8_t gid[16]{};
};

Endpoint QueryEndpoint(struct ibv_qp* qp, struct ibv_context* ctx)
{
    Endpoint endpoint{};
    struct ibv_port_attr portAttr{};
    (void)ibv_query_port(ctx, kIbvPort, &portAttr);
    endpoint.qpn = qp->qp_num;
    endpoint.lid = portAttr.lid;
    (void)ibv_query_gid(ctx, kIbvPort, 0, reinterpret_cast<union ibv_gid*>(endpoint.gid));
    return endpoint;
}

void ConnectRcQp(struct ibv_qp* qp, const Endpoint& remote, bool isRoce)
{
    {
        struct ibv_qp_attr attr{};
        attr.qp_state = IBV_QPS_INIT;
        attr.pkey_index = 0;
        attr.port_num = kIbvPort;
        attr.qp_access_flags =
            IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE;
        if (ibv_modify_qp(qp, &attr,
                          IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS) !=
            0) {
            throw std::runtime_error("failed to move RC QP to INIT");
        }
    }
    {
        struct ibv_qp_attr attr{};
        attr.qp_state = IBV_QPS_RTR;
        attr.path_mtu = IBV_MTU_4096;
        attr.dest_qp_num = remote.qpn;
        attr.rq_psn = 0;
        attr.max_dest_rd_atomic = 1;
        attr.min_rnr_timer = 12;
        if (isRoce) {
            attr.ah_attr.is_global = 1;
            attr.ah_attr.grh.hop_limit = 64;
            attr.ah_attr.grh.sgid_index = 0;
            std::memcpy(&attr.ah_attr.grh.dgid, remote.gid, sizeof(remote.gid));
        } else {
            attr.ah_attr.is_global = 0;
            attr.ah_attr.dlid = remote.lid;
        }
        attr.ah_attr.sl = 0;
        attr.ah_attr.src_path_bits = 0;
        attr.ah_attr.port_num = kIbvPort;
        if (ibv_modify_qp(qp, &attr,
                          IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
                              IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER) !=
            0) {
            throw std::runtime_error("failed to move RC QP to RTR");
        }
    }
    {
        struct ibv_qp_attr attr{};
        attr.qp_state = IBV_QPS_RTS;
        attr.timeout = 14;
        attr.retry_cnt = 7;
        attr.rnr_retry = 7;
        attr.sq_psn = 0;
        attr.max_rd_atomic = 1;
        if (ibv_modify_qp(qp, &attr,
                          IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
                              IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC) != 0) {
            throw std::runtime_error("failed to move RC QP to RTS");
        }
    }
}

bool RangeContains(uint64_t base, size_t totalSize, uint64_t addr, size_t len)
{
    if (addr < base) { return false; }
    const auto offset = addr - base;
    if (offset > totalSize) { return false; }
    return len <= totalSize - offset;
}

struct ibv_mr* FindRegisteredMrRange(const std::vector<RegisteredMrRange>& ranges, uint64_t addr,
                                     size_t len)
{
    for (const auto& range : ranges) {
        if (RangeContains(range.addr, range.len, addr, len)) { return range.mr; }
    }
    return nullptr;
}

struct ibv_mr* FindRegisteredMrExact(const std::vector<RegisteredMrRange>& ranges, uint64_t addr,
                                     size_t len)
{
    for (const auto& range : ranges) {
        if (range.addr == addr && range.len == len) { return range.mr; }
    }
    return nullptr;
}

struct ibv_mr* ReleaseRegisteredMrExact(std::vector<RegisteredMrRange>& ranges, uint64_t addr,
                                        size_t len)
{
    for (auto it = ranges.begin(); it != ranges.end(); ++it) {
        if (it->addr != addr || it->len != len) { continue; }
        auto* mr = it->mr;
        ranges.erase(it);
        return mr;
    }
    return nullptr;
}

void ClearRegisteredMrRanges(std::vector<RegisteredMrRange>& ranges)
{
    for (const auto& range : ranges) {
        if (range.mr) { ibv_dereg_mr(range.mr); }
    }
    ranges.clear();
}

class GdrCopyChannelImpl : public GdrCopyChannel {
public:
    GdrCopyChannelImpl(int gpuId, std::string nicName) : gpuId_{gpuId}, nicName_{std::move(nicName)}
    {
        try {
            const auto cudaRet = cudaSetDevice(gpuId_);
            if (cudaRet != cudaSuccess) {
                throw std::runtime_error(std::string("cudaSetDevice failed: ") +
                                         cudaGetErrorString(cudaRet));
            }
            int nDev = 0;
            struct ibv_device** devices = ibv_get_device_list(&nDev);
            if (!devices || nDev == 0) { throw std::runtime_error("no RDMA device found"); }

            struct ibv_device* target = nullptr;
            for (int i = 0; i < nDev; ++i) {
                if (nicName_ == ibv_get_device_name(devices[i])) {
                    target = devices[i];
                    break;
                }
            }
            if (!target) {
                ibv_free_device_list(devices);
                throw std::runtime_error("target RDMA device not found");
            }

            ctx_ = ibv_open_device(target);
            ibv_free_device_list(devices);
            if (!ctx_) { throw std::runtime_error("ibv_open_device failed"); }

            struct ibv_port_attr portAttr{};
            if (ibv_query_port(ctx_, kIbvPort, &portAttr) != 0) {
                throw std::runtime_error("ibv_query_port failed");
            }
            isRoce_ = portAttr.lid == 0;

            pd_ = ibv_alloc_pd(ctx_);
            if (!pd_) { throw std::runtime_error("ibv_alloc_pd failed"); }
            for (const auto& buffer : UC::Trans::HostBufferRegistry::Snapshot()) {
                RegisterHostBuffer(buffer.addr, buffer.size);
            }
            std::vector<UC::Trans::DeviceBufferInfo> invalidGpuBuffers;
            for (const auto& buffer : UC::Trans::DeviceBufferRegistry::Snapshot()) {
                try {
                    RegisterDeviceBuffer(buffer.addr, buffer.size);
                } catch (const std::exception& e) {
                    UC_WARN("Skip pre-registered device MR at addr(0x{:x}) size({}): {}.",
                            buffer.addr, buffer.size, e.what());
                    invalidGpuBuffers.push_back(buffer);
                }
            }
            for (const auto& buffer : invalidGpuBuffers) {
                UC::Trans::DeviceBufferRegistry::Unregister(reinterpret_cast<void*>(buffer.addr));
            }

            struct ibv_device_attr deviceAttr{};
            if (ibv_query_device(ctx_, &deviceAttr) != 0) {
                throw std::runtime_error("ibv_query_device failed");
            }

            compChannel_ = ibv_create_comp_channel(ctx_);
            if (!compChannel_) { throw std::runtime_error("ibv_create_comp_channel failed"); }

            completionWakeupFd_ = eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
            if (completionWakeupFd_ < 0) { throw std::runtime_error("eventfd failed"); }

            const int cqDepth =
                std::max(1, std::min(kTargetCqDepth, static_cast<int>(deviceAttr.max_cqe)));
            cq_ = ibv_create_cq(ctx_, cqDepth, nullptr, compChannel_, 0);
            if (!cq_) { throw std::runtime_error("ibv_create_cq failed"); }

            struct ibv_qp_init_attr initAttr{};
            initAttr.send_cq = cq_;
            initAttr.recv_cq = cq_;
            initAttr.cap.max_send_wr =
                std::max(1, std::min(kTargetSendWr, static_cast<int>(deviceAttr.max_qp_wr == 0
                                                                         ? kTargetSendWr
                                                                         : deviceAttr.max_qp_wr)));
            initAttr.cap.max_recv_wr = 1;
            initAttr.cap.max_send_sge = 1;
            initAttr.cap.max_recv_sge = 1;
            initAttr.qp_type = IBV_QPT_RC;
            initAttr.sq_sig_all = 0;
            qp_ = ibv_create_qp(pd_, &initAttr);
            if (!qp_) { throw std::runtime_error("ibv_create_qp failed"); }

            maxInflightWr_ = std::max(1, static_cast<int>(initAttr.cap.max_send_wr) - 1);

            const auto endpoint = QueryEndpoint(qp_, ctx_);
            ConnectRcQp(qp_, endpoint, isRoce_);
        } catch (...) {
            Cleanup();
            throw;
        }
    }

    ~GdrCopyChannelImpl() override { Cleanup(); }

    void RegisterHostBuffer(uint64_t addr, size_t len)
    {
        std::lock_guard<std::mutex> lock{mutex_};
        RegisterHostBufferLocked(addr, len);
    }

    void UnregisterHostBuffer(uint64_t addr, size_t len)
    {
        std::lock_guard<std::mutex> lock{mutex_};
        auto* mr = ReleaseRegisteredMrExact(hostMrRanges_, addr, len);
        if (mr) { ibv_dereg_mr(mr); }
    }

    void RegisterDeviceBuffer(uint64_t addr, size_t len)
    {
        std::lock_guard<std::mutex> lock{mutex_};
        RegisterDeviceBufferLocked(addr, len);
    }

    void UnregisterDeviceBuffer(uint64_t addr, size_t len)
    {
        std::lock_guard<std::mutex> lock{mutex_};
        auto* mr = ReleaseRegisteredMrExact(gpuMrRanges_, addr, len);
        if (mr) { ibv_dereg_mr(mr); }
    }

    int GdrMemcpyAsync(void* dst, const void* src, size_t bytes, GdrCopyKind kind,
                       uint64_t* reqId) override
    {
        try {
            if (bytes == 0) {
                if (reqId) { *reqId = 0; }
                return 0;
            }
            const uint64_t localReqId = nextReqId_.fetch_add(1, std::memory_order_relaxed);
            const auto rc = GdrMemcpyAsyncWithReqId(dst, src, bytes, kind, localReqId);
            if (rc != 0) { return rc; }
            if (reqId) { *reqId = localReqId; }
            return 0;
        } catch (...) {
            return -EIO;
        }
    }

    int GdrMemcpyAsyncWithReqId(void* dst, const void* src, size_t bytes, GdrCopyKind kind,
                                uint64_t reqId) override
    {
        try {
            return GdrMemcpyAsyncWithReqIdImpl(dst, src, bytes, kind, reqId);
        } catch (...) {
            return -EIO;
        }
    }

    GdrCompletionPollResult PollCompletion(uint64_t* reqId) override
    {
        struct ibv_wc wc{};
        const int polledCompletionCount = ibv_poll_cq(cq_, 1, &wc);
        if (polledCompletionCount < 0) { return GdrCompletionPollResult::Error; }
        if (polledCompletionCount == 0) { return GdrCompletionPollResult::Empty; }
        if (wc.status != IBV_WC_SUCCESS) { return GdrCompletionPollResult::Error; }

        if (reqId) { *reqId = wc.wr_id; }
        if (!ReleaseInflightSlot(wc.wr_id)) { return GdrCompletionPollResult::UnknownRequest; }
        inflightWr_.fetch_sub(1, std::memory_order_acq_rel);
        return GdrCompletionPollResult::Completed;
    }

    int RequestCompletionNotification() override
    {
        if (ibv_req_notify_cq(cq_, 0) != 0) { return -EIO; }
        return 0;
    }

    int WaitForCompletionEvent() override
    {
        struct pollfd fds[2]{};
        fds[0].fd = compChannel_->fd;
        fds[0].events = POLLIN;
        fds[1].fd = completionWakeupFd_;
        fds[1].events = POLLIN;

        for (;;) {
            const int rc = poll(fds, 2, -1);
            if (rc < 0) {
                if (errno == EINTR) { continue; }
                return -EIO;
            }

            if (fds[1].revents & POLLIN) {
                ClearCompletionWakeup();
                return -ECANCELED;
            }
            if (fds[1].revents & (POLLERR | POLLNVAL)) { return -EIO; }

            if (fds[0].revents & POLLIN) {
                struct ibv_cq* eventCq = nullptr;
                void* eventContext = nullptr;
                if (ibv_get_cq_event(compChannel_, &eventCq, &eventContext) != 0) { return -EIO; }
                (void)eventContext;
                if (!eventCq) { return -EIO; }
                ibv_ack_cq_events(eventCq, 1);
                if (eventCq != cq_) { return -EIO; }
                return 0;
            }
            if (fds[0].revents & (POLLERR | POLLNVAL | POLLHUP)) { return -EIO; }
        }
    }

    void InterruptCompletionWait() override
    {
        if (completionWakeupFd_ < 0) { return; }

        const uint64_t wakeup = 1;
        const auto bytes = write(completionWakeupFd_, &wakeup, sizeof(wakeup));
        if (bytes < 0 && errno != EAGAIN) {
            UC_WARN("failed to interrupt GDR completion wait: errno({})", errno);
        }
    }

private:
    void Cleanup() noexcept
    {
        ClearRegisteredMrRanges(hostMrRanges_);
        ClearRegisteredMrRanges(gpuMrRanges_);
        for (size_t i = 0; i < kInflightRingCapacity; ++i) {
            ReleaseInflightFallbackMrs(inflightSlots_[i]);
            inflightSlots_[i].reqId.store(0, std::memory_order_relaxed);
        }
        if (qp_) {
            ibv_destroy_qp(qp_);
            qp_ = nullptr;
        }
        if (cq_) {
            ibv_destroy_cq(cq_);
            cq_ = nullptr;
        }
        if (compChannel_) {
            ibv_destroy_comp_channel(compChannel_);
            compChannel_ = nullptr;
        }
        if (completionWakeupFd_ >= 0) {
            close(completionWakeupFd_);
            completionWakeupFd_ = -1;
        }
        if (pd_) {
            ibv_dealloc_pd(pd_);
            pd_ = nullptr;
        }
        if (ctx_) {
            ibv_close_device(ctx_);
            ctx_ = nullptr;
        }
    }

    MrRef GetGpuMr(uint64_t addr, size_t len)
    {
        uint64_t mrAddr = addr;
        size_t mrLen = len;
        if (UC::Trans::DeviceBufferRegistry::Resolve(reinterpret_cast<void*>(addr), len, &mrAddr,
                                                     &mrLen)) {
            if (auto* mr = FindRegisteredMrExact(gpuMrRanges_, mrAddr, mrLen)) {
                return MrRef{mr, false};
            }
            RegisterDeviceBufferLocked(mrAddr, mrLen);
            if (auto* mr = FindRegisteredMrExact(gpuMrRanges_, mrAddr, mrLen)) {
                return MrRef{mr, false};
            }
        }
        UC_WARN("GPU MR cache miss at addr(0x{:x}) size({}); fallback to per-transfer MR.", addr,
                len);
        const int flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;
        auto* mr = ibv_reg_mr(pd_, reinterpret_cast<void*>(addr), len, flags);
        if (!mr) { throw std::runtime_error("ibv_reg_mr on GPU memory failed"); }
        return MrRef{mr, true};
    }

    MrRef GetHostMr(uint64_t addr, size_t len)
    {
        uint64_t mrAddr = addr;
        size_t mrLen = len;
        if (UC::Trans::HostBufferRegistry::Resolve(reinterpret_cast<void*>(addr), len, &mrAddr,
                                                   &mrLen)) {
            if (auto* mr = FindRegisteredMrExact(hostMrRanges_, mrAddr, mrLen)) {
                return MrRef{mr, false};
            }
            RegisterHostBufferLocked(mrAddr, mrLen);
            if (auto* mr = FindRegisteredMrExact(hostMrRanges_, mrAddr, mrLen)) {
                return MrRef{mr, false};
            }
        }
        UC_WARN("Host MR cache miss at addr(0x{:x}) size({}); fallback to per-transfer MR.", addr,
                len);
        const int flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;
        auto* mr = ibv_reg_mr(pd_, reinterpret_cast<void*>(addr), len, flags);
        if (!mr) { throw std::runtime_error("ibv_reg_mr on host memory failed"); }
        return MrRef{mr, true};
    }

    void RegisterHostBufferLocked(uint64_t addr, size_t len)
    {
        if (FindRegisteredMrExact(hostMrRanges_, addr, len)) { return; }
        const int flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;
        auto* mr = ibv_reg_mr(pd_, reinterpret_cast<void*>(addr), len, flags);
        if (!mr) { throw std::runtime_error("ibv_reg_mr on host memory failed"); }
        hostMrRanges_.push_back({addr, len, mr});
    }

    void RegisterDeviceBufferLocked(uint64_t addr, size_t len)
    {
        if (FindRegisteredMrExact(gpuMrRanges_, addr, len)) { return; }
        const int flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;
        auto* mr = ibv_reg_mr(pd_, reinterpret_cast<void*>(addr), len, flags);
        if (!mr) { throw std::runtime_error("ibv_reg_mr on GPU memory failed"); }
        gpuMrRanges_.push_back({addr, len, mr});
    }

    bool TryGetRegisteredGpuMr(uint64_t addr, size_t len, MrRef* out)
    {
        auto* mr = FindRegisteredMrRange(gpuMrRanges_, addr, len);
        if (!mr) { return false; }
        *out = MrRef{mr, false};
        return true;
    }

    bool TryGetRegisteredHostMr(uint64_t addr, size_t len, MrRef* out)
    {
        auto* mr = FindRegisteredMrRange(hostMrRanges_, addr, len);
        if (!mr) { return false; }
        *out = MrRef{mr, false};
        return true;
    }

    int GdrMemcpyAsyncWithReqIdImpl(void* dst, const void* src, size_t bytes, GdrCopyKind kind,
                                    uint64_t reqId)
    {
        if (!dst || !src) { return -EINVAL; }
        if (bytes == 0) { return 0; }
        if (reqId == 0) { return -EINVAL; }
        if (kind != GdrMemcpyHostToDevice && kind != GdrMemcpyDeviceToHost) { return -EINVAL; }
        if (bytes > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) { return -E2BIG; }

        MrRef gpuMr{};
        MrRef hostMr{};
        if (kind == GdrMemcpyHostToDevice) {
            if (TryGetRegisteredGpuMr(reinterpret_cast<uint64_t>(dst), bytes, &gpuMr) &&
                TryGetRegisteredHostMr(reinterpret_cast<uint64_t>(src), bytes, &hostMr)) {
                return PostCopyWithInflightSlot(dst, src, bytes, kind, reqId, hostMr, gpuMr);
            }
        } else {
            if (TryGetRegisteredHostMr(reinterpret_cast<uint64_t>(dst), bytes, &hostMr) &&
                TryGetRegisteredGpuMr(reinterpret_cast<uint64_t>(src), bytes, &gpuMr)) {
                return PostCopyWithInflightSlot(dst, src, bytes, kind, reqId, hostMr, gpuMr);
            }
        }

        std::lock_guard<std::mutex> lock{mutex_};
        try {
            if (kind == GdrMemcpyHostToDevice) {
                gpuMr = GetGpuMr(reinterpret_cast<uint64_t>(dst), bytes);
                hostMr = GetHostMr(reinterpret_cast<uint64_t>(src), bytes);
            } else {
                hostMr = GetHostMr(reinterpret_cast<uint64_t>(dst), bytes);
                gpuMr = GetGpuMr(reinterpret_cast<uint64_t>(src), bytes);
            }
        } catch (...) {
            ReleaseOwnedMr(hostMr);
            ReleaseOwnedMr(gpuMr);
            throw;
        }
        return PostCopyWithInflightSlot(dst, src, bytes, kind, reqId, hostMr, gpuMr);
    }

    int PostCopyWithInflightSlot(void* dst, const void* src, size_t bytes, GdrCopyKind kind,
                                 uint64_t reqId, MrRef hostMr, MrRef gpuMr)
    {
        if (!gpuMr.mr || !hostMr.mr) {
            ReleaseOwnedMr(hostMr);
            ReleaseOwnedMr(gpuMr);
            return -EIO;
        }

        const int oldInflight = inflightWr_.fetch_add(1, std::memory_order_acq_rel);
        if (oldInflight + 1 > maxInflightWr_) {
            inflightWr_.fetch_sub(1, std::memory_order_acq_rel);
            ReleaseOwnedMr(hostMr);
            ReleaseOwnedMr(gpuMr);
            return -EAGAIN;
        }

        auto& slot = inflightSlots_[reqId & kInflightRingMask];
        if (slot.reqId.load(std::memory_order_acquire) != 0) {
            inflightWr_.fetch_sub(1, std::memory_order_acq_rel);
            ReleaseOwnedMr(hostMr);
            ReleaseOwnedMr(gpuMr);
            return -EAGAIN;
        }
        slot.fallbackHostMr = hostMr.owned ? hostMr.mr : nullptr;
        slot.fallbackGpuMr = gpuMr.owned ? gpuMr.mr : nullptr;
        slot.reqId.store(reqId, std::memory_order_release);

        int rc = 0;
        if (kind == GdrMemcpyHostToDevice) {
            rc = PostWrite(reinterpret_cast<uint64_t>(dst), gpuMr.mr->rkey,
                           reinterpret_cast<uint64_t>(src), hostMr.mr->lkey, bytes, reqId);
        } else {
            rc = PostRead(reinterpret_cast<uint64_t>(dst), hostMr.mr->lkey,
                          reinterpret_cast<uint64_t>(src), gpuMr.mr->rkey, bytes, reqId);
        }
        if (rc != 0) {
            ReleaseInflightFallbackMrs(slot);
            slot.reqId.store(0, std::memory_order_release);
            inflightWr_.fetch_sub(1, std::memory_order_acq_rel);
            return rc;
        }
        return 0;
    }

    static void ReleaseOwnedMr(const MrRef& mr)
    {
        if (mr.owned && mr.mr) { ibv_dereg_mr(mr.mr); }
    }

    struct InflightSlot {
        std::atomic<uint64_t> reqId{0};
        struct ibv_mr* fallbackHostMr{nullptr};
        struct ibv_mr* fallbackGpuMr{nullptr};
    };

    static void ReleaseInflightFallbackMrs(InflightSlot& slot)
    {
        if (slot.fallbackHostMr) {
            ibv_dereg_mr(slot.fallbackHostMr);
            slot.fallbackHostMr = nullptr;
        }
        if (slot.fallbackGpuMr) {
            ibv_dereg_mr(slot.fallbackGpuMr);
            slot.fallbackGpuMr = nullptr;
        }
    }

    bool ReleaseInflightSlot(uint64_t reqId)
    {
        auto& slot = inflightSlots_[reqId & kInflightRingMask];
        if (slot.reqId.load(std::memory_order_acquire) != reqId) { return false; }
        ReleaseInflightFallbackMrs(slot);
        slot.reqId.store(0, std::memory_order_release);
        return true;
    }

    void ClearCompletionWakeup()
    {
        if (completionWakeupFd_ < 0) { return; }

        uint64_t wakeup = 0;
        for (;;) {
            const auto bytes = read(completionWakeupFd_, &wakeup, sizeof(wakeup));
            if (bytes == static_cast<ssize_t>(sizeof(wakeup))) { continue; }
            if (bytes < 0 && errno == EINTR) { continue; }
            break;
        }
    }

    int PostWrite(uint64_t remoteAddr, uint32_t rkey, uint64_t localAddr, uint32_t lkey,
                  size_t bytes, uint64_t reqId)
    {
        struct ibv_sge sge{};
        sge.addr = localAddr;
        sge.length = static_cast<uint32_t>(bytes);
        sge.lkey = lkey;

        struct ibv_send_wr wr{};
        wr.wr_id = reqId;
        wr.opcode = IBV_WR_RDMA_WRITE;
        wr.sg_list = &sge;
        wr.num_sge = 1;
        wr.send_flags = IBV_SEND_SIGNALED;
        wr.wr.rdma.remote_addr = remoteAddr;
        wr.wr.rdma.rkey = rkey;

        struct ibv_send_wr* bad = nullptr;
        if (ibv_post_send(qp_, &wr, &bad) != 0) { return -EIO; }
        return 0;
    }

    int PostRead(uint64_t localAddr, uint32_t lkey, uint64_t remoteAddr, uint32_t rkey,
                 size_t bytes, uint64_t reqId)
    {
        struct ibv_sge sge{};
        sge.addr = localAddr;
        sge.length = static_cast<uint32_t>(bytes);
        sge.lkey = lkey;

        struct ibv_send_wr wr{};
        wr.wr_id = reqId;
        wr.opcode = IBV_WR_RDMA_READ;
        wr.sg_list = &sge;
        wr.num_sge = 1;
        wr.send_flags = IBV_SEND_SIGNALED;
        wr.wr.rdma.remote_addr = remoteAddr;
        wr.wr.rdma.rkey = rkey;

        struct ibv_send_wr* bad = nullptr;
        if (ibv_post_send(qp_, &wr, &bad) != 0) { return -EIO; }
        return 0;
    }

private:
    struct ibv_context* ctx_{nullptr};
    struct ibv_pd* pd_{nullptr};
    struct ibv_comp_channel* compChannel_{nullptr};
    struct ibv_cq* cq_{nullptr};
    struct ibv_qp* qp_{nullptr};
    std::vector<RegisteredMrRange> gpuMrRanges_;
    std::vector<RegisteredMrRange> hostMrRanges_;
    int gpuId_{-1};
    std::string nicName_;
    bool isRoce_{false};
    int maxInflightWr_{1};
    std::atomic<int> inflightWr_{0};
    int completionWakeupFd_{-1};
    std::atomic<uint64_t> nextReqId_{1};
    InflightSlot inflightSlots_[kInflightRingCapacity];
    std::mutex mutex_;
};

std::mutex gChannelMutex;
std::vector<std::weak_ptr<GdrCopyChannelImpl>> gChannels;

template <class Fn>
void ForEachLiveChannel(Fn&& fn)
{
    std::lock_guard<std::mutex> lock{gChannelMutex};
    auto out = gChannels.begin();
    for (auto it = gChannels.begin(); it != gChannels.end(); ++it) {
        if (auto channel = it->lock()) {
            fn(*channel);
            *out++ = *it;
        }
    }
    gChannels.erase(out, gChannels.end());
}

}  // namespace

std::shared_ptr<GdrCopyChannel> GdrCopyLib::Open(int gpuId, const std::string& nicName)
{
    if (nicName.empty()) { throw std::runtime_error("empty RDMA nic name"); }
    auto channel = std::make_shared<GdrCopyChannelImpl>(gpuId, nicName);
    {
        std::lock_guard<std::mutex> lock{gChannelMutex};
        gChannels.emplace_back(channel);
    }
    return channel;
}

void GdrCopyLib::RegisterHostBuffer(void* host, size_t size)
{
    UC::Trans::HostBufferRegistry::Register(host, size);
    ForEachLiveChannel([host, size](GdrCopyChannelImpl& channel) {
        try {
            channel.RegisterHostBuffer(reinterpret_cast<uint64_t>(host), size);
        } catch (...) {
        }
    });
}

void GdrCopyLib::UnregisterHostBuffer(void* host)
{
    UC::Trans::HostBufferInfo info{};
    const bool found = UC::Trans::HostBufferRegistry::Lookup(host, &info);
    if (found) {
        ForEachLiveChannel([&info](GdrCopyChannelImpl& channel) {
            channel.UnregisterHostBuffer(info.addr, info.size);
        });
    }
    UC::Trans::HostBufferRegistry::Unregister(host);
}

UC::Status GdrCopyLib::RegisterDeviceBuffer(void* device, size_t size)
{
    if (!device || size == 0) {
        return UC::Status::InvalidParam("invalid device buffer({},{})", fmt::ptr(device), size);
    }
    UC::Trans::DeviceBufferRegistry::Register(device, size);
    UC::Status status = UC::Status::OK();
    std::vector<std::shared_ptr<GdrCopyChannelImpl>> registeredChannels;
    {
        std::lock_guard<std::mutex> lock{gChannelMutex};
        auto out = gChannels.begin();
        for (auto it = gChannels.begin(); it != gChannels.end(); ++it) {
            auto channel = it->lock();
            if (!channel) { continue; }
            try {
                channel->RegisterDeviceBuffer(reinterpret_cast<uint64_t>(device), size);
                registeredChannels.push_back(channel);
            } catch (const std::exception& e) {
                if (status.Success()) {
                    status = UC::Status::OsApiError(
                        fmt::format("failed to register device MR at addr(0x{:x}) size({}): {}",
                                    reinterpret_cast<uint64_t>(device), size, e.what()));
                }
            }
            *out++ = *it;
        }
        gChannels.erase(out, gChannels.end());
    }
    if (status.Success()) { return status; }
    for (auto& channel : registeredChannels) {
        channel->UnregisterDeviceBuffer(reinterpret_cast<uint64_t>(device), size);
    }
    UC::Trans::DeviceBufferRegistry::Unregister(device);
    return status;
}

void GdrCopyLib::UnregisterDeviceBuffer(void* device)
{
    UC::Trans::DeviceBufferInfo info{};
    const bool found = UC::Trans::DeviceBufferRegistry::Lookup(device, &info);
    if (found) {
        ForEachLiveChannel([&info](GdrCopyChannelImpl& channel) {
            channel.UnregisterDeviceBuffer(info.addr, info.size);
        });
    }
    UC::Trans::DeviceBufferRegistry::Unregister(device);
}
