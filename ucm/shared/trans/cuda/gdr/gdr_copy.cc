#include "gdr_copy.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <limits>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>
#include <infiniband/verbs.h>

#include "gdr_buffer_registry.h"

namespace {

constexpr int kIbvPort = 1;
constexpr int kTargetCqDepth = 4096;
constexpr int kTargetSendWr = 4096;

struct MRKey {
    uint64_t addr;
    size_t len;

    bool operator==(const MRKey& other) const noexcept
    {
        return addr == other.addr && len == other.len;
    }
};

struct MRKeyHash {
    size_t operator()(const MRKey& key) const noexcept
    {
        size_t hash = 14695981039346656037ULL;
        auto mix = [&hash](uint64_t value) {
            for (int i = 0; i < 8; ++i) {
                hash ^= (value & 0xff);
                hash *= 1099511628211ULL;
                value >>= 8;
            }
        };
        mix(key.addr);
        mix(static_cast<uint64_t>(key.len));
        return hash;
    }
};

using RegisteredMrTable = std::unordered_map<MRKey, struct ibv_mr*, MRKeyHash>;

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
    struct ibv_port_attr portAttr {};
    (void)ibv_query_port(ctx, kIbvPort, &portAttr);
    endpoint.qpn = qp->qp_num;
    endpoint.lid = portAttr.lid;
    (void)ibv_query_gid(ctx, kIbvPort, 0, reinterpret_cast<union ibv_gid*>(endpoint.gid));
    return endpoint;
}

void ConnectRcQp(struct ibv_qp* qp, const Endpoint& remote, bool isRoce)
{
    {
        struct ibv_qp_attr attr {};
        attr.qp_state = IBV_QPS_INIT;
        attr.pkey_index = 0;
        attr.port_num = kIbvPort;
        attr.qp_access_flags =
            IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE;
        if (ibv_modify_qp(qp, &attr, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT |
                                        IBV_QP_ACCESS_FLAGS) != 0) {
            throw std::runtime_error("failed to move RC QP to INIT");
        }
    }
    {
        struct ibv_qp_attr attr {};
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
        if (ibv_modify_qp(qp, &attr, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU |
                                        IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                                        IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER) !=
            0) {
            throw std::runtime_error("failed to move RC QP to RTR");
        }
    }
    {
        struct ibv_qp_attr attr {};
        attr.qp_state = IBV_QPS_RTS;
        attr.timeout = 14;
        attr.retry_cnt = 7;
        attr.rnr_retry = 7;
        attr.sq_psn = 0;
        attr.max_rd_atomic = 1;
        if (ibv_modify_qp(qp, &attr, IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
                                        IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN |
                                        IBV_QP_MAX_QP_RD_ATOMIC) != 0) {
            throw std::runtime_error("failed to move RC QP to RTS");
        }
    }
}

struct ibv_mr* FindRegisteredMr(RegisteredMrTable& table, uint64_t addr, size_t len)
{
    const auto it = table.find(MRKey{addr, len});
    if (it == table.end()) { return nullptr; }
    return it->second;
}

struct ibv_mr* ReleaseRegisteredMr(RegisteredMrTable& table, uint64_t addr, size_t len)
{
    const auto it = table.find(MRKey{addr, len});
    if (it == table.end()) { return nullptr; }
    auto* mr = it->second;
    table.erase(it);
    return mr;
}

template <class Fn>
void ClearRegisteredMrs(RegisteredMrTable& table, Fn&& fn)
{
    for (auto& [key, mr] : table) {
        (void)key;
        fn(mr);
    }
    table.clear();
}

class GdrCopyChannelImpl : public GdrCopyChannel {
public:
    GdrCopyChannelImpl(int gpuId, std::string nicName)
        : gpuId_{gpuId},
          nicName_{std::move(nicName)}
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

            struct ibv_port_attr portAttr {};
            if (ibv_query_port(ctx_, kIbvPort, &portAttr) != 0) {
                throw std::runtime_error("ibv_query_port failed");
            }
            isRoce_ = portAttr.lid == 0;

            pd_ = ibv_alloc_pd(ctx_);
            if (!pd_) { throw std::runtime_error("ibv_alloc_pd failed"); }
            for (const auto& buffer : UC::Trans::HostBufferRegistry::Snapshot()) {
                RegisterHostBuffer(buffer.addr, buffer.size);
            }
            for (const auto& buffer : UC::Trans::DeviceBufferRegistry::Snapshot()) {
                RegisterDeviceBuffer(buffer.addr, buffer.size);
            }

            struct ibv_device_attr deviceAttr {};
            if (ibv_query_device(ctx_, &deviceAttr) != 0) {
                throw std::runtime_error("ibv_query_device failed");
            }

            const int cqDepth =
                std::max(1, std::min(kTargetCqDepth, static_cast<int>(deviceAttr.max_cqe)));
            cq_ = ibv_create_cq(ctx_, cqDepth, nullptr, nullptr, 0);
            if (!cq_) { throw std::runtime_error("ibv_create_cq failed"); }

            struct ibv_qp_init_attr initAttr {};
            initAttr.send_cq = cq_;
            initAttr.recv_cq = cq_;
            initAttr.cap.max_send_wr = std::max(
                1, std::min(kTargetSendWr, static_cast<int>(deviceAttr.max_qp_wr == 0
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
        auto* mr = ReleaseRegisteredMr(hostMrs_, addr, len);
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
        auto* mr = ReleaseRegisteredMr(gpuMrs_, addr, len);
        if (mr) { ibv_dereg_mr(mr); }
    }

    int GdrMemcpyAsync(void* dst, const void* src, size_t bytes, GdrCopyKind kind,
                       uint64_t* reqId) override
    {
        try {
            if (!dst || !src) { return -EINVAL; }
            if (bytes == 0) {
                if (reqId) { *reqId = 0; }
                return 0;
            }
            if (kind != GdrMemcpyHostToDevice && kind != GdrMemcpyDeviceToHost) {
                return -EINVAL;
            }
            if (bytes > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) { return -E2BIG; }

            std::lock_guard<std::mutex> lock{mutex_};
            if (inflightWr_ + 1 > maxInflightWr_) { return -EAGAIN; }

            MrRef gpuMr{};
            MrRef hostMr{};
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
            if (!gpuMr.mr || !hostMr.mr) {
                ReleaseOwnedMr(hostMr);
                ReleaseOwnedMr(gpuMr);
                return -EIO;
            }

            const uint64_t localReqId = nextReqId_++;
            pendingReqs_.insert(localReqId);
            inflightWr_ += 1;
            if (hostMr.owned) { pendingReqMrs_[localReqId].push_back(hostMr.mr); }
            if (gpuMr.owned) { pendingReqMrs_[localReqId].push_back(gpuMr.mr); }

            int rc = 0;
            if (kind == GdrMemcpyHostToDevice) {
                rc = PostWrite(reinterpret_cast<uint64_t>(dst), gpuMr.mr->rkey,
                               reinterpret_cast<uint64_t>(src), hostMr.mr->lkey, bytes,
                               localReqId);
            } else {
                rc = PostRead(reinterpret_cast<uint64_t>(dst), hostMr.mr->lkey,
                              reinterpret_cast<uint64_t>(src), gpuMr.mr->rkey, bytes,
                              localReqId);
            }
            if (rc != 0) {
                CleanupReqFallbackMrs(localReqId);
                pendingReqs_.erase(localReqId);
                inflightWr_ = std::max(0, inflightWr_ - 1);
                return rc;
            }

            if (reqId) { *reqId = localReqId; }
            return 0;
        } catch (...) {
            return -EIO;
        }
    }

    int PollCompletion(uint64_t* reqId) override
    {
        std::lock_guard<std::mutex> lock{mutex_};
        struct ibv_wc wc {};
        const int n = ibv_poll_cq(cq_, 1, &wc);
        if (n < 0) { return -EIO; }
        if (n == 0) { return -EAGAIN; }
        if (wc.status != IBV_WC_SUCCESS) { return -EIO; }

        inflightWr_ = std::max(0, inflightWr_ - 1);
        const auto it = pendingReqs_.find(wc.wr_id);
        if (it == pendingReqs_.end()) { return -ENOENT; }

        if (reqId) { *reqId = wc.wr_id; }
        CleanupReqFallbackMrs(wc.wr_id);
        pendingReqs_.erase(it);
        return 0;
    }

private:
    void Cleanup() noexcept
    {
        ClearRegisteredMrs(hostMrs_, [](struct ibv_mr* mr) {
            if (mr) { ibv_dereg_mr(mr); }
        });
        ClearRegisteredMrs(gpuMrs_, [](struct ibv_mr* mr) {
            if (mr) { ibv_dereg_mr(mr); }
        });
        for (auto& [reqId, mrs] : pendingReqMrs_) {
            (void)reqId;
            for (auto* mr : mrs) {
                if (mr) { ibv_dereg_mr(mr); }
            }
        }
        pendingReqMrs_.clear();
        if (qp_) {
            ibv_destroy_qp(qp_);
            qp_ = nullptr;
        }
        if (cq_) {
            ibv_destroy_cq(cq_);
            cq_ = nullptr;
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
            if (auto* mr = FindRegisteredMr(gpuMrs_, mrAddr, mrLen)) { return MrRef{mr, false}; }
            RegisterDeviceBufferLocked(mrAddr, mrLen);
            if (auto* mr = FindRegisteredMr(gpuMrs_, mrAddr, mrLen)) { return MrRef{mr, false}; }
        }
        const int flags =
            IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;
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
            if (auto* mr = FindRegisteredMr(hostMrs_, mrAddr, mrLen)) { return MrRef{mr, false}; }
            RegisterHostBufferLocked(mrAddr, mrLen);
            if (auto* mr = FindRegisteredMr(hostMrs_, mrAddr, mrLen)) { return MrRef{mr, false}; }
        }
        const int flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;
        auto* mr = ibv_reg_mr(pd_, reinterpret_cast<void*>(addr), len, flags);
        if (!mr) { throw std::runtime_error("ibv_reg_mr on host memory failed"); }
        return MrRef{mr, true};
    }

    void RegisterHostBufferLocked(uint64_t addr, size_t len)
    {
        if (FindRegisteredMr(hostMrs_, addr, len)) { return; }
        const int flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;
        auto* mr = ibv_reg_mr(pd_, reinterpret_cast<void*>(addr), len, flags);
        if (!mr) { throw std::runtime_error("ibv_reg_mr on host memory failed"); }
        hostMrs_[MRKey{addr, len}] = mr;
    }

    void RegisterDeviceBufferLocked(uint64_t addr, size_t len)
    {
        if (FindRegisteredMr(gpuMrs_, addr, len)) { return; }
        const int flags =
            IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;
        auto* mr = ibv_reg_mr(pd_, reinterpret_cast<void*>(addr), len, flags);
        if (!mr) { throw std::runtime_error("ibv_reg_mr on GPU memory failed"); }
        gpuMrs_[MRKey{addr, len}] = mr;
    }

    static void ReleaseOwnedMr(const MrRef& mr)
    {
        if (mr.owned && mr.mr) { ibv_dereg_mr(mr.mr); }
    }

    void CleanupReqFallbackMrs(uint64_t reqId)
    {
        const auto it = pendingReqMrs_.find(reqId);
        if (it == pendingReqMrs_.end()) { return; }
        for (auto* mr : it->second) {
            if (mr) { ibv_dereg_mr(mr); }
        }
        pendingReqMrs_.erase(it);
    }

    int PostWrite(uint64_t remoteAddr, uint32_t rkey, uint64_t localAddr, uint32_t lkey,
                  size_t bytes, uint64_t reqId)
    {
        struct ibv_sge sge {};
        sge.addr = localAddr;
        sge.length = static_cast<uint32_t>(bytes);
        sge.lkey = lkey;

        struct ibv_send_wr wr {};
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
        struct ibv_sge sge {};
        sge.addr = localAddr;
        sge.length = static_cast<uint32_t>(bytes);
        sge.lkey = lkey;

        struct ibv_send_wr wr {};
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
    struct ibv_cq* cq_{nullptr};
    struct ibv_qp* qp_{nullptr};
    RegisteredMrTable gpuMrs_;
    RegisteredMrTable hostMrs_;
    int gpuId_{-1};
    std::string nicName_;
    bool isRoce_{false};
    int maxInflightWr_{1};
    int inflightWr_{0};
    uint64_t nextReqId_{1};
    std::unordered_set<uint64_t> pendingReqs_;
    std::unordered_map<uint64_t, std::vector<struct ibv_mr*>> pendingReqMrs_;
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

void GdrCopyLib::RegisterDeviceBuffer(void* device, size_t size)
{
    UC::Trans::DeviceBufferRegistry::Register(device, size);
    ForEachLiveChannel([device, size](GdrCopyChannelImpl& channel) {
        try {
            channel.RegisterDeviceBuffer(reinterpret_cast<uint64_t>(device), size);
        } catch (...) {
        }
    });
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
