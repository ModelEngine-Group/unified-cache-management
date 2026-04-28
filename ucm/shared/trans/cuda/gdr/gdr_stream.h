#ifndef UNIFIEDCACHE_TRANS_GDR_STREAM_H
#define UNIFIEDCACHE_TRANS_GDR_STREAM_H

#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#include <cuda_runtime_api.h>
#include "gdr_copy.h"
#include "trans/stream.h"

namespace UC::Trans {

class GdrStream : public Stream {
public:
    ~GdrStream() override;

    Status Setup() override;

    Status DeviceToHost(void* device, void* host, size_t size) override;
    Status DeviceToHost(void* device[], void* host[], size_t size, size_t number) override;
    Status DeviceToHost(void* device[], void* host, size_t size, size_t number) override;
    Status DeviceToHostAsync(void* device, void* host, size_t size) override;
    Status DeviceToHostAsync(void* device[], void* host[], size_t size, size_t number) override;
    Status DeviceToHostAsync(void* device[], void* host, size_t size, size_t number) override;

    Status HostToDevice(void* host, void* device, size_t size) override;
    Status HostToDevice(void* host[], void* device[], size_t size, size_t number) override;
    Status HostToDevice(void* host, void* device[], size_t size, size_t number) override;
    Status HostToDeviceAsync(void* host, void* device, size_t size) override;
    Status HostToDeviceAsync(void* host[], void* device[], size_t size, size_t number) override;
    Status HostToDeviceAsync(void* host, void* device[], size_t size, size_t number) override;

    Status AppendCallback(std::function<void(bool)> cb) override;
    Status Synchronized() override;
    Status WaitEvent(void* event) override;

private:
    struct BarrierOp {
        uint64_t seq;
        cudaEvent_t event;
    };

    struct DeferredCopy {
        uint64_t seq;
        void* dst;
        const void* src;
        size_t size;
        GdrCopyKind kind;
    };

    Status SubmitAsync(void* dst, const void* src, size_t size, GdrCopyKind kind);
    Status AdvanceLocked(uint64_t targetSeq);
    Status PollOneCompletionLocked();
    bool HasPendingBarrierLocked(uint64_t targetSeq) const;
    bool HasDeferredCopyLocked(uint64_t targetSeq) const;
    bool HasInflightLocked(uint64_t targetSeq) const;
    bool HasOutstandingLocked(uint64_t targetSeq) const;
    void ClearPendingLocked();

private:
    std::shared_ptr<GdrCopyChannel> channel_{nullptr};
    std::mutex mutex_;
    std::deque<BarrierOp> pendingBarriers_;
    std::deque<DeferredCopy> deferredCopies_;
    std::unordered_map<uint64_t, uint64_t> inflightReqSeq_;
    uint64_t nextSeq_{1};
    std::string nicName_{"mlx5_0"};
    bool useOdp_{false};
};

}  // namespace UC::Trans

#endif
