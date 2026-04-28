#include "gdr_stream.h"

#include <cerrno>
#include <cstdlib>
#include <limits>
#include <string>
#include <thread>

#include <cuda_runtime.h>

#include "logger/logger.h"

namespace {

bool ParseBoolEnv(const char* name, bool defaultValue)
{
    const auto* value = std::getenv(name);
    if (!value || value[0] == '\0') { return defaultValue; }
    const std::string str{value};
    return str == "1" || str == "true" || str == "TRUE" || str == "on" || str == "ON";
}

std::string ParseStringEnv(const char* name, const char* defaultValue)
{
    const auto* value = std::getenv(name);
    if (!value || value[0] == '\0') { return defaultValue; }
    return value;
}

UC::Status MakeGdrStatus(const char* op, int rc)
{
    return UC::Status::OsApiError(fmt::format("{} failed({})", op, rc));
}

}  // namespace

namespace UC::Trans {

GdrStream::~GdrStream()
{
    if (channel_) { (void)Synchronized(); }
}

Status GdrStream::Setup()
{
    nicName_ = ParseStringEnv("UCM_GDR_NIC_NAME", "mlx5_0");
    useOdp_ = ParseBoolEnv("UCM_GDR_USE_ODP", false);

    int deviceId = -1;
    const auto ret = cudaGetDevice(&deviceId);
    if (ret != cudaSuccess) { return Status{ret, cudaGetErrorString(ret)}; }

    try {
        channel_ = GdrCopyLib::Open(deviceId, nicName_, useOdp_);
        UC_INFO("Enable GDR stream on device({}) with nic({}), use_odp={}.", deviceId, nicName_,
                useOdp_);
        return Status::OK();
    } catch (const std::exception& e) {
        return Status::OsApiError(
            fmt::format("failed to open GDR channel on device({}) with nic({}): {}", deviceId,
                        nicName_, e.what()));
    }
}

Status GdrStream::DeviceToHost(void* device, void* host, size_t size)
{
    auto status = DeviceToHostAsync(device, host, size);
    if (status.Failure()) { return status; }
    return Synchronized();
}

Status GdrStream::DeviceToHost(void* device[], void* host[], size_t size, size_t number)
{
    auto status = DeviceToHostAsync(device, host, size, number);
    if (status.Failure()) { return status; }
    return Synchronized();
}

Status GdrStream::DeviceToHost(void* device[], void* host, size_t size, size_t number)
{
    auto status = DeviceToHostAsync(device, host, size, number);
    if (status.Failure()) { return status; }
    return Synchronized();
}

Status GdrStream::DeviceToHostAsync(void* device, void* host, size_t size)
{
    return SubmitAsync(host, device, size, GdrMemcpyDeviceToHost);
}

Status GdrStream::DeviceToHostAsync(void* device[], void* host[], size_t size, size_t number)
{
    for (size_t i = 0; i < number; ++i) {
        auto status = DeviceToHostAsync(device[i], host[i], size);
        if (status.Failure()) { return status; }
    }
    return Status::OK();
}

Status GdrStream::DeviceToHostAsync(void* device[], void* host, size_t size, size_t number)
{
    for (size_t i = 0; i < number; ++i) {
        auto* pHost = static_cast<void*>(static_cast<int8_t*>(host) + size * i);
        auto status = DeviceToHostAsync(device[i], pHost, size);
        if (status.Failure()) { return status; }
    }
    return Status::OK();
}

Status GdrStream::HostToDevice(void* host, void* device, size_t size)
{
    auto status = HostToDeviceAsync(host, device, size);
    if (status.Failure()) { return status; }
    return Synchronized();
}

Status GdrStream::HostToDevice(void* host[], void* device[], size_t size, size_t number)
{
    auto status = HostToDeviceAsync(host, device, size, number);
    if (status.Failure()) { return status; }
    return Synchronized();
}

Status GdrStream::HostToDevice(void* host, void* device[], size_t size, size_t number)
{
    auto status = HostToDeviceAsync(host, device, size, number);
    if (status.Failure()) { return status; }
    return Synchronized();
}

Status GdrStream::HostToDeviceAsync(void* host, void* device, size_t size)
{
    return SubmitAsync(device, host, size, GdrMemcpyHostToDevice);
}

Status GdrStream::HostToDeviceAsync(void* host[], void* device[], size_t size, size_t number)
{
    for (size_t i = 0; i < number; ++i) {
        auto status = HostToDeviceAsync(host[i], device[i], size);
        if (status.Failure()) { return status; }
    }
    return Status::OK();
}

Status GdrStream::HostToDeviceAsync(void* host, void* device[], size_t size, size_t number)
{
    for (size_t i = 0; i < number; ++i) {
        auto* pHost = static_cast<void*>(static_cast<int8_t*>(host) + size * i);
        auto status = HostToDeviceAsync(pHost, device[i], size);
        if (status.Failure()) { return status; }
    }
    return Status::OK();
}

Status GdrStream::AppendCallback(std::function<void(bool)> cb)
{
    (void)cb;
    return Status::OK();
}

Status GdrStream::Synchronized()
{
    const uint64_t targetSeq = [&]() {
        std::lock_guard<std::mutex> lock{mutex_};
        return nextSeq_ - 1;
    }();
    for (;;) {
        bool hasInflight = false;
        bool done = false;
        {
            std::lock_guard<std::mutex> lock{mutex_};
            auto status = AdvanceLocked(targetSeq);
            if (status.Failure()) { return status; }
            done = !HasOutstandingLocked(targetSeq);
            hasInflight = HasInflightLocked(targetSeq);
        }
        if (done) { return Status::OK(); }

        if (hasInflight) {
            {
                std::lock_guard<std::mutex> lock{mutex_};
                auto status = PollOneCompletionLocked();
                if (status.Failure()) { return status; }
            }
            std::this_thread::yield();
        } else {
            std::this_thread::yield();
        }
    }
}

Status GdrStream::WaitEvent(void* event)
{
    if (!event) { return Status::OK(); }
    std::lock_guard<std::mutex> lock{mutex_};
    pendingBarriers_.push_back(BarrierOp{nextSeq_++, static_cast<cudaEvent_t>(event)});
    return Status::OK();
}

Status GdrStream::SubmitAsync(void* dst, const void* src, size_t size, GdrCopyKind kind)
{
    if (!channel_) { return Status::Error("GDR channel is not ready"); }

    std::lock_guard<std::mutex> lock{mutex_};
    const uint64_t seq = nextSeq_++;
    const DeferredCopy op{seq, dst, src, size, kind};

    if (!pendingBarriers_.empty() || !deferredCopies_.empty()) {
        deferredCopies_.push_back(op);
        return AdvanceLocked(std::numeric_limits<uint64_t>::max());
    }

    uint64_t reqId = 0;
    auto rc = channel_->GdrMemcpyAsync(dst, src, size, kind, &reqId);
    if (rc == 0) {
        if (reqId != 0) { inflightReqSeq_[reqId] = seq; }
        return Status::OK();
    }
    if (rc != -EAGAIN) { return MakeGdrStatus("GdrMemcpyAsync", rc); }

    auto status = PollOneCompletionLocked();
    if (status.Failure()) { return status; }

    reqId = 0;
    rc = channel_->GdrMemcpyAsync(dst, src, size, kind, &reqId);
    if (rc == 0) {
        if (reqId != 0) { inflightReqSeq_[reqId] = seq; }
        return Status::OK();
    }
    if (rc != -EAGAIN) { return MakeGdrStatus("GdrMemcpyAsync", rc); }

    deferredCopies_.push_back(op);
    return AdvanceLocked(std::numeric_limits<uint64_t>::max());
}

Status GdrStream::AdvanceLocked(uint64_t targetSeq)
{
    while (true) {
        const bool hasBarrier =
            !pendingBarriers_.empty() && pendingBarriers_.front().seq <= targetSeq;
        const bool hasDeferred =
            !deferredCopies_.empty() && deferredCopies_.front().seq <= targetSeq;

        if (!hasBarrier && !hasDeferred) { return Status::OK(); }

        const bool processBarrier =
            hasBarrier && (!hasDeferred || pendingBarriers_.front().seq < deferredCopies_.front().seq);
        if (processBarrier) {
            const auto ret = cudaEventQuery(pendingBarriers_.front().event);
            if (ret == cudaSuccess) {
                pendingBarriers_.pop_front();
                continue;
            }
            if (ret == cudaErrorNotReady) { return Status::OK(); }
            ClearPendingLocked();
            return Status{ret, cudaGetErrorString(ret)};
        }

        const auto& op = deferredCopies_.front();
        uint64_t reqId = 0;
        const auto rc = channel_->GdrMemcpyAsync(op.dst, op.src, op.size, op.kind, &reqId);
        if (rc == 0) {
            if (reqId != 0) { inflightReqSeq_[reqId] = op.seq; }
            deferredCopies_.pop_front();
            continue;
        }
        if (rc == -EAGAIN) {
            auto status = PollOneCompletionLocked();
            if (status.Failure()) { return status; }
            return Status::OK();
        }

        ClearPendingLocked();
        return MakeGdrStatus("GdrMemcpyAsync", rc);
    }
}

Status GdrStream::PollOneCompletionLocked()
{
    uint64_t reqId = 0;
    const auto rc = channel_->PollCompletion(&reqId);
    if (rc == 0) {
        inflightReqSeq_.erase(reqId);
        return Status::OK();
    }
    if (rc == -EAGAIN) { return Status::OK(); }

    ClearPendingLocked();
    return MakeGdrStatus("PollCompletion", rc);
}

bool GdrStream::HasPendingBarrierLocked(uint64_t targetSeq) const
{
    return !pendingBarriers_.empty() && pendingBarriers_.front().seq <= targetSeq;
}

bool GdrStream::HasDeferredCopyLocked(uint64_t targetSeq) const
{
    return !deferredCopies_.empty() && deferredCopies_.front().seq <= targetSeq;
}

bool GdrStream::HasInflightLocked(uint64_t targetSeq) const
{
    for (const auto& [reqId, seq] : inflightReqSeq_) {
        (void)reqId;
        if (seq <= targetSeq) { return true; }
    }
    return false;
}

bool GdrStream::HasOutstandingLocked(uint64_t targetSeq) const
{
    return HasPendingBarrierLocked(targetSeq) || HasDeferredCopyLocked(targetSeq) ||
           HasInflightLocked(targetSeq);
}

void GdrStream::ClearPendingLocked()
{
    pendingBarriers_.clear();
    deferredCopies_.clear();
    inflightReqSeq_.clear();
}

}  // namespace UC::Trans
